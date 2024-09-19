# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_obs, 120, device=device),
            nn.ReLU(),
            nn.Linear(120, 84, device=device),
            nn.ReLU(),
            nn.Linear(84, n_act, device=device),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int):
    slope = (end_e - start_e) / duration
    slope = torch.tensor(slope, device=device)
    while True:
        yield slope.clamp_min(end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    wandb.init(
        project="dqn",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    n_act = envs.single_action_space.n
    n_obs = math.prod(envs.single_observation_space.shape)

    q_network = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
    q_network_detach = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
    params_vals = from_module(q_network).detach()
    params_vals.to_module(q_network_detach)

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, capturable=args.cudagraphs and not args.compile)

    target_network = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
    target_params = params_vals.clone().lock_()
    target_params.to_module(target_network)

    def update(data):
        with torch.no_grad():
            target_max, _ = target_network(data["next_observations"]).max(dim=1)
            td_target = data["rewards"].flatten() + args.gamma * target_max * (~data["dones"].flatten()).float()
        old_val = q_network(data["observations"]).gather(1, data["actions"].unsqueeze(-1)).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach()

    def policy(obs, epsilon):
        q_values = q_network_detach(obs)
        actions = torch.argmax(q_values, dim=1)
        actions_random = torch.rand(actions.shape, device=actions.device).mul(n_act).floor().to(torch.long)
        # actions_random = torch.randint_like(actions, n_act)
        use_policy = torch.rand(actions.shape, device=actions.device).gt(epsilon)
        return torch.where(use_policy, actions, actions_random)

    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update = torch.compile(update, mode=mode)
        policy = torch.compile(policy, mode=mode, fullgraph=True)

    if args.cudagraphs:
        update = CudaGraphModule(update)
        policy = CudaGraphModule(policy)

    start_time = None

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    eps_schedule = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps)
    avg_returns = deque(maxlen=20)

    pbar = tqdm.tqdm(range(args.total_timesteps))
    transitions = []
    for global_step in pbar:
        if global_step == args.learning_starts + args.measure_burnin:
            start_time = time.time()
            global_step_start = global_step

        # ALGO LOGIC: put action logic here
        epsilon = next(eps_schedule)
        actions = policy(obs, epsilon)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    avg_returns.append(info["episode"]["r"])
            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean()}"

        next_obs = torch.as_tensor(next_obs, dtype=torch.float).to(device, non_blocking=True)
        terminations = torch.as_tensor(terminations, dtype=torch.bool).to(device, non_blocking=True)
        rewards = torch.as_tensor(rewards, dtype=torch.float).to(device, non_blocking=True)

        real_next_obs = None
        for idx, trunc in enumerate(truncations):
            if trunc:
                if real_next_obs is None:
                    real_next_obs = next_obs.clone()
                real_next_obs[idx] = torch.as_tensor(infos["final_observation"][idx], device=device, dtype=torch.float)
        if real_next_obs is None:
            real_next_obs = next_obs
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
        transitions.append(
            TensorDict._new_unsafe(
                observations=obs,
                next_observations=real_next_obs,
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                dones=terminations,
                batch_size=obs.shape[:1],
                device=device,
            )
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                rb.extend(torch.cat(transitions))
                transitions = []
                data = rb.sample(args.batch_size)
                loss = update(data)
            # update target network
            if global_step % args.target_network_frequency == 0:
                target_params.lerp_(params_vals, args.tau)

        if global_step % 100 == 0 and start_time is not None:
            speed = (global_step - global_step_start) / (time.time() - start_time)
            pbar.set_description(f"speed: {speed: 4.2f} sps, " f"epsilon: {epsilon.cpu().item(): 4.2f}, " + desc)
            with torch.no_grad():
                logs = {
                    "episode_return": torch.tensor(avg_returns).mean(),
                    "loss": loss.mean(),
                    "epsilon": epsilon,
                }
            wandb.log(
                {
                    "speed": speed,
                    **logs,
                },
                step=global_step,
            )

    envs.close()
