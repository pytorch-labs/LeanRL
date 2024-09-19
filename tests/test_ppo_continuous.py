import subprocess


def test_ppo_continuous_action():
    subprocess.run(
        "python leanrl/ppo_continuous_action.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_continuous_action_torchcompile():
    subprocess.run(
        "python leanrl/ppo_continuous_action_torchcompile.py --num-envs 1 --num-steps 64 --total-timesteps 256 --compile --cudagraphs",
        shell=True,
        check=True,
    )
