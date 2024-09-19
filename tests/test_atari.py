import subprocess


def test_ppo():
    subprocess.run(
        "python leanrl/ppo_atari.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_envpool():
    subprocess.run(
        "python leanrl/ppo_atari_envpool.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_torchcompile():
    subprocess.run(
        "python leanrl/ppo_atari_envpool_torchcompile.py --num-envs 1 --num-steps 64 --total-timesteps 256 --compile --cudagraphs",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax():
    subprocess.run(
        "python leanrl/ppo_atari_envpool_xla_jax.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )
