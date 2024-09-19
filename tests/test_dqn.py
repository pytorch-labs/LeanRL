import subprocess


def test_dqn():
    subprocess.run(
        "python leanrl/dqn.py --num-envs 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_dqn_jax():
    subprocess.run(
        "python leanrl/dqn_jax.py --num-envs 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_dqn_torchcompile():
    subprocess.run(
        "python leanrl/dqn_torchcompile.py --num-envs 1 --total-timesteps 256 --compile --cudagraphs",
        shell=True,
        check=True,
    )
