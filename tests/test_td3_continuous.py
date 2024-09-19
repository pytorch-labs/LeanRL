import subprocess


def test_td3_continuous_action():
    subprocess.run(
        "python leanrl/td3_continuous_action.py --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_td3_continuous_action_jax():
    subprocess.run(
        "python leanrl/td3_continuous_action_jax.py --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_td3_continuous_action_torchcompile():
    subprocess.run(
        "python leanrl/td3_continuous_action_torchcompile.py --total-timesteps 256 --compile --cudagraphs",
        shell=True,
        check=True,
    )
