#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    leanrl/ppo_continuous_action.py
    leanrl/ppo_continuous_action_torchcompile.py
    leanrl/dqn.py
    leanrl/dqn_jax.py
    leanrl/dqn_torchcompile.py
    leanrl/td3_continuous_action_jax.py
    leanrl/td3_continuous_action.py
    leanrl/td3_continuous_action_torchcompile.py
    leanrl/ppo_atari_envpool.py
    leanrl/ppo_atari_envpool_torchcompile.py
    leanrl/ppo_atari_envpool_xla_jax.py
    leanrl/sac_continuous_action.py
    leanrl/sac_continuous_action_torchcompile.py
)
for script in "${scripts[@]}"; do
    for seed in 21 31 41; do
        if [[ $script == *_torchcompile.py ]]; then
            python $script --seed=$seed --cudagraphs
            python $script --seed=$seed --cudagraphs --compile
            python $script --seed=$seed --compile
            python $script --seed=$seed
        else
            python $script --seed=$seed
        fi
    done
done
