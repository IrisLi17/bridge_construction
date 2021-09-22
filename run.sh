#!/bin/bash
python run.py --env_id FetchBridgeBullet7Blocks-v1 --no_adaptive_number --algo ppg --policy_arch shared --exclude_time --random_size \
              --reward_type onestep --num_workers 4 --num_timesteps 2e7 --gamma 0.97 --noptepochs 10 --action_scale 0.6 \
              --num_bin 64 --rotation_low -0.5 --friction_low 0.5 --friction_high 0.5 --aux_freq 1 --bc_coef 1 --ewma 0.995 --inf_horizon \
              --restart_rate 0.0 --priority_type td --priority_decay 0.0 --noop --clip_priority --auxiliary_task inverse_dynamics \
              --auxiliary_coef 0.1 --force_scale 0 --robot xarm --primitive --n_steps 128