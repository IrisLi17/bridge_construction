#!/bin/bash
python run.py --env_id FetchBridgeBullet7Blocks-v1 --algo ppg --policy_arch shared --random_size \
              --num_workers 64 --num_timesteps 2e7 --noptepochs 10 --action_scale 0.6 \
              --aux_freq 1 --ewma 0.995 --inf_horizon \
              --restart_rate 0.5 --priority_type td --priority_decay 0.0 --noop --clip_priority --auxiliary_task inverse_dynamics \
              --force_scale 0 --robot xarm