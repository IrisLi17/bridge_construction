import os
import gym
import re
import numpy as np
# from env.bridge_construction import FetchBridgeConstructionHighEnv
# from env.bridge_construction_scratch import FetchBridgeConstructionFromScratchHighEnv
from env.bridge_construction_bullet import BulletBridgeConstructionHigh
from utils.monitor import Monitor
from utils.wrapper import DoneWhenSuccessWrapper, RewardScaleWrapper
from torch_algorithms import logger


# ENTRY_POINT = {
#     'FetchBridge3Blocks-v0': FetchBridgeConstructionHighEnv,
# }


def make_env(env_id, rank=0, seed=0, max_episode_steps=100, log_dir=None, done_when_success=False, use_monitor=False,
             reward_scale=1.0, bonus_weight=0.0, env_kwargs=None):
    if env_id not in gym.envs.registry.env_specs:
        if re.match("FetchBridgeBullet\d+Blocks", env_id) is not None:
            entry_point = BulletBridgeConstructionHigh
        else:
            raise NotImplementedError
        gym.register(env_id, entry_point=entry_point, max_episode_steps=max_episode_steps, kwargs=env_kwargs)
    env = gym.make(env_id)
    env.seed(seed + rank)
    if done_when_success:
        env = DoneWhenSuccessWrapper(env)
    env = RewardScaleWrapper(env, scale=reward_scale, bias=0.0, bonus_weight=bonus_weight)
    if use_monitor:
        env = Monitor(env, os.path.join(log_dir, "%d.monitor.csv" % rank), info_keywords=("is_success",))
    return env


def get_env_kwargs(env_id, horizon=50, random_size=False, min_num_blocks=3, discrete_height=False, random_mode="split",
                   action_scale=0.4, restart_rate=0., noop=False, robot=None, force_scale=0, adaptive_primitive=False):
    if re.match("FetchBridge\d+Blocks", env_id) is not None or re.match("FetchBridgeBullet\d+Blocks", env_id) is not None:
        if re.match("FetchBridge\d+Blocks", env_id):
            num_blocks = int(re.search("(?<=FetchBridge)(\d+)", env_id).group(0))
            _v = int(env_id.split("-v")[1])
            # -v1: thin block, large cliff
            # -v2: fat block, large cliff
            # -v3: thin block, thin cliff
            thickness = 0.05 if _v == 2 else 0.025
            cliff_thickness = 0.025 if _v == 3 else 0.5
        else:
            num_blocks = int(re.search("(?<=FetchBridgeBullet)(\d+)", env_id).group(0))
            thickness = 0.025
            cliff_thickness = 0.1
        cliff_height = 0.1
        env_kwargs = dict(max_episode_steps=horizon, num_blocks=num_blocks,
                          random_size=random_size,
                          min_num_blocks=min_num_blocks,
                          discrete=discrete_height,
                          random_mode=random_mode, action_scale=action_scale, block_thickness=thickness,
                          restart_rate=restart_rate, cliff_thickness=cliff_thickness, cliff_height=cliff_height,
                          noop=noop, robot=robot,
                          force_scale=force_scale,
                          adaptive_primitive=adaptive_primitive)
    # elif env_id == "FetchBridge3Blocks-v0":
    #     # TODO: parse number of objects
    #     env_kwargs = dict(max_episode_steps=horizon, num_blocks=3, stable_reward_coef=stable_reward_coef,
    #                       rotation_penalty_coef=rotation_penalty_coef, height_coef=height_coef,
    #                       reward_type=reward_type, include_size=include_size)
    else:
        raise NotImplementedError
    return env_kwargs


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)