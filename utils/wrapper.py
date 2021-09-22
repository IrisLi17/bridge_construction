import gym
from gym.spaces import Box
import numpy as np
import torch
from vec_env.base_vec_env import VecEnvWrapper


class DoneWhenSuccessWrapper(gym.Wrapper):
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if info['is_success']:
            done = True
        return next_obs, reward, done, info


class FlattenDictWrapper(gym.ObservationWrapper):
    def __init__(self, env, key_list):
        super(FlattenDictWrapper, self).__init__(env)
        self.key_list = key_list
        spaces = [env.observation_space[k] for k in self.key_list]
        self.observation_space = Box(low=np.concatenate([s.low for s in spaces]),
                                     high=np.concatenate([s.high for s in spaces]))

    def observation(self, observation):
        return np.concatenate([observation[k] for k in self.key_list])


class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env, scale=1.0, bias=0.0, bonus_weight=1.0):
        super(RewardScaleWrapper, self).__init__(env)
        self.scale = scale
        self.bias = bias
        self.bonus_weight = bonus_weight

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        reward = self.scale * (reward + self.bias) + self.bonus_weight * info['is_success']
        return next_obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        r, success = self.env.reward_and_success(achieved_goal, desired_goal, info)
        return self.scale * (r + self.bias) + self.bonus_weight * success

    def reward_and_success(self, achieved_goal, desired_goal, info):
        r, success = self.env.reward_and_success(achieved_goal, desired_goal, info)
        return self.scale * (r + self.bias) + self.bonus_weight * success, success


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        # self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        # obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        # actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        # obs = torch.from_numpy(obs).float().to(self.device)
        # reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        reward = np.expand_dims(reward, axis=1).astype(np.float32)
        return obs, reward, done, info
