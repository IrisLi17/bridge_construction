import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_algorithms import logger
from vec_env.base_vec_env import VecEnv
from torch_algorithms.storage import RolloutStorage
from collections import deque
from utils.math_utils import safe_mean
import numpy as np


class PPO(object):
    def __init__(self, env, policy: nn.Module, device="cpu", n_steps=1024, nminibatches=32, noptepochs=10,
                 gamma=0.99, lam=0.95, learning_rate=2.5e-4, cliprange=0.2, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, eps=1e-5, use_gae=True, use_clipped_value_loss=True, use_linear_lr_decay=False,
                 use_linear_clip_decay=False, recompute_adv=False):
        self.env = env
        self.policy = policy
        self.device = device
        self.n_steps = n_steps
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cur_cliprange = cliprange
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_linear_clip_decay = use_linear_clip_decay
        self.recompute_adv = recompute_adv

        if isinstance(self.env, VecEnv):
            self.n_envs = self.env.num_envs
        else:
            self.n_envs = 1

        self.rollouts = RolloutStorage(self.n_steps, self.n_envs,
                                       self.env.observation_space.shape, self.env.action_space,
                                       self.policy.recurrent_hidden_state_size)

        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=eps)

    def learn(self, total_timesteps, callback=None):
        obs = self.env.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        episode_rewards = deque(maxlen=100)
        ep_infos = deque(maxlen=100)
        if "FetchBridge" in self.env.get_attr("spec")[0].id:
            detailed_sr = [deque(maxlen=100) for _ in range(self.env.get_attr("num_blocks")[0])]
        else:
            detailed_sr = []
        self.num_timesteps = 0
        loss_names = ["value_loss", "policy_loss", "entropy", "grad_norm", "param_norm"]

        start = time.time()
        num_updates = int(total_timesteps) // self.n_steps // self.n_envs
        for j in range(num_updates):
            if callable(callback):
                callback(locals(), globals())
            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                update_linear_schedule(self.optimizer, j, num_updates, self.learning_rate)
            if self.use_linear_clip_decay:
                self.cur_cliprange = update_linear_clip(j, num_updates, self.cliprange)

            for step in range(self.n_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.policy.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = self.env.step(action)
                self.num_timesteps += self.n_envs

                for info in infos:
                    maybe_ep_info = info.get('episode')
                    if maybe_ep_info is not None:
                        ep_infos.append(maybe_ep_info)
                        episode_rewards.append(maybe_ep_info['r'])
                        maybe_cur_num_blocks = info.get('cur_num_objects')
                        if maybe_cur_num_blocks is not None:
                            detailed_sr[int(maybe_cur_num_blocks) - 1].append(info['is_success'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.policy.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.lam)

            losses = self.update()

            self.rollouts.after_update()
            fps = int(self.num_timesteps / (time.time() - start))
            current_hard_ratio = np.nan
            current_max_blocks = np.nan
            if "FetchBridge" in self.env.get_attr("spec")[0].id:
                current_hard_ratio = self.env.env_method('get_hard_ratio')[0]
                current_max_blocks = self.env.get_attr('cur_max_blocks')[0]
            # TODO: log
            logger.logkv("serial_timesteps", j * self.n_steps)
            logger.logkv("n_updates", j)
            logger.logkv("total_timesteps", self.num_timesteps)
            logger.logkv("fps", fps)
            if len(ep_infos) > 0 and len(ep_infos[0]) > 0:
                logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_infos]))
                logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_infos]))
                if "is_success" in ep_infos[0]:
                    logger.logkv('success_rate', safe_mean([ep_info['is_success'] for ep_info in ep_infos]))
            logger.logkv('time_elapsed', time.time() - start)
            for idx, loss_name in enumerate(loss_names):
                logger.logkv(loss_name, losses[idx])
            logger.logkv('hard_ratio', current_hard_ratio)
            logger.logkv('cur_max_blocks', current_max_blocks)
            for i in range(len(detailed_sr)):
                logger.logkv('%d_success_rate' % i, safe_mean(detailed_sr[i]))
            logger.dumpkvs()

    def update(self):
        if not self.recompute_adv:
            advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = []
        action_loss_epoch = []
        dist_entropy_epoch = []
        total_grad_norm_epoch = []
        total_param_norm_epoch = []

        for e in range(self.noptepochs):
            if self.recompute_adv:
                with torch.no_grad():
                    rec_values = self.policy.get_value(
                        self.rollouts.obs.view(-1, *self.rollouts.obs.size()[2:]),
                        self.rollouts.recurrent_hidden_states, self.rollouts.masks).detach()
                    rec_values = torch.reshape(rec_values, [-1, self.n_envs, 1])
                self.rollouts.value_preds.copy_(rec_values)

                self.rollouts.compute_returns(rec_values[-1], self.use_gae, self.gamma, self.lam)
                advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
                advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-5)

            if self.policy.is_recurrent:
                data_generator = self.rollouts.recurrent_generator(
                    advantages, self.nminibatches)
            else:
                data_generator = self.rollouts.feed_forward_generator(
                    advantages, self.nminibatches)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, *_ = sample
                # # TODO: debug. see what kind of observation gets big advantage
                # with torch.no_grad():
                #     adv_sorted_idx = torch.argsort(adv_targ.squeeze())
                #     print(adv_sorted_idx)
                #     cliff_distances = obs_batch[adv_sorted_idx, -2] - obs_batch[adv_sorted_idx, -3]
                #     time_observation = obs_batch[adv_sorted_idx, -1]
                #     import matplotlib.pyplot as plt
                #     plt.plot(cliff_distances.squeeze().numpy(), label='cliff distance')
                #     plt.plot(value_preds_batch[adv_sorted_idx].squeeze().numpy(), label='value preds')
                #     plt.plot(return_batch[adv_sorted_idx].squeeze().numpy(), label='returns')
                #     # plt.plot(adv_targ[adv_sorted_idx].squeeze().numpy(), label='adv')
                #     plt.plot(time_observation.squeeze().numpy() / 20, label='time observation')
                #     plt.legend()
                #     plt.show()

                # Reshape to do in a single forward pass for all steps
                action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.cur_cliprange,
                                    1.0 + self.cur_cliprange) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                values = self.policy.get_value(obs_batch, recurrent_hidden_states_batch, masks_batch)
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.cur_cliprange, self.cur_cliprange)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.vf_coef + action_loss -
                 dist_entropy * self.ent_coef).backward()
                total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # print('total norm', total_norm)
                self.optimizer.step()
                params = list(filter(lambda p: p[1].grad is not None, self.policy.named_parameters()))
                param_norm = torch.norm(
                    torch.stack([torch.norm(p[1].detach().to(self.device)) for p in params]))

                value_loss_epoch.append(value_loss.detach().item())
                action_loss_epoch.append(action_loss.detach().item())
                dist_entropy_epoch.append(dist_entropy.detach().item())
                total_grad_norm_epoch.append(total_norm.detach().item())
                total_param_norm_epoch.append(param_norm.detach().item())

        value_loss_epoch, action_loss_epoch, dist_entropy_epoch, total_grad_norm_epoch, total_param_norm_epoch = \
            map(safe_mean, [value_loss_epoch, action_loss_epoch, dist_entropy_epoch, total_grad_norm_epoch,
                            total_param_norm_epoch])

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, total_grad_norm_epoch, total_param_norm_epoch

    def save(self, save_path):
        torch.save({'policy': self.policy.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, save_path)

    def load(self, load_pth, eval=True):
        checkpoint = torch.load(load_pth, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if eval:
            self.policy.eval()
        else:
            self.policy.train()


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_linear_clip(epoch, total_num_epochs, initial_clip):
    cur_clip = initial_clip - (initial_clip * (epoch / float(total_num_epochs)))
    return cur_clip
