import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_algorithms import logger
from vec_env.base_vec_env import VecEnv
from torch_algorithms.storage import RolloutStorage, NpRolloutStorage
from collections import deque
from utils.math_utils import safe_mean
import numpy as np
from torch_algorithms.policies import CNNPolicy, MLPPolicy, HybridAttentionPolicy, MultiDiscreteAttentionPolicy
from torch_algorithms.policies.distributions import FixedCategorical, FixedNormal
from copy import deepcopy
from utils.replay_buffer import PrioritizedReplayBuffer, PriorityQueue
from utils.evaluation import get_success_rate
from env.bullet_rotations import quat_rot_vec, euler2quat


class EwmaModel(nn.Module):  # TODO: should inherit policy
    """
    An EWMA-lagged copy of a PpoModel.
    """

    def __init__(self, model, ewma_decay, exp_update=False):
        super().__init__()
        self.model = model
        self.ewma_decay = ewma_decay
        self.model_ewma = deepcopy(model)
        self.exp_update = exp_update
        if self.exp_update:
            self.total_weight = 1

    def evaluate_actions(self, *args, **kwargs):
        with torch.no_grad():
            return self.model_ewma.evaluate_actions(*args, **kwargs)

    def forward_policy(self, *args, **kwargs):
        with torch.no_grad():
            return self.model_ewma.forward_policy(*args, **kwargs)

    def update(self, decay=None):
        if decay is None:
            decay = self.ewma_decay
        if self.exp_update:
            new_total_weight = decay * self.total_weight + 1
            decayed_weight_ratio = decay * self.total_weight / new_total_weight
            with torch.no_grad():
                for p, p_ewma in zip(self.model.parameters(), self.model_ewma.parameters()):
                    p_ewma.data.mul_(decayed_weight_ratio).add_(p.data / new_total_weight)
            self.total_weight = new_total_weight
        else:
            # Momemtum-like update
            with torch.no_grad():
                for p, p_ewma in zip(self.model.parameters(), self.model_ewma.parameters()):
                    p_ewma.data.mul_(decay).add_(p.data * (1 - decay))

    def reset(self):
        self.update(decay=0)


class PPO_dev(object):
    def __init__(self, env, policy: nn.Module, device="cpu", n_steps=1024, nminibatches=32, noptepochs=10,
                 gamma=0.99, lam=0.95, learning_rate=2.5e-4, cliprange=0.2, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, eps=1e-5, use_gae=True, use_clipped_value_loss=True, use_linear_lr_decay=False,
                 bc_coef=1., n_vf_rollout=4, nvfepochs=10, ewma_decay=None, kl_beta=0., use_linear_clip_decay=False,
                 auxiliary_task=None, aux_coef=0.0, exp_update=False, inf_horizon=False, eval_env=None,
                 priority_type=None, optimizer="adam", manual_filter_state=False,
                 state_replay_size=100_000, filter_priority=0, nvfminibatches=32,
                 priority_decay=0.0, clip_priority=False):
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        if ewma_decay is not None:
            self.policy_ewma = EwmaModel(policy, ewma_decay, exp_update=exp_update)
        else:
            self.policy_ewma = None
        self.device = device
        self.n_steps = n_steps
        self.nminibatches = nminibatches
        self.nvfminibatches = nvfminibatches
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
        self.bc_coef = bc_coef
        self.n_vf_rollout = n_vf_rollout
        self.nvfepochs = nvfepochs
        self.kl_beta = kl_beta
        self.inf_horizon = inf_horizon
        if auxiliary_task == "skyline":
            from torch_algorithms.aux_task.skyline_prediction import SkylinePrediction
            self.auxiliary_task = SkylinePrediction(self.policy, env.get_attr("skyline_dim")[0])
        elif auxiliary_task == "next_skyline":
            from torch_algorithms.aux_task.skyline_prediction import NextSkylinePrediction
            self.auxiliary_task = NextSkylinePrediction(self.policy, env.action_space.shape[0], env.get_attr("skyline_dim")[0])
        elif auxiliary_task == "next_state":
            from torch_algorithms.aux_task.skyline_prediction import NextStatePrediction
            self.auxiliary_task = NextStatePrediction(self.policy, env.action_space.shape[0],
                                                      env.observation_space.shape[0], task="classification")
        elif auxiliary_task == "inverse_dynamics":
            from torch_algorithms.aux_task.inverse_dynamics_prediction import InverseDynamicsPrediction
            self.auxiliary_task = InverseDynamicsPrediction(self.policy, env.action_space.shape[0],
                                                            env.observation_space.shape[0])
        elif auxiliary_task is None:
            self.auxiliary_task = None
        self.auxiliary_task_name = auxiliary_task
        self.aux_coef = aux_coef
        self.aux_out_shape = (self.auxiliary_task.out_dim,) if self.auxiliary_task is not None else None

        if isinstance(self.env, VecEnv):
            self.n_envs = self.env.num_envs
        else:
            self.n_envs = 1

        self.rollouts = NpRolloutStorage(self.n_steps, self.n_envs,
                                       self.env.observation_space.shape, self.env.action_space,
                                       self.policy.recurrent_hidden_state_size,
                                       self.policy.n_values,
                                       self.aux_out_shape,
                                       n_action_level=2 if self.policy.refined_action or self.policy.bilevel_action else 1)

        # TODO: reshape to support recurrent version
        '''
        self.vf_rollouts = dict(
            obs=torch.zeros(self.n_steps * self.n_vf_rollout, self.n_envs, *self.env.observation_space.shape),
            recurrent_hidden_state=torch.zeros(self.n_steps * self.n_vf_rollout, self.n_envs, self.policy.recurrent_hidden_state_size),
            recurrent_mask=torch.zeros(self.n_steps * self.n_vf_rollout, self.n_envs, 1),
            value_targ=torch.zeros(self.n_steps * self.n_vf_rollout, self.n_envs, self.policy.n_values, 1)
        )
        if self.aux_out_shape is not None:
            self.vf_rollouts['aux_info'] = torch.zeros(self.n_steps * self.n_vf_rollout, self.n_envs, *self.aux_out_shape)
        if self.auxiliary_task_name == "next_skyline" or self.auxiliary_task_name == "next_state":
            self.vf_rollouts['actions'] = torch.zeros(self.n_steps * self.n_vf_rollout, self.n_envs,
                                                      self.env.action_space.shape[0])
        '''
        
        self.vf_rollouts = dict(
            obs=np.zeros((self.n_steps * self.n_vf_rollout, self.n_envs, *self.env.observation_space.shape)),
            recurrent_hidden_state=np.zeros((self.n_steps * self.n_vf_rollout, self.n_envs, self.policy.recurrent_hidden_state_size)),
            recurrent_mask=np.zeros((self.n_steps * self.n_vf_rollout, self.n_envs, 1)),
            value_targ=np.zeros((self.n_steps * self.n_vf_rollout, self.n_envs, self.policy.n_values, 1)),
        )
        if self.aux_out_shape is not None:
            self.vf_rollouts['aux_info'] = np.zeros((self.n_steps * self.n_vf_rollout, self.n_envs, *self.aux_out_shape))
        if self.auxiliary_task_name == "next_skyline" or self.auxiliary_task_name == "next_state" or self.auxiliary_task_name == "inverse_dynamics":
            self.vf_rollouts['actions'] = np.zeros((self.n_steps * self.n_vf_rollout, self.n_envs,
                                                      self.env.action_space.shape[0]))
        if self.auxiliary_task_name == "inverse_dynamics":
            self.vf_rollouts['reset_actions'] = np.zeros((self.n_steps * self.n_vf_rollout, self.n_envs, 1))
        # self.vf_rollouts = dict(
        #     obs=torch.zeros(self.n_steps * self.n_envs * self.n_vf_rollout, *self.env.observation_space.shape),
        #     recurrent_hidden_state=torch.zeros(self.n_steps * self.n_envs * self.n_vf_rollout, self.policy.recurrent_hidden_state_size),
        #     recurrent_mask=torch.zeros(self.n_steps * self.n_envs * self.n_vf_rollout, 1),
        #     value_targ=torch.zeros(self.n_steps * self.n_envs * self.n_vf_rollout, 1))
        # if self.aux_out_shape is not None:
        #     self.vf_rollouts['aux_info'] = torch.zeros(self.n_steps * self.n_envs * self.n_vf_rollout, *self.aux_out_shape)
        # if self.auxiliary_task_name == "next_skyline":
        #     self.vf_rollouts['actions'] = torch.zeros(self.n_steps * self.n_envs * self.n_vf_rollout,
        #                                               self.env.action_space.shape[0])
        opt_keys = ["pi", "vf"]
        if optimizer == "adam":
            self.optimizer = {k: optim.Adam(policy.parameters(), lr=learning_rate, eps=eps) for k in opt_keys}
        else:
            raise NotImplementedError

        # State replay buffer for restart
        # self.state_replay = PrioritizedReplayBuffer(size=300000, alpha=1.0, unique=unique)
        # todo: use hard priority, a priority queue with max length
        self.state_replay = PriorityQueue(size=state_replay_size, decay=priority_decay)  # 0.0 correspond to always new priority
        self.priority_type = priority_type
        self.manual_filter_state = manual_filter_state
        self.filter_priority = filter_priority
        self.clip_priority = clip_priority  # Used for td priority, if true, reward and value are clipped to [0, 1]

    def learn(self, total_timesteps, callback=None):
        obs = self.env.reset()
        last_obs = obs
        if 'FetchBridge' in self.env.get_attr("spec")[0].id:
            state = self.env.env_method("get_state")
            self.rollouts.states[0] = state
        self.rollouts.obs[0] = obs.copy()
        # self.rollouts.to(self.device)
        # for _key in self.vf_rollouts:
        #     self.vf_rollouts[_key] = self.vf_rollouts[_key].to(self.device)

        episode_rewards = deque(maxlen=100)
        ep_infos = deque(maxlen=100)
        low_level_success = 0
        if "FetchBridge" in self.env.get_attr("spec")[0].id:
            detailed_sr = [deque(maxlen=100) for _ in range(self.env.get_attr("num_blocks")[0])]
        else:
            detailed_sr = []
        self.num_timesteps = 0
        loss_names = ["value_loss", "policy_loss", "entropy", "kl_loss",  "aux_task_loss", "pi_grad_norm", "vf_grad_norm",
                      "rec2old_ratio", "cur2rec_ratio", "supervision_action_loss", "priority_mean", "priority_std",
                      "priority_max", "restart_value_mean", "rec2old_max", "cur2rec_max", "clipped_ratio",
                      "adv_max", "adv_min"]

        start = time.time()
        num_updates = int(total_timesteps) // self.n_steps // self.n_envs
    
        for j in range(num_updates):
            # print("Begining of update", get_memory_usage())
            
            if callable(callback):
                callback(locals(), globals())
            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                for k in self.optimizer:
                    update_linear_schedule(self.optimizer[k], j, num_updates, self.learning_rate)
            if self.use_linear_clip_decay:
                self.cur_cliprange = update_linear_clip(j, num_updates, self.cliprange)
            # if self.policy_ewma is not None:
            #     self.policy_ewma.reset()

            # value = torch.from_numpy(np.random.rand(self.n_envs, 1, 1))
            # action = torch.from_numpy(np.zeros((self.n_envs, 4)))
            # action_log_prob = torch.from_numpy(np.random.rand(2, self.n_envs, 1))
            # recurrent_hidden_states = self.rollouts.recurrent_hidden_states[0]
            # print("Before roll out", get_memory_usage())
            
            # # Debugging mujoco state leakage
            # for step in range(100000):
            #     action = np.random.rand(self.n_envs, 4)
            #     action[:, 0] = 1
            #     self.env.step(action)
            #     state_dict = self.env.env_method("get_state")
            #     if not isinstance(state_dict, np.ndarray):
            #         state_dict = np.asarray(state_dict)
            #     self.rollouts.states[step % self.rollouts.states.shape[0]] = state_dict
            #     if step % 50 == 0:
            #         print(get_memory_usage())
            #     if step % self.rollouts.states.shape[0] == self.rollouts.states.shape[0] - 1:
            #         # Looks like del is bad. Just overwrite, the memory keeps stable.
            #         _shape = self.rollouts.states.shape
            #         last_state = self.rollouts.states[-1]
            #         del self.rollouts.states
            #         self.rollouts.states = np.empty(_shape, dtype=object)
            #         self.rollouts.states[0] = last_state
            # exit()
            for step in range(self.n_steps):
                # Sample actions
                # if j > 1:
                #     print("before predict action", get_memory_usage())
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.policy.act(
                        torch.from_numpy(self.rollouts.obs[step]).float().to(self.device), 
                        torch.from_numpy(self.rollouts.recurrent_hidden_states[step]).to(self.device),
                        torch.from_numpy(self.rollouts.masks[step]).to(self.device))
                    # value = torch.rand((self.n_envs, 1, 1))
                    # action = torch.rand((self.n_envs, 4)) * 0
                    # action_log_prob = torch.rand((2, self.n_envs, 1))
                    # recurrent_hidden_states = self.rollouts.recurrent_hidden_states[step]
                # if j > 1:
                #     print("predict action", get_memory_usage())

                # Obser reward and next obs
                obs, reward, done, infos = self.env.step(action.cpu().numpy())
                if "FetchBridge" in self.env.get_attr("spec")[0].id:
                    state_dict = self.env.env_method("get_state")
                    # state_dict = None
                else:
                    state_dict = None
                self.num_timesteps += self.n_envs

                for info in infos:
                    maybe_ep_info = info.get('episode')
                    low_level_success += (info.get('low_level_result', None) == 0)
                    # maybe_debug_action = info.get('debug_supervised_action')
                    if maybe_ep_info is not None:
                        ep_infos.append(maybe_ep_info)
                        episode_rewards.append(maybe_ep_info['r'])
                        maybe_cur_num_blocks = info.get('cur_num_objects')
                        # The trajectories restarting from buffer should not affect the adaptive curriculum
                        restart_from_buffer = info.get("restart_from_buffer", False)
                        if (maybe_cur_num_blocks is not None) and (not restart_from_buffer):
                            detailed_sr[int(maybe_cur_num_blocks) - 1].append(info['is_success'])
                        # if restart_from_buffer:
                        #     # Store two trajectories for self imitation
                        #     self.state_replay.get_history(info["initial_state"])
                        #     self.rollouts.obs, self.rollouts.actions

                aux_infos = None
                if self.auxiliary_task is not None:
                    aux_infos = []
                    if self.auxiliary_task_name == "next_state" or self.auxiliary_task_name == "inverse_dynamics":
                        if np.sum(done) > 0:
                            aux_infos = np.stack([infos[i]['terminal_observation'] for i in range(len(infos))], axis=0)
                            # aux_infos = torch.from_numpy(aux_infos).float().to(self.device)
                        else:
                            aux_infos = obs  # the next observation,
                    else:
                        for info in infos:
                            if self.auxiliary_task_name == "skyline":
                                aux_infos.append(info.get("skyline"))
                            elif self.auxiliary_task_name == "next_skyline":
                                aux_infos.append(info.get("next_skyline"))
                        # aux_infos = torch.FloatTensor(aux_infos)
                        aux_infos = np.stack(aux_infos, axis=0)
                
                # if j > 1:
                #     print("taken action", get_memory_usage())

                # If done then clean the history of observations.
                # masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                masks = np.array([[0.0] if done_ else [1.0] for done_ in done])
                # bad_masks = torch.FloatTensor(
                #     [[0.0] if 'bad_transition' in info.keys() else [1.0]
                #      for info in infos])
                bad_masks = np.array([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                # success_mask = torch.FloatTensor([[info.get("is_success", 0.)] for info in infos])
                success_mask = np.array([[info.get("is_success", 0.)] for info in infos])
                reset_action = np.array([[info.get("out_of_reach", 0.)] for info in infos])

                # Hack to make infinite horizon. Add value of next state into reward of last step.
                # Fixed horizon case. We can do it in a batch.
                if self.inf_horizon and done[0]:
                    for i in range(len(done)):
                        assert done[i]
                    terminal_obs = np.stack([infos[i]["terminal_observation"] for i in range(len(infos))], axis=0)
                    # TODO: recurrent support
                    with torch.no_grad():
                        recurrent_masks = torch.from_numpy(np.ones_like(masks)).to(self.device)
                        _next_values = self.policy.get_value(torch.from_numpy(terminal_obs).float().to(self.device),
                                                             recurrent_hidden_states, recurrent_masks)
                        # reward += self.gamma * _next_values.mean(dim=1).to(reward.device)
                        reward += self.gamma * _next_values.mean(dim=1).cpu().numpy()

                # if j > 1:
                #     print("process mask", get_memory_usage())

                self.rollouts.insert(obs, recurrent_hidden_states.cpu().numpy(), action.cpu().numpy(),
                                action_log_prob.cpu().numpy(), value.cpu().numpy(), reward, masks, bad_masks, aux_infos,
                                     state_dict=state_dict, success_masks=success_mask, reset_action=reset_action)
                
                # if j > 1:
                #     print("inserted", get_memory_usage())
                # deprecated
                # TODO: the first steps in episodes should not be saved
                # cur_obs = obs
                # if done[0]:
                #     cur_obs = torch.from_numpy(np.stack([infos[i]["terminal_observation"] for i in range(len(infos))])).float().to(self.device)
                # states = np.array(self.env.env_method("get_state"))
                # fail_mask = np.array([not info["is_success"] for info in infos])
                # if np.sum(fail_mask) > 0:
                #     fail_idxs = np.where(fail_mask)[0]
                #     # store recurrent state into state_replay
                #     self.state_replay.extend(states[fail_idxs], last_obs[fail_idxs], cur_obs[fail_idxs],
                #                              self.rollouts.recurrent_hidden_states[step], recurrent_hidden_states,
                #                              self.rollouts.masks[step], masks.to(self.device))
                # last_obs = obs

            # print("After roll out", get_memory_usage())
            # objgraph.show_growth()
            
            with torch.no_grad():
                next_value = self.policy.get_value(
                    torch.from_numpy(self.rollouts.obs[-1]).float().to(self.device), self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            self.rollouts.compute_returns(next_value.cpu().numpy(), self.use_gae, self.gamma, self.lam)
            self.rollouts.compute_step_to_go()

            # print("After compute", get_memory_usage())

            # Save (s, value_targ) for aux phase
            # TODO: rewrite shape
            self.vf_rollouts['obs'][
                self.n_steps * (j % self.n_vf_rollout): self.n_steps * (j % self.n_vf_rollout + 1)] = (self.rollouts.obs[:-1]).copy()
            self.vf_rollouts['recurrent_hidden_state'][
                self.n_steps * (j % self.n_vf_rollout): self.n_steps * (j % self.n_vf_rollout + 1)] = (self.rollouts.recurrent_hidden_states[:-1]).copy()
            self.vf_rollouts['recurrent_mask'][
                self.n_steps * (j % self.n_vf_rollout): self.n_steps * (j % self.n_vf_rollout + 1)] = (self.rollouts.masks[:-1]).copy()
            self.vf_rollouts['value_targ'][
                self.n_steps * (j % self.n_vf_rollout): self.n_steps * (j % self.n_vf_rollout + 1)] = (self.rollouts.returns[:-1]).copy()
            if self.auxiliary_task is not None:
                self.vf_rollouts['aux_info'][
                    self.n_steps * (j % self.n_vf_rollout): self.n_steps * (j % self.n_vf_rollout + 1)] = (self.rollouts.aux).copy()
            if "actions" in self.vf_rollouts:
                self.vf_rollouts['actions'][
                    self.n_steps * (j % self.n_vf_rollout): self.n_steps * (j % self.n_vf_rollout + 1)] = (self.rollouts.actions).copy()
            if "reset_actions" in self.vf_rollouts:
                self.vf_rollouts['reset_actions'][
                    self.n_steps * (j % self.n_vf_rollout): self.n_steps * (j % self.n_vf_rollout + 1)] = (self.rollouts.reset_actions).copy()
            self.n_valid_rollouts = np.minimum(j + 1, self.n_vf_rollout)

            losses = self.update(j)

            self.rollouts.after_update()

            # if j > 0:
            #     tensor_count = 0
            #     for obj in gc.get_objects():
            #         try:
            #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #                 tensor_count += 1
            #         except: pass
            #     print('tensor count', tensor_count)
            
            # print("After update", get_memory_usage())

            fps = int(self.num_timesteps / (time.time() - start))
            current_hard_ratio = [np.nan]
            current_max_blocks = np.nan
            if "FetchBridge" in self.env.get_attr("spec")[0].id:
                current_hard_ratio = self.env.env_method('get_hard_ratio')[0]
                current_max_blocks = self.env.get_attr('cur_max_blocks')[0]
                # eval_n_objects = [3, 5, 7]
                eval_n_objects = list(range(3, self.env.get_attr("num_blocks")[0] + 1, 2))
                eval_success_rate = []
                for n_object in eval_n_objects:
                    eval_success_rate.append(get_success_rate(self.eval_env, self.policy, n_object,
                                                              n_episode=self.eval_env.num_envs * 2, device=self.device))
            # print("After eval", get_memory_usage())

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
            logger.logkv('hard_ratio', current_hard_ratio[-1])
            logger.logkv('cur_max_blocks', current_max_blocks)
            for i in range(len(detailed_sr)):
                logger.logkv('%d_success_rate' % i, safe_mean(detailed_sr[i]))
            if "FetchBridge" in self.env.get_attr("spec")[0].id:
                for i, n in enumerate(eval_n_objects):
                    logger.logkv("eval_%d_sr" % n, eval_success_rate[i])
                logger.logkv("low_level_sr", low_level_success / self.num_timesteps)
            logger.dumpkvs()

            # if j > 0:
                # objgraph.show_growth()
                # print('leaking obj', len(objgraph.get_leaking_objects()))  # 1942
                # print(len(objgraph.by_type('Tensor')))  # 40
                # # for ii in range(len(objgraph.by_type('Tensor'))):
                # #     print(objgraph.by_type('Tensor')[ii].shape, 
                #               # objgraph.find_backref_chain(objgraph.by_type('Tensor')[ii], lambda x: isinstance(x, PPO_dev))
                #               # )
                #         # objgraph.show_chain(
                #         #     objgraph.find_backref_chain(
                #         #         objgraph.by_type('Tensor')[ii],
                #         #         objgraph.is_proper_module))
                #     # for ii in range(4):
                #     #     objgraph.show_chain(
                #     #         objgraph.find_backref_chain(
                #     #             objgraph.by_type('list')[ii],
                #     #             objgraph.is_proper_module))
                #     # objgraph.show_chain(
                #     #     objgraph.find_backref_chain(
                #     #         objgraph.by_type('tuple')[0],
                #     #         objgraph.is_proper_module))
                # print("length of list", len(objgraph.by_type("list")))  # 5332
                # for ii in range(min(10, len(objgraph.by_type("list")))):
                #     print(len(objgraph.by_type("list")[ii]))
                # print("length of tuple", len(objgraph.by_type("tuple"))) # 1e4
                # for ii in range(min(len(objgraph.by_type("tuple")), 10)):
                #     print(len(objgraph.by_type("tuple")[ii]))
                # if j == 3:
                #     exit()

    def update(self, n_update):
        # Take the first critic to compute advantage
        advantages = self.rollouts.returns[:-1, :, 0] - self.rollouts.value_preds[:-1, :, 0]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        debug_priority_mean, debug_priority_max, debug_priority_std = np.nan, np.nan, np.nan
        restart_value_mean = np.nan
        if self.priority_type is not None:
            # Insert states into state_replay with priority
            cur_value = self.rollouts.value_preds[:-2].copy()
            next_value = self.rollouts.value_preds[1: -1].copy()
            # TODO:replace cur_value and next_value with those predicted by priority_head
            rewards = self.rollouts.rewards[:-1].copy()
            masks = self.rollouts.masks[1: -1].copy()
            # for aux_error
            obs = self.rollouts.obs[:-2].copy()
            next_obs = self.rollouts.obs[1:-1].copy()
            actions = self.rollouts.actions[:-1].copy()
            reset_actions = self.rollouts.reset_actions[:-1].copy()

            priority = self.compute_priority(rewards, cur_value, next_value, masks, obs, next_obs, actions, reset_actions)
            # history_starts = compute_history_start(self.rollouts.masks)
            
            priority = np.maximum(priority, 1e-5)
            debug_priority_mean = np.mean(priority)
            debug_priority_std = np.std(priority)
            debug_priority_max = np.max(priority)
            # self.state_replay.update_priorities(idxes, np.maximum(priority, 1e-5))

            priority_threshold = np.sort(priority.reshape(-1))[int(self.filter_priority * len(priority.reshape(-1)))]

            # print("In ppo, after priority", get_memory_usage())

            cliff_height = self.env.env_method("get_cliff_height")[0]
            block_thickness = self.env.env_method("get_block_thickness")[0]
            object_dim =self.env.get_attr("object_dim")[0]
            for i in range(rewards.shape[0]):
                # only store unsuccessful ones
                # TODO: only store entries with top 1/8 priorities?
                valid_mask = (1 - self.rollouts.success[i]).squeeze(axis=-1).astype(np.bool)
                not_terminate_mask = self.rollouts.masks[i + 1].squeeze(axis=-1).astype(np.bool)
                valid_mask = np.logical_and(valid_mask, not_terminate_mask)
                priority_mask = priority[i] >= priority_threshold
                if self.manual_filter_state:
                    n_obj = self.env.get_attr('num_blocks')[0]
                    # valid_mask = torch.logical_and(valid_mask, manual_validation_fn(self.rollouts.obs[i], n_obj))
                    valid_mask = np.logical_and(valid_mask, manual_validation_fn(
                        self.rollouts.obs[i], n_obj, cliff_height, block_thickness, object_dim))
                else:
                    # valid_mask = torch.logical_and(valid_mask, priority_mask)
                    valid_mask = np.logical_and(valid_mask, priority_mask)
                    # pass
                if np.sum(valid_mask) > 0:
                    self.state_replay.extend(
                        self.rollouts.states[i][valid_mask], self.rollouts.obs[i + 1][valid_mask],
                        self.rollouts.obs[i][valid_mask], self.rollouts.rewards[i][valid_mask], [n_update] * int(np.sum(valid_mask)),
                        self.rollouts.recurrent_hidden_states[i + 1][valid_mask], self.rollouts.recurrent_hidden_states[i][valid_mask],
                        self.rollouts.masks[i + 1][valid_mask], self.rollouts.masks[i][valid_mask], priority[i][valid_mask],
                        self.rollouts.step_to_go[i][valid_mask], self.rollouts.actions[i][valid_mask],
                        self.rollouts.reset_actions[i][valid_mask]
                    )
            print("after extend", len(self.state_replay))
            # print("In ppo, after extend", get_memory_usage())

            # sample state from buffer and set into the cache of env
            _n_state_per_env = 30
            if self.state_replay.can_sample(self.n_envs * _n_state_per_env):
                replay_state, _, replay_obs, _, _, replay_rnn_hxs, \
                    _, replay_rnn_mask, _, _, _, _, _, _ = \
                    self.state_replay.sample(self.n_envs * _n_state_per_env, beta=0.4, uniform=(self.priority_type == "uniform"))
                # Also get their backtraced neighbours.

                with torch.no_grad():
                    restart_value_mean = self.policy.get_value(
                        torch.from_numpy(replay_obs).float().to(self.device), torch.from_numpy(replay_rnn_hxs).to(self.device), 
                        torch.from_numpy(replay_rnn_mask).to(self.device)).mean().item()
                # print(weights, replay_idxes)
                self.env.env_method("clear_state_replay")
                dispatch_state = np.reshape(replay_state, (self.n_envs, _n_state_per_env))
                # print(type(dispatch_state[0]), len(dispatch_state[0]))
                self.env.dispatch_env_method("add_restart_states", *[dispatch_state[i] for i in range(self.n_envs)])
                del replay_state, replay_obs, replay_rnn_hxs, replay_rnn_mask
                # print("In ppo, after sample and dispatch", get_memory_usage())

        value_loss_epoch = []
        action_loss_epoch = []
        dist_entropy_epoch = []
        # pi_kl_epoch = []
        kl_loss_epoch = []
        aux_task_loss_epoch = []
        pi_grad_norm_epoch = []
        vf_grad_norm_epoch = []
        rec2old_ratio_epoch = []
        cur2rec_ratio_epoch = []
        debug_supervision_action_epoch = []
        clipped_ratio_epoch = []
        adv_max_epoch, adv_min_epoch = [], []

        # seperate policy and value training here.
        for e in range(self.noptepochs):
            # continue
            if self.policy.is_recurrent:
                data_generator = self.rollouts.recurrent_generator(
                    advantages, self.nminibatches)
            else:
                data_generator = self.rollouts.feed_forward_generator(
                    advantages, self.nminibatches, with_aux=(self.auxiliary_task!=None))

            for sample in data_generator:
                # obs_batch, recurrent_hidden_states_batch, actions_batch, \
                #    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                #         adv_targ, aux_info_batch, debug_supervised_actions_batch, debug_supervised_actions_idx = sample
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = \
                    map(lambda x: torch.from_numpy(x).to(self.device), [obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, 
                                                                        return_batch, masks_batch, old_action_log_probs_batch, adv_targ])
                obs_batch = obs_batch.float()

                # Reshape to do in a single forward pass for all steps
                
                action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                
                # del obs_batch, actions_batch, recurrent_hidden_states_batch, masks_batch
                
                
                if self.policy_ewma is not None:
                # if False:
                    rec_action_log_probs_batch, _, _ = self.policy_ewma.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch,
                        compute_entropy=False
                    )
                else:
                    rec_action_log_probs_batch = old_action_log_probs_batch
                logratio = action_log_probs - rec_action_log_probs_batch
                if self.kl_beta == 0:
                    ratio = torch.exp(logratio.sum(dim=0))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.cur_cliprange, 1.0 + self.cur_cliprange) * adv_targ
                    with torch.no_grad():
                        clipped_ratio = torch.sum((torch.min(surr1, surr2) - surr1).abs() > 1e-4).float() / surr1.shape[0]
                        clipped_ratio_epoch.append(clipped_ratio.detach().item())
                    action_loss = (-torch.min(surr1, surr2) * torch.exp(
                        (rec_action_log_probs_batch - old_action_log_probs_batch).sum(dim=0))).mean()
                else:
                    # Try kl_penalty version
                    # TODO: do we need to use true kl?
                    pi_kl = 0.5 * (logratio ** 2).mean()
                    action_loss = (-torch.exp(action_log_probs - old_action_log_probs_batch) * adv_targ).mean() \
                                + self.kl_beta * pi_kl
                
                # HACK: supervised action loss
                # if len(debug_supervised_actions_idx) > 0:
                #     debug_supervised_action_log_probs, _, _ = self.policy.evaluate_actions(
                #         obs_batch[debug_supervised_actions_idx], recurrent_hidden_states_batch[debug_supervised_actions_idx],
                #         masks_batch[debug_supervised_actions_idx], debug_supervised_actions_batch[debug_supervised_actions_idx]
                #     )
                #     action_supervision_loss = -debug_supervised_action_log_probs.mean()
                #     # action_supervision_loss = ((actions_batch[debug_supervised_actions_idx] -
                #     #                             debug_supervised_actions_batch[debug_supervised_actions_idx])**2).mean()
                # else:
                #     action_supervision_loss = torch.FloatTensor([0]).to(self.device)
                
                self.optimizer['pi'].zero_grad()
                # (action_log_probs.sum() + dist_entropy.sum()).backward()
                
                if self.policy.arch == "shared":
                    # (action_loss - dist_entropy * self.ent_coef + action_supervision_loss).backward()
                    (action_loss - dist_entropy * self.ent_coef).backward()
                elif self.policy.arch == "dual":
                    values = self.policy.get_value(obs_batch, recurrent_hidden_states_batch, masks_batch)
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                                             (values - value_preds_batch).clamp(-self.cliprange, self.cliprange)
                        value_losses = (values - return_batch).pow(2).sum(dim=1)
                        value_losses_clipped = (
                                value_pred_clipped - return_batch).pow(2).sum(dim=1)
                        value_loss = 0.5 * torch.max(value_losses,
                                                     value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).sum(dim=1).mean()
                    # (value_loss * self.vf_coef + action_loss - dist_entropy * self.ent_coef + action_supervision_loss).backward()
                    (value_loss * self.vf_coef + action_loss - dist_entropy * self.ent_coef).backward()
                    value_loss_epoch.append(value_loss.detach().item())
                else:
                    raise NotImplementedError
                
                # import IPython
                # IPython.embed()
                # exit()
                if self.max_grad_norm > 0:
                    total_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                               self.max_grad_norm)
                else:
                    total_grad_norm = get_grad_norm(self.policy.parameters())
                self.optimizer['pi'].step()
                
                # if False:
                if self.policy_ewma is not None:
                    self.policy_ewma.update()
                
                
                # value_loss_epoch += value_loss.detach().item()
                action_loss_epoch.append(action_loss.detach().item())
                dist_entropy_epoch.append(dist_entropy.detach().item())
                # pi_kl_epoch += (logratio ** 2).mean().detach().item()
                pi_grad_norm_epoch.append(total_grad_norm.detach().item())
                rec2old_ratio_epoch.append(torch.exp(
                    rec_action_log_probs_batch - old_action_log_probs_batch).mean().detach().item())
                cur2rec_ratio_epoch.append(torch.exp(logratio).mean().detach().item())
                # debug_supervision_action_epoch.append(action_supervision_loss.detach().item())
                adv_max_epoch.append(adv_targ.max().detach().item())
                adv_min_epoch.append(adv_targ.min().detach().item())
                
        # print("In update, after ppo", get_memory_usage())
        
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except: pass
        
        if n_update % self.n_vf_rollout == 0:
            
            with torch.no_grad():
                if isinstance(self.policy, CNNPolicy) or isinstance(self.policy, MLPPolicy) \
                        or isinstance(self.policy, HybridAttentionPolicy) \
                        or isinstance(self.policy, MultiDiscreteAttentionPolicy):
                    if self.policy_ewma is not None:
                        # add recurrent support
                        old_dist, _ = self.policy_ewma.forward_policy(
                            torch.from_numpy(self.vf_rollouts['obs'].reshape((-1, *self.vf_rollouts['obs'].shape[2:]))).float().to(self.device),
                            torch.from_numpy(self.vf_rollouts['recurrent_hidden_state'].reshape((-1, *self.vf_rollouts['recurrent_hidden_state'].shape[2:]))).to(self.device),
                            torch.from_numpy(self.vf_rollouts['recurrent_mask'].reshape((-1, 1))).to(self.device))
                    else:
                        old_dist, _ = self.policy.forward_policy(
                            torch.from_numpy(self.vf_rollouts['obs'].reshape((-1, *self.vf_rollouts['obs'].shape[2:]))).float().to(self.device),
                            torch.from_numpy(self.vf_rollouts['recurrent_hidden_state'].reshape((-1, *self.vf_rollouts['recurrent_hidden_state'].shape[2:]))).to(self.device),
                            torch.from_numpy(self.vf_rollouts['recurrent_mask'].reshape((-1, 1))).to(self.device))
                    # old_dist = tuple([FixedCategorical(logit) for logit in old_dist])
                else:
                    raise NotImplementedError
            # print("In ppg, after computing old dist", get_memory_usage())
            
            for e in range(self.nvfepochs):
                if self.policy.is_recurrent:
                    indices = np.arange(self.n_valid_rollouts * self.n_steps * self.n_envs).reshape((self.n_steps * self.n_valid_rollouts, self.n_envs))
                    # Permute the columns
                    indices = np.swapaxes(indices, 0, 1)
                    np.random.shuffle(indices)
                    indices = np.swapaxes(indices, 0, 1).reshape([-1])
                else:
                    indices = np.arange(self.n_valid_rollouts * self.n_steps * self.n_envs)
                    np.random.shuffle(indices)
                batch_size = self.n_steps * self.n_envs // self.nvfminibatches
                for m in range(self.nvfminibatches * self.n_valid_rollouts):
                    if self.policy.is_recurrent:
                        m1 = m // self.nvfminibatches  # Which rollout we are in
                        m2 = m % self.nvfminibatches  # Which chunk of envs we are in
                        mb_indices = indices.reshape([self.n_steps * self.n_valid_rollouts, self.n_envs])[
                                     self.n_steps * m1: self.n_steps * (m1 + 1),
                                     self.n_envs // self.nvfminibatches * m2: self.n_envs // self.nvfminibatches * (m2 + 1)]
                        mb_indices = np.reshape(mb_indices, [-1])

                        recurrent_hidden_states_batch = \
                            torch.from_numpy(self.vf_rollouts['recurrent_hidden_state'][self.n_steps * m1,
                                self.n_envs // self.nvfminibatches * m2: self.n_envs // self.nvfminibatches * (m2 + 1)]).to(self.device)
                    else:
                        mb_indices = indices[m * batch_size: (m + 1) * batch_size]
                        recurrent_hidden_states_batch = torch.from_numpy(
                            self.vf_rollouts['recurrent_hidden_state'].reshape((-1, self.vf_rollouts['recurrent_hidden_state'].shape[2]))[mb_indices])
                    # print(mb_indices)
                    obs_batch = torch.from_numpy(self.vf_rollouts['obs'].reshape((-1, *self.vf_rollouts['obs'].shape[2:]))[mb_indices]).float().to(self.device)
                    return_batch = torch.from_numpy(self.vf_rollouts['value_targ'].reshape((-1, self.policy.n_values, 1))[mb_indices]).to(self.device)
                    masks_batch = torch.from_numpy(self.vf_rollouts['recurrent_mask'].reshape((-1, 1))[mb_indices]).to(self.device)
                    # recurrent_hidden_states_batch = torch.zeros(len(mb_indices), self.policy.recurrent_hidden_state_size)
                    # masks_batch = torch.ones(len(mb_indices), 1)
                    if self.policy.arch == "shared":
                        # Value loss comes from the value head on the shared backbone
                        # values = self.policy.get_value(obs_batch, recurrent_hidden_states_batch, masks_batch)
                        # value_loss = 0.5 * (return_batch - values).pow(2).sum(dim=1).mean()
                        # cur_dist, _ = self.policy.forward_policy(obs_batch, recurrent_hidden_states_batch, masks_batch)
                        values, cur_dist = self.policy.forward_policy_and_value(obs_batch, recurrent_hidden_states_batch, masks_batch)
                        value_loss = 0.5 * (return_batch - values).pow(2).sum(dim=1).mean()
                    elif self.policy.arch == "dual":
                        # Value loss comes from the aux head and (optionally) the separate value network
                        cur_dist, aux_values = self.policy.forward_policy_with_aux_head(obs_batch, recurrent_hidden_states_batch, masks_batch)
                        values = self.policy.get_value(obs_batch, recurrent_hidden_states_batch, masks_batch)
                        value_loss = 0.5 * (return_batch[:, 0] - aux_values).pow(2).mean() + 0.5 * (return_batch - values).pow(2).sum(dim=1).mean()
                    else:
                        raise NotImplementedError
                    # del recurrent_hidden_states_batch, masks_batch
                    cur_dist = tuple([FixedCategorical(logits=logit) for logit in cur_dist])
                    # TODO: run auxiliary task, get loss
                    aux_task_loss = torch.FloatTensor([np.nan]).to(self.device)
                    if self.auxiliary_task is not None:
                        if self.auxiliary_task_name == "skyline":
                            aux_info_batch = self.vf_rollouts['aux_info'].view(-1, *self.vf_rollouts['aux_info'].shape[2:])[mb_indices]
                            aux_task_loss = self.auxiliary_task(obs_batch, aux_info_batch)
                        elif self.auxiliary_task_name == "next_skyline" or self.auxiliary_task_name == "next_state":
                            actions_batch = torch.from_numpy(self.vf_rollouts['actions'].reshape((-1, *self.vf_rollouts['actions'].shape[2:]))[mb_indices]).to(self.device)
                            aux_info_batch = torch.from_numpy(self.vf_rollouts['aux_info'].reshape((-1, *self.vf_rollouts['aux_info'].shape[2:]))[mb_indices]).float().to(self.device)
                            aux_task_loss = self.auxiliary_task(obs_batch, actions_batch, aux_info_batch)
                        elif self.auxiliary_task_name == "inverse_dynamics":
                            actions_batch = torch.from_numpy(
                                self.vf_rollouts['actions'].reshape((-1, *self.vf_rollouts['actions'].shape[2:]))[
                                    mb_indices]).to(self.device)
                            reset_actions_batch = torch.from_numpy(
                                self.vf_rollouts['reset_actions']
                                    .reshape((-1, *self.vf_rollouts['reset_actions'].shape[2:]))[mb_indices]
                            ).squeeze(dim=-1).bool().to(self.device)
                            aux_info_batch = torch.from_numpy(
                                self.vf_rollouts['aux_info'].reshape((-1, *self.vf_rollouts['aux_info'].shape[2:]))[
                                    mb_indices]).float().to(self.device)
                            aux_task_loss = self.auxiliary_task(obs_batch, actions_batch, aux_info_batch, reset_actions_batch)
                    
                    if isinstance(self.policy, CNNPolicy):
                        mb_old_dist = FixedCategorical(probs=old_dist.probs[mb_indices])
                    elif isinstance(self.policy, MLPPolicy):
                        mb_old_dist = FixedNormal(loc=old_dist.loc[mb_indices], scale=old_dist.scale[mb_indices])
                    elif isinstance(self.policy, HybridAttentionPolicy):
                        mb_old_dist = (FixedCategorical(probs=old_dist[0].probs[mb_indices]),
                                       FixedNormal(loc=old_dist[1].loc[mb_indices], scale=old_dist[1].scale[mb_indices]))
                    elif isinstance(self.policy, MultiDiscreteAttentionPolicy):
                        with torch.no_grad():
                            mb_old_dist = tuple([FixedCategorical(logits=old_dist[i][mb_indices]) for i in range(len(old_dist))])
                    else:
                        raise NotImplementedError
                    

                    # kl_loss = torch.distributions.kl.kl_divergence(mb_old_dist, cur_dist).mean()
                    # kl_loss = kl_divergence(mb_old_dist, cur_dist)
                    kl_loss = joint_kl(mb_old_dist[0], cur_dist[0], mb_old_dist[1:], cur_dist[1:])
                    # kl_loss = torch.FloatTensor([0.]).to(self.device)
                    # print(m, "forward", get_memory_usage())
                    self.optimizer['vf'].zero_grad()
                    (value_loss * self.vf_coef + kl_loss * self.bc_coef + aux_task_loss * self.aux_coef).backward()
                    # if m == 0:
                    #     print(value_loss, kl_loss, aux_task_loss)
                    #     import IPython
                    #     IPython.embed()
                    #     exit()
                    if self.max_grad_norm > 0:
                        total_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                                   self.max_grad_norm)
                    else:
                        total_grad_norm = get_grad_norm(self.policy.parameters())
                    self.optimizer['vf'].step()
                    value_loss_epoch.append(value_loss.detach().item())
                    kl_loss_epoch.append(kl_loss.detach().item())
                    aux_task_loss_epoch.append(aux_task_loss.detach().item())
                    vf_grad_norm_epoch.append(total_grad_norm.detach().item())
                    # print(m, "backward", get_memory_usage())
            del old_dist
            
            if self.policy_ewma is not None:
                self.policy_ewma.reset()
            
            # print("In update, after ppg", get_memory_usage())
            
        # num_pi_updates = self.noptepochs * self.nminibatches
        # num_vf_updates = self.nvfepochs * self.nminibatches * self.n_valid_rollouts

        value_loss_epoch = safe_mean(value_loss_epoch)
        action_loss_epoch = safe_mean(action_loss_epoch)
        dist_entropy_epoch = safe_mean(dist_entropy_epoch)
        kl_loss_epoch = safe_mean(kl_loss_epoch)
        aux_task_loss_epoch = safe_mean(aux_task_loss_epoch)
        pi_grad_norm_epoch = safe_mean(pi_grad_norm_epoch)
        vf_grad_norm_epoch = safe_mean(vf_grad_norm_epoch)
        rec2old_ratio_mean = safe_mean(rec2old_ratio_epoch)
        cur2rec_ratio_mean = safe_mean(cur2rec_ratio_epoch)
        rec2old_ratio_max = np.max(rec2old_ratio_epoch)
        cur2rec_ratio_max = np.max(cur2rec_ratio_epoch)
        clipped_ratio_epoch = safe_mean(clipped_ratio_epoch)
        debug_supervision_action_epoch = safe_mean(debug_supervision_action_epoch)
        adv_max_epoch = safe_mean(adv_max_epoch)
        adv_min_epoch = safe_mean(adv_min_epoch)

        if self.priority_type is not None:
            # TODO: update all the priorities?
            # chunk to avoid large memory cost
            _chunk_size = 2_000
            for chunk_start in range(0, len(self.state_replay), _chunk_size):
                _, _, data, _, _ = zip(*self.state_replay.storage[chunk_start: chunk_start + _chunk_size])
                next_obs_buf, obs_buf, reward_buf, next_hxs_buf, hxs_buf, next_mask_buf, mask_buf, step_to_go_buf, \
                action_buf, reset_action_buf, time_buf = zip(*data)
                # next_obs_buf = torch.stack(next_obs_buf, dim=0)
                # obs_buf = torch.stack(obs_buf, dim=0)
                # hxs_buf = torch.stack(hxs_buf, dim=0)
                # next_hxs_buf = torch.stack(next_hxs_buf, dim=0)
                # mask_buf = torch.stack(mask_buf, dim=0)
                # next_mask_buf = torch.stack(next_mask_buf, dim=0)
                # reward_buf = torch.stack(reward_buf, dim=0)
                rewards = np.stack(reward_buf, axis=0)
                step_to_go_buf = np.stack(step_to_go_buf, axis=0)
                next_obs_buf, obs_buf, hxs_buf, next_hxs_buf, mask_buf, next_mask_buf, action_buf, reset_action_buf = \
                    map(lambda x: torch.from_numpy(np.stack(x, axis=0)).to(self.device), 
                    [next_obs_buf, obs_buf, hxs_buf, next_hxs_buf, mask_buf, next_mask_buf, action_buf, reset_action_buf])
                obs_buf, next_obs_buf = obs_buf.float(), next_obs_buf.float()
                reset_action_buf = torch.squeeze(reset_action_buf, dim=-1).bool()
                replay_idxes = np.arange(chunk_start, chunk_start + len(obs_buf))
                with torch.no_grad():
                    # TODO: compute with priority_head
                    cur_value = self.policy.get_value(obs_buf, hxs_buf, mask_buf).cpu().numpy()
                    next_value = self.policy.get_value(next_obs_buf, next_hxs_buf, next_mask_buf).cpu().numpy()
                # rewards = reward_buf.cpu().numpy()
                next_masks = next_mask_buf.cpu().numpy()
                # Old code.
                '''
                with torch.no_grad():
                    # TODO: for recurrent policy, using old rnn_hxs will introduce bias
                    cur_value = self.policy.get_value(replay_obs, replay_rnn_hxs, replay_rnn_mask).cpu().numpy()
                    next_value = self.policy.get_value(replay_next_obs, replay_next_rnn_hxs, replay_next_rnn_mask).cpu().numpy()
                    rewards = replay_reward.cpu().numpy()
                '''
                priority = self.compute_priority(rewards, cur_value, next_value, next_masks, obs_buf, next_obs_buf,
                                                 action_buf, reset_action_buf)
                # if self.priority_type == "diff":
                #     priority = np.squeeze(np.abs(next_value[:, 0] - cur_value[:, 0]), axis=-1)
                # elif self.priority_type == "pos_diff":
                #     priority = np.squeeze(np.maximum(next_value[:, 0] - cur_value[:, 0], 0), axis=-1)
                # elif self.priority_type == "value":
                #     priority = np.squeeze(cur_value[:, 0], axis=-1)
                # elif self.priority_type == "td":
                #     if self.clip_priority:
                #         priority = np.squeeze(np.abs(np.clip(rewards, 0., 1.) + self.gamma * np.clip(next_value[:, 0], 0., 1.) * next_masks - np.clip(cur_value[:, 0], 0., 1.)), axis=-1)
                #     else:
                #         priority = np.squeeze(np.abs(rewards + self.gamma * next_value[:, 0] * next_masks - cur_value[:, 0]), axis=-1)
                # elif self.priority_type == "td_step_to_go":
                #     step_to_go_bonus = compute_step_to_go_bonus(step_to_go_buf, 30)
                #     priority = np.squeeze(np.abs(rewards + self.gamma * next_value[:, 0] * next_masks - cur_value[:, 0])
                #                           + step_to_go_bonus, axis=1)
                # elif self.priority_type == "v_std":
                #     priority = np.squeeze(np.std(cur_value, axis=1), axis=-1)
                # elif self.priority_type == "task_grad":
                #     n_obj = self.env.get_attr('num_blocks')[0]
                #     obj_dim = self.env.get_attr('object_dim')[0]
                #     switched_obs = switch_task_obs(obs_buf, n_obj, obj_dim)
                #     original_shape = switched_obs.shape
                #     flat_obs = switched_obs.view(-1, switched_obs.shape[-1])
                #     with torch.no_grad():
                #         switched_values = self.policy.get_value(flat_obs, None, None)[:, 0].cpu().numpy()
                #     switched_values = switched_values.reshape(original_shape[:-1] + (1,))
                #     # print('switched values shape', switched_values.shape)
                #     priority = np.squeeze((np.max(switched_values, axis=0) - switched_values[0]) / np.abs(switched_values[0]), axis=-1)
                #     # print('updated priority shape', priority.shape)
                # else:
                #     priority = None
                if priority is not None:
                    priority = np.maximum(priority, 1e-5)
                    # print("updated priority", priority)
                    self.state_replay.update_priorities(replay_idxes, priority)
            self.state_replay.force_sort()

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kl_loss_epoch, aux_task_loss_epoch, \
               pi_grad_norm_epoch, vf_grad_norm_epoch, rec2old_ratio_mean, cur2rec_ratio_mean, \
               debug_supervision_action_epoch, debug_priority_mean, debug_priority_std, debug_priority_max, restart_value_mean, \
               rec2old_ratio_max, cur2rec_ratio_max, clipped_ratio_epoch, adv_max_epoch, adv_min_epoch

    def compute_priority(self, rewards, cur_value, next_value, masks, obs, next_obs, actions, reset_actions):
        if self.priority_type == "diff":
            priority = np.abs(next_value[..., 0, 0] - cur_value[..., 0, 0])
        elif self.priority_type == "pos_diff":
            priority = np.maximum(next_value[..., 0, 0] - cur_value[..., 0, 0], 0.)
        elif self.priority_type == "value":
            priority = cur_value[..., 0, 0]
        elif self.priority_type == "td":
            if self.clip_priority:
                priority = np.abs(np.clip(rewards[..., 0] + \
                    self.gamma * next_value[..., 0, 0] * masks[..., 0], -np.inf, 1.) - np.clip(cur_value[..., 0, 0], -np.inf, 1.))
            else:
                priority = np.abs(rewards[..., 0] + self.gamma * next_value[..., 0, 0] * masks[..., 0] - cur_value[..., 0, 0])
        elif self.priority_type == "td_step_to_go":
            # penalize cases already very close to success <5 steps
            step_to_go_bonus = compute_step_to_go_bonus(self.rollouts.step_to_go[:-2].copy(), 30)
            priority = np.abs(rewards[..., 0] + self.gamma * next_value[..., 0, 0] * masks[..., 0] \
                - cur_value[..., 0, 0]) + step_to_go_bonus[..., 0]
            # print("step_to_go", step_to_go_bonus, "priority", priority)
        elif self.priority_type == "v_std":
            priority = np.std(cur_value[..., 0], axis=-1)
        elif self.priority_type == "uniform":
            priority = np.zeros_like(rewards).squeeze(axis=-1)
        elif self.priority_type == "aux_error":
            with torch.no_grad():
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).to(self.device).float()
                    next_obs = torch.from_numpy(next_obs).to(self.device).float()
                if isinstance(actions, np.ndarray):
                    actions = torch.from_numpy(actions).to(self.device)
                    reset_actions = torch.from_numpy(reset_actions).to(self.device)
                    reset_actions = torch.squeeze(reset_actions, dim=-1).bool()
                priority = self.auxiliary_task.get_error(obs, actions, next_obs, reset_actions).cpu().numpy()
        elif self.priority_type == "task_grad":
            # Switch tasks, calculate imagined values,
            n_obj = self.env.get_attr('num_blocks')[0]
            obj_dim = self.env.get_attr('object_dim')[0]
            switched_obs = switch_task_obs(self.rollouts.obs[:-2].copy(), n_obj, obj_dim)
            original_shape = switched_obs.shape
            # print('switched obs shape', switched_obs.shape)
            flat_obs = switched_obs.reshape((-1, switched_obs.shape[-1]))
            # print('flat obs shape', flat_obs.shape)
            with torch.no_grad():
                switched_values = self.policy.get_value(torch.from_numpy(flat_obs).float().to(self.device), None, None)[:, 0].cpu().numpy()  # Take the first value head
            switched_values = switched_values.reshape(original_shape[:-1] + (1,))
            # print('switched values shape', switched_values.shape, 'other value shape', cur_value.shape)

            # todo: check shape
            # take the max??
            priority = np.squeeze((np.max(switched_values, axis=0) - switched_values[0]) / np.maximum(np.abs(switched_values[0]), 1e-5), axis=-1)
            # print('priority shape', priority.shape)
        else:
            priority = None
        return priority
    
    def save(self, save_path):
        save_dict = {'policy': self.policy.state_dict()}
        optimizers = {k: self.optimizer[k].state_dict() for k in self.optimizer}
        save_dict.update(optimizers)
        # TODO: save more information
        if self.state_replay.can_sample(128):
            sample_state, batch_next_obs, batch_obs, batch_reward, batch_next_rnn_hxs, batch_rnn_hxs, \
            batch_next_rnn_mask, batch_rnn_mask, batch_actions, batch_reset_actions, batch_priority, batch_encounter, time_buf, _ = \
                self.state_replay.sample(128, beta=0.4, uniform=(self.priority_type == "uniform"))
            # with torch.no_grad():
            #     # state_value = self.policy.get_value(batch_obs, batch_rnn_hxs, batch_rnn_mask).cpu().numpy()
            #     batch_obs = batch_obs.cpu().numpy()
            #     batch_next_obs = batch_next_obs.cpu().numpy()
            #     batch_reward = batch_reward.cpu().numpy()
            #     batch_next_rnn_mask = batch_next_rnn_mask.cpu().numpy()
        else:
            sample_state = []
            # state_value = None
            batch_obs, batch_next_obs, batch_reward, time_buf, batch_next_rnn_mask, batch_priority, batch_encounter \
                = None, None, None, None, None, None, None
        save_dict.update(dict(state_buffer=sample_state, obs_buffer=batch_obs, next_obs_buffer=batch_next_obs,
                              reward_buffer=batch_reward, time_buffer=time_buf, next_mask_buffer=batch_next_rnn_mask,
                              priority_buffer=batch_priority, encounter_buffer=batch_encounter))
        torch.save(save_dict, save_path)

    def load(self, load_pth, eval=True):
        checkpoint = torch.load(load_pth, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        if self.policy_ewma is not None:
            self.policy_ewma.reset()
        for k in self.optimizer:
            self.optimizer[k].load_state_dict(checkpoint[k])
        if eval:
            self.policy.eval()
        else:
            self.policy.train()


def kl_divergence(dist1, dist2):
    if isinstance(dist1, tuple):
        assert len(dist1) == len(dist2)
        kl = 0
        for i in range(len(dist1)):
            temp = torch.distributions.kl.kl_divergence(dist1[i], dist2[i])
            if len(temp.shape) == 2:  # (batch_size, seq_len)
                temp = temp.mean(dim=-1)
            kl += temp
        kl = kl.mean()
    else:
        kl = torch.distributions.kl.kl_divergence(dist1, dist2).mean()
    return kl


def joint_kl(dist1: torch.distributions.Categorical, dist2, cond_dist1, cond_dist2):
    kl = torch.distributions.kl.kl_divergence(dist1, dist2)
    cond_kl = 0
    assert len(cond_dist1) == len(cond_dist2)
    for i in range(len(cond_dist1)):
        cond_kl += torch.distributions.kl.kl_divergence(cond_dist1[i], cond_dist2[i])
    prob1 = dist1.probs[:, 1:]  # Exclude dim for no-op
    assert prob1.shape == cond_kl.shape, (prob1.shape, cond_kl.shape)
    kl = (kl + (prob1 * cond_kl).sum(dim=-1)).mean()
    return kl


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_linear_clip(epoch, total_num_epochs, initial_clip):
    cur_clip = initial_clip - (initial_clip * (epoch / float(total_num_epochs)))
    return cur_clip


def get_grad_norm(parameters):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    return torch.norm(torch.stack([torch.norm(g) for g in grads]))


def manual_validation_fn(obs_arr, n_obj, cliff_height=0.1, block_thickness=0.025, object_dim=17):
    # TODO: remove hard coded numbers
    valid = np.ones(obs_arr.shape[0]).astype(np.bool)
    for i, obs in enumerate(obs_arr):
        for obj_id in range(n_obj):
            if np.linalg.norm(obs[object_dim * obj_id: object_dim * obj_id + 3]) < 1e-3:
                # Run out of objects
                break
            if np.linalg.norm(obs[object_dim * obj_id: object_dim * obj_id + 3] + 1) > 1e-3:
                if abs(obs[17 * obj_id] - 1.3) > 0.01:
                    # Not in the y-z plane
                    valid[i] = False
                    break
                long_axis = quat_rot_vec(euler2quat(obs[object_dim * obj_id + 3: object_dim * obj_id + 6]), np.array([0, 1, 0]))
                if np.abs(np.dot(long_axis, np.array([0, 1, 0]))) < 0.97 and \
                        np.abs(np.dot(long_axis, np.array([0, 0, 1]))) < 0.97:
                    # Orientation
                    valid[i] = False
                    break
                if np.abs(np.dot(long_axis, np.array([0, 1, 0]))) > 0.97 and \
                        abs(obs[object_dim * obj_id + 2] - (2 * cliff_height + block_thickness)) > cliff_height / 5:
                    # Horizontal block in the wrong place
                    valid[i] = False
                    break
                if np.abs(np.dot(long_axis, np.array([0, 0, 1]))) > 0.97 and \
                        abs(obs[object_dim * obj_id + 2] - cliff_height) > cliff_height / 5:
                    # Vertical block in the wrong place
                    valid[i] = False
                    break
                # buggy when working with primitive
                '''
                if np.linalg.norm(obs[object_dim * obj_id + 4: object_dim * obj_id + 6]) > 0.03:  # 0.03 is consistent with gen_random_q
                    # Orientation around y, z is wrong
                    valid[i] = False
                    break
                if abs(obs[object_dim * obj_id + 3] - np.round(obs[object_dim * obj_id + 3] / (np.pi / 2)) * (np.pi / 2)) > 0.05:
                    # Orientation around x is wrong
                    valid[i] = False
                    break
                if np.round(obs[object_dim * obj_id + 3] / 1.5708).astype(np.int) % 2 == 0 and \
                        abs(obs[object_dim * obj_id + 2] - (2 * cliff_height + block_thickness)) > cliff_height / 5:
                    # Horizontal block in the wrong place
                    valid[i] = False
                    break
                if np.round(obs[object_dim * obj_id + 3] / 1.5708).astype(np.int) % 2 == 1 and \
                        abs(obs[object_dim * obj_id + 2] - cliff_height) > cliff_height / 5:
                    # Vertical block in the wrong place
                    valid[i] = False
                    break
                '''
            # if torch.norm(obs[17 * obj_id: 17 * obj_id + 3] + 1).item() > 1e-3 and \
            #         (abs(obs[17 * obj_id] - 1.3) > 0.01 or obs[17 * obj_id + 2] > 0.23 or
            #          abs(obs[17 * obj_id + 3] - torch.round(obs[17 * obj_id + 3] / 1.5708) * 1.5708) > 0.05 or
            #          torch.norm(obs[17 * obj_id + 4: 17 * obj_id + 6]).item() > 0.01):
            #     valid[i] = False
            #     break
    return valid


def switch_task_obs(rollout_obs: torch.Tensor, n_obj, obj_size):
    obs = rollout_obs.unsqueeze(dim=0).repeat(5, *tuple([1] * len(rollout_obs.shape)))
    # device = obs.device
    # obs[:, :, :, n_obj * obj_size + 1] += torch.from_numpy(np.arange(0., 0.01 * 5, 0.01)).to(device).view(-1, 1, 1)
    # obs[:, :, :, (n_obj + 1) * obj_size + 1] -= torch.from_numpy(np.arange(0., 0.01 * 5, 0.01)).to(device).view(-1, 1, 1)
    obs[..., n_obj * obj_size + 1] += \
        torch.min(0.02 * torch.randn_like(obs[..., 0]),
                  (obs[..., (n_obj + 1) * obj_size + 1] - obs[..., n_obj * obj_size + 1]) / 2)
    obs[..., (n_obj + 1) * obj_size + 1] += \
        torch.max(0.02 * torch.randn_like(obs[..., 0]),
                  -(obs[..., (n_obj + 1) * obj_size + 1] - obs[..., n_obj * obj_size + 1]) / 2)
    assert torch.all(obs[..., n_obj * obj_size + 1] < obs[..., (n_obj + 1) * obj_size + 1])
    return obs


def compute_step_to_go_bonus(step_to_go: np.ndarray, horizon=30):
    bonus = np.clip(1 - np.abs(step_to_go - horizon / 2) / (horizon / 2), 0., 1)
    return bonus


def compute_history_start(masks: torch.Tensor):
    np_masks = masks.cpu().numpy().squeeze(axis=-1)
    starts = np.ones([masks.size(0), masks.size(1)])
    for step in range(1, starts.shape[0]):
        starts[step] = starts[step - 1] * np_masks[step] + step * (1 - np_masks[step])
    return starts

'''
def get_memory_usage():
    import os, psutil
    process = psutil.Process()
    mem = process.memory_info().rss
    main_mem = mem
    # print("main pid", process.pid, "name", process.name(), "cmdline", process.cmdline())
    child_count = 0
    mems = []
    for child in process.children(recursive=True):
        # print("pid", child.pid, "cmdline", child.cmdline(), "parent", child.ppid(), "mem", child.memory_info().rss / (1024 ** 3))
        child_count += 1
        mems.append(child.memory_info().rss / (1024 ** 3))
        mem += child.memory_info().rss
    # print(mem / (1024 ** 3), child_count)
    return mem / (1024 ** 3), main_mem, sum(mems[3: 67]), sum(mems[67:])
'''