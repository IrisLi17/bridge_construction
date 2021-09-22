import torch
import torch.nn as nn
from torch_algorithms.policies import MultiDiscreteAttentionPolicy
from torch_algorithms.policies.attention_model import SelfAttentionBase


class InverseDynamicsPrediction(object):
    def __init__(self, policy: MultiDiscreteAttentionPolicy, action_dim, obs_dim, task="regression"):
        self.policy = policy
        self.feature_dim = self.policy.feature_dim
        self.num_bin = self.policy.num_bin
        self.bilevel_action = self.policy.bilevel_action
        self.action_dim = action_dim
        self.out_dim = obs_dim  # Used in ppo_dev to allocate buffer
        self.object_state_dim = 3
        assert task in ["regression", "classification"]
        self.task = task
        self._set_up()

    def _set_up(self):
        self.policy.__setattr__("inverse_object_id_linear", nn.Linear(2 * self.feature_dim, 1))
        if self.policy.noop:
            self.policy.__setattr__("inverse_no_op_linear", nn.Linear(2 * self.feature_dim, 1))
        else:
            self.policy.__setattr__("inverse_no_op_linear", None)
        self.policy.__setattr__("inverse_object_state", nn.ModuleList(
            # another 1 or 2 bins for out_of_reach prediction
            [nn.Linear(2 * self.feature_dim, self.num_bin[i] + 2 if self.bilevel_action else self.num_bin[i] + 1)
             for i in range(len(self.num_bin))]))
        # Initialize
        self.policy.to(self.policy.object_id_linear.weight.device)

    def _predict(self, obs_batch, next_obs_batch):
        # obj_and_others, _, _ = self.policy.parse_input(obs_batch)
        # parsed_obs = torch.narrow(obj_and_others, -1, start=1, length=self.object_state_dim)
        all_obs_batch = torch.cat([obs_batch, next_obs_batch], dim=0)
        actor_latent, rnn_hxs, mask, action_mask = self.policy._encode(all_obs_batch)
        (cur_latent, next_latent) = torch.split(actor_latent, actor_latent.size(0) // 2, dim=0)
        cat_latent = torch.cat([cur_latent, next_latent], dim=-1)  # (batch_size, seq_len, 2 * feature_dim)
        id_logit = self.policy.inverse_object_id_linear(cat_latent).squeeze(dim=-1)  # (batch_size, seq_len)
        if self.policy.noop:
            noop_logit = self.policy.inverse_no_op_linear(torch.mean(cat_latent, dim=1))  # (batch_size, 1)
            id_logit = torch.cat([noop_logit, id_logit], dim=-1)
        object_state_logit = [self.policy.inverse_object_state[i](cat_latent) for i in range(len(self.num_bin))]  # 3 * (batch_size, seq_len, num_bin)
        # print("before reshape logit", len(object_state_logit), object_state_logit[0].shape, object_state_logit[0][:, 0, 0], object_state_logit[0][:, 0, 8])
        if self.bilevel_action:
            coarse_state_logit = [object_state_logit[i][..., :self.num_bin[i] // 2 + 1] for i in range(len(self.num_bin))]  # 3 * (batch_size, seq_len, num_bin / 2)
            refined_state_logit = [object_state_logit[i][..., self.num_bin[i] // 2 + 1:] for i in range(len(self.num_bin))]  # 3 * (batch_size, seq_len, num_bin / 2)
            object_state_logit = torch.cat(coarse_state_logit + refined_state_logit, dim=0)  # (6 * batch_size, seq_len, num_bin / 2 + 1). x_c, y_c, th_c, x_r, y_c, th_r
            # print("after reshape logit", object_state_logit.shape, object_state_logit[:, 0, 0])
        else:
            object_state_logit = torch.cat(object_state_logit, dim=0)  # (3 * batch_size, seq_len, num_bin + 1)
        return id_logit, object_state_logit

    def _process(self, obs_batch, action_batch, next_obs_batch, reset_action_batch):
        device = obs_batch.device
        id_logit, states_logit = self._predict(obs_batch, next_obs_batch)
        gt_id = action_batch[..., 0].long()  # (batch_size,)
        noop_mask = (gt_id == -1).float()
        if self.bilevel_action:
            noop_mask = noop_mask.unsqueeze(dim=0).expand(2 * len(self.num_bin), -1).reshape(-1)
        else:
            noop_mask = noop_mask.unsqueeze(dim=0).expand(len(self.num_bin), -1).reshape(-1)
        if self.policy.noop:
            gt_id += 1
        cond_onehot = torch.zeros(gt_id.shape[0], states_logit.shape[1] + 1).to(device)
        cond_onehot.scatter_(1, gt_id.unsqueeze(dim=-1), 1)
        cond_onehot = cond_onehot[:, 1:]
        if self.bilevel_action:
            cond_onehot = cond_onehot.unsqueeze(dim=0).expand(2 * len(self.num_bin), -1, -1).reshape(
                (-1, states_logit.shape[1]))  # (6 * batch_size, seq_len)
        else:
            cond_onehot = cond_onehot.unsqueeze(dim=0).expand(len(self.num_bin), -1, -1).reshape(
                (-1, states_logit.shape[1]))  # (3 * batch_size, seq_len)
        # print('cond onehot', cond_onehot.shape, cond_onehot)
        states_logit = (cond_onehot.unsqueeze(dim=-1) * states_logit).sum(dim=1)  # (6 * batch_size, num_bin / 2)
        gt_states = action_batch[..., 1:]  # (batch_size, 3)
        # print("before reshape gt", gt_states)
        num_bin = torch.from_numpy(self.num_bin).to(device)
        if self.bilevel_action:
            action_object_states = ((gt_states + 1) / 2 * num_bin.unsqueeze(dim=0) / 2)  # (batch_size, 3)
            coarse_action_states = (action_object_states + 1e-4).int()
            refined_action_states = (
                        (action_object_states - coarse_action_states) * num_bin.unsqueeze(dim=0) / 2 + 1e-4).int()

            coarse_action_states += 1
            refined_action_states += 1
            # Apply mask on out_of_reach actions
            coarse_action_states[reset_action_batch] = 0.
            refined_action_states[reset_action_batch] = 0.

            coarse_action_states = coarse_action_states.transpose(0, 1).reshape(-1)
            refined_action_states = refined_action_states.transpose(0, 1).reshape(-1)
            gt_states = torch.cat([coarse_action_states, refined_action_states],
                                  dim=0).long()  # (6 * batch_size,)  x_c, y_c, th_c, x_r, y_r, th_r
        else:
            action_object_states = (gt_states + 1) / 2 * (num_bin - 1).unsqueeze(dim=0)
            action_object_states = (action_object_states + 1e-4).long() + 1
            action_object_states[reset_action_batch] = 0.
            gt_states = action_object_states.transpose(0, 1).reshape(-1)
        # print("after reshape gt", gt_states)
        return id_logit, gt_id, noop_mask, states_logit, gt_states

    def get_error(self, obs_batch, action_batch, next_obs_batch, reset_action_batch):
        batch_size = obs_batch.shape[:-1]
        obs_batch = torch.reshape(obs_batch, (-1, obs_batch.shape[-1]))
        action_batch = torch.reshape(action_batch, (-1, action_batch.shape[-1]))
        next_obs_batch = torch.reshape(next_obs_batch, (-1, next_obs_batch.shape[-1]))
        reset_action_batch = torch.reshape(reset_action_batch, (-1,))
        id_logit, gt_id, noop_mask, states_logit, gt_states = self._process(
            obs_batch, action_batch, next_obs_batch, reset_action_batch)
        # y, z, angle
        errors = nn.functional.cross_entropy(id_logit, gt_id, reduction="none")
        for i in range(len(self.num_bin)):
            errors += (1 - noop_mask[i * id_logit.shape[0]: (i + 1) * id_logit.shape[0]]) * nn.functional.cross_entropy(
                states_logit[i * id_logit.shape[0]: (i + 1) * id_logit.shape[0]],
                gt_states[i * id_logit.shape[0]: (i + 1) * id_logit.shape[0]], reduction="none")
        errors = torch.reshape(errors, (*batch_size,))
        return errors

    def __call__(self, obs_batch, action_batch, next_obs_batch, reset_action_batch):
        device = obs_batch.device
        id_logit, states_logit = self._predict(obs_batch, next_obs_batch)
        gt_id = action_batch[..., 0].long()  # (batch_size,)
        noop_mask = (gt_id == -1).float()
        if self.bilevel_action:
            noop_mask = noop_mask.unsqueeze(dim=0).expand(2 * len(self.num_bin), -1).reshape(-1)
        else:
            noop_mask = noop_mask.unsqueeze(dim=0).expand(len(self.num_bin), -1).reshape(-1)
        if self.policy.noop:
            gt_id += 1
        cond_onehot = torch.zeros(gt_id.shape[0], states_logit.shape[1] + 1).to(device)
        cond_onehot.scatter_(1, gt_id.unsqueeze(dim=-1), 1)
        cond_onehot = cond_onehot[:, 1:]
        if self.bilevel_action:
            cond_onehot = cond_onehot.unsqueeze(dim=0).expand(2 * len(self.num_bin), -1, -1).reshape((-1, states_logit.shape[1]))  # (6 * batch_size, seq_len)
        else:
            cond_onehot = cond_onehot.unsqueeze(dim=0).expand(len(self.num_bin), -1, -1).reshape((-1, states_logit.shape[1]))  # (3 * batch_size, seq_len)
        # print('cond onehot', cond_onehot.shape, cond_onehot)
        states_logit = (cond_onehot.unsqueeze(dim=-1) * states_logit).sum(dim=1)  # (6 * batch_size, num_bin / 2)
        gt_states = action_batch[..., 1:]  # (batch_size, 3)
        # print("before reshape gt", gt_states)
        num_bin = torch.from_numpy(self.num_bin).to(device)
        if self.bilevel_action:
            action_object_states = ((gt_states + 1) / 2 * num_bin.unsqueeze(dim=0) / 2)  # (batch_size, 3)
            coarse_action_states = (action_object_states + 1e-4).int()
            refined_action_states = ((action_object_states - coarse_action_states) * num_bin.unsqueeze(dim=0) / 2 + 1e-4).int()

            coarse_action_states += 1
            refined_action_states += 1
            # Apply mask on out_of_reach actions
            coarse_action_states[reset_action_batch] = 0.
            refined_action_states[reset_action_batch] = 0.

            coarse_action_states = coarse_action_states.transpose(0, 1).reshape(-1)
            refined_action_states = refined_action_states.transpose(0, 1).reshape(-1)
            gt_states = torch.cat([coarse_action_states, refined_action_states], dim=0).long()  # (6 * batch_size,)  x_c, y_c, th_c, x_r, y_r, th_r
        else:
            action_object_states = (gt_states + 1) / 2 * (num_bin - 1).unsqueeze(dim=0)
            action_object_states = (action_object_states + 1e-4).long() + 1
            action_object_states[reset_action_batch] = 0.
            gt_states = action_object_states.transpose(0, 1).reshape(-1)
        # print("after reshape gt", gt_states)
        loss = nn.functional.cross_entropy(id_logit, gt_id) + ((1 - noop_mask) * nn.functional.cross_entropy(
            states_logit, gt_states, reduction='none')).sum() / (1 - noop_mask).sum()
        return loss

    '''
    def predict(self, obs_batch, next_obs_batch):
        if self.bilevel_action:
            with torch.no_grad():
                id_logit, object_state_logit = self._predict(obs_batch, next_obs_batch)
                id_pred = (torch.argmax(id_logit, dim=-1) - 1).unsqueeze(dim=-1)  # (batch_size, 1)
                state_pred = torch.argmax(object_state_logit, dim=-1).reshape(2, len(self.num_bin), -1)  # (2, 3, batch_size)
                num_bin = torch.from_numpy(self.num_bin).unsqueeze(dim=-1).float().to(id_logit.device)  # (3, 1)
                coarse = state_pred[0]  # (3, batch_size)
                refined = state_pred[1] / num_bin * 2
                state_pred = (coarse + refined) / num_bin * 2 - 1
                state_pred = torch.transpose(state_pred, 0, 1)  # (batch_size, 3)
                prediction = torch.cat([id_pred, state_pred], dim=-1)  # (batch_size, 4)
        else:
            with torch.no_grad():
                id_logit, object_state_logit = self._predict(obs_batch, next_obs_batch)
                id_pred = (torch.argmax(id_logit, dim=-1) - 1).unsqueeze(dim=-1)
                state_pred = torch.argmax(object_state_logit, dim=-1)
                num_bin = torch.from_numpy(self.num_bin).unsqueeze(dim=-1).float().to(id_logit.device)
                state_pred = state_pred / (num_bin - 1) * 2 - 1
                state_pred = torch.transpose(state_pred, 0, 1)
                prediction = torch.cat([id_pred, state_pred], dim=-1)
        return prediction
    '''
