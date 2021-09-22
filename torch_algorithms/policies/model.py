import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from .distributions import Bernoulli, Categorical, DiagGaussian, init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs=None, masks=None):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    # def entropy(self, inputs, rnn_hxs=None, masks=None):
    #     _, actor_features, _ = self.base(inputs, rnn_hxs, masks)
    #     dist = self.dist(actor_features)
    #     dist_entropy = dist.entropy().mean()
    #     return dist_entropy

    # def neglogp(self, inputs, action):
    #     _, actor_features, _ = self.base(inputs, None, None)
    #     dist = self.dist(actor_features)
    #     action_log_probs = dist.log_probs(action)
    #     return -action_log_probs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class PhasicPolicy(nn.Module):
    def forward(self, obs, rnn_hxs=None, masks=None, forward_policy=True, forward_value=True, aux_phase=False, priority_head=False) -> (Any, Any):
        raise NotImplementedError

    def forward_policy(self, x, rnn_hxs=None, masks=None) -> (Any, Any):
        _, dist, rnn_hxs = self.forward(x, rnn_hxs, masks, forward_value=False)
        return dist, None
    
    def forward_policy_and_value(self, x, rnn_hxs=None, masks=None) -> (Any, Any):
        value, dist, rnn_hxs = self.forward(x, rnn_hxs, masks, forward_policy=True, forward_value=True)
        del rnn_hxs
        return value, dist

    def forward_policy_with_aux_head(self, x, rnn_hxs=None, masks=None) -> (Any, Any):
        aux_value, dist, rnn_hxs = self.forward(x, rnn_hxs, masks, forward_value=True, aux_phase=True)
        return dist, aux_value

    def act(self, x, rnn_hxs, masks, deterministic=False) -> (Any, Any, Any, Any):
        raise NotImplementedError

    def get_value(self, x, rnn_hxs, masks, priority_head=False) -> Any:
        # TODO: leakage since we also forward the action?
        value, _, rnn_hxs = self.forward(x, rnn_hxs, masks, forward_policy=False, forward_value=True, aux_phase=False, priority_head=priority_head)
        del rnn_hxs
        return value

    def evaluate_actions(self, x, rnn_hxs, masks, action, compute_entropy=True) -> (Any, Any, Any):
        raise NotImplementedError


class RecurrentPhasicPolicy(nn.Module):
    def forward(self, obs, rnn_hxs, masks, forward_value=True, aux_phase=False) -> (Any, Any):
        raise NotImplementedError

    def forward_policy(self, x, rnn_hxs, masks) -> (Any, Any):
        _, dist = self.forward(x, rnn_hxs, masks, forward_value=False)
        return dist, None

    def forward_policy_with_aux_head(self, x, rnn_hxs, masks) -> (Any, Any):
        aux_value, dist = self.forward(x, rnn_hxs, masks, forward_value=True, aux_phase=True)
        return dist, aux_value

    def act(self, x, rnn_hxs, masks, deterministic=False) -> (Any, Any, Any, Any):
        raise NotImplementedError

    def get_value(self, x, rnn_hxs, masks) -> Any:
        value, _ = self.forward(x, rnn_hxs, masks, forward_value=True, aux_phase=False)
        return value

    def evaluate_actions(self, x, rnn_hxs, masks, action) -> (Any, Any, Any):
        raise NotImplementedError


class HybridPolicy(nn.Module):
    def __init__(self, obs_shape, discrete_action_dim, continuous_action_shape, base=None, base_kwargs=None):
        super(HybridPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        self.continuous_action_shape = continuous_action_shape

        self.dist_discrete = Categorical(self.base.output_size, discrete_action_dim)  # num of blocks
        self.dist_continuous = DiagGaussian(self.base.output_size, continuous_action_shape) # dim of continuous action

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # dist = self.dist(actor_features)
        dist_discrete = self.dist_discrete(actor_features)
        dist_continuous = self.dist_continuous(actor_features)

        if deterministic:
            action_discrete = dist_discrete.mode()
            action_continuous = dist_continuous.mode()
            # action = torch.cat([dist_discrete.mode().float(), dist_continuous.mode()], dim=-1)
        else:
            action_discrete = dist_discrete.sample()
            action_continuous = dist_continuous.sample()
            # action = torch.cat([dist_discrete.sample().float(), dist_continuous.sample()], dim=-1)

        action = torch.cat([action_discrete.float(), action_continuous], dim=-1)
        action_discrete_log_probs = dist_discrete.log_probs(action_discrete)
        action_continuous_log_probs = dist_continuous.log_probs(action_continuous)
        # action_log_probs = dist.log_probs(action)
        action_log_probs = action_discrete_log_probs + action_continuous_log_probs
        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # dist = self.dist(actor_features)
        dist_discrete = self.dist_discrete(actor_features)
        dist_continuous = self.dist_continuous(actor_features)

        action_discrete = torch.narrow(action, dim=-1, start=0, length=1)
        action_continuous = torch.narrow(action, dim=-1, start=1, length=self.continuous_action_shape)
        # action_log_probs = dist.log_probs(action)
        action_discrete_log_probs = dist_discrete.log_probs(action_discrete)
        action_continuous_log_probs = dist_continuous.log_probs(action_continuous)
        action_log_probs = action_discrete_log_probs + action_continuous_log_probs
        # dist_entropy = dist.entropy().mean()
        dist_discrete_entropy = dist_discrete.entropy()
        dist_continuous_entropy = dist_continuous.entropy()
        dist_entropy = (dist_discrete_entropy + dist_continuous_entropy.sum(dim=-1).unsqueeze(dim=0)).mean() / (self.continuous_action_shape + 1)

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        # self.actor[0].weight.retain_grad()
        # self.actor[2].weight.retain_grad()

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        # self.critic[0].weight.retain_grad()

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        self.hidden_critic = hidden_critic
        self.hidden_actor = hidden_actor
        # self.hidden_actor.retain_grad()
        # self.hidden_critic.retain_grad()

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


if  __name__ == "__main__":
    from gym.spaces import Box
    policy = Policy(obs_shape=(5,), action_space=Box(low=-1., high=1., shape=(4,)), base_kwargs={'recurrent': False})
    optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4, eps=1e-5)
    obs = np.random.uniform(-1., 1., size=(10, 5))
    with torch.no_grad():
        action, value, old_neglogpac = policy.step(torch.from_numpy(obs).float())
    action = action.numpy()
    value = value.numpy()
    # advs = np.random.uniform(-1., 1., size=(10,))
    advs = np.ones((10,), dtype=np.float32)
    print('test advs', advs)
    neglogpac = policy.neglogp(torch.from_numpy(obs).float(), torch.from_numpy(action))
    print('neglogpac before', neglogpac)
    for i in range(100):
        neglogpac = policy.neglogp(torch.from_numpy(obs).float(), torch.from_numpy(action))
        ratio = torch.exp(old_neglogpac.detach() - neglogpac)
        loss = (-torch.from_numpy(advs).float().detach() * ratio).mean()
        if i % 20 == 0:
            print('loss', loss.squeeze(), 'ratio', ratio.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after_neglogpac = policy.neglogp(torch.from_numpy(obs).float(), torch.from_numpy(action))
    print('neglogpac after', after_neglogpac)
