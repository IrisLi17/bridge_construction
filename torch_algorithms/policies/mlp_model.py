import torch
from torch import nn
import torch.nn.functional as F
from .distributions import FixedNormal


class MLPPolicy(nn.Module):
    def __init__(self, obs_shape, action_shape, layers=(64, 64), aux_head=True, arch="shared"):
        super(MLPPolicy, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.aux_head = aux_head
        self.arch = arch
        # self.detach_value_head = (arch == "shared" and aux_head)
        self.detach_value_head = False
        self.layers = nn.ModuleList()
        in_dim = self.obs_shape[0]
        for fdim in layers:
            self.layers.append(nn.Linear(in_dim, fdim))
            in_dim = fdim
        if arch == "dual":
            self.critic_layers = nn.ModuleList()
            in_dim = self.obs_shape[0]
            for fdim in layers:
                self.critic_layers.append(nn.Linear(in_dim, fdim))
                in_dim = fdim
        self.policy_linear = nn.Linear(layers[-1], self.action_shape[0])  # TODO: change to continuous action
        self.policy_logstd = nn.Parameter(torch.zeros(self.action_shape[0]))
        self.value_linear = nn.Linear(layers[-1], 1)
        if self.aux_head:
            self.aux_value_linear = nn.Linear(layers[-1], 1)
        self.n_values = 1

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return 1

    def _encode(self, x, encode_critic=False):
        x = x.to(torch.float32)
        if not encode_critic:
            for layer in self.layers:
                x = layer(x)
                x = F.relu(x)
        else:
            for layer in self.critic_layers:
                x = layer(x)
                x = F.relu(x)
        return x

    def forward(self, obs, rnn_hxs=None, rnn_mask=None, forward_value=True, aux_phase=False):
        policy_x = self._encode(obs)
        policy_logit = self.policy_linear(policy_x)
        dist = FixedNormal(loc=policy_logit, scale=torch.exp(self.policy_logstd))
        value = None
        if forward_value:
            if aux_phase:
                value = self.aux_value_linear(policy_x)
            else:
                critic_x = policy_x if self.arch == "shared" else self._encode(obs, encode_critic=True)
                value = self.value_linear(critic_x.detach() if self.detach_value_head else critic_x)
            value = value.unsqueeze(dim=1)

        return value, dist

    def forward_policy(self, x, rnn_hxs=None, rnn_mask=None):
        _, dist = self.forward(x, rnn_hxs, rnn_mask, forward_value=False)
        return dist, None

    def forward_policy_with_aux_head(self, x, rnn_hxs=None, rnn_mask=None):
        aux_value, dist = self.forward(x, rnn_hxs, rnn_mask, forward_value=True, aux_phase=True)
        return dist, aux_value

    def act(self, x, rnn_hxs, masks, deterministic=False):
        value, dist = self.forward(x, rnn_hxs, masks, forward_value=True, aux_phase=False)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, x, rnn_hxs, masks):
        value, _ = self.forward(x, rnn_hxs, masks, forward_value=True, aux_phase=False)
        return value

    def evaluate_actions(self, x, rnn_hxs, masks, actions):
        _, dist = self.forward(x, rnn_hxs, masks, forward_value=False)
        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()
        return action_log_probs, dist_entropy, rnn_hxs
