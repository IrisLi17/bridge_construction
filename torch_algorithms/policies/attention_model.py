import torch
from torch import nn
from .model import PhasicPolicy, RecurrentPhasicPolicy
from .distributions import FixedCategorical, FixedNormal
import numpy as np


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
    dk = k.shape[-1]
    scaled_qk = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_qk += (mask * -1e9)  # 1: we don't want it, 0: we want it
    attention_weights = nn.functional.softmax(scaled_qk, dim=-1)  # (..., seq_len_q, seq_len_k)
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, feature_dim)
    return output, attention_weights


def ffn(feed_forward_network, x):
    for i in range(len(feed_forward_network) - 1):
        x = nn.functional.relu(feed_forward_network[i](x))
    x = feed_forward_network[-1](x)
    return x


class SelfAttentionBase(nn.Module):
    def __init__(self, input_dim, feature_dim, n_heads=1):
        super(SelfAttentionBase, self).__init__()
        self.n_heads = n_heads
        self.q_linear = nn.Linear(input_dim, feature_dim)
        self.k_linear = nn.Linear(input_dim, feature_dim)
        self.v_linear = nn.Linear(input_dim, feature_dim)
        # self.qkv_linear = nn.Linear(input_dim, feature_dim)
        self.dense = nn.Linear(feature_dim, feature_dim)
        # self.layer_norm = nn.LayerNorm(feature_dim)

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return 1

    def split_head(self, x):
        x_size = x.size()
        assert isinstance(x_size[2] // self.n_heads, int)
        x = torch.reshape(x, [-1, x_size[1], self.n_heads, x_size[2] // self.n_heads])
        x = torch.transpose(x, 1, 2)  # (batch_size, n_heads, seq_len, depth)
        return x

    def forward(self, q, k, v, mask):
        assert len(q.size()) == 3
        # x = self.qkv_linear(x)  # (batch_size, seq_len, feature_dim * n_heads)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q_heads = self.split_head(q)
        k_heads = self.split_head(k)
        v_heads = self.split_head((v))
        # x_heads = self.split_head(x)  # (batch_size, n_heads, seq_len, depth)
        # mask = torch.unsqueeze(mask, dim=1).unsqueeze(dim=-1)  # (batch_size, 1, seq_len, 1)
        mask = torch.unsqueeze(mask, dim=1).unsqueeze(dim=2)  # (batch_size, 1, 1, seq_len)
        attention_out, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        attention_out = torch.transpose(attention_out, 1, 2)  # (batch_size, seq_len_q, n_heads, depth)
        out_size = attention_out.size()
        attention_out = torch.reshape(attention_out, [-1, out_size[1], out_size[2] * out_size[3]])
        attention_out = self.dense(attention_out)
        # out = self.layer_norm(x + attention_out)
        return attention_out


class SelfAttentionFeatureExtractor(nn.Module):
    def __init__(self, obs_shape, n_object, feature_dim, n_attention_blocks, object_dim, base_kwargs=None):
        super().__init__()
        self.obs_embed = nn.ModuleList(
            [nn.Linear(object_dim + obs_shape[0] - object_dim * n_object, feature_dim),
             nn.Linear(feature_dim, feature_dim)])
        self.embed_layernorm = nn.LayerNorm(feature_dim)
        self.base = nn.ModuleList(
            [SelfAttentionBase(input_dim=feature_dim, feature_dim=feature_dim, **base_kwargs)
             for _ in range(n_attention_blocks)])
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(n_attention_blocks)])
        self.feed_forward_network = nn.ModuleList([nn.ModuleList([nn.Linear(feature_dim, feature_dim),
                                                                  nn.Linear(feature_dim, feature_dim)]) for _
                                                   in range(n_attention_blocks)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(n_attention_blocks)])
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self.lstm = None

    def forward(self, x, mask, rnn_hxs=None, rnn_masks=None):
        obs_embed = x
        for i in range(len(self.obs_embed) - 1):
            obs_embed = nn.functional.relu(self.obs_embed[i](obs_embed))
        obs_embed = self.obs_embed[-1](obs_embed)
        obs_embed = self.embed_layernorm(obs_embed)

        latent = obs_embed
        for i in range(len(self.base)):
            attn_output = self.base[i](latent, latent, latent,
                                       mask)  # (batch_size, seq_len, feature_dim)
            out1 = self.layer_norm1[i](latent + attn_output)
            ffn_output = ffn(self.feed_forward_network[i], out1)
            latent = self.layer_norm2[i](ffn_output)

        return latent, rnn_hxs


class HybridAttentionPolicy(PhasicPolicy):
    def __init__(self, obs_shape, discrete_action_dim, continuous_action_shape, feature_dim, n_attention_blocks,
                 object_dim, has_cliff=True, aux_head=True, arch="dual", base_kwargs=None):
        super(HybridAttentionPolicy, self).__init__()
        self.obs_shape = obs_shape
        self.has_cliff = has_cliff
        self.n_object = discrete_action_dim + 2 * has_cliff  # 2 refers to two cliffs as objects
        self.object_dim = object_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_shape = continuous_action_shape
        self.feature_dim = feature_dim
        self.n_attention_blocks = n_attention_blocks
        self.arch = arch
        self.attention_feature_extractor = \
            SelfAttentionFeatureExtractor(obs_shape, self.n_object, feature_dim, n_attention_blocks, object_dim,
                                          base_kwargs)
        if arch == "dual":
            self.critic_attention_feature_extractor = \
                SelfAttentionFeatureExtractor(obs_shape, self.n_object, feature_dim, n_attention_blocks, object_dim,
                                              base_kwargs)
        # self.obs_embed = nn.ModuleList([nn.Linear(self.object_dim + obs_shape[0] - self.object_dim * self.n_object, self.feature_dim),
        #                                 nn.Linear(self.feature_dim, self.feature_dim)])
        # self.critic_obs_embed = nn.ModuleList([nn.Linear(self.object_dim + obs_shape[0] - self.object_dim * self.n_object, self.feature_dim),
        #                                        nn.Linear(self.feature_dim, self.feature_dim)])
        # self.base = SelfAttentionBase(input_dim=self.feature_dim, feature_dim=self.feature_dim, **base_kwargs)
        # # TODO: try to use separate networks for actor and critic
        # self.base = nn.ModuleList([SelfAttentionBase(input_dim=self.feature_dim, feature_dim=self.feature_dim, **base_kwargs)
        #                            for _ in range(self.n_attention_blocks)])
        # self.layer_norm1 = nn.ModuleList([nn.LayerNorm(self.feature_dim) for _ in range(self.n_attention_blocks)])
        # self.feed_forward_network = nn.ModuleList([nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim),
        #                                            nn.Linear(self.feature_dim, self.feature_dim)]) for _ in range(self.n_attention_blocks)])
        # self.layer_norm2 = nn.ModuleList([nn.LayerNorm(self.feature_dim) for _ in range(self.n_attention_blocks)])
        # self.critic_base = nn.ModuleList([SelfAttentionBase(input_dim=self.feature_dim, feature_dim=self.feature_dim, **base_kwargs)
        #                            for _ in range(self.n_attention_blocks)])
        # self.critic_layer_norm1 = nn.ModuleList([nn.LayerNorm(self.feature_dim) for _ in range(self.n_attention_blocks)])
        # self.critic_feed_forward_network = nn.ModuleList([nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim),
        #                                            nn.Linear(self.feature_dim, self.feature_dim)]) for _ in range(self.n_attention_blocks)])
        # self.critic_layer_norm2 = nn.ModuleList([nn.LayerNorm(self.feature_dim) for _ in range(self.n_attention_blocks)])

        self.critic_linears = nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim),
                                             nn.Linear(self.feature_dim, self.feature_dim)])
        self.critic_final = nn.Linear(self.feature_dim, 1)

        if aux_head:
            self.aux_head_linears = nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim),
                                                   nn.Linear(self.feature_dim, self.feature_dim)])
            self.aux_head_final = nn.Linear(self.feature_dim, 1)
        else:
            self.aux_head_linears = self.aux_head_final = None

        self.discrete_action_linear = nn.Linear(self.feature_dim, 1)
        self.continuous_action_q = nn.Parameter(torch.normal(mean=0., std=0.01, size=(self.feature_dim + 1,)))
        self.actor_linears = nn.ModuleList([nn.Linear(self.feature_dim + 1, self.feature_dim),
                                            nn.Linear(self.feature_dim, self.feature_dim)])
        self.continuous_mean = nn.Linear(self.feature_dim, continuous_action_shape)
        self.continuous_logstd = nn.Parameter(torch.zeros(continuous_action_shape))
        self._initialize()

    def _initialize(self):
        for f in self.critic_linears:
            nn.init.orthogonal_(f.weight, gain=np.sqrt(2))
            nn.init.constant_(f.bias, 0.)
        nn.init.orthogonal_(self.critic_final.weight, gain=np.sqrt(2))
        nn.init.constant_(self.critic_final.bias, 0.)
        # TODO: value head in pi
        if self.aux_head_linears is not None:
            for f in self.aux_head_linears:
                nn.init.orthogonal_(f.weight, gain=np.sqrt(2))
                nn.init.constant_(f.bias, 0.)
            nn.init.orthogonal_(self.aux_head_final.weight, gain=np.sqrt(2))
            nn.init.constant_(self.aux_head_final.bias, 0.)

        nn.init.orthogonal_(self.discrete_action_linear.weight, gain=0.01)
        nn.init.constant_(self.discrete_action_linear.bias, 0.)
        for f in self.actor_linears:
            nn.init.orthogonal_(f.weight, gain=np.sqrt(2))
            nn.init.constant_(f.bias, 0.)
        nn.init.orthogonal_(self.continuous_mean.weight)
        nn.init.constant_(self.continuous_mean.bias, 0.)

    @property
    def is_recurrent(self):
        return self.attention_feature_extractor.base[0].is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.attention_feature_extractor.base[0].recurrent_hidden_state_size

    def parse_input(self, x):
        # x: (batch_size, observation_size)
        objects = torch.narrow(x, dim=-1, start=0, length=self.object_dim * self.n_object).reshape([-1, self.n_object, self.object_dim])
        others = torch.narrow(x, dim=-1, start=self.object_dim * self.n_object, length=self.obs_shape[0] - self.object_dim * self.n_object)\
            .unsqueeze(dim=1).expand(-1, self.n_object, -1)
        objects_and_others = torch.cat([objects, others], dim=-1)  # (batch_size, n_object, input_dim)
        if self.has_cliff:
            masks = torch.norm(objects[:, :, :-1], dim=-1) < 1e-3  # (batch_size, n_object)?
            action_masks = torch.logical_or(torch.norm(objects[:, :, -1:], dim=-1) > 1e-3, masks)  # We assume movable objects are with type 0.
        else:
            masks = torch.norm(objects, dim=-1) < 1e-3
            action_masks = masks
        assert len(masks.size()) == 2 and masks.size()[0] == x.size()[0] and masks.size()[1] == self.n_object, masks.size()
        assert len(action_masks.size()) == 2 and action_masks.size()[0] == x.size()[0] and action_masks.size()[1] == self.n_object, action_masks.size()
        return objects_and_others, masks.detach(), action_masks.detach()

    def _encode(self, x, encode_critic=False):
        x, mask, action_mask = self.parse_input(x)
        if not encode_critic:
            return self.attention_feature_extractor.forward(x, mask), mask, action_mask
        else:
            return self.critic_attention_feature_extractor.forward(x, mask), mask, action_mask

    # check the following API
    def forward(self, obs, forward_value=True, aux_phase=False):
        actor_latent, mask, action_mask = self._encode(obs)
        discrete_action_latent = self.discrete_action_linear(actor_latent)  # (batch_size, seq_len, 1)
        discrete_action_latent -= torch.unsqueeze(action_mask, dim=-1) * 1e9
        dist_discrete = FixedCategorical(logits=discrete_action_latent.squeeze(dim=-1))
        norm_discrete_latent = nn.functional.softmax(discrete_action_latent, dim=1)
        aug_attention_latent = torch.cat([actor_latent, norm_discrete_latent],
                                         dim=-1)  # (batch_size, seq_len, 1 + feature_dim)
        actor_q = torch.unsqueeze(self.continuous_action_q, dim=0).unsqueeze(dim=1)
        actor_attention_out, _ = scaled_dot_product_attention(actor_q, aug_attention_latent,
                                                              aug_attention_latent, mask.unsqueeze(dim=1))
        continuous_action_latent = torch.squeeze(actor_attention_out, dim=1)  # (batch_size, 1 + feature_dim)
        for f in self.actor_linears:
            continuous_action_latent = nn.functional.relu(f(continuous_action_latent))
        continuous_mean = self.continuous_mean(continuous_action_latent)
        dist_continuous = FixedNormal(loc=continuous_mean, scale=torch.exp(self.continuous_logstd))
        value = None
        if forward_value:
            if aux_phase:
                critic_latent = actor_latent
                critic_out = torch.sum(critic_latent * (1. - torch.unsqueeze(mask.float(), dim=-1)), dim=1) / torch.sum(
                    1. - torch.unsqueeze(mask.float(), dim=-1), dim=1)
                for f in self.aux_head_linears:
                    critic_out = nn.functional.relu(f(critic_out))
                value = self.aux_head_final(critic_out)
            else:
                critic_latent = self._encode(obs, encode_critic=True)[0] if self.arch == "dual" else actor_latent
                critic_out = torch.sum(critic_latent * (1. - torch.unsqueeze(mask.float(), dim=-1)), dim=1) / torch.sum(
                    1. - torch.unsqueeze(mask.float(), dim=-1), dim=1)
                for f in self.critic_linears:
                    critic_out = nn.functional.relu(f(critic_out))
                value = self.critic_final(critic_out)

        return value, (dist_discrete, dist_continuous)

    def act(self, x, rnn_hxs, masks, deterministic=False):
        value, dist = self.forward(x, forward_value=True, aux_phase=False)
        if deterministic:
            action_discrete = dist[0].mode()
            action_continuous = dist[1].mode()
        else:
            action_discrete = dist[0].sample()
            action_continuous = dist[1].sample()

        action = torch.cat([action_discrete.float(), action_continuous], dim=-1)
        action_discrete_log_probs = dist[0].log_probs(action_discrete)
        action_continuous_log_probs = dist[1].log_probs(action_continuous)
        action_log_probs = action_discrete_log_probs + action_continuous_log_probs
        return value, action, action_log_probs, rnn_hxs

    def evaluate_actions(self, x, rnn_hxs, masks, action):
        _, dist = self.forward(x, forward_value=False)
        action_discrete = torch.narrow(action, dim=-1, start=0, length=1)
        action_continuous = torch.narrow(action, dim=-1, start=1, length=self.continuous_action_shape)
        # action_log_probs = dist.log_probs(action)
        action_discrete_log_probs = dist[0].log_probs(action_discrete)
        action_continuous_log_probs = dist[1].log_probs(action_continuous)
        action_log_probs = action_discrete_log_probs + action_continuous_log_probs
        dist_discrete_entropy = dist[0].entropy()
        dist_continuous_entropy = dist[1].entropy()
        dist_entropy = (dist_discrete_entropy + dist_continuous_entropy.sum(dim=-1).unsqueeze(dim=0)).mean() / (
                    self.continuous_action_shape + 1)

        # return value, action_log_probs, dist_entropy, rnn_hxs
        return action_log_probs, dist_entropy, rnn_hxs


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

class MultiDiscreteAttentionPolicy(PhasicPolicy):
    '''
    Discretizing continuous dimensions into several bins
    '''
    def __init__(self, obs_shape, discrete_action_dim, continuous_action_shape, num_bin, feature_dim,
                 n_attention_blocks, object_dim, has_cliff=True, aux_head=True, arch="dual", base_kwargs=None,
                 noop=False, n_values=1, refined_action=False, bilevel_action=False, priority_head=False):
        '''
        :param obs_shape:
        :param discrete_action_dim: for the original categorical action dimension
        :param num_bin: for original continuous action dimensions
        :param base:
        :param base_kwargs:
        '''
        super(MultiDiscreteAttentionPolicy, self).__init__()
        self.obs_shape = obs_shape
        self.has_cliff = has_cliff
        self.n_object = discrete_action_dim + 2 * has_cliff  # 2 refers to two cliffs as objects
        self.object_dim = object_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_shape = continuous_action_shape
        if isinstance(num_bin, int):
            num_bin = [num_bin] * continuous_action_shape
        self.num_bin = np.asarray(num_bin, dtype=np.int)
        # self.num_bin = num_bin
        self.feature_dim = feature_dim
        self.n_attention_blocks = n_attention_blocks
        self.arch = arch
        self.noop = noop
        self.attention_feature_extractor = \
            SelfAttentionFeatureExtractor(obs_shape, self.n_object, feature_dim, n_attention_blocks, object_dim,
                                          base_kwargs)
        if arch == "dual":
            self.critic_attention_feature_extractor = \
                SelfAttentionFeatureExtractor(obs_shape, self.n_object, feature_dim, n_attention_blocks, object_dim,
                                              base_kwargs)

        # self.critic_linears = nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim),
        #                                      nn.Linear(self.feature_dim, self.feature_dim)])
        # self.critic_final = nn.Linear(self.feature_dim, 1)

        # An ensemble of values
        self.n_values = n_values
        self.critic_linears = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim),
                           nn.Linear(self.feature_dim, self.feature_dim)])
            for _ in range(n_values)])
        self.critic_final = nn.ModuleList([
            nn.Linear(self.feature_dim, 1) for _ in range(n_values)
        ])

        if aux_head:
            self.aux_head_linears = nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim),
                                                   nn.Linear(self.feature_dim, self.feature_dim)])
            self.aux_head_final = nn.Linear(self.feature_dim, 1)
        else:
            self.aux_head_linears = self.aux_head_final = None

        # Only used for computing td priority with special return
        if priority_head:
            self.priority_critic_linears = nn.ModuleList([
                nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim),
                               nn.Linear(self.feature_dim, self.feature_dim)])
                for _ in range(n_values)])
            self.priority_critic_final = nn.ModuleList([
                nn.Linear(self.feature_dim, 1) for _ in range(n_values)
            ])
        else:
            self.priority_critic_linears = None
            self.priority_critic_final = None

        # self.discrete_action_linear = nn.Linear(self.feature_dim, 1)
        # self.aggregate_action_q = nn.Parameter(torch.normal(mean=0., std=0.01, size=(self.feature_dim + 1,)))
        # self.actor_linears = nn.ModuleList([nn.Linear(self.feature_dim + 1, self.feature_dim),
        #                                     nn.Linear(self.feature_dim, self.feature_dim)])
        # TODO: revert to shallow network and same bin for all dims to see if correct ewma works
        self.object_id_linear = nn.Linear(self.feature_dim, 1)
        # self.object_id_linear = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
        #                                       nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
        #                                       nn.Linear(self.feature_dim, 1))
        if noop:
            self.no_op_linear = nn.Linear(self.feature_dim, 1)
        else:
            self.no_op_linear = None
        self.refined_action = refined_action
        self.bilevel_action = bilevel_action
        if self.bilevel_action:
            assert not self.refined_action
        if self.refined_action:
            # TODO: hierarchical action space
            self.coarse_states_linear = nn.ModuleList([nn.Linear(self.feature_dim + 1, self.num_bin[i])
                                                       for i in range(continuous_action_shape)])
            self.refine_states_linear = nn.ModuleList([nn.Linear(self.feature_dim + 1 + self.num_bin[i], self.num_bin[i])
                                                       for i in range(continuous_action_shape)])
        else:
            self.object_states_linear = nn.ModuleList([nn.Linear(self.feature_dim, self.num_bin[i])
                                                       for i in range(continuous_action_shape)])
        # self.object_states_linear = nn.ModuleList(
        #     [nn.Sequential(nn.Linear(self.feature_dim + 1, self.feature_dim), nn.ReLU(),
        #                    nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
        #                    nn.Linear(self.feature_dim, self.num_bin[i])) for i in range(continuous_action_shape)]
        # )
        # self.continuous_mean = nn.Linear(self.feature_dim, continuous_action_shape)
        # self.continuous_logstd = nn.Parameter(torch.zeros(continuous_action_shape))
        self._initialize()
        # self.dist_object_id = Categorical(self.base.output_size, self.discrete_action_dim)
        # self.dist_object_states = [Categorical(self.base.output_size, self.num_bin) for _ in range(continuous_action_shape)]

    def _initialize(self):
        for i, critic in enumerate(self.critic_linears):
            for f in critic:
                nn.init.orthogonal_(f.weight, gain=np.sqrt(2))
                nn.init.constant_(f.bias, 0.)
            nn.init.orthogonal_(self.critic_final[i].weight, gain=np.sqrt(2))
            nn.init.constant_(self.critic_final[i].bias, 0.)
        # TODO: value head in pi
        if self.aux_head_linears is not None:
            for f in self.aux_head_linears:
                nn.init.orthogonal_(f.weight, gain=np.sqrt(2))
                nn.init.constant_(f.bias, 0.)
            nn.init.orthogonal_(self.aux_head_final.weight, gain=np.sqrt(2))
            nn.init.constant_(self.aux_head_final.bias, 0.)

        if isinstance(self.object_id_linear, nn.Sequential):
            for f in self.object_id_linear:
                if isinstance(f, nn.Linear):
                    nn.init.orthogonal_(f.weight, gain=0.01)
                    nn.init.constant_(f.bias, 0.)
        else:
            nn.init.orthogonal_(self.object_id_linear.weight, gain=0.01)
            nn.init.constant_(self.object_id_linear.bias, 0.)
        if self.no_op_linear is not None:
            nn.init.orthogonal_(self.no_op_linear.weight, gain=0.01)
            nn.init.constant_(self.no_op_linear.bias, 0.)

        if self.refined_action:
            for net in self.coarse_states_linear:
                nn.init.orthogonal_(net.weight, gain=0.005)
                nn.init.constant_(net.bias, 0.)
            for net in self.refine_states_linear:
                nn.init.orthogonal_(net.weight, gain=0.005)
                nn.init.constant_(net.bias, 0.)
        else:
            for net in self.object_states_linear:
                nn.init.orthogonal_(net.weight, gain=0.01)
                nn.init.constant_(net.bias, 0.)

    @property
    def is_recurrent(self):
        return self.attention_feature_extractor.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.attention_feature_extractor.recurrent_hidden_state_size

    def parse_input(self, x):
        # x: (batch_size, observation_size)
        with torch.no_grad():
            objects = torch.narrow(x, dim=-1, start=0, length=self.object_dim * self.n_object).reshape([-1, self.n_object, self.object_dim])
            others = torch.narrow(x, dim=-1, start=self.object_dim * self.n_object, length=self.obs_shape[0] - self.object_dim * self.n_object)\
                .unsqueeze(dim=1).expand(-1, self.n_object, -1)
            if self.has_cliff:
                masks = torch.norm(objects[:, :, :-1], dim=-1) < 1e-3  # (batch_size, n_object)?
                action_masks = torch.logical_or(torch.norm(objects[:, :, -1:], dim=-1) > 1e-3, masks)  # We assume movable objects are with type 0.
            else:
                masks = torch.norm(objects, dim=-1) < 1e-3  # If there is no cliff, there is no object_type in observation
                action_masks = masks
            objects_and_others = torch.cat([objects, others], dim=-1)  # (batch_size, n_object, input_dim)

        assert len(masks.size()) == 2 and masks.size()[0] == x.size()[0] and masks.size()[1] == self.n_object, masks.size()
        assert len(action_masks.size()) == 2 and action_masks.size()[0] == x.size()[0] and action_masks.size()[1] == self.n_object, action_masks.size()
        # TODO: apply normalization
        # _mean = torch.from_numpy(np.zeros(objects_and_others.shape[-1]))
        # _mean[0] = 1.3
        # _mean[1] = 0.6
        # _mean[self.object_dim * self.n_object:] = 0.2  # Assume all other dimensions are skyline
        # _std = torch.from_numpy(np.ones(objects_and_others.shape[-1]))
        # _std[1] = 0.6
        # _std[2] = 0.1
        # objects_and_others = ((objects_and_others - _mean.to(objects_and_others.device)) / _std.to(objects_and_others.device)).float()
        return objects_and_others, masks.detach(), action_masks.detach()

    def _encode(self, x, encode_critic=False, rnn_hxs=None, rnn_masks=None):
        x, mask, action_mask = self.parse_input(x)
        if not encode_critic:
            latent, rnn_hxs = self.attention_feature_extractor.forward(x, mask, rnn_hxs, rnn_masks)
            return latent, rnn_hxs, mask, action_mask
        else:
            latent, rnn_hxs = self.critic_attention_feature_extractor.forward(x, mask, rnn_hxs, rnn_masks)
            return latent, rnn_hxs, mask, action_mask

    def forward(self, obs, rnn_hxs=None, rnn_masks=None, forward_policy=True, forward_value=False, aux_phase=False, priority_head=False):
        # recurrent support
        actor_latent, rnn_hxs, mask, action_mask = self._encode(obs, rnn_hxs=rnn_hxs, rnn_masks=rnn_masks)
        dist = None
        if forward_policy:
            # TODO: add a no-op action to dist_id
            object_id_logit = self.object_id_linear(actor_latent)  # (batch_size, seq_len, 1)
            object_id_logit -= torch.unsqueeze(action_mask, dim=-1) * 1e9
            if self.noop:
                no_op_logit = self.no_op_linear(torch.mean(actor_latent, dim=1))  # (batch_size, feature_dim) -> (batch_size, 1)
                total_id_logit = torch.cat([no_op_logit, object_id_logit.squeeze(dim=-1)], dim=-1)  # (batch_size, 1 + seq_len)
            else:
                total_id_logit = object_id_logit.squeeze(dim=-1)
            # dist_id = FixedCategorical(logits=total_id_logit)
            # norm_discrete_latent = nn.functional.softmax(object_id_logit, dim=1)  # (batch_size, seq_len, 1)
            # # Shift the real action dimension
            # mb_active_objects = torch.sum((1 - action_mask.float()), dim=-1, keepdim=True)  # (batch_size, seq_len)
            # center_discrete_latent = norm_discrete_latent - torch.unsqueeze((1 - action_mask.float()) / mb_active_objects, dim=-1)
            # aug_attention_latent = torch.cat([actor_latent, center_discrete_latent],
            #                                  dim=-1)  # (batch_size, seq_len, 1 + feature_dim)
            # actor_q = torch.unsqueeze(self.aggregate_action_q, dim=0).unsqueeze(dim=1)
            # actor_attention_out, _ = scaled_dot_product_attention(actor_q, aug_attention_latent,
            #                                                       aug_attention_latent, mask.unsqueeze(dim=1))
            # agg_action_latent = torch.squeeze(actor_attention_out, dim=1)  # (batch_size, feature_dim + 1)

            if not self.refined_action:
                object_states_logit = [self.object_states_linear[i](actor_latent) for i in range(self.continuous_action_shape)]  # [(batch_size, seq_len, n_bin), (),..]
                if self.bilevel_action:
                    # coarse_states_logit = [object_states_logit[i][..., :self.num_bin[i] // 2] for i in range(self.continuous_action_shape)]
                    # refined_states_logit = [object_states_logit[i][..., self.num_bin[i] // 2:] for i in range(self.continuous_action_shape)]
                    coarse_states_logit = [torch.narrow(object_states_logit[i], -1, 0, self.num_bin[i] // 2) for i in range(self.continuous_action_shape)]
                    refined_states_logit = [torch.narrow(object_states_logit[i], -1, self.num_bin[i] // 2, self.num_bin[i] // 2) for i in range(self.continuous_action_shape)]
                    # dist_coarse_states = [FixedCategorical(logits=coarse_states_logit[i]) for i in
                    #                       range(self.continuous_action_shape)]
                    # dist_refined_states = [FixedCategorical(logits=refined_states_logit[i]) for i in
                    #                        range(self.continuous_action_shape)]
                    # dist = tuple([dist_id] + dist_coarse_states + dist_refined_states)
                    dist = tuple([total_id_logit] + coarse_states_logit + refined_states_logit)
                else:
                    # dist_states = [FixedCategorical(logits=object_states_logit[i]) for i in range(self.continuous_action_shape)]
                    # dist = tuple([dist_id] + dist_states)
                    dist = tuple([total_id_logit] + object_states_logit)
            else:
                raise NotImplementedError
                # TODO: create a hierarchical dist_state
                coarse_states_logit = [self.coarse_states_linear[i](agg_action_latent) for i in range(self.continuous_action_shape)]
                refined_action_input = [torch.cat([agg_action_latent, nn.functional.softmax(coarse_states_logit[i])], dim=-1)
                                        for i in range(self.continuous_action_shape)]
                refined_states_logit = [self.refine_states_linear[i](refined_action_input[i]) for i in range(self.continuous_action_shape)]
                dist_coarse_states = [FixedCategorical(logits=coarse_states_logit[i]) for i in range(self.continuous_action_shape)]
                dist_refined_states = [FixedCategorical(logits=refined_states_logit[i]) for i in range(self.continuous_action_shape)]
                dist = tuple([dist_id] + dist_coarse_states + dist_refined_states)

        value = None
        if forward_value:
            if aux_phase:
                critic_latent = actor_latent
                critic_out = torch.sum(critic_latent * (1. - torch.unsqueeze(mask.float(), dim=-1)), dim=1) / torch.sum(
                    1. - torch.unsqueeze(mask.float(), dim=-1), dim=1)
                for f in self.aux_head_linears:
                    critic_out = nn.functional.relu(f(critic_out))
                value = self.aux_head_final(critic_out)
            else:
                critic_latent = self._encode(obs, encode_critic=True)[0] if self.arch == "dual" else actor_latent
                critic_out = torch.sum(critic_latent * (1. - torch.unsqueeze(mask.float(), dim=-1)), dim=1) / torch.sum(
                    1. - torch.unsqueeze(mask.float(), dim=-1), dim=1)
                # TODO: an ensemble of values
                value = []
                shared_feature = critic_out
                if priority_head:
                    for critic_id in range(self.n_values):
                        _out = shared_feature if critic_id == 0 else shared_feature.detach()
                        for f in self.priority_critic_linears[critic_id]:
                            _out = nn.functional.relu(f(_out))
                        value.append(self.priority_critic_final[critic_id](_out))
                else:
                    for critic_id in range(self.n_values):
                        _out = shared_feature if critic_id == 0 else shared_feature.detach()
                        for f in self.critic_linears[critic_id]:
                            _out = nn.functional.relu(f(_out))
                        value.append(self.critic_final[critic_id](_out))
                value = torch.stack(value, dim=1)  # (batch_size, n_values, 1)
                # for f in self.critic_linears:
                #     critic_out = nn.functional.relu(f(critic_out))
                # value = self.critic_final(critic_out)

        return value, dist, rnn_hxs

    def act(self, x, rnn_hxs, masks, deterministic=False, verbose=False):
        # TODO: recurrent support
        value, dist, rnn_hxs = self.forward(x, rnn_hxs, masks, forward_value=True, aux_phase=False)
        if verbose:
            print("In act, after forward", get_memory_usage())
        dist_object_id = dist[0]
        dist_object_id = FixedCategorical(logits=dist_object_id)
        dist_object_states = dist[1:]

        # Sample object ID
        if deterministic:
            object_id = dist_object_id.mode()
        else:
            object_id = dist_object_id.sample()

        if self.noop:
            # Multiple by one hot
            onehot = torch.zeros(dist_object_states[0].size(0), dist_object_states[0].size(1) + 1).to(object_id.device)
            onehot.scatter_(1, object_id, 1)
            onehot = onehot[:, 1:]
            reconstr_logits = [(dist * onehot.unsqueeze(dim=-1).detach()).sum(dim=1) for dist in dist_object_states]
            
            # # Construct the dist again
            # if verbose:
            #     print("In act, before reconstr", get_memory_usage())
            # all_logits = [
            #     torch.cat([torch.zeros(dist.logits.size(0), 1, dist.logits.size(2), device=dist.logits.device),
            #                dist.logits], dim=1) for dist in dist_object_states]
            # all_inds = torch.from_numpy(np.arange(dist_object_states[0].logits.size(0))).to(
            #     dist_object_states[0].logits.device)
            # reconstr_logits = [logit[all_inds, object_id.long().squeeze(dim=-1)] for logit in all_logits]
            dist_object_states = [FixedCategorical(logits=logit) for logit in reconstr_logits]
            if verbose:
                print("In act, after reconstr", get_memory_usage())
        else:
            raise NotImplementedError

        # Sample object state
        if deterministic:
            object_states = [dist.mode() for dist in dist_object_states]
        else:
            object_states = [dist.sample() for dist in dist_object_states]

        num_bin = torch.from_numpy(self.num_bin).to(object_id.device)
        if self.refined_action:
            unscaled_states = [c + r / num_bin[i].float() for i, (c, r) in enumerate(zip(object_states[:self.continuous_action_shape], object_states[self.continuous_action_shape:]))]  # unsclaed action in [0, num_bin)
        elif self.bilevel_action:
            unscaled_states = [c + r / (num_bin[i].float() / 2) for i, (c, r) in enumerate(zip(object_states[:self.continuous_action_shape], object_states[self.continuous_action_shape:]))]
        else:
            unscaled_states = object_states
        if self.noop:
            action = torch.cat([object_id - 1] + unscaled_states, dim=-1).float()  # -1 is reserved for no-op
        else:
            action = torch.cat([object_id] + unscaled_states, dim=-1).float()
        if self.refined_action:
            action[:, 1:] = action[:, 1:] / num_bin * 2 - 1
        elif self.bilevel_action:
            action[:, 1:] = action[:, 1:] / (num_bin / 2.0) * 2 - 1
        else:
            action[:, 1:] = action[:, 1:] / (num_bin - 1) * 2 - 1
        # if (not self.refined_action) and (not self.bilevel_action):
        #     action[:, 1:] = action[:, 1:] / (num_bin - 1) * 2 - 1  # action[1:] should fall within [-1, 1]
        # else:
        #     action[:, 1:] = action[:, 1:] / num_bin * 2 - 1  # action[1:] should fall within [-1, 1)
        # action_log_probs = [dist_object_id.log_probs(object_id)] + [dist_object_states[i].log_probs(object_states[i]) for i in range(len(object_states))]
        # action_log_probs = torch.sum(torch.stack(action_log_probs, dim=1), dim=1)
        # Separate bi-level action log probs
        level_action_log_probs = []
        object_id_log_prob = dist_object_id.log_probs(object_id)
        for level in range(len(object_states) // self.continuous_action_shape):
            per_level_log_probs = [object_id_log_prob] + \
                                  [dist_object_states[i].log_probs(object_states[i])
                                   for i in range(level * self.continuous_action_shape,
                                                  (level + 1) * self.continuous_action_shape)]
            level_action_log_probs.append(torch.sum(torch.stack(per_level_log_probs, dim=1), dim=1))
        level_action_log_probs = torch.stack(level_action_log_probs)  # (n_level, n_process, 1)
        if verbose:
            print("In act, before return", get_memory_usage())
        return value, action, level_action_log_probs, rnn_hxs
    
    '''
    def evaluate_actions(self, x, rnn_hxs, masks, action, compute_entropy=True):
        # TODO: recurrent support
        _, dist, rnn_hxs = self.forward(x, rnn_hxs, masks, forward_value=False)
        dist_object_id = dist[0]
        dist_object_id = FixedCategorical(logits=dist_object_id)
        dist_object_states = dist[1:]
        action_object_id = torch.narrow(action, dim=-1, start=0, length=1).int()
        # action_object_states = torch.narrow(action, dim=-1, start=1, length=self.continuous_action_shape)
        action_object_states = torch.narrow(action, dim=-1, start=1, length=action.shape[1] - 1)
        num_bin = torch.from_numpy(self.num_bin).to(action.device)
        # action_object_states fall within [-1, 1]
        if self.refined_action:
            action_object_states = (action_object_states + 1) / 2 * num_bin  # [0, num_bin)
            coarse_action_states = (action_object_states + 1e-4).int()
            refined_action_states = ((action_object_states - coarse_action_states) * num_bin + 1e-4).int()
            action_object_states = torch.cat([coarse_action_states, refined_action_states], dim=-1)
        elif self.bilevel_action:
            action_object_states = (action_object_states + 1) / 2 * num_bin / 2
            coarse_action_states = (action_object_states + 1e-4).int()
            refined_action_states = ((action_object_states - coarse_action_states) * num_bin / 2 + 1e-4).int()
            action_object_states = torch.cat([coarse_action_states, refined_action_states], dim=-1)
        else:
            # action_object_states = ((action_object_states + 1) / 2 * (num_bin - 1)).int()
            action_object_states = torch.round((action_object_states + 1) / 2 * (num_bin - 1)).int()
        if self.noop:
            _object_id = (action_object_id + 1).long()
            # Multiple by one hot
            onehot = torch.from_numpy(np.zeros((dist_object_states[0].size(0), dist_object_states[0].size(1) + 1))).to(_object_id.device)
            onehot.scatter_(1, _object_id, 1)
            onehot = onehot[:, 1:]
            reconstr_logits = [(dist * onehot.unsqueeze(dim=-1).detach()).sum(dim=1) for dist in dist_object_states]
            # # Construct the dist again, Check if the gradient remains
            # all_logits = [torch.cat([torch.zeros(dist.logits.size(0), 1, dist.logits.size(2), device=dist.logits.device),
            #                          dist.logits], dim=1) for dist in dist_object_states]
            # all_inds = torch.from_numpy(np.arange(dist_object_states[0].logits.size(0))).to(dist_object_states[0].logits.device)
            # reconstr_logits = [logit[all_inds, _object_id.long().squeeze(dim=-1)] for logit in all_logits]
            dist_object_states = [FixedCategorical(logits=logit) for logit in reconstr_logits]
            # del onehot, reconstr_logits
        else:
            raise NotImplementedError
            _object_id = action_object_id
            # action_log_probs = dist_object_id.log_probs(action_object_id)
        # print(action_log_probs.shape)
        log_probs = []
        object_id_log_prob = dist_object_id.log_probs(_object_id)
        for level in range(len(dist_object_states) // self.continuous_action_shape):
            level_action_log_probs = [object_id_log_prob] + [dist_object_states[j].log_probs(action_object_states[:, j].detach())
                                      for j in range(level * self.continuous_action_shape,
                                                     (level + 1) * self.continuous_action_shape)]
            log_probs.append(torch.sum(torch.stack(level_action_log_probs, dim=1), dim=1))  # TODO: bug, not exactly the same as that from act
        log_probs = torch.stack(log_probs)  # (n_level, batch_size, 1)
        # for j in range(len(dist_object_states)):
        #     action_log_probs += dist_object_states[j].log_probs(action_object_states[:, j: j + 1])
        dist_entropy = None
        if compute_entropy:
            dist_entropy = dist_object_id.entropy()
            for j in range(len(dist_object_states)):
                dist_entropy += dist_object_states[j].entropy()
            # print(dist_entropy.shape, dist_entropy[0])
            dist_entropy = dist_entropy.mean()
        # del dist_object_states
        return log_probs, dist_entropy, rnn_hxs
    '''

    def evaluate_actions(self, x, rnn_hxs, masks, action, compute_entropy=True):
        _, dist, rnn_hxs = self.forward(x, rnn_hxs, masks, forward_value=False)
        # return dist[0].sum(dim=-1), torch.stack(dist[1:]).mean(), rnn_hxs # No leakage
        # actor_latent, rnn_hxs, mask, action_mask = self._encode(x, rnn_hxs=rnn_hxs, rnn_masks=masks) # No leakage
        # # return actor_latent.sum(), action_mask.sum(), rnn_hxs
        # object_id_logit = self.object_id_linear(actor_latent)  # (batch_size, seq_len, 1)  # No Leakage
        # # return object_id_logit.sum(), action_mask.sum(), rnn_hxs
        # object_id_logit -= torch.unsqueeze(action_mask, dim=-1) * 1e9
        # no_op_logit = self.no_op_linear(torch.mean(actor_latent, dim=1))  # (batch_size, feature_dim) -> (batch_size, 1)
        # total_id_logit = torch.cat([no_op_logit, object_id_logit.squeeze(dim=-1)], dim=-1)  # (batch_size, 1 + seq_len)
        # return total_id_logit.sum(), action_mask.sum(), rnn_hxs # No Leakage
        dist_object_id = dist[0]
        dist_object_id = FixedCategorical(logits=dist_object_id)
        dist_object_states = dist[1:]
        action_object_id = torch.narrow(action, dim=-1, start=0, length=1).int()
        # action_object_states = torch.narrow(action, dim=-1, start=1, length=self.continuous_action_shape)
        action_object_states = torch.narrow(action, dim=-1, start=1, length=action.shape[1] - 1)
        num_bin = torch.from_numpy(self.num_bin).to(action.device)
        # action_object_states fall within [-1, 1]
        if False:
            action_object_states = (action_object_states + 1) / 2 * num_bin  # [0, num_bin)
            coarse_action_states = (action_object_states + 1e-4).int()
            refined_action_states = ((action_object_states - coarse_action_states) * num_bin + 1e-4).int()
            action_object_states = torch.cat([coarse_action_states, refined_action_states], dim=-1)
        elif self.bilevel_action:
            action_object_states = (action_object_states + 1) / 2 * num_bin / 2
            coarse_action_states = (action_object_states + 1e-4).int()
            refined_action_states = ((action_object_states - coarse_action_states) * num_bin / 2 + 1e-4).int()
            action_object_states = torch.cat([coarse_action_states, refined_action_states], dim=-1)
        else:
            # action_object_states = ((action_object_states + 1) / 2 * (num_bin - 1)).int()
            action_object_states = torch.round((action_object_states + 1) / 2 * (num_bin - 1)).int()
        if self.noop:
            _object_id = (action_object_id + 1).long()
            # Multiple by one hot
            onehot = torch.from_numpy(np.zeros((dist_object_states[0].size(0), dist_object_states[0].size(1) + 1))).to(_object_id.device)
            onehot.scatter_(1, _object_id, 1)
            onehot = onehot[:, 1:]
            reconstr_logits = [(dist * onehot.unsqueeze(dim=-1).detach()).sum(dim=1) for dist in dist_object_states]
            # # Construct the dist again, Check if the gradient remains
            # all_logits = [torch.cat([torch.zeros(dist.logits.size(0), 1, dist.logits.size(2), device=dist.logits.device),
            #                          dist.logits], dim=1) for dist in dist_object_states]
            # all_inds = torch.from_numpy(np.arange(dist_object_states[0].logits.size(0))).to(dist_object_states[0].logits.device)
            # reconstr_logits = [logit[all_inds, _object_id.long().squeeze(dim=-1)] for logit in all_logits]
            dist_object_states = [FixedCategorical(logits=logit) for logit in reconstr_logits]
            # del onehot, reconstr_logits
        else:
            raise NotImplementedError
            _object_id = action_object_id
            # action_log_probs = dist_object_id.log_probs(action_object_id)
        # print(action_log_probs.shape)
        log_probs = []
        object_id_log_prob = dist_object_id.log_probs(_object_id)
        for level in range(len(dist_object_states) // self.continuous_action_shape):
            level_action_log_probs = [object_id_log_prob] + [dist_object_states[j].log_probs(action_object_states[:, j].detach())
                                      for j in range(level * self.continuous_action_shape,
                                                     (level + 1) * self.continuous_action_shape)]
            log_probs.append(torch.sum(torch.stack(level_action_log_probs, dim=1), dim=1))  # TODO: bug, not exactly the same as that from act
        log_probs = torch.stack(log_probs)  # (n_level, batch_size, 1)
        # for j in range(len(dist_object_states)):
        #     action_log_probs += dist_object_states[j].log_probs(action_object_states[:, j: j + 1])
        dist_entropy = None
        if compute_entropy:
            dist_entropy = dist_object_id.entropy()
            for j in range(len(dist_object_states)):
                dist_entropy += dist_object_states[j].entropy()
            # print(dist_entropy.shape, dist_entropy[0])
            dist_entropy = dist_entropy.mean()
        # del dist_object_states
        return log_probs, dist_entropy, rnn_hxs # No leakage

