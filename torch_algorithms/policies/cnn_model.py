import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributions import FixedCategorical
import math
import numpy as np


def NormedConv2d(*args, scale=1., **kwargs):
    out = nn.Conv2d(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def NormedLinear(*args, scale=1.0, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    """
    out = nn.Linear(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


class CnnBasicBlock(nn.Module):
    def __init__(self, in_channel, scale=1., batch_norm=False):
        super(CnnBasicBlock, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = NormedConv2d(self.in_channel, self.in_channel, 3, padding=1, scale=s)
        self.conv1 = NormedConv2d(self.in_channel, self.in_channel, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.in_channel)
            self.bn1 = nn.BatchNorm2d(self.in_channel)

    def residual(self, x):
        # inplace should be False for the first relu, so that it does not change the input,
        # which will be used for skip connection.
        # getattr is for backwards compatibility with loaded models
        if getattr(self, "batch_norm", False):
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if getattr(self, "batch_norm", False):
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, in_channel, nblock, out_channel, scale=1., pool=True, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pool = pool
        self.firstconv = NormedConv2d(in_channel, out_channel, 3, padding=1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList(
            [CnnBasicBlock(out_channel, scale=s, **kwargs) for _ in range(nblock)]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if getattr(self, "pool", True):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.in_channel
        if getattr(self, "pool", True):
            return (self.out_channel, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.out_channel, h, w)


class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, action_shape, image_feature_dim, chans=(16, 32, 32), activation_fn=nn.Tanh,
                 aux_head=True, arch="shared"):
        super(CNNPolicy, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.image_feature_dim = image_feature_dim
        self.activation_fn = activation_fn
        self.scale_ob = 255.0
        self.final_relu = True
        self.aux_head = aux_head
        self.detach_value_head = (arch == "shared" and aux_head)
        self.arch = arch
        n_input_channels = self.obs_shape.shape[-1]
        # TODO: try normalized conv, residual, multiple blocks
        s = 1 / math.sqrt(len(chans))  # per stack scale
        curshape = (n_input_channels,) + self.obs_shape.shape[:2]
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = CnnDownStack(
                curshape[0], nblock=2, out_channel=outchan, scale=s
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = NormedLinear(np.prod(curshape), self.image_feature_dim, scale=1.4)
        if self.arch == "dual":
            curshape = (n_input_channels,) + self.obs_shape.shape[:2]
            self.critic_stacks = nn.ModuleList()
            for outchan in chans:
                stack = CnnDownStack(
                    curshape[0], nblock=2, out_channel=outchan, scale=s
                )
                self.critic_stacks.append(stack)
                curshape = stack.output_shape(curshape)
            self.critic_dense = NormedLinear(np.prod(curshape), self.image_feature_dim, scale=1.4)

        # # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     n_flatten = self.cnn(torch.as_tensor(self.obs_shape.sample()[None]).permute((0, 3, 1, 2)).float()).shape[1]
        #
        # self.feature_linear = nn.Sequential(nn.Linear(n_flatten, self.image_feature_dim), nn.ReLU())

        self.policy_linear = NormedLinear(self.image_feature_dim, self.action_shape.n, scale=0.1)
        self.value_linear = NormedLinear(self.image_feature_dim, 1, scale=0.1)
        if self.aux_head:
            self.aux_value_linear = NormedLinear(self.image_feature_dim, 1, scale=0.1)

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return 1

    def _encode(self, x, encode_critic=False):
        if encode_critic:
            assert hasattr(self, "critic_stacks")
        x = x.to(dtype=torch.float32) / self.scale_ob
        x = x.permute((0, 3, 1, 2))
        if not encode_critic:
            for layer in self.stacks:
                x = layer(x)
            # x = x.reshape(b, t, *x.shape[1:])
            x = nn.Flatten()(x)
            x = F.relu(x)
            x = self.dense(x)
        else:
            for layer in self.critic_stacks:
                x = layer(x)
            x = nn.Flatten()(x)
            x = F.relu(x)
            x = self.critic_dense(x)
        if self.final_relu:
            x = F.relu(x)
        return x

    def forward(self, obs, forward_value=True, aux_phase=False):
        policy_x = self._encode(obs)
        policy_logit = self.policy_linear(policy_x)
        dist = FixedCategorical(logits=policy_logit)
        value = None
        if forward_value:
            if aux_phase:
                value = self.aux_value_linear(policy_x)
            else:
                critic_x = policy_x if self.arch == "shared" else self._encode(obs, encode_critic=True)
                value = self.value_linear(critic_x.detach() if self.detach_value_head else critic_x)

        return value, dist

    def forward_policy(self, x):
        _, dist = self.forward(x, forward_value=False)
        return dist, None

    def forward_policy_with_aux_head(self, x):
        aux_value, dist = self.forward(x, forward_value=True, aux_phase=True)
        return dist, aux_value

    def act(self, x, rnn_hxs, masks, deterministic=False):
        value, dist = self.forward(x, forward_value=True, aux_phase=False)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, x, rnn_hxs, masks):
        value, _ = self.forward(x, forward_value=True, aux_phase=False)
        return value

    def evaluate_actions(self, x, rnn_hxs, masks, actions):
        _, dist = self.forward(x, forward_value=False)
        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()
        return action_log_probs, dist_entropy, rnn_hxs
