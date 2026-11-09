import torch
from torch import nn

from rsl_rl.modules.actor_critic_mimic import Actor
from .geometry import increments, integrate, quat_inv, quat_rotate


class RunningNorm(nn.Module):
    def __init__(self, size, eps=1e-4):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('var', torch.ones(size))
        self.register_buffer('count', torch.tensor(0.))
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel()).float()
        if len(x) == 0:
            raise ValueError('不能用空样本更新归一化')
        count = self.count+len(x)
        delta = x.mean(0)-self.mean
        self.var.copy_((self.var*self.count+x.var(0, unbiased=False)*len(x)
                        +delta.square()*self.count*len(x)/count)/count)
        self.mean.add_(delta*len(x)/count)
        self.count.copy_(count)

    def std(self):
        return (self.var+self.eps**2).sqrt()

    def forward(self, x):
        return (x-self.mean)/self.std()


def mlp(size, hidden, output):
    layers = []
    for width in hidden:
        layers += [nn.Linear(size, width), nn.LayerNorm(width), nn.SiLU()]
        size = width
    return nn.Sequential(*layers, nn.Linear(size, output))


def policy():
    return Actor(num_observations=1155, num_motion_observations=31, num_motion_steps=1,
                 motion_latent_dim=128, num_actions=23, actor_hidden_dims=[512, 512, 256, 128],
                 activation=nn.SiLU(), layer_norm=True, tanh_encoder_output=False)


class DeploymentPolicy(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observations):
        # 旧部署入口附加的历史和航向字段保留在输入中，原生策略读取前1155维。
        return self.actor(observations[:, :1155])


class WorldModel(nn.Module):
    def __init__(self, width=2048, amp=True):
        super().__init__()
        self.amp = amp
        self.net = mlp(107, [width]*3, 58)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.input_norm = RunningNorm(107, .01)
        self.output_norm = RunningNorm(58)

    def inputs(self, state, action, context):
        inv = quat_inv(state[..., 3:7])
        gravity = torch.zeros_like(state[..., :3])
        gravity[..., 2] = -1
        return torch.cat((state[..., 2:3], quat_rotate(inv, gravity),
                          quat_rotate(inv, state[..., 7:10]), quat_rotate(inv, state[..., 10:13]),
                          state[..., 13:], action, context), -1)

    @torch.no_grad()
    def update_norm(self, state, sn, action, context):
        self.input_norm.update(self.inputs(state, action, context))
        self.output_norm.update(increments(state, sn))

    def forward(self, state, action, context):
        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16,
                            enabled=self.amp and state.is_cuda):
            prediction = self.net(self.input_norm(self.inputs(state, action, context)))
        delta = prediction.float()*self.output_norm.std()+self.output_norm.mean
        return integrate(state, delta)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RunningNorm(1324, .01)
        self.net = mlp(1324, [256, 128], 1)
        self.register_buffer('return_mean', torch.tensor(0.))
        self.register_buffer('return_std', torch.tensor(1.))
        self.register_buffer('return_count', torch.tensor(0.))

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1)

    def value(self, x):
        return self(x)*self.return_std+self.return_mean

    @torch.no_grad()
    def targets(self, x):
        mean, std = x.mean(), x.std(unbiased=False).clamp_min(.01)
        alpha = 1. if self.return_count == 0 else .01
        self.return_mean.lerp_(mean, alpha)
        self.return_std.lerp_(std, alpha)
        self.return_count.add_(len(x))
        return ((x-self.return_mean)/self.return_std).clamp(-20, 20)
