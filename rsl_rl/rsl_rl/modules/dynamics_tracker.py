"""Independent sensor/command contract and fully trainable tracking policy.

PPO stores pre-tanh samples. The environment/runtime maps them to absolute
joint targets; histories and the dynamics decoder use those sent targets.
No pretrained TWIST weights, normalizer or privileged actor inputs are used.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


NUM_JOINTS = 23
STATE_DIM = 52  # gyro*.25, projected gravity, absolute q, dq*.05
REFERENCE_DIM = 31  # height, roll/pitch, relative heading, local v/wz, q_ref
TASK_DIM = 54  # reference + causal reference dq*.05
CURRENT_DIM = STATE_DIM + TASK_DIM + NUM_JOINTS
HISTORY_FRAME_DIM = STATE_DIM + NUM_JOINTS + 1  # sensor, sent target, validity


def append_history(history, state, command):
    frame = torch.cat((state, command, torch.ones_like(state[:, :1])), dim=-1)
    return torch.cat((history[:, 1:], frame.unsqueeze(1)), dim=1)


def reference_features(reference, previous_reference, initialized, dt):
    """Backward difference only: identical causal input in sim and deployment."""
    velocity = ((reference[:, 8:] - previous_reference[:, 8:]) / dt).clamp(-20., 20.)
    velocity = torch.where(initialized[:, None], velocity, torch.zeros_like(velocity))
    return torch.cat((reference, velocity * 0.05), dim=-1)


def mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.SiLU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class TargetTransform(nn.Module):
    """The only interpretation of a raw policy sample in the new task."""
    def __init__(self, lower, upper, action_scale, action_mode="reference"):
        super().__init__()
        if action_mode not in ("reference", "direct"):
            raise ValueError("action_mode must be reference or direct")
        lower, upper, scale = [torch.as_tensor(v, dtype=torch.float32) for v in
                               (lower, upper, action_scale)]
        if any(v.shape != (NUM_JOINTS,) for v in (lower, upper, scale)):
            raise ValueError("joint limits and action scales must each contain 23 entries")
        if not all(torch.isfinite(v).all() for v in (lower, upper, scale)):
            raise ValueError("target transform contains non-finite values")
        if torch.any(upper <= lower) or torch.any(scale <= 0):
            raise ValueError("invalid joint range or action scale")
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.register_buffer("scale", scale)
        self.action_mode = action_mode

    def forward(self, raw_sample, reference_q):
        if self.action_mode == "reference":
            target = reference_q + self.scale * torch.tanh(raw_sample)
        else:
            target = (self.lower + self.upper) * 0.5 + (self.upper - self.lower) * 0.5 * torch.tanh(raw_sample)
        return torch.maximum(torch.minimum(target, self.upper), self.lower)


class DynamicsHistoryEncoder(nn.Module):
    def __init__(self, history_len=79, latent_dim=128):
        super().__init__()
        length = ((history_len - 9) // 5 + 1 - 6) // 3 + 1
        if length < 1:
            raise ValueError("history is too short for the dynamics encoder")
        self.conv = nn.Sequential(nn.Conv1d(HISTORY_FRAME_DIM, 64, 9, 5), nn.SiLU(),
                                  nn.Conv1d(64, 64, 6, 3), nn.SiLU())
        self.output = nn.Linear(64 * length, latent_dim)

    def forward(self, history):
        # Invalid startup padding must not masquerade as actual transitions.
        valid = history[:, :, -1:]
        history = torch.cat((history[:, :, :-1] * valid, valid), dim=-1)
        return self.output(self.conv(history.transpose(1, 2)).flatten(1))


class SensorDynamicsModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=(256, 256)):
        super().__init__()
        self.net = mlp(STATE_DIM + NUM_JOINTS + latent_dim, hidden_dims, STATE_DIM)
        nn.init.normal_(self.net[-1].weight, std=0.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, command, latent):
        prediction = state + self.net(torch.cat((state, command, latent), dim=-1))
        gravity = F.normalize(prediction[:, 3:6], dim=-1, eps=1e-6)
        # q and dq have separate clean targets; endpoint velocity is not the
        # control-interval average and is not forced to integrate q exactly.
        return torch.cat((prediction[:, :3], gravity, prediction[:, 6:]), dim=-1)


class DynamicsTrackerActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_observations, num_critic_observations, num_actions=23,
                 history_len=79, latent_dim=128, actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128), world_model_hidden_dims=(256, 256),
                 init_noise_std=0.3, use_dynamics_latent=False, **kwargs):
        super().__init__()
        self.history_len = int(history_len)
        self.latent_dim = int(latent_dim)
        self.use_dynamics_latent = bool(use_dynamics_latent)
        self.obs_dim = CURRENT_DIM + self.history_len * HISTORY_FRAME_DIM
        if num_actions != NUM_JOINTS or num_observations != self.obs_dim:
            raise ValueError("dynamics tracker observation/action contract mismatch")
        if init_noise_std <= 0:
            raise ValueError("init_noise_std must be positive")
        self.history_encoder = DynamicsHistoryEncoder(history_len, latent_dim)
        self.world_model = SensorDynamicsModel(latent_dim, world_model_hidden_dims)
        self.actor = mlp(CURRENT_DIM + 2 * NUM_JOINTS + latent_dim, actor_hidden_dims, NUM_JOINTS)
        nn.init.normal_(self.actor[-1].weight, std=0.001)
        nn.init.zeros_(self.actor[-1].bias)
        self.critic = mlp(num_critic_observations, critic_hidden_dims, 1)
        self.log_std = nn.Parameter(torch.full((NUM_JOINTS,), math.log(init_noise_std)))
        self.distribution = None

    def split_obs(self, obs):
        if obs.shape[-1] != self.obs_dim:
            raise ValueError("invalid dynamics tracker observation dimension")
        return obs[:, :CURRENT_DIM], obs[:, CURRENT_DIM:].reshape(-1, self.history_len, HISTORY_FRAME_DIM)

    def actor_mean(self, obs):
        current, history = self.split_obs(obs)
        latent = self.history_encoder(history).detach()
        if not self.use_dynamics_latent:
            latent = torch.zeros_like(latent)
        state, task = current[:, :STATE_DIM], current[:, STATE_DIM:STATE_DIM + TASK_DIM]
        q_error = task[:, 8:31] - state[:, 6:29]
        dq_error = task[:, 31:] - state[:, 29:52]
        return self.actor(torch.cat((current, q_error, dq_error, latent), dim=-1))

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def update_distribution(self, obs):
        self.distribution = Normal(self.actor_mean(obs), self.std)

    def act_inference(self, obs):
        return self.actor_mean(obs)

    def forward(self, obs):
        return self.actor_mean(obs)

    @staticmethod
    def tanh_log_jacobian(raw):
        return 2. * (math.log(2.) - raw - F.softplus(-2. * raw))

    def get_actions_log_prob(self, raw):
        # Evaluate the tanh-normal at its PRE-tanh sample, avoiding atanh and
        # numerical saturation. The Jacobian cancels in PPO's same-action ratio.
        return (self.distribution.log_prob(raw) - self.tanh_log_jacobian(raw)).sum(-1)

    @property
    def std(self):
        return self.log_std.clamp(math.log(0.03), math.log(0.7)).exp()

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        sample = self.distribution.rsample()
        return (self.distribution.entropy() + self.tanh_log_jacobian(sample)).sum(-1)

    def evaluate(self, obs, **kwargs):
        return self.critic(obs)

    def if_fix_std(self):
        return False

    def reset(self, dones=None):
        pass
