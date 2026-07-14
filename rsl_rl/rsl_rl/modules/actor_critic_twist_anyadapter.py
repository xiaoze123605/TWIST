"""
TWIST + AnyAdapter-style policy module.

Drop this file into:
    TWIST/rsl_rl/rsl_rl/modules/actor_critic_twist_anyadapter.py

Design:
    action = frozen_base_actor(obs_base) + adapter(obs_base, history_embedding)

The history encoder and world model are trained with an auxiliary forward-dynamics
loss.  The frozen base actor is loaded from a TorchScript/JIT TWIST student actor,
so this module can be added without rewriting the original TWIST actor.

Important assumptions:
    1. The environment provides an augmented observation:
           [base_obs, history_flat]
       where history_flat is history_len * history_frame_dim.
    2. Each history frame is:
           [hist_state, previous_action]
       where hist_state is a selected subset of base_obs used for dynamics ID.
    3. base_actor_jit_path points to an exported TWIST student JIT policy.

You must set base_obs_dim, history_len, history_frame_dim, hist_state_dim and
wm_target_indices in the TWIST config.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


Normal.set_default_validate_args = False


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "lrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


def mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    act = get_activation(activation)
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), act.__class__()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def zero_init_last_linear(module: nn.Module) -> None:
    """Make a network initially output almost zero, preserving the base policy."""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return
    nn.init.zeros_(linears[-1].weight)
    nn.init.zeros_(linears[-1].bias)


class HistoryEncoder(nn.Module):
    """Encode recent robot response history into a dynamics embedding z_t."""

    def __init__(
        self,
        history_frame_dim: int,
        history_len: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        activation: str = "elu",
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.history_frame_dim = history_frame_dim
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.use_conv = use_conv
        act = get_activation(activation)
        if use_conv:
            self.net = nn.Sequential(
                nn.Conv1d(history_frame_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                act.__class__(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                act.__class__(),
                nn.Flatten(),
                nn.Linear(hidden_dim * ((history_len + 3) // 4), latent_dim),
            )
        else:
            self.net = mlp(history_frame_dim * history_len, [hidden_dim, hidden_dim], latent_dim, activation)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, H, D]
        if history.ndim != 3:
            raise ValueError(f"history must be [B,H,D], got {tuple(history.shape)}")
        if self.use_conv:
            return self.net(history.transpose(1, 2))
        return self.net(history.reshape(history.shape[0], -1))


class WorldModel(nn.Module):
    """
    One-step forward dynamics proxy model.

    It predicts current/next selected robot state from previous selected state,
    previous action and dynamics embedding.  This is not a planner; it only makes
    the embedding dynamics-aware.
    """

    def __init__(
        self,
        hist_state_dim: int,
        num_actions: int,
        latent_dim: int,
        target_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "elu",
        predict_delta: bool = True,
    ) -> None:
        super().__init__()
        self.hist_state_dim = hist_state_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.predict_delta = predict_delta
        self.net = mlp(hist_state_dim + num_actions + latent_dim, hidden_dims, target_dim, activation)

    def forward(self, prev_state: torch.Tensor, prev_action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        pred = self.net(torch.cat([prev_state, prev_action, z], dim=-1))
        if self.predict_delta:
            # If target_dim <= hist_state_dim, interpret prediction as state delta.
            base = prev_state[..., : self.target_dim]
            pred = base + pred
        return pred


class ResidualAdapter(nn.Module):
    """Small zero-initialized residual correction branch."""

    def __init__(
        self,
        base_obs_dim: int,
        latent_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "elu",
        delta_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.net = mlp(base_obs_dim + latent_dim, hidden_dims, num_actions, activation)
        zero_init_last_linear(self.net)

    def forward(self, base_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.delta_scale * torch.tanh(self.net(torch.cat([base_obs, z], dim=-1)))


class TwistAnyAdapterActorCritic(nn.Module):
    """
    RSL-RL compatible actor-critic wrapper.

    Actor:
        frozen TWIST JIT base actor + trainable residual adapter.
    Critic:
        trainable MLP using critic observations.  You may replace this with the
        original TWIST critic if you want a stricter continuation of training.
    """

    is_recurrent = False

    def __init__(
        self,
        num_prop: Optional[int] = None,
        num_critic_obs: Optional[int] = None,
        num_priv_latent: int = 0,
        num_hist: int = 0,
        num_actions: Optional[int] = None,
        base_actor_jit_path: Optional[str] = None,
        base_obs_dim: int = -1,
        history_len: int = 20,
        history_frame_dim: Optional[int] = None,
        hist_state_dim: Optional[int] = None,
        wm_target_indices: Optional[Sequence[int]] = None,
        latent_dim: int = 32,
        adapter_hidden_dims: Sequence[int] = (128, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        world_model_hidden_dims: Sequence[int] = (256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.2,
        fix_action_std: bool = False,
        action_delta_scale: float = 0.25,
        adapter_gain: float = 1.0,
        default_ref_dof_pos: Optional[Sequence[float]] = None,
        use_conv_history: bool = True,
        freeze_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_critic_obs is None:
            num_critic_obs = kwargs.pop("num_critic_observations", None)
        if num_critic_obs is None:
            num_critic_obs = kwargs.pop("num_observations", None)
        if base_actor_jit_path is None:
            raise ValueError("base_actor_jit_path must point to an exported TWIST student JIT actor.")
        if base_obs_dim <= 0:
            raise ValueError("base_obs_dim must be the original TWIST student obs dim before AnyAdapter history.")
        if num_actions is None:
            raise ValueError("num_actions must be provided by the runner.")
        if num_critic_obs is None:
            raise ValueError("num_critic_obs or num_critic_observations must be provided by the runner.")
        self.num_actions = num_actions
        self.base_obs_dim = int(base_obs_dim)
        self.history_len = int(history_len)
        self.history_frame_dim = int(history_frame_dim or (self.base_obs_dim + num_actions))
        self.hist_state_dim = int(hist_state_dim or (self.history_frame_dim - num_actions))
        if self.history_len <= 0:
            raise ValueError("history_len must be positive for AnyAdapter.")
        if self.history_frame_dim <= 0:
            raise ValueError("history_frame_dim must be len(anyadapter_state_indices) + num_actions.")
        if self.hist_state_dim <= 0:
            raise ValueError("hist_state_dim must be len(anyadapter_state_indices).")
        self.latent_dim = int(latent_dim)
        self.fix_action_std = bool(fix_action_std)
        self.wm_target_indices = None if wm_target_indices is None else torch.as_tensor(wm_target_indices, dtype=torch.long)
        self.adapter_gain = float(adapter_gain)
        if default_ref_dof_pos is None:
            default_ref_dof_pos = [0.0] * num_actions
        if len(default_ref_dof_pos) != num_actions:
            raise ValueError(f"default_ref_dof_pos must have {num_actions} values, got {len(default_ref_dof_pos)}.")
        self.register_buffer("default_ref_dof_pos", torch.as_tensor(default_ref_dof_pos, dtype=torch.float32))

        self.base_actor = torch.jit.load(base_actor_jit_path, map_location="cpu")
        self.base_actor.eval()
        if freeze_base:
            for p in self.base_actor.parameters():
                p.requires_grad_(False)

        self.history_encoder = HistoryEncoder(
            history_frame_dim=self.history_frame_dim,
            history_len=self.history_len,
            latent_dim=latent_dim,
            hidden_dim=128,
            activation=activation,
            use_conv=use_conv_history,
        )
        self.adapter = ResidualAdapter(
            base_obs_dim=self.base_obs_dim,
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden_dims=adapter_hidden_dims,
            activation=activation,
            delta_scale=action_delta_scale,
        )
        target_dim = len(wm_target_indices) if wm_target_indices is not None else self.hist_state_dim
        self.world_model = WorldModel(
            hist_state_dim=self.hist_state_dim,
            num_actions=num_actions,
            latent_dim=latent_dim,
            target_dim=target_dim,
            hidden_dims=world_model_hidden_dims,
            activation=activation,
            predict_delta=True,
        )
        # Per-component world model loss splits.
        # The target state selected by wm_target_indices is laid out as:
        #   [base_ang_vel(3), roll/pitch(2), dof_pos(23), dof_vel(23)] = 51 dims
        # Components  →  (start, end) in the 51-dim target vector.
        self.wm_component_splits = {
            "ang_vel":     (0, 3),   # base angular velocity
            "orientation": (3, 5),   # roll / pitch
            "dof_pos":     (5, 28),  # joint positions
            "dof_vel":     (28, 51), # joint velocities
        }
        self.wm_component_weights = {
            "ang_vel":     1.0,
            "orientation": 2.0,
            "dof_pos":     1.0,
            "dof_vel":     1.0,
        }
        self.critic = mlp(num_critic_obs, critic_hidden_dims, 1, activation)

        if self.fix_action_std:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=False)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Optional[Normal] = None

    def split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_obs = obs[:, : self.base_obs_dim]
        hist_flat = obs[:, self.base_obs_dim : self.base_obs_dim + self.history_len * self.history_frame_dim]
        if hist_flat.numel() == 0:
            raise RuntimeError(
                "Observation does not contain history. Add AnyAdapterHistoryMixin to the TWIST env "
                "or check base_obs_dim/history_len/history_frame_dim."
            )
        history = hist_flat.reshape(obs.shape[0], self.history_len, self.history_frame_dim)
        return base_obs, history

    def encode_history_for_policy(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history).detach()

    def encode_history_for_world_model(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history)

    def base_action(self, observations: torch.Tensor) -> torch.Tensor:
        base_obs, _ = self.split_obs(observations)
        with torch.no_grad():
            return self.base_actor(base_obs)

    def action_delta(self, observations: torch.Tensor, detach_history: bool = True) -> torch.Tensor:
        base_obs, history = self.split_obs(observations)
        if detach_history:
            z = self.encode_history_for_policy(history)
        else:
            z = self.encode_history_for_world_model(history)
        return self.adapter(base_obs, z)

    def actor_mean(self, observations: torch.Tensor) -> torch.Tensor:
        return self.base_action(observations) + self.adapter_gain * self.action_delta(observations, detach_history=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(observations)

    def get_adapter_delta(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_delta(observations, detach_history=True)

    def predict_world_model(self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, history = self.split_obs(observations)
        z = self.encode_history_for_world_model(history)
        prev_frame = history[:, -1]
        prev_state = prev_frame[:, : self.hist_state_dim]
        if actions is None:
            prev_action = prev_frame[:, self.hist_state_dim : self.hist_state_dim + self.num_actions]
        else:
            prev_action = actions
        return self.world_model(prev_state, prev_action, z)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor_mean(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor, eval: bool = False, **kwargs) -> torch.Tensor:
        return self.actor_mean(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Call act() or update_distribution() before get_actions_log_prob().")
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def reset(self, dones=None) -> None:
        pass

    def if_fix_std(self) -> bool:
        return self.fix_action_std

    def update_std(self, std: float) -> None:
        self.std.data[:] = std

    def world_model_loss(self, observations: torch.Tensor, loss_type: str = "smooth_l1") -> Tuple[torch.Tensor, dict]:
        base_obs, history = self.split_obs(observations)
        z = self.encode_history_for_world_model(history)
        prev_frame = history[:, -1]
        prev_state = prev_frame[:, : self.hist_state_dim]
        prev_action = prev_frame[:, self.hist_state_dim : self.hist_state_dim + self.num_actions]

        if self.wm_target_indices is None:
            target = base_obs[:, : self.hist_state_dim]
        else:
            idx = self.wm_target_indices.to(base_obs.device)
            target = base_obs.index_select(dim=1, index=idx)

        pred = self.predict_world_model(observations)
        if loss_type == "mse":
            loss = torch.mean((pred - target) ** 2)
        else:
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
        return loss, {"wm_loss": float(loss.detach().cpu())}

    def adapter_regularization_loss(self, observations: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        delta = self.get_adapter_delta(observations)
        loss = (delta ** 2).mean()
        delta_l2 = torch.norm(delta, p=2, dim=-1).mean()
        return loss, {
            "adapter_delta_l2": float(delta_l2.detach().cpu()),
            "adapter_reg_loss": float(loss.detach().cpu()),
            "max_abs_delta_action": float(delta.detach().abs().max().cpu()),
            "mean_abs_delta_action": float(delta.detach().abs().mean().cpu()),
        }
