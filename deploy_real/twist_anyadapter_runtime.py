"""
Runtime wrapper for TWIST + AnyAdapter JIT policy.

Drop into:
    TWIST/deploy_real/twist_anyadapter_runtime.py

Use this from server_low_level_g1_real.py or server_low_level_g1_sim.py after you
export the adapter actor to JIT.  It maintains the history buffer required by
the adapter policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch


@dataclass
class AnyAdapterRuntimeConfig:
    base_obs_dim: int
    num_actions: int
    history_len: int
    state_indices: Sequence[int]
    policy_path: str
    device: str = "cpu"
    action_clip: float = 1.0
    adapter_input_is_augmented: bool = True
    adapter_context_dim: int = 0
    fill_history_on_first_observation: bool = False
    # EMA smoothing factor for action output (0 = no smoothing, must be < 1).
    # 0.3–0.5 recommended to suppress high-frequency jitter from adapter.
    action_ema_alpha: float = 0.0


class AnyAdapterRuntime:
    def __init__(self, cfg: AnyAdapterRuntimeConfig):
        self.cfg = cfg
        if cfg.base_obs_dim <= 0 or cfg.num_actions <= 0 or cfg.history_len <= 0:
            raise ValueError("base_obs_dim, num_actions, and history_len must be positive")
        if not 0.0 <= float(cfg.action_ema_alpha) < 1.0:
            raise ValueError("action_ema_alpha must be in [0, 1)")
        self.device = torch.device(cfg.device)
        self.policy = torch.jit.load(cfg.policy_path, map_location=self.device)
        self.policy.eval()
        self.state_indices = np.asarray(cfg.state_indices, dtype=np.int64)
        if self.state_indices.size == 0:
            raise ValueError("state_indices must not be empty")
        if self.state_indices.min() < 0 or self.state_indices.max() >= cfg.base_obs_dim:
            raise ValueError("state_indices must index only the base observation")
        self.hist_state_dim = len(self.state_indices)
        self.frame_dim = self.hist_state_dim + cfg.num_actions
        self.policy_obs_dim = (
            cfg.base_obs_dim
            + cfg.history_len * self.frame_dim
            + cfg.adapter_context_dim
        )
        self.history = np.zeros((cfg.history_len, self.frame_dim), dtype=np.float32)
        self.prev_action = np.zeros((cfg.num_actions,), dtype=np.float32)
        # EMA smoothing state
        self._ema_action: Optional[np.ndarray] = None
        self._ema_alpha = float(cfg.action_ema_alpha)
        self._history_initialized = False
        with torch.no_grad():
            probe = self.policy(
                torch.zeros(1, self.policy_obs_dim, device=self.device)
            )
        if probe.numel() != cfg.num_actions:
            raise ValueError(
                f"Policy output has {probe.numel()} values, expected {cfg.num_actions}."
            )

    def reset(self):
        self.history[:] = 0.0
        self.prev_action[:] = 0.0
        self._ema_action = None
        self._history_initialized = False

    def _build_augmented_obs(
        self,
        base_obs: np.ndarray,
        adapter_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)
        if base_obs.shape[0] != self.cfg.base_obs_dim:
            raise ValueError(
                f"Expected base observation dim {self.cfg.base_obs_dim}, "
                f"got {base_obs.shape[0]}."
            )
        dyn_state = base_obs[self.state_indices].astype(np.float32)
        new_frame = np.concatenate([dyn_state, self.prev_action], axis=0)
        if self.cfg.fill_history_on_first_observation and not self._history_initialized:
            self.history[:] = new_frame
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = new_frame
        self._history_initialized = True
        if self.cfg.adapter_context_dim > 0:
            if adapter_context is None:
                raise ValueError(
                    f"This policy requires {self.cfg.adapter_context_dim} adapter context values."
                )
            adapter_context = np.asarray(adapter_context, dtype=np.float32).reshape(-1)
            if adapter_context.shape[0] != self.cfg.adapter_context_dim:
                raise ValueError(
                    f"Expected adapter context dim {self.cfg.adapter_context_dim}, "
                    f"got {adapter_context.shape[0]}."
                )
        else:
            adapter_context = np.zeros(0, dtype=np.float32)
        return np.concatenate([
            base_obs.astype(np.float32),
            self.history.reshape(-1),
            adapter_context,
        ], axis=0)

    @torch.no_grad()
    def act(
        self,
        base_obs: np.ndarray,
        adapter_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        obs = self._build_augmented_obs(base_obs, adapter_context)
        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        action = self.policy(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
        action = action.reshape(-1)
        if action.shape[0] != self.cfg.num_actions:
            raise RuntimeError(
                f"Policy output has {action.shape[0]} values, expected {self.cfg.num_actions}."
            )
        action = np.clip(action, -self.cfg.action_clip, self.cfg.action_clip)

        # EMA smoothing to suppress high-frequency adapter jitter.
        if self._ema_alpha > 0.0:
            if self._ema_action is None:
                self._ema_action = action.copy()
            else:
                self._ema_action = (
                    self._ema_alpha * self._ema_action + (1.0 - self._ema_alpha) * action
                )
            action = self._ema_action.copy()

        self.prev_action = action.copy()
        return action
