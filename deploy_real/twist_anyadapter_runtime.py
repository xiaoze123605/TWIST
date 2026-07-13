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
    # EMA smoothing factor for action output (0 = no smoothing, 1 = no update).
    # 0.3–0.5 recommended to suppress high-frequency jitter from adapter.
    action_ema_alpha: float = 0.0


class AnyAdapterRuntime:
    def __init__(self, cfg: AnyAdapterRuntimeConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = torch.jit.load(cfg.policy_path, map_location=self.device)
        self.policy.eval()
        self.state_indices = np.asarray(cfg.state_indices, dtype=np.int64)
        self.hist_state_dim = len(self.state_indices)
        self.frame_dim = self.hist_state_dim + cfg.num_actions
        self.history = np.zeros((cfg.history_len, self.frame_dim), dtype=np.float32)
        self.prev_action = np.zeros((cfg.num_actions,), dtype=np.float32)
        # EMA smoothing state
        self._ema_action: Optional[np.ndarray] = None
        self._ema_alpha = float(cfg.action_ema_alpha)

    def reset(self):
        self.history[:] = 0.0
        self.prev_action[:] = 0.0
        self._ema_action = None

    def _build_augmented_obs(self, base_obs: np.ndarray) -> np.ndarray:
        dyn_state = base_obs[self.state_indices].astype(np.float32)
        new_frame = np.concatenate([dyn_state, self.prev_action], axis=0)
        self.history[:-1] = self.history[1:]
        self.history[-1] = new_frame
        return np.concatenate([base_obs.astype(np.float32), self.history.reshape(-1)], axis=0)

    @torch.no_grad()
    def act(self, base_obs: np.ndarray) -> np.ndarray:
        obs = self._build_augmented_obs(base_obs)
        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        action = self.policy(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
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
