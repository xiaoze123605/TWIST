"""Frozen, batched Motion-WM reference preprocessing for small training pilots.

Corruption intentionally reuses the deployment NumPy implementation per env.
This preserves its distribution and reset semantics, but entails a CPU/device
transfer per control step; benchmark before increasing the environment count.
"""
from __future__ import annotations

import numpy as np
import torch

from .runtime import MotionReferenceRefiner, RuntimeReferenceCorruptor, unwrap_angle_near


class TrainingReferencePipeline:
    MODES = ("clean", "corrupt", "wm")

    def __init__(self, checkpoint, num_envs, device="cpu", seed=42,
                 mode_probabilities=(1/3, 1/3, 1/3), preset="formal", control_dt=0.02,
                 artificial_corruption=True):
        if num_envs <= 0 or abs(control_dt - 0.02) > 1e-8:
            raise ValueError("Motion-WM requires positive num_envs and 50 Hz control")
        probabilities = np.asarray(mode_probabilities, dtype=float)
        if probabilities.shape != (3,) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
            raise ValueError("Expected clean/corrupt/wm probabilities summing to one")
        loaded = MotionReferenceRefiner(checkpoint, device=device)
        self.model = loaded.model.requires_grad_(False).eval()
        self.mean, self.std = loaded.mean, loaded.std
        self.device = loaded.mean.device
        self.num_envs, self.seed, self.preset = num_envs, int(seed), preset
        self.artificial_corruption = bool(artificial_corruption)
        self.probabilities = probabilities
        self.history = torch.zeros(num_envs, 25, 31, device=self.device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.mode = torch.zeros_like(self.count)
        self.episodes = np.zeros(num_envs, dtype=np.int64)
        self.previous_yaw = [None] * num_envs
        self.corruptors = [None] * num_envs
        self.mode_rngs = [np.random.default_rng(seed + 7919*(i+1)) for i in range(num_envs)]
        self.clean = torch.zeros(num_envs, 31, device=self.device)
        self.corrupted = torch.zeros_like(self.clean)
        self.processed = torch.zeros_like(self.clean)
        self.total_refined_frames = 0
        self.reset(range(num_envs))

    def reset(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        ids = list(env_ids)
        for i in ids:
            self.corruptors[i] = (
                RuntimeReferenceCorruptor.from_preset(
                    self.preset,
                    seed=self.seed + 9973*i + 1000003*int(self.episodes[i]),
                )
                if self.artificial_corruption else None
            )
            self.episodes[i] += 1
            self.previous_yaw[i] = None
            self.mode[i] = int(self.mode_rngs[i].choice(3, p=self.probabilities))
        self.history[ids] = 0
        self.count[ids] = 0
        self.clean[ids] = 0
        self.corrupted[ids] = 0
        self.processed[ids] = 0

    @torch.no_grad()
    def process(self, clean):
        if clean.shape != (self.num_envs, 31) or clean.device != self.device:
            raise ValueError("Expected [num_envs,31] clean reference on pipeline device")
        if not torch.isfinite(clean).all():
            raise ValueError("Non-finite clean training reference")
        self.clean.copy_(clean)
        continuous = []
        for i, reference in enumerate(clean.detach().cpu().numpy()):
            value = (
                self.corruptors[i].corrupt(reference).copy()
                if self.artificial_corruption else reference.copy()
            )
            value[3] = unwrap_angle_near(float(value[3]), self.previous_yaw[i])
            self.previous_yaw[i] = float(value[3])
            continuous.append(value)
        current = torch.as_tensor(np.stack(continuous), device=self.device)
        self.history = torch.cat((self.history[:, 1:], current[:, None]), dim=1)
        self.count.add_(1).clamp_(max=25)
        self.corrupted.copy_(current)
        self.corrupted[:, 3] = torch.atan2(torch.sin(current[:, 3]), torch.cos(current[:, 3]))
        output = self.corrupted.clone()
        ready = (self.count >= 25) & (self.mode == 2)
        if ready.any():
            self.model.eval()
            prediction = self.model((self.history[ready] - self.mean) / self.std)[:, 0]
            restored = prediction * self.std[0, 0] + self.mean[0, 0]
            if not torch.isfinite(restored).all():
                raise RuntimeError("Non-finite Motion-WM output; training aborted")
            restored[:, 3] = torch.atan2(torch.sin(restored[:, 3]), torch.cos(restored[:, 3]))
            output[ready] = restored
            self.total_refined_frames += int(ready.sum().item())
        output[self.mode == 0] = clean[self.mode == 0]
        self.processed.copy_(output)
        return output
