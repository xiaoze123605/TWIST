"""Export the 7001-D deployable part of Motion-WM + AnyAdapter-OpenTrack."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "rsl_rl")]

import torch

from rsl_rl.modules.actor_critic_twist_anyadapter_opentrack import (
    TwistAnyAdapterOpenTrackActorCritic,
)

BASE_OBS_DIM, HISTORY_LEN, HISTORY_FRAME_DIM = 1155, 79, 74
POLICY_OBS_DIM = BASE_OBS_DIM + HISTORY_LEN * HISTORY_FRAME_DIM
NUM_ACTIONS = 23
WM_TARGET_INDICES = list(range(31, 82))
BASE_JIT = str(ROOT / "legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt")


class DeployActor(torch.nn.Module):
    def __init__(self, actor, latent_mode="normal", activity_gate=None):
        super().__init__()
        if latent_mode not in ("normal", "zero"):
            raise ValueError("latent_mode must be normal or zero")
        self.latent_mode = latent_mode
        self.activity_gate = activity_gate
        # Register deployment-only modules. Keeping the complete actor here
        # would also serialize the training-only dynamics WM and critic.
        self.history_encoder = actor.history_encoder
        self.layerwise_actor = actor.layerwise_actor

    def forward(self, observations):
        if observations.shape[-1] != POLICY_OBS_DIM:
            raise RuntimeError("AnyAdapter-OpenTrack JIT requires exactly 7001-D input")
        base_obs = observations[:, :BASE_OBS_DIM]
        history = observations[:, BASE_OBS_DIM:].reshape(
            observations.shape[0], HISTORY_LEN, HISTORY_FRAME_DIM
        )
        embedding = self.history_encoder(history)
        if self.latent_mode == "zero":
            embedding = torch.zeros_like(embedding)
        adapted = self.layerwise_actor(base_obs, embedding)
        if self.activity_gate is None:
            return adapted
        low, high = self.activity_gate
        planar_speed = torch.linalg.vector_norm(base_obs[:, 4:6], dim=-1, keepdim=True)
        yaw_speed = base_obs[:, 7:8].abs()
        activity = torch.maximum(planar_speed, yaw_speed)
        gate = ((activity - low) / (high - low)).clamp(0.0, 1.0)
        base = self.layerwise_actor.base_forward(base_obs)
        return base + gate * (adapted - base)


def build_actor(checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    trained_gain = state.get("training_config", {}).get("policy", {}).get("adapter_gain", 1.0)
    actor = TwistAnyAdapterOpenTrackActorCritic(
        num_actions=NUM_ACTIONS, num_critic_observations=1318,
        base_actor_jit_path=BASE_JIT, base_obs_dim=BASE_OBS_DIM,
        history_len=HISTORY_LEN, history_frame_dim=HISTORY_FRAME_DIM,
        hist_state_dim=51, wm_target_indices=WM_TARGET_INDICES, latent_dim=128,
        world_model_hidden_dims=(512, 512, 256, 256, 256, 128),
        critic_hidden_dims=(512, 256, 128), activation="silu", freeze_base=True,
        adapter_gain=trained_gain,
    )
    actor.load_state_dict(state["model_state_dict"], strict=True)
    return actor.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-gain", type=float, default=None,
                        help="Override checkpoint training gain in [0,1]; zero exactly recovers TWIST.")
    parser.add_argument("--latent-mode", choices=("normal", "zero"), default="normal",
                        help="Zero is a diagnostic ablation of dynamics-history conditioning.")
    parser.add_argument("--activity-gate", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        help="Blend base to adapted action as reference planar/yaw speed rises.")
    args = parser.parse_args()
    if args.adapter_gain is not None and (not math.isfinite(args.adapter_gain)
                                          or not 0.0 <= args.adapter_gain <= 1.0):
        parser.error('--adapter-gain must be finite and in [0,1]')
    if args.activity_gate is not None and (
            not all(math.isfinite(value) for value in args.activity_gate) or
            args.activity_gate[0] < 0 or args.activity_gate[1] <= args.activity_gate[0]):
        parser.error('--activity-gate requires finite 0 <= LOW < HIGH')
    actor = build_actor(args.checkpoint)
    if args.adapter_gain is not None:
        actor.layerwise_actor.adapter_gain = args.adapter_gain
    gain = actor.layerwise_actor.adapter_gain
    eager = DeployActor(actor, latent_mode=args.latent_mode,
                        activity_gate=tuple(args.activity_gate)
                        if args.activity_gate is not None else None).eval()
    sample = torch.randn(2, POLICY_OBS_DIM)
    scripted = torch.jit.trace(eager, sample)
    error = float((eager(sample) - scripted(sample)).abs().max())
    if error >= 1e-5:
        raise RuntimeError(f"eager/JIT parity failed: max_abs_error={error:.3e}")
    if args.activity_gate is not None:
        for speed in (0.0, args.activity_gate[1] + 0.1):
            probe = sample.clone()
            probe[:, 4:8] = 0.0
            probe[:, 4] = speed
            probe_error = float((eager(probe) - scripted(probe)).abs().max())
            if probe_error >= 1e-5:
                raise RuntimeError(f"activity gate JIT parity failed: {probe_error:.3e}")
    if gain == 0.0:
        base = torch.jit.load(BASE_JIT, map_location='cpu').eval()
        base_error = float((eager(sample) - base(sample[:, :BASE_OBS_DIM])).abs().max())
        if base_error >= 2e-5:
            raise RuntimeError(f"zero-gain/base parity failed: max_abs_error={base_error:.3e}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    print(f"saved {output}; adapter_gain={gain:g}; latent_mode={args.latent_mode}; "
          f"activity_gate={args.activity_gate}; "
          f"eager/JIT max_abs_error={error:.3e}")


if __name__ == "__main__":
    main()
