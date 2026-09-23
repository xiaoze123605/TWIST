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
    def __init__(self, actor):
        super().__init__()
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
        return self.layerwise_actor(base_obs, embedding)


def build_actor(checkpoint):
    actor = TwistAnyAdapterOpenTrackActorCritic(
        num_actions=NUM_ACTIONS, num_critic_observations=1318,
        base_actor_jit_path=BASE_JIT, base_obs_dim=BASE_OBS_DIM,
        history_len=HISTORY_LEN, history_frame_dim=HISTORY_FRAME_DIM,
        hist_state_dim=51, wm_target_indices=WM_TARGET_INDICES, latent_dim=128,
        world_model_hidden_dims=(512, 512, 256, 256, 256, 128),
        critic_hidden_dims=(512, 256, 128), activation="silu", freeze_base=True,
    )
    state = torch.load(checkpoint, map_location="cpu")
    actor.load_state_dict(state["model_state_dict"], strict=True)
    return actor.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-gain", type=float, default=1.0,
                        help="Layer-adapter strength in [0,1]; zero exactly recovers TWIST.")
    args = parser.parse_args()
    if not math.isfinite(args.adapter_gain) or not 0.0 <= args.adapter_gain <= 1.0:
        parser.error('--adapter-gain must be finite and in [0,1]')
    actor = build_actor(args.checkpoint)
    actor.layerwise_actor.adapter_gain = args.adapter_gain
    eager = DeployActor(actor).eval()
    sample = torch.randn(2, POLICY_OBS_DIM)
    scripted = torch.jit.trace(eager, sample)
    error = float((eager(sample) - scripted(sample)).abs().max())
    if error >= 1e-5:
        raise RuntimeError(f"eager/JIT parity failed: max_abs_error={error:.3e}")
    if args.adapter_gain == 0.0:
        base = torch.jit.load(BASE_JIT, map_location='cpu').eval()
        base_error = float((eager(sample) - base(sample[:, :BASE_OBS_DIM])).abs().max())
        if base_error >= 2e-5:
            raise RuntimeError(f"zero-gain/base parity failed: max_abs_error={base_error:.3e}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    print(f"saved {output}; adapter_gain={args.adapter_gain:g}; "
          f"eager/JIT max_abs_error={error:.3e}")


if __name__ == "__main__":
    main()
