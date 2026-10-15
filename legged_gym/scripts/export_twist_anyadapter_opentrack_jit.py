"""Export the 7001-D deployable part of Motion-WM + AnyAdapter-OpenTrack."""

from __future__ import annotations

import argparse
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
        self.policy_obs_dim = actor.EXPECTED_POLICY_OBS_DIM
        self.heading_enabled = self.policy_obs_dim == 7003
        self.history_encoder = actor.history_encoder
        self.layerwise_actor = actor.layerwise_actor

    def forward(self, observations):
        observations = observations.reshape(-1, self.policy_obs_dim)
        if observations.shape[-1] != self.policy_obs_dim:
            raise RuntimeError("AnyAdapter-OpenTrack JIT requires exactly 7001-D input")
        base_obs = observations[:, :BASE_OBS_DIM]
        history = observations[:, BASE_OBS_DIM:POLICY_OBS_DIM].reshape(
            observations.shape[0], HISTORY_LEN, HISTORY_FRAME_DIM
        )
        embedding = self.history_encoder(history)
        action = self.layerwise_actor(base_obs, embedding)
        if self.heading_enabled:
            action = action + .2*torch.tanh(self.layerwise_actor.adapter[5](torch.cat((embedding, observations[:, -2:]), -1)))
        return action


def build_actor(checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    actor_class = TwistAnyAdapterOpenTrackActorCritic
    if "layerwise_actor.adapter.5.weight" in state["model_state_dict"]:
        from tools.heading23500_actor import TwistHeadingSupportActorCritic
        actor_class = TwistHeadingSupportActorCritic
    actor = actor_class(
        num_actions=NUM_ACTIONS, num_critic_observations=state["model_state_dict"]["critic.0.weight"].shape[1],
        base_actor_jit_path=BASE_JIT, base_obs_dim=BASE_OBS_DIM,
        history_len=HISTORY_LEN, history_frame_dim=HISTORY_FRAME_DIM,
        hist_state_dim=51, wm_target_indices=WM_TARGET_INDICES, latent_dim=128,
        world_model_hidden_dims=(512, 512, 256, 256, 256, 128),
        critic_hidden_dims=(512, 256, 128), activation="silu", freeze_base=True,
    )
    actor.load_state_dict(state["model_state_dict"], strict=True)
    actor.layerwise_actor.adapter_gain = float(state.get("recovery1250", {}).get("adapter_gain", 1.0))
    return actor.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-gain", type=float, default=None)
    args = parser.parse_args()
    if args.adapter_gain is not None and not 0. <= args.adapter_gain <= 1.:
        parser.error("adapter gain must be between0 and1")
    actor = build_actor(args.checkpoint)
    if args.adapter_gain is not None:
        actor.layerwise_actor.adapter_gain = args.adapter_gain
    gain = actor.layerwise_actor.adapter_gain
    eager = DeployActor(actor).eval()
    sample = torch.randn(2, eager.policy_obs_dim)
    scripted = torch.jit.trace(eager, sample)
    error = float((eager(sample) - scripted(sample)).abs().max())
    if error >= 1e-5:
        raise RuntimeError(f"eager/JIT parity failed: max_abs_error={error:.3e}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    import hashlib, json
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    adapter_gain=gain, input_dim=eager.policy_obs_dim, output_dim=NUM_ACTIONS,
                    eager_jit_max_abs_error=error)
    if gain == 0. and not eager.heading_enabled:
        base = torch.jit.load(BASE_JIT, map_location='cpu').eval()
        base_error = float((scripted(sample)-base(sample[:, :BASE_OBS_DIM])).abs().max())
        if base_error >= 1e-5:
            raise RuntimeError('gain0 does not recover TWIST: '+str(base_error))
        metadata['gain0_base_max_abs_error'] = base_error
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(f"saved {output}; eager/JIT max_abs_error={error:.3e}")


if __name__ == "__main__":
    main()
