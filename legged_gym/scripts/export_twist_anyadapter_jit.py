"""
Export a trained TWIST+AnyAdapter actor to TorchScript.

This is a template because TWIST checkpoints may store keys differently across
experiments.  Use it after training with TwistAnyAdapterActorCritic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rsl_rl.modules.actor_critic_twist_anyadapter import TwistAnyAdapterActorCritic


DEFAULT_BASE_ACTOR_JIT_PATH = (
    "/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/"
    "0721_twist_rlbcstu/traced/0721_twist_rlbcstu-23500-jit.pt"
)
DEFAULT_BASE_OBS_DIM = 1155
DEFAULT_NUM_ACTIONS = 23
DEFAULT_NUM_CRITIC_OBS = 2635
DEFAULT_HISTORY_LEN = 20
DEFAULT_HISTORY_FRAME_DIM = 74
DEFAULT_HIST_STATE_DIM = 51
DEFAULT_LATENT_DIM = 32
DEFAULT_ACTION_DELTA_SCALE = 0.02


def default_output_path(ckpt_path: str) -> str:
    ckpt = Path(ckpt_path)
    checkpoint_tag = ckpt.stem.replace("model_", "")
    run_dir = ckpt.parent
    return str(run_dir / "traced" / f"{run_dir.name}-{checkpoint_tag}-jit.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to TWIST adapter checkpoint .pt")
    parser.add_argument("--out", default=None, help="Output JIT actor path. Defaults to <run_dir>/traced/<run>-<ckpt>-jit.pt")
    parser.add_argument("--base_actor_jit_path", default=DEFAULT_BASE_ACTOR_JIT_PATH)
    parser.add_argument("--base_obs_dim", type=int, default=DEFAULT_BASE_OBS_DIM)
    parser.add_argument("--num_actions", type=int, default=DEFAULT_NUM_ACTIONS)
    parser.add_argument("--num_critic_obs", type=int, default=DEFAULT_NUM_CRITIC_OBS)
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--history_frame_dim", type=int, default=DEFAULT_HISTORY_FRAME_DIM)
    parser.add_argument("--hist_state_dim", type=int, default=DEFAULT_HIST_STATE_DIM)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--action_delta_scale", type=float, default=DEFAULT_ACTION_DELTA_SCALE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.out is None:
        args.out = default_output_path(args.ckpt)

    model = TwistAnyAdapterActorCritic(
        num_prop=args.base_obs_dim,
        num_critic_obs=args.num_critic_obs,
        num_priv_latent=0,
        num_hist=args.history_len,
        num_actions=args.num_actions,
        base_actor_jit_path=args.base_actor_jit_path,
        base_obs_dim=args.base_obs_dim,
        history_len=args.history_len,
        history_frame_dim=args.history_frame_dim,
        hist_state_dim=args.hist_state_dim,
        latent_dim=args.latent_dim,
        action_delta_scale=args.action_delta_scale,
    ).to(args.device)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # Remove keys that are irrelevant for inference (critic, world_model) so
    # shape mismatches on those modules do not block the export.
    inference_keys = {
        k for k in state
        if not k.startswith("critic.") and not k.startswith("world_model.")
    }
    state = {k: v for k, v in state.items() if k in inference_keys}

    model.load_state_dict(state, strict=False)
    model.eval()

    class ActorOnly(torch.nn.Module):
        def __init__(self, ac):
            super().__init__()
            self.ac = ac
        def forward(self, obs):
            return self.ac.act_inference(obs)

    actor = ActorOnly(model).to(args.device).eval()
    example_dim = args.base_obs_dim + args.history_len * args.history_frame_dim
    example = torch.zeros(1, example_dim, device=args.device)
    traced = torch.jit.trace(actor, example)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.out)
    print(f"Saved TWIST+AnyAdapter JIT actor to {args.out}")


if __name__ == "__main__":
    main()
