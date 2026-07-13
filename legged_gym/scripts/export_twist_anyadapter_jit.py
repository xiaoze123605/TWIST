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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to TWIST adapter checkpoint .pt")
    parser.add_argument("--out", required=True, help="Output JIT actor path")
    parser.add_argument("--base_actor_jit_path", required=True)
    parser.add_argument("--base_obs_dim", type=int, required=True)
    parser.add_argument("--num_actions", type=int, required=True)
    parser.add_argument("--num_critic_obs", type=int, required=True)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--history_frame_dim", type=int, required=True)
    parser.add_argument("--hist_state_dim", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

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
