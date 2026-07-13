from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

import torch

from rsl_rl.modules import TwistAnyAdapterActorCritic


class DummyBaseActor(torch.nn.Module):
    def __init__(self, base_obs_dim: int, num_actions: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(base_obs_dim, num_actions)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


def make_dummy_base_actor_jit(base_obs_dim: int, num_actions: int, path: str) -> None:
    actor = DummyBaseActor(base_obs_dim, num_actions).eval()
    example = torch.zeros(1, base_obs_dim)
    traced = torch.jit.trace(actor, example)
    traced.save(path)


def main() -> None:
    torch.manual_seed(0)

    base_obs_dim = 100
    history_len = 20
    hist_state_dim = 40
    num_actions = 29
    history_frame_dim = hist_state_dim + num_actions
    total_obs_dim = base_obs_dim + history_len * history_frame_dim

    obs = torch.randn(4, total_obs_dim)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_actor_jit_path = os.path.join(tmpdir, "dummy_base_actor.pt")
        make_dummy_base_actor_jit(base_obs_dim, num_actions, base_actor_jit_path)

        actor_critic = TwistAnyAdapterActorCritic(
            num_prop=base_obs_dim,
            num_critic_obs=total_obs_dim,
            num_priv_latent=0,
            num_hist=history_len,
            num_actions=num_actions,
            base_actor_jit_path=base_actor_jit_path,
            base_obs_dim=base_obs_dim,
            history_len=history_len,
            history_frame_dim=history_frame_dim,
            hist_state_dim=hist_state_dim,
            latent_dim=16,
            adapter_hidden_dims=[32, 32],
            critic_hidden_dims=[64, 32],
            world_model_hidden_dims=[64, 32],
            use_conv_history=True,
            freeze_base=True,
        )
        actor_critic.eval()

        with torch.no_grad():
            action = actor_critic.act_inference(obs)
            value = actor_critic.evaluate(obs)
            adapter_delta = actor_critic.get_adapter_delta(obs)
            world_pred = actor_critic.predict_world_model(obs)

        assert list(action.shape) == [4, num_actions], action.shape
        assert list(value.shape) == [4, 1], value.shape
        assert list(adapter_delta.shape) == [4, num_actions], adapter_delta.shape
        assert world_pred.shape[0] == 4, world_pred.shape
        assert abs(adapter_delta.mean().item()) < 1e-6, adapter_delta.mean().item()

    print("twist anyadapter shape test ok")


if __name__ == "__main__":
    main()
