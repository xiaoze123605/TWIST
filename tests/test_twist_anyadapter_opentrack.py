"""Fast invariants for the paper-style TWIST AnyAdapter path."""

from pathlib import Path

import isaacgym  # Must precede torch when importing legged_gym environment modules.
import torch

from rsl_rl.algorithms.ppo_any2track import PPOAny2Track
from rsl_rl.modules.actor_critic_twist_anyadapter_opentrack import (
    TwistAnyAdapterOpenTrackActorCritic,
)
from legged_gym.envs.g1.anyadapter_history_mixin import AnyAdapterHistoryMixin


ROOT = Path(__file__).resolve().parents[1]
BASE_JIT = ROOT / "legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt"
STATE_INDICES = list(range(31, 82))


def _actor():
    if not BASE_JIT.is_file():
        raise RuntimeError(f"required frozen TWIST JIT is missing: {BASE_JIT}")
    return TwistAnyAdapterOpenTrackActorCritic(
        num_actions=23, num_critic_observations=1318,
        base_actor_jit_path=str(BASE_JIT), base_obs_dim=1155,
        history_len=79, history_frame_dim=74, hist_state_dim=51,
        wm_target_indices=STATE_INDICES, latent_dim=128,
        world_model_hidden_dims=(64, 64), critic_hidden_dims=(64, 32),
        activation="silu", freeze_base=True,
    )


def test_observation_shape_and_zero_init_identity():
    actor = _actor().eval()
    obs = torch.randn(2, 7001)
    with torch.no_grad():
        adapted = actor.act_inference(obs)
        base = actor.layerwise_actor.base_forward(obs[:, :1155])
    assert adapted.shape == (2, 23)
    assert (adapted - base).abs().max().item() < 2e-5


def test_frozen_base_and_gradient_ownership():
    actor = _actor()
    obs = torch.randn(2, 7001)
    before = [parameter.detach().clone() for parameter in actor.layerwise_actor.base_layers.parameters()]
    optimizer = torch.optim.Adam(actor.adapter.parameters(), lr=1e-3)
    # A policy-side loss must reach adapters but not the detached history encoder.
    loss = actor.actor_mean(obs).square().mean()
    optimizer.zero_grad()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.adapter.parameters())
    assert all(p.grad is None for p in actor.history_encoder.parameters())
    optimizer.step()
    assert all(not p.requires_grad for p in actor.layerwise_actor.base_layers.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, actor.layerwise_actor.base_layers.parameters()))

    # The dynamics-WM loss must train both the encoder and its own model.
    _, history = actor.split_obs(obs)
    state = actor.selected_state(obs)
    prediction = actor.world_model(state, torch.randn(2, 23), actor.encode_history_for_world_model(history))
    wm_loss = prediction.abs().mean()
    actor.zero_grad(set_to_none=True)
    wm_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.history_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.world_model.parameters())


def test_jit_parity_and_input_contract():
    actor = _actor().eval()
    obs = torch.randn(2, 7001)
    eager = actor.act_inference(obs)
    scripted = torch.jit.trace_module(actor, {"act_inference": obs})
    assert (eager - scripted.act_inference(obs)).abs().max().item() < 1e-5
    try:
        actor.act_inference(torch.randn(1, 7000))
    except RuntimeError as error:
        assert "7001" in str(error)
    else:
        raise AssertionError("invalid policy observation shape was accepted")


def test_continuation_optimizer_overrides_and_std_clamp():
    class DummyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.std = torch.nn.Parameter(torch.tensor([0.01, 0.20, 0.77]))

    algorithm = object.__new__(PPOAny2Track)
    algorithm.actor_critic = DummyActor()
    wm_parameter = torch.nn.Parameter(torch.ones(()))
    algorithm.ppo_optimizer = torch.optim.Adam(
        [algorithm.actor_critic.std], lr=1e-5
    )
    algorithm.wm_optimizer = torch.optim.Adam(
        [wm_parameter], lr=1e-4, weight_decay=1e-4
    )
    algorithm.policy_learning_rate = 5e-6
    algorithm.world_model_learning_rate = 3e-5
    algorithm.action_std_min = 0.03
    algorithm.action_std_max = 0.35

    algorithm.on_optimizer_state_loaded()

    assert algorithm.learning_rate == 5e-6
    assert algorithm.ppo_optimizer.param_groups[0]["lr"] == 5e-6
    assert algorithm.wm_optimizer.param_groups[0]["lr"] == 3e-5
    assert algorithm.wm_optimizer.param_groups[0]["weight_decay"] == 0.0
    assert torch.equal(
        algorithm.actor_critic.std.detach(), torch.tensor([0.03, 0.20, 0.35])
    )


def test_history_pairs_all_79_past_states_and_actions():
    class Dummy(AnyAdapterHistoryMixin):
        pass

    env = Dummy()
    env.use_anyadapter = True
    env.num_envs = 1
    env.num_actions = 23
    env.anyadapter_history_len = 79
    env.anyadapter_frame_dim = 74
    env.anyadapter_context_dim = 0
    env.anyadapter_fill_history_on_reset = True
    env.use_tracking_error_history = False
    env.anyadapter_state_indices = torch.arange(51)
    env.anyadapter_history = torch.zeros(1, 79, 74)
    env.anyadapter_prev_actions = torch.zeros(1, 23)
    env.anyadapter_pre_step_state = torch.zeros(1, 51)
    env.anyadapter_pre_step_state_valid = torch.zeros(1, dtype=torch.bool)
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)

    observation = None
    for step in range(83):
        env.episode_length_buf.fill_(step + 2)
        base = torch.full((1, 1155), float(step))
        action = torch.full((1, 23), float(step))
        observation = env._append_anyadapter_history(base)
        if step < 82:
            env._commit_anyadapter_transition(action)

    history = observation[:, 1155:].reshape(1, 79, 74)[0]
    expected = torch.arange(3, 82, dtype=torch.float32)
    for index in range(79):
        assert torch.all(history[index, :51] == expected[index])
        assert torch.all(history[index, 51:] == expected[index])
    assert torch.equal(history[:, 0], history[:, 51])
    assert not torch.any(history[:, 0] == 82)

    env.anyadapter_history.fill_(999.0)
    env.anyadapter_prev_actions.fill_(999.0)
    env.anyadapter_pre_step_state.fill_(999.0)
    env.anyadapter_pre_step_state_valid.fill_(True)
    env._reset_anyadapter_history(torch.tensor([0]))
    assert torch.all(env.anyadapter_history == 0)
    assert torch.all(env.anyadapter_prev_actions == 0)
    assert torch.all(env.anyadapter_pre_step_state == 0)
    assert not torch.any(env.anyadapter_pre_step_state_valid)
