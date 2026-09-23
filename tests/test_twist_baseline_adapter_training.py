"""Regression checks for the anchored adapter's causal WM contract."""

import isaacgym  # Isaac Gym must load before torch in this checkout.
import torch

from rsl_rl.algorithms.ppo_twist_baseline_adapter import PPOTwistBaselineAdapter


class RecordingActor:
    num_actions = 1
    wm_component_splits = {"state": (0, 1)}

    def __init__(self):
        self.calls = []

    def split_obs(self, obs):
        return obs[:, :1], obs[:, 1:].reshape(-1, 2, 2)

    def selected_state(self, obs):
        return obs[:, :1]

    def predict_world_model(self, obs, action, *, state, history):
        self.calls.append((state.clone(), action.clone(), history.clone()))
        return state + 1


def test_world_model_uses_sent_action_and_pretransition_state():
    actor = RecordingActor()
    algorithm = object.__new__(PPOTwistBaselineAdapter)
    algorithm.actor_critic = actor
    algorithm.world_model_loss_type = "smooth_l1"
    algorithm.world_model_component_weights = {"state": 1.0}
    # Each observation is [current state, two (state, sent action) frames].
    obs = torch.tensor([[[10., 0., 0., 0., 0.],
                         [11., 0., 0., 10., 3.],
                         [12., 10., 3., 11., 4.]]])
    next_obs = torch.tensor([[[11., 0., 0., 10., 3.],
                              [12., 10., 3., 11., 4.],
                              [13., 11., 4., 12., 5.]]])
    # PPO samples differ from sent commands, as can occur with delay/clipping.
    sampled = torch.full((1, 3, 1), 99.)
    loss, _, valid = algorithm._autoregressive_world_model_loss(
        obs, sampled, next_obs, torch.zeros(1, 3, 1), torch.ones(1, 3, 1),
    )
    assert valid == 3
    assert loss.item() == 0.
    assert [call[1].item() for call in actor.calls] == [3., 4., 5.]
    assert actor.calls[1][2][0, -1].tolist() == [10., 3.]
    assert actor.calls[2][2][0, -1].tolist() == [11., 4.]


def test_world_model_resets_history_after_terminal_transition():
    actor = RecordingActor()
    algorithm = object.__new__(PPOTwistBaselineAdapter)
    algorithm.actor_critic = actor
    algorithm.world_model_loss_type = "smooth_l1"
    algorithm.world_model_component_weights = {"state": 1.0}
    obs = torch.tensor([[[10., 0., 0., 0., 0.], [20., 20., 0., 20., 0.]]])
    next_obs = torch.tensor([[[20., 20., 0., 20., 0.], [21., 20., 0., 20., 6.]]])
    _, _, valid = algorithm._autoregressive_world_model_loss(
        obs, torch.zeros(1, 2, 1), next_obs,
        torch.tensor([[[1.], [0.]]]), torch.ones(1, 2, 1),
    )
    assert valid == 1
    assert actor.calls[1][0].item() == 20.
    assert actor.calls[1][2][0, -1].tolist() == [20., 0.]


def test_refinement_penalizes_large_action_corrections():
    algorithm = object.__new__(PPOTwistBaselineAdapter)
    algorithm.adapter_tail_threshold = 0.12
    algorithm.adapter_tail_coef = 8.0
    small_loss, small_tail = algorithm._adapter_regularization(torch.tensor([[0.05, -0.10]]))
    large_loss, large_tail = algorithm._adapter_regularization(torch.tensor([[0.05, -0.30]]))
    assert small_tail.item() == 0.0
    assert large_tail.item() > 0.0
    assert large_loss.item() > small_loss.item()
