from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from rsl_rl.algorithms import PPODTERA
from rsl_rl.modules import TwistDTERAActorCritic


NUM_ACTIONS = 4
BASE_SINGLE_DIM = 30
BASE_HISTORY_LEN = 1
BASE_OBS_DIM = BASE_SINGLE_DIM * (BASE_HISTORY_LEN + 1)
HISTORY_LEN = 4
HIST_STATE_DIM = 51
DYN_FRAME_DIM = HIST_STATE_DIM + NUM_ACTIONS
ERROR_FRAME_DIM = 2 * NUM_ACTIONS + 7
TOTAL_OBS_DIM = (
    BASE_OBS_DIM
    + HISTORY_LEN * DYN_FRAME_DIM
    + HISTORY_LEN * ERROR_FRAME_DIM
)


class DummyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(BASE_OBS_DIM, NUM_ACTIONS)

    def forward(self, observations):
        return self.linear(observations)


def make_actor(path, branch_mode="full", gate_mode="full"):
    return TwistDTERAActorCritic(
        num_critic_obs=TOTAL_OBS_DIM,
        num_actions=NUM_ACTIONS,
        base_actor_jit_path=path,
        base_obs_dim=BASE_OBS_DIM,
        base_single_obs_dim=BASE_SINGLE_DIM,
        base_history_len=BASE_HISTORY_LEN,
        history_len=HISTORY_LEN,
        history_frame_dim=DYN_FRAME_DIM,
        hist_state_dim=HIST_STATE_DIM,
        wm_target_indices=list(range(HIST_STATE_DIM)),
        latent_dim=8,
        tracking_history_len=HISTORY_LEN,
        tracking_error_frame_dim=ERROR_FRAME_DIM,
        tracking_latent_dim=8,
        adapter_hidden_dims=[16, 16],
        critic_hidden_dims=[32, 16],
        world_model_hidden_dims=[16, 16],
        error_predictor_hidden_dims=[16, 16],
        risk_predictor_hidden_dims=[16, 16],
        default_ref_dof_pos=[0.0] * NUM_ACTIONS,
        adapter_branch_mode=branch_mode,
        gate_mode=gate_mode,
        tracking_error_scales=[1.0] * 5,
        freeze_base=True,
    )


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.square().sum().item()
    return total ** 0.5


class DTERATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base_path = str(Path(cls.tmp.name) / "base.pt")
        traced = torch.jit.trace(DummyBase().eval(), torch.zeros(1, BASE_OBS_DIM))
        traced.save(cls.base_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_history_and_latent_shapes(self):
        actor = make_actor(self.base_path)
        obs = torch.randn(3, TOTAL_OBS_DIM)
        base, dynamics, tracking = actor.split_dtera_obs(obs)
        self.assertEqual(base.shape, (3, BASE_OBS_DIM))
        self.assertEqual(dynamics.shape, (3, HISTORY_LEN, DYN_FRAME_DIM))
        self.assertEqual(tracking.shape, (3, HISTORY_LEN, ERROR_FRAME_DIM))
        self.assertEqual(actor.history_encoder(dynamics).shape, (3, 8))
        self.assertEqual(actor.tracking_error_history_encoder(tracking).shape, (3, 8))
        self.assertEqual(
            actor.predict_world_model_members(obs).shape,
            (3, 3, HIST_STATE_DIM),
        )

    def test_zero_init_and_base_only(self):
        obs = torch.randn(5, TOTAL_OBS_DIM)
        actor = make_actor(self.base_path)
        diag = actor.action_diagnostics(obs)
        self.assertLess(diag["delta_dyn"].abs().max().item(), 1e-8)
        self.assertLess(diag["delta_err"].abs().max().item(), 1e-8)
        self.assertTrue(torch.equal(actor.actor_mean(obs), actor.base_action(obs)))

        actor.adapter_branch_mode = "base_only"
        self.assertTrue(torch.equal(actor.actor_mean(obs), actor.base_action(obs)))
        self.assertEqual(actor.action_delta(obs).abs().max().item(), 0.0)

    def test_branch_feature_isolation(self):
        torch.manual_seed(7)
        actor = make_actor(self.base_path, gate_mode="off")
        with torch.no_grad():
            actor.adapter.dynamics_branch.net[-1].weight.fill_(0.01)
            actor.adapter.tracking_branch.net[-1].weight.fill_(0.01)
        obs = torch.randn(3, TOTAL_OBS_DIM)
        dyn0, err0 = actor.get_adapter_delta_components(obs)

        tracking_changed = obs.clone()
        tracking_changed[:, actor.tracking_history_offset:] += 3.0
        dyn1, err1 = actor.get_adapter_delta_components(tracking_changed)
        self.assertTrue(torch.equal(dyn0, dyn1))
        self.assertFalse(torch.equal(err0, err1))

        dynamics_changed = obs.clone()
        dynamics_changed[:, BASE_OBS_DIM:actor.tracking_history_offset] += 3.0
        dyn2, err2 = actor.get_adapter_delta_components(dynamics_changed)
        self.assertFalse(torch.equal(dyn0, dyn2))
        self.assertTrue(torch.equal(err0, err2))

    def test_gate_mathematics(self):
        actor = make_actor(self.base_path)
        zeros = torch.zeros(4, ERROR_FRAME_DIM)
        self.assertTrue(torch.equal(actor.tracking_demand(zeros), torch.zeros(4)))

        consistent = torch.zeros(3, 4, HIST_STATE_DIM)
        disagreement = consistent.clone()
        disagreement[0] = 4.0
        disagreement[1] = -4.0
        confidence0, _ = actor.confidence_from_members(consistent)
        confidence1, _ = actor.confidence_from_members(disagreement)
        self.assertTrue(torch.all(confidence1 < confidence0))

        risks = torch.tensor([-0.2, 0.0, 0.1, 0.4])
        safety = actor.safety_from_delta_risk(risks)
        self.assertTrue(torch.equal(safety[:2], torch.ones(2)))
        self.assertGreater(safety[2].item(), safety[3].item())

        diag = actor.action_diagnostics(torch.randn(9, TOTAL_OBS_DIM))
        self.assertTrue(torch.all((diag["gate"] >= 0) & (diag["gate"] <= 1)))

    def test_gate_off_matches_dual_formula(self):
        actor = make_actor(self.base_path, gate_mode="off")
        with torch.no_grad():
            actor.adapter.dynamics_branch.net[-1].bias.fill_(0.3)
            actor.adapter.tracking_branch.net[-1].bias.fill_(0.2)
        obs = torch.randn(4, TOTAL_OBS_DIM)
        diag = actor.action_diagnostics(obs)
        expected = (
            actor.dynamics_branch_gain * diag["delta_dyn"]
            + actor.tracking_branch_gain * diag["delta_err"]
        )
        self.assertTrue(torch.equal(diag["candidate_delta"], expected))
        self.assertTrue(torch.equal(diag["applied_delta"], expected))

    def test_wm_and_error_auxiliary_gradients(self):
        actor = make_actor(self.base_path)
        obs = torch.randn(6, TOTAL_OBS_DIM)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(6, NUM_ACTIONS)

        actor.zero_grad(set_to_none=True)
        actor.predict_world_model_members(obs, actions).square().mean().backward()
        self.assertGreater(grad_norm(actor.history_encoder.parameters()), 1e-8)
        self.assertGreater(grad_norm(actor.world_model.parameters()), 1e-8)

        actor.zero_grad(set_to_none=True)
        loss, _ = actor.error_prediction_loss(
            obs, actions, next_obs, torch.ones(6, 1)
        )
        loss.backward()
        self.assertGreater(
            grad_norm(actor.tracking_error_history_encoder.parameters()), 1e-8
        )
        self.assertGreater(grad_norm(actor.error_trend_predictor.parameters()), 1e-8)

    def test_optimizer_ownership_and_weight_decay(self):
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(
            object(), actor,
            world_model_loss_coef=0.1,
            error_prediction_loss_coef=0.05,
            weight_decay=1e-4,
        )
        expected = {
            "dynamics_encoder_in_ppo": False,
            "dynamics_encoder_in_wm": True,
            "tracking_encoder_in_ppo": True,
            "tracking_encoder_in_wm": False,
            "risk_predictor_only_in_risk": True,
        }
        self.assertEqual(algorithm.optimizer_ownership, expected)
        self.assertTrue(all(g["weight_decay"] == 0 for g in algorithm.wm_optimizer.param_groups))
        self.assertTrue(all(g["weight_decay"] == 0 for g in algorithm.risk_optimizer.param_groups))
        tracking_ids = {id(p) for p in actor.tracking_error_history_encoder.parameters()}
        tracking_groups = [
            group for group in algorithm.ppo_optimizer.param_groups
            if any(id(p) in tracking_ids for p in group["params"])
        ]
        self.assertTrue(tracking_groups)
        self.assertTrue(all(group["weight_decay"] == 0 for group in tracking_groups))

    def test_end_to_end_auxiliary_update(self):
        torch.manual_seed(11)
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(
            object(), actor,
            num_learning_epochs=1,
            num_mini_batches=1,
            world_model_loss_coef=0.1,
            error_prediction_loss_coef=0.05,
            adapter_reg_coef=0.02,
            adapter_bias_reg_coef=0.1,
            stand_anchor_coef=0.0,
            synthetic_stand_anchor_coef=2.0,
        )
        algorithm.init_storage(
            2, 4, [TOTAL_OBS_DIM], [TOTAL_OBS_DIM], [NUM_ACTIONS]
        )
        storage = algorithm.storage
        storage.observations.normal_()
        storage.next_observations.normal_()
        storage.next_observations_available.fill_(1)
        storage.privileged_observations.copy_(storage.observations)
        storage.actions.normal_()
        storage.dones.zero_()
        storage.timeouts.zero_()
        storage.values.normal_()
        storage.returns.normal_()
        storage.advantages.normal_()
        storage.mu.normal_()
        storage.sigma.fill_(0.5)
        storage.actions_log_prob.zero_()

        result = algorithm.update()
        self.assertEqual(len(result), 6)
        for key in (
            "world_model_loss", "error_prediction_loss", "risk_loss",
            "risk_positive_ratio", "gate_mean", "candidate_delta_l2",
            "applied_delta_l2", "tracking_encoder_aux_grad_norm",
        ):
            self.assertIn(key, algorithm.anyadapter_metrics)
            self.assertTrue(torch.isfinite(torch.tensor(
                algorithm.anyadapter_metrics[key]
            )))

    def test_future_risk_targets_ignore_timeout_and_stop_at_reset(self):
        actor = make_actor(self.base_path)
        algorithm = PPODTERA(object(), actor, risk_horizon=5)
        algorithm.init_storage(
            1, 5, [TOTAL_OBS_DIM], [TOTAL_OBS_DIM], [NUM_ACTIONS]
        )
        algorithm.storage.dones.zero_()
        algorithm.storage.timeouts.zero_()
        algorithm.storage.dones[2, 0, 0] = 1
        algorithm.storage.timeouts[2, 0, 0] = 1
        algorithm.storage.dones[4, 0, 0] = 1
        targets = algorithm._future_risk_targets()[:, 0, 0]
        self.assertTrue(torch.equal(
            targets, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        ))

    def test_synthetic_stand_and_checkpoint_buffers(self):
        actor = make_actor(self.base_path)
        stand = actor.build_synthetic_stand_observation(
            torch.randn(2, TOTAL_OBS_DIM)
        )
        first = stand[:, :BASE_SINGLE_DIM]
        second = stand[:, BASE_SINGLE_DIM:BASE_OBS_DIM]
        self.assertTrue(torch.equal(first, second))
        _, dynamics, tracking = actor.split_dtera_obs(stand)
        self.assertTrue(torch.equal(dynamics, torch.zeros_like(dynamics)))
        self.assertTrue(torch.equal(tracking, torch.zeros_like(tracking)))
        diag = actor.action_diagnostics(stand)
        self.assertLess(diag["candidate_delta"].abs().max().item(), 1e-8)
        self.assertLess(diag["applied_delta"].abs().max().item(), 1e-8)

        actor.wm_variance_ema.fill_(2.5)
        actor.wm_variance_ema_updates.fill_(17)
        restored = make_actor(self.base_path)
        restored.load_state_dict(actor.state_dict())
        self.assertTrue(torch.equal(restored.wm_variance_ema, actor.wm_variance_ema))
        self.assertTrue(torch.equal(
            restored.wm_variance_ema_updates, actor.wm_variance_ema_updates
        ))


class HistoryMixinTest(unittest.TestCase):
    def test_reset_repeated_fill_for_both_histories(self):
        path = REPO_ROOT / "legged_gym/legged_gym/envs/g1/anyadapter_history_mixin.py"
        spec = importlib.util.spec_from_file_location("history_mixin_for_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Env(module.AnyAdapterHistoryMixin):
            pass

        env = Env()
        env.cfg = type("Cfg", (), {})()
        env.cfg.env = type("EnvCfg", (), {
            "use_anyadapter": True,
            "anyadapter_history_len": 4,
            "anyadapter_state_indices": [0, 1, 2],
            "anyadapter_hist_state_dim": 3,
            "anyadapter_history_frame_dim": 3 + NUM_ACTIONS,
            "anyadapter_context_dim": 0,
            "anyadapter_fill_history_on_reset": True,
            "use_tracking_error_history": True,
            "tracking_error_history_len": 4,
            "tracking_error_frame_dim": ERROR_FRAME_DIM,
        })()
        env.device = "cpu"
        env.num_envs = 2
        env.num_actions = NUM_ACTIONS
        env.num_obs = 6
        env.obs_buf = torch.zeros(2, 6)
        env.episode_length_buf = torch.zeros(2)
        env.actions = torch.ones(2, NUM_ACTIONS)
        env.yaw = torch.zeros(2)
        env.roll = torch.zeros(2)
        env.pitch = torch.zeros(2)
        env.dof_pos = torch.zeros(2, NUM_ACTIONS)
        env.dof_vel = torch.zeros(2, NUM_ACTIONS)
        env._ref_dof_pos = torch.ones(2, NUM_ACTIONS)
        env._ref_dof_vel = torch.ones(2, NUM_ACTIONS)
        env.root_states = torch.zeros(2, 13)
        env._ref_root_vel = torch.ones(2, 3)
        env._ref_root_ang_vel = torch.zeros(2, 3)
        env._ref_root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
        env._init_anyadapter_history()
        base = torch.arange(12, dtype=torch.float32).reshape(2, 6) + 1.0
        result = env._append_anyadapter_history(base, env.actions)

        self.assertEqual(result.shape, (2, 6 + 4 * 7 + 4 * ERROR_FRAME_DIM))
        for history in (env.anyadapter_history, env.tracking_error_history):
            self.assertTrue(torch.equal(
                history, history[:, :1].expand_as(history)
            ))
        self.assertGreater(env.anyadapter_history[:, :, :3].abs().sum().item(), 0)
        self.assertEqual(
            env.anyadapter_history[:, :, 3:].abs().max().item(), 0.0
        )
        self.assertGreater(env.tracking_error_history.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
