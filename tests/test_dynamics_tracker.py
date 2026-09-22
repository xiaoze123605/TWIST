"""Contract and optimization tests for the independent tracker."""
import isaacgym  # Import order required by the existing repo.
import torch
import pytest

from rsl_rl.modules.dynamics_tracker import (
    DynamicsTrackerActorCritic, TargetTransform, append_history, reference_features,
)
from rsl_rl.modules.dynamics_tracker_runtime import DeploymentPolicy, DynamicsTrackerRuntime
from rsl_rl.algorithms.ppo_dynamics_tracker import PPODynamicsTracker
from tools.export_dynamics_tracker import export
from rsl_rl.runners.dynamics_tracker_runner import DynamicsTrackerRunner
from rsl_rl.datasets.dynamics_dagger_buffer import (
    DynamicsDaggerBuffer, SOURCE_CURRENT_STUDENT, SOURCE_OLD_STUDENT, SOURCE_TEACHER,
)


def actor(latent=True):
    return DynamicsTrackerActorCritic(6133, 135, use_dynamics_latent=latent,
        actor_hidden_dims=(32,), critic_hidden_dims=(32,), world_model_hidden_dims=(32,))


def spec(mode='reference'):
    return dict(version=1, policy_kind='dynamics_tracker', history_len=79,
                observation_dim=6133, use_dynamics_latent=True,
                joint_names=[str(i) for i in range(23)], control_dt=.02,
                joint_lower=[-2.] * 23, joint_upper=[2.] * 23,
                target_scales=[.5] * 23, action_mode=mode)


@pytest.mark.parametrize('mode', ['reference', 'direct'])
def test_targets_and_export_runtime_roundtrip(tmp_path, mode):
    ac = actor().eval()
    schema = spec(mode)
    config = dict(actor_hidden_dims=(32,), critic_hidden_dims=(32,),
                  world_model_hidden_dims=(32,), use_dynamics_latent=True)
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(dict(model_state_dict=ac.state_dict(), deployment_spec=schema,
                    train_cfg={'policy': config}, critic_dim=135), checkpoint)
    path = tmp_path / 'policy.pt'
    scripted = export(checkpoint, path)
    runtime = DynamicsTrackerRuntime(path, schema['joint_names'])
    runtime.reset(torch.zeros(23))
    history = torch.zeros(1, 79, 76)
    prev = torch.zeros(1, 31)
    initialized = torch.zeros(1, dtype=torch.bool)
    target = torch.zeros(1, 23)
    transform = TargetTransform(schema['joint_lower'], schema['joint_upper'], [.5]*23, mode)
    for _ in range(4):
        state = torch.randn(1, 52)
        ref = torch.randn(1, 31)
        task = reference_features(ref, prev, initialized, .02)
        obs = torch.cat((state, task, target, history.flatten(1)), -1)
        expected = transform(ac.actor_mean(obs), ref[:, 8:])
        actual = runtime.propose(state, ref)
        torch.testing.assert_close(actual, expected[0])
        torch.testing.assert_close(scripted(obs), expected)
        assert (actual.abs() <= 2).all()
        with pytest.raises(RuntimeError):
            runtime.propose(state, ref)
        # Deliberate external projection must be stored, rather than proposal.
        target = actual[None] * .9
        runtime.commit(target)
        history = append_history(history, state, target)
        torch.testing.assert_close(runtime.history, history)
        prev = ref
        initialized.fill_(True)
    runtime.reset(torch.ones(23))
    assert runtime.history.count_nonzero() == 0
    with pytest.raises(ValueError):
        DynamicsTrackerRuntime(path, list(reversed(schema['joint_names'])))


def test_gradient_ownership_and_stable_probability():
    ac = actor()
    obs = torch.randn(3, 6133)
    ac.act(obs)
    loss = -ac.get_actions_log_prob(torch.full((3,23), 100.)).mean() - ac.entropy.mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.actor.parameters())
    assert all(p.grad is None for p in ac.history_encoder.parameters())
    assert all(p.grad is None for p in ac.world_model.parameters())


def test_autoregression_appends_pre_state_and_masks_reset():
    ac = actor()
    alg = PPODynamicsTracker(None, ac, world_model_sequence_length=3)
    seen = []
    hook = ac.history_encoder.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
    state = torch.randn(2,52)
    state[:,3:6] = torch.tensor([0.,0.,-1.])
    commands = torch.randn(2,3,23)
    targets = state[:,None].repeat(1,3,1)
    done = torch.tensor([[True,False,False],[False,False,False]])
    history = torch.zeros(2,79,76)
    loss, count = alg.sequence_loss(history, state, commands, targets, done)
    assert count == 4
    torch.testing.assert_close(seen[1][:,-1,:52], state[1:])
    torch.testing.assert_close(seen[1][:,-1,52:75], commands[1:,0])
    poisoned = targets.clone()
    poisoned[0,1:] = float('nan')
    other, _ = alg.sequence_loss(history, state, commands, poisoned, done)
    torch.testing.assert_close(loss, other)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.history_encoder.parameters())
    assert all(p.grad is None for p in ac.actor.parameters())
    hook.remove()


def test_rollout_updates_full_actor_and_world_model():
    torch.manual_seed(8)
    ac = actor()
    alg = PPODynamicsTracker(None, ac, world_model_sequence_length=3,
        horizon_curriculum_updates=0, num_learning_epochs=1, num_mini_batches=1)
    alg.init_storage(2,4,[6133],[135],[23])
    before_actor = ac.actor[0].weight.detach().clone()
    before_encoder = ac.history_encoder.conv[0].weight.detach().clone()
    with torch.no_grad():
        for i in range(4):
            obs = torch.randn(2,6133)
            alg.act(obs, torch.randn(2,135), {})
            state = torch.randn(2,52)
            state[:,3:6] = torch.tensor([0.,0.,-1.])
            alg.process_env_step(torch.randn(2), torch.zeros(2,dtype=torch.bool), dict(
                dynamics_state=state, dynamics_next_state=state+.01,
                sent_target=torch.full((2,23),.123), time_outs=torch.zeros(2,dtype=torch.bool)))
        alg.compute_returns(torch.randn(2,135))
    assert torch.all(alg.dyn_command == .123)
    alg.update()
    assert not torch.equal(ac.actor[0].weight, before_actor)
    assert not torch.equal(ac.history_encoder.conv[0].weight, before_encoder)
    assert alg.metrics['world_model_horizon'] == 3


def test_stage_switch_preserves_action_and_resume_rejects_contract(tmp_path):
    import copy
    class Environment:
        device = 'cpu'
        num_obs, num_privileged_obs, num_actions, num_envs = 6133, 135, 23, 2
        def deployment_spec(self):
            return spec()
    cfg = dict(policy=dict(use_dynamics_latent=False, actor_hidden_dims=[32],
               critic_hidden_dims=[32], world_model_hidden_dims=[32]),
               algorithm=dict(world_model_sequence_length=3), runner=dict(num_steps_per_env=4))
    nominal = DynamicsTrackerRunner(Environment(), cfg)
    path = tmp_path/'nominal.pt'
    nominal.save(path)
    robust_cfg = copy.deepcopy(cfg)
    robust_cfg['policy']['use_dynamics_latent'] = True
    robust = DynamicsTrackerRunner(Environment(), robust_cfg)
    with pytest.raises(ValueError):
        robust.load(path)
    robust.load(path, warm_start=True)
    obs = torch.randn(5,6133)
    torch.testing.assert_close(robust.alg.actor_critic(obs), nominal.alg.actor_critic(obs), rtol=0, atol=0)


def test_sensor_assembly_matches_simulator_quaternions():
    from isaacgym.torch_utils import quat_rotate_inverse
    from rsl_rl.modules.dynamics_tracker_runtime import measured_state_and_reference
    quat = torch.nn.functional.normalize(torch.randn(5,4), dim=-1)
    q, dq, gyro, ref = torch.randn(5,23), torch.randn(5,23), torch.randn(5,3), torch.randn(5,31)
    state, relative = measured_state_and_reference(q, dq, gyro, quat[:,[3,0,1,2]], ref)
    expected_gravity = quat_rotate_inverse(quat, torch.tensor([[0.,0.,-1.]]).repeat(5,1))
    torch.testing.assert_close(state[:,3:6], expected_gravity, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(state[:,29:], dq*.05)
    assert (relative[:,3].abs() <= torch.pi).all()


def test_timeout_bootstraps_terminal_critic_not_reset_state():
    ac = actor()
    alg = PPODynamicsTracker(None, ac, world_model_sequence_length=1, gamma=.9)
    alg.init_storage(2,2,[6133],[135],[23])
    with torch.no_grad():
        alg.act(torch.zeros(2,6133), torch.zeros(2,135), {})
        terminal = torch.randn(2,135)
        expected = 1. + .9 * ac.evaluate(terminal)[0,0]
        alg.process_env_step(torch.ones(2), torch.ones(2,dtype=torch.bool), dict(
            dynamics_state=torch.zeros(2,52), dynamics_next_state=torch.ones(2,52),
            sent_target=torch.ones(2,23), time_outs=torch.tensor([True,False]),
            terminal_critic_observation=terminal))
    torch.testing.assert_close(alg.storage.rewards[0,0,0], expected)
    assert alg.storage.rewards[0,1,0] == 1
    assert alg.dyn_next[0].eq(1).all()


def _dagger_round(round_id, source, count=20):
    return dict(
        actor_input=torch.randn(count, 303), teacher_target=torch.randn(count, 23),
        motion_id=torch.zeros(count, dtype=torch.long), motion_time=torch.linspace(0, 1, count),
        motion_phase=torch.linspace(0, 1, count),
        episode_id=torch.arange(count, dtype=torch.long) + round_id*100,
        source=torch.full((count,), source, dtype=torch.long),
        steps_to_failure=torch.tensor(([30, 40, -1, -1]*((count+3)//4))[:count]),
        failure_margin_m=torch.linspace(-.01, .2, count),
        seed=torch.full((count,), 42, dtype=torch.long),
        action_mode=torch.zeros(count, dtype=torch.long),
        target_scale_version=torch.ones(count, dtype=torch.long),
        collection_round=torch.full((count,), round_id, dtype=torch.long))


def test_cumulative_dagger_replay_and_stratification(tmp_path):
    replay = DynamicsDaggerBuffer()
    replay.append_round(_dagger_round(0, SOURCE_TEACHER), {'round': 0, 'beta': 1.0})
    replay.append_round(_dagger_round(1, SOURCE_CURRENT_STUDENT), {'round': 1, 'beta': .7})
    replay.append_round(_dagger_round(2, SOURCE_CURRENT_STUDENT), {'round': 2, 'beta': .5})
    assert len(replay) == 60
    assert replay.source_counts() == {'teacher': 20, 'old_student': 20, 'current_student': 20}
    path = tmp_path/'dagger.pt'
    file_hash = replay.save(path)
    loaded = DynamicsDaggerBuffer.load(path)
    assert file_hash == loaded.file_sha256(path)
    train, validation = loaded.split_train_validation()
    assert train.numel() + validation.numel() == len(loaded)
    ids = loaded.stratified_indices(torch.arange(len(loaded)), 1000, latest_round=2,
                                    generator=torch.Generator().manual_seed(3))
    counts = torch.bincount(loaded.tensors['source'][ids], minlength=3)
    assert counts.tolist() == [400, 300, 300]
    with pytest.raises(ValueError):
        loaded.append_round(_dagger_round(2, SOURCE_CURRENT_STUDENT), {'round': 2})
