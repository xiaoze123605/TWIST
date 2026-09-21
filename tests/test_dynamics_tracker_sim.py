"""Opt-in GPU contract check: RUN_DYNAMICS_SIM_TEST=1 pytest this file."""
import os
import sys
from pathlib import Path

import isaacgym
import torch
import pytest


@pytest.mark.skipif(os.environ.get('RUN_DYNAMICS_SIM_TEST') != '1', reason='requires Isaac Gym GPU')
def test_simulator_delay_reset_clean_labels_and_nominal_mixture(monkeypatch):
    from legged_gym.envs import task_registry
    from legged_gym.gym_utils import get_args
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys, 'argv', ['simcheck', '--task', 'g1_dynamics_tracker_robust',
        '--headless', '--num_envs', '4', '--motion_file',
        str(root/'track_dataset/twist_motion_dataset/accad/B3___walk1.pkl')])
    args = get_args()
    env, _ = task_registry.make_env(args.task, args=args)
    try:
        assert env.nominal_env_count == 1
        assert env.motor_strength[:,0].eq(1).all()
        assert env.mass_params_tensor[0].eq(0).all()
        original_velocity = env.root_states[:,7:9].clone()
        env._push_robots()
        torch.testing.assert_close(env.root_states[0,7:9], original_velocity[0], rtol=0, atol=0)
        assert not torch.equal(env.root_states[1:,7:9], original_velocity[1:])
        env.max_episode_length = 5
        reset_count, timeout_count = 0, 0
        for _ in range(20):
            env.command_delay[1:] = 1
            clean = env.clean_state().clone()
            sensor = env.sensor_state.clone()
            last_queued = env.target_queue[:,-1].clone()
            raw = torch.randn(env.num_envs,23,device=env.device)*.03
            expected_sent = env.target_transform(raw, env.current_reference[:,8:31])
            _, _, _, done, info = env.step(raw)
            torch.testing.assert_close(info['dynamics_state'], clean, rtol=0, atol=0)
            torch.testing.assert_close(info['sent_target'], expected_sent)
            torch.testing.assert_close(info['applied_target'][0], expected_sent[0])
            torch.testing.assert_close(info['applied_target'][1:], last_queued[1:])
            if (~done).any():
                torch.testing.assert_close(info['dynamics_next_state'][~done], env.clean_state()[~done])
                torch.testing.assert_close(env.dynamics_history[~done,-1,:52], sensor[~done])
                torch.testing.assert_close(env.dynamics_history[~done,-1,52:75], expected_sent[~done])
            if done.any():
                assert env.dynamics_history[done].count_nonzero() == 0
                assert not torch.equal(info['dynamics_next_state'][done], env.clean_state()[done])
            reset_count += int(done.sum())
            timeout_count += int(info['time_outs'].sum())
        assert reset_count > 0 and timeout_count > 0
    finally:
        env.gym.destroy_sim(env.sim)
