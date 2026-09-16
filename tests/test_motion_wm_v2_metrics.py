"""Exercise V2 aggregation without importing Isaac Gym or creating a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class Base:
    def compute_reward(self):
        pass

    def reset_idx(self, env_ids, motion_ids=None):
        self.motion_reference_pipeline.mode[env_ids] = 2


source = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/envs/g1/g1_motion_wm_dtera.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'G1MotionWMDTERAV2')
namespace = dict(torch=torch, G1MotionWMDTERA=Base)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
Environment = namespace['G1MotionWMDTERAV2']


class ModeMetricsTests(unittest.TestCase):
    def test_sample_weighted_metrics_reset_attribution_and_empty_modes(self):
        env = Environment()
        env.motion_reference_pipeline = SimpleNamespace(
            mode=torch.tensor([0, 0, 1]), MODES=('clean', 'corrupt', 'wm'))
        env._mode_metrics = torch.zeros(3, 5)
        env._mode_episode_steps = torch.zeros(3)
        env.dof_pos = torch.tensor([[1., 1.], [3., 3.], [2., 2.]])
        env._ref_dof_pos = torch.zeros(3, 2)
        env.rew_buf = torch.tensor([2., 4., 9.])
        env.compute_reward()
        env.compute_reward()
        env.reset_idx(torch.tensor([0, 2]))
        metrics = env.pop_training_metrics()
        self.assertEqual(metrics['MotionReference/clean/steps'], 4)
        self.assertEqual(metrics['MotionReference/clean/reward_per_step'], 3)
        self.assertAlmostEqual(metrics['MotionReference/clean/clean_joint_rmse'], 5**0.5)
        self.assertEqual(metrics['MotionReference/corrupt/clean_joint_rmse'], 2)
        self.assertEqual(metrics['MotionReference/clean/episode_length_steps'], 2)
        self.assertEqual(metrics['MotionReference/corrupt/completed_episodes'], 1)
        self.assertEqual(metrics['MotionReference/wm/steps'], 0)
        self.assertNotIn('MotionReference/wm/clean_joint_rmse', metrics)
        self.assertEqual(env._mode_episode_steps.tolist(), [0, 2, 0])
        self.assertTrue(all(v == 0 for v in env.pop_training_metrics().values()))


if __name__ == '__main__':
    unittest.main()
