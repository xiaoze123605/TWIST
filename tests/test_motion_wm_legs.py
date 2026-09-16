import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch


source = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/envs/g1/g1_motion_wm_dtera.py'
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == 'G1MotionWMDTERALegs')


class Base:
    def _init_buffers(self):
        pass


ns = dict(torch=torch, G1MotionWMDTERAV2=Base)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)


class LegRewardTests(unittest.TestCase):
    def test_named_mapping_clean_targets_and_bilateral_symmetry(self):
        env = ns['G1MotionWMDTERALegs']()
        env.device = 'cpu'
        names = [s+'_'+j+'_joint' for s in ('left','right')
                 for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        env.dof_names = ['arm'] + names[::-1]
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(leg_position_sigma=.2, leg_velocity_sigma=2.))
        env._init_buffers()
        env.dof_pos = torch.zeros(2,13)
        env._ref_dof_pos = torch.zeros(2,13)
        env.dof_vel = torch.zeros(2,13)
        env._ref_dof_vel = torch.zeros(2,13)
        torch.testing.assert_close(env._reward_tracking_leg_position(), torch.ones(2))
        env.dof_pos[0,env.dof_names.index('left_knee_joint')] = .2
        env.dof_pos[1,env.dof_names.index('right_knee_joint')] = .2
        rewards = env._reward_tracking_leg_position()
        self.assertLess(float(rewards[0]), 1.)
        torch.testing.assert_close(rewards[0], rewards[1])
        env.dof_pos[:,0] = 100
        torch.testing.assert_close(rewards, env._reward_tracking_leg_position())
        env.dof_vel[:,env._leg_ids] = 2.
        torch.testing.assert_close(env._reward_tracking_leg_velocity(), torch.full((2,), torch.exp(torch.tensor(-1.)).item()))
        env.dof_names.remove('left_knee_joint')
        with self.assertRaises(ValueError):
            env._init_buffers()


if __name__ == '__main__':
    unittest.main()
