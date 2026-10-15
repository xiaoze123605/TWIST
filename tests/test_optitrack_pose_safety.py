"""Run in the gmr environment: python -m unittest discover -s tests -p 'test_optitrack_pose_safety.py'."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy_real'))
from server_optitrack_redis import OptiTrackRedisServer, _pose_safety_reason


class CrouchSafetyTest(unittest.TestCase):
    def setUp(self):
        self.bridge = OptiTrackRedisServer.__new__(OptiTrackRedisServer)
        self.bridge.args = SimpleNamespace(
            min_root_height=.35, max_root_height=1.05, max_root_tilt=.5,
            max_knee=2.4, max_knee_asymmetry=1.2,
        )
        self.pose = np.zeros(36)
        self.pose[2] = .57
        self.pose[3] = 1.
        self.pose[[10, 16]] = 1.8

    def test_crouch_that_old_standing_envelope_rejected(self):
        self.assertIn('root_z', _pose_safety_reason(self.pose, min_root_height=.65))
        self.assertIsNone(self.bridge._pose_safety_reason(self.pose))

    def test_height_is_checked_in_ground_coordinates(self):
        # Mocap's world origin can place raw IK below the robot floor envelope.
        self.pose[2] = .2
        self.assertIsNone(self.bridge._pose_safety_reason(self.pose, ground_aligned=False))
        self.assertIn('root_z', self.bridge._pose_safety_reason(self.pose))
        self.pose[2] = 1.2
        self.assertIn('root_z', self.bridge._pose_safety_reason(self.pose))

    def test_default_envelope_allows_independent_leg_motion(self):
        self.pose[[10, 16]] = [1.96, .004]
        self.assertIn('knee asymmetry', _pose_safety_reason(
            self.pose, max_knee_asymmetry=1.2))
        self.assertIsNone(_pose_safety_reason(self.pose))

    def test_default_envelope_allows_crouch_torso_lean(self):
        self.pose[3:7] = [np.cos(.3), 0., np.sin(.3), 0.]
        self.assertIn('root tilt', _pose_safety_reason(self.pose, max_root_tilt=.5))
        self.assertIsNone(_pose_safety_reason(self.pose))
        self.pose[3:7] = [np.cos(.55), 0., np.sin(.55), 0.]
        self.assertIn('root tilt', _pose_safety_reason(self.pose))

    def test_corrupt_and_folded_poses_remain_rejected(self):
        self.pose[0] = np.nan
        self.assertEqual(self.bridge._pose_safety_reason(self.pose, ground_aligned=False), 'NaN/Inf')
        self.pose[0] = 0.
        self.pose[10] = 2.5
        self.assertIn('knee limit', self.bridge._pose_safety_reason(self.pose, ground_aligned=False))
        self.pose[10] = .2
        self.assertIn('knee asymmetry', self.bridge._pose_safety_reason(self.pose, ground_aligned=False))
        self.pose[[10, 16]] = 1.8
        self.pose[3:7] = [np.cos(.4), np.sin(.4), 0., 0.]
        self.assertIn('root tilt', self.bridge._pose_safety_reason(self.pose, ground_aligned=False))


if __name__ == '__main__':
    unittest.main()
