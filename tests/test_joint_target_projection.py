import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy_real'))
from joint_target_projection import safe_bounds,project_numpy,project_torch,xml_bounds

class ProjectionTests(unittest.TestCase):
    def test_knee_by_name_and_numpy_torch_parity(self):
        names=['right_hip_yaw_joint','right_knee_joint','left_knee_joint']
        lo,hi=safe_bounds([-2,-.087,-.087],[2,2.88,2.88],names)
        x=np.array([[.1,-.2,1.2],[3,3,-.5]],dtype=np.float32)
        a=project_numpy(x,lo,hi);b=project_torch(torch.tensor(x),torch.tensor(lo),torch.tensor(hi)).numpy()
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a[0,2],np.float32(1.2));self.assertEqual(a[0,1],0.)
        self.assertTrue((a>=lo).all() and (a<=hi).all())
    def test_invalid_bounds_and_nonfinite_target(self):
        with self.assertRaises(ValueError):safe_bounds([0],[.01],['knee'])
        with self.assertRaises(ValueError):project_numpy([np.nan],[0],[1])
    def test_named_xml_subset_can_be_reordered(self):
        p=Path(__file__).resolve().parents[1]/'assets/g1/g1_sim2sim_with_wrist_roll.xml'
        lo,hi=xml_bounds(p,['right_knee_joint','left_hip_yaw_joint'])
        self.assertEqual(lo[0],0.);self.assertAlmostEqual(lo[1],-2.7576+.02)
        self.assertAlmostEqual(hi[0],2.8798-.02)
if __name__=='__main__':unittest.main()
