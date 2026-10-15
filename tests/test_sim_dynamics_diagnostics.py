import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy_real'))
from sim_dynamics_diagnostics import hull_xy,support_margin,quaternion_from_euler
class DynamicsGeometryTests(unittest.TestCase):
 def test_margin_and_degenerate_support(self):
  points=[[1,1],[-1,1],[-1,-1],[1,-1],[0,0]]
  self.assertAlmostEqual(support_margin([0,0],points),1.)
  self.assertAlmostEqual(support_margin([2,0],points),-1.)
  self.assertIsNone(support_margin([0,0],[[0,0],[1,1]]))
 def test_reference_quaternion(self):
  np.testing.assert_allclose(quaternion_from_euler([0,0,0]),[1,0,0,0])
  np.testing.assert_allclose(quaternion_from_euler([0,np.pi/2,0]),[2**-.5,0,2**-.5,0],atol=1e-7)
if __name__=='__main__':unittest.main()
