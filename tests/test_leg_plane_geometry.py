import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from leg_plane_geometry import plane_normals,plane_alignment_cost

class LegPlaneTests(unittest.TestCase):
    def test_rotation_and_translation(self):
        points=torch.tensor([[[0.,0.,0.],[0.,0.,-1.],[1.,0.,-1.]]]);ref=plane_normals(points)
        self.assertAlmostEqual(float(plane_alignment_cost(plane_normals(points+7.),ref)),0.)
        rotated=torch.tensor([[[0.,0.,0.],[0.,0.,-1.],[0.,1.,-1.]]])
        self.assertAlmostEqual(float(plane_alignment_cost(plane_normals(rotated),ref)),1.)
    def test_degenerate_reference_and_actual(self):
        zero=torch.zeros(1,3);ref=torch.tensor([[0.,1.,0.]])
        self.assertEqual(float(plane_alignment_cost(ref,zero)),0.)
        self.assertEqual(float(plane_alignment_cost(zero,ref)),1.)
if __name__=='__main__':unittest.main()
