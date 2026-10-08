import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy_real'))
from heading_context import heading_features

class HeadingContextTests(unittest.TestCase):
    def test_startup_frame_alignment(self):
        ref,robot=.03,1.37
        np.testing.assert_allclose(heading_features(ref,robot,robot-ref),[0,0],atol=1e-7)
        moved=heading_features(ref,robot+.1,robot-ref)
        self.assertLess(moved[0],0)
        self.assertGreater(moved[1],0)
    def test_wrap_boundary_is_continuous(self):
        a=heading_features(-np.pi+.01,np.pi-.01)
        b=heading_features(.02,0.)
        np.testing.assert_allclose(a,b,atol=1e-7)
if __name__=='__main__':unittest.main()
