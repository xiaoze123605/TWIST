import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from heading300_teacher import complete_teacher_action
class TeacherTests(unittest.TestCase):
 def test_learned_heading_included_once(self):
  class Teacher:
   def layerwise_actor(self,base,z):return base*.5
   def heading_delta(self,z,context):return z+context
  base=torch.tensor([[2.,4.]]);z=torch.tensor([[.1,.2]]);context=torch.tensor([[.3,.4]])
  torch.testing.assert_close(complete_teacher_action(Teacher(),base,z,context),torch.tensor([[1.4,2.6]]))
 def test_zero_heading_preserves_old_teacher(self):
  class Teacher:
   def layerwise_actor(self,base,z):return base
   def heading_delta(self,z,context):return torch.zeros_like(z)
  value=torch.ones(2,23)
  self.assertTrue(torch.equal(complete_teacher_action(Teacher(),value,value,value),value))
if __name__=='__main__':unittest.main()
