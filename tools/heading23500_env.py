import torch
from isaacgym.torch_utils import get_euler_xyz
from support23500_env import Support23500Env

class Heading23500Env(Support23500Env):
    def _reward_heading_tracking(self):
        _,_,ref_yaw=get_euler_xyz(self._ref_root_rot)
        error=torch.atan2(torch.sin(self.yaw-ref_yaw),torch.cos(self.yaw-ref_yaw))
        return 1.-torch.cos(error)
