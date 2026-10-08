import torch
from isaacgym.torch_utils import quat_rotate_inverse
from heading23500_env import Heading23500Env
from leg_plane_geometry import plane_normals,plane_alignment_cost

class Plane23500Env(Heading23500Env):
    def _reward_leg_plane_alignment(self):
        if not hasattr(self,'leg_plane_body_ids'):
            names=[[side+'_hip_yaw_link',side+'_knee_link',side+'_ankle_pitch_link'] for side in ['left','right']]
            ids=[[self.gym.find_actor_rigid_body_handle(self.envs[0],self.actor_handles[0],n) for n in leg] for leg in names]
            if min(v for leg in ids for v in leg)<0:raise ValueError('leg-plane link mapping missing')
            self.leg_plane_body_ids=torch.tensor(ids,device=self.device,dtype=torch.long)
        actual=plane_normals(self.rigid_body_states[:,self.leg_plane_body_ids,:3])
        ref=plane_normals(self._ref_body_pos[:,self.leg_plane_body_ids])
        aq=self.base_quat[:,None,:].expand(-1,2,-1).reshape(-1,4)
        rq=self._ref_root_rot[:,None,:].expand(-1,2,-1).reshape(-1,4)
        actual=quat_rotate_inverse(aq,actual.reshape(-1,3)).reshape(-1,2,3)
        ref=quat_rotate_inverse(rq,ref.reshape(-1,3)).reshape(-1,2,3)
        knee_ids=self._joint_ids('knee')
        squat=(self._ref_dof_pos[:,knee_ids].mean(-1)>.8)&((self._ref_root_pos[:,2]-self.episode_init_origin[:,2])<.7)
        return (plane_alignment_cost(actual,ref)*self.ref_stance_confidence).sum(-1)*squat.float()
