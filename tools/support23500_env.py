"""Support-aware23500 environment; all controlled joints remain free."""
from pathlib import Path
import sys
import torch
from train_1250_coordinated import CoordinatedEnv
from legged_gym.envs.g1.g1_mimic_distill import g1_body_from_38_to_52
from legged_gym.envs.base.humanoid_char import convert_to_global_root_body_pos
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy_real'))
from joint_target_projection import xml_bounds,project_torch

class Support23500Env(CoordinatedEnv):
    def _target_bounds(self):
        if not hasattr(self,'_safe_target_lower'):
            lo,hi=xml_bounds(ROOT/'assets/g1/g1_sim2sim_with_wrist_roll.xml',self.dof_names,
                             self.cfg.control.projection_margin,self.cfg.control.projection_knee_min)
            self._safe_target_lower=torch.tensor(lo,device=self.device,dtype=self.dof_pos.dtype)
            self._safe_target_upper=torch.tensor(hi,device=self.device,dtype=self.dof_pos.dtype)
        return self._safe_target_lower,self._safe_target_upper

    def _compute_torques(self,actions):
        target=self.default_dof_pos_all+actions*self.cfg.control.action_scale
        lo,hi=self._target_bounds();safe=project_torch(target,lo,hi)
        # Keep command action history; model the same projection in training.
        return super()._compute_torques((safe-self.default_dof_pos_all)/self.cfg.control.action_scale)

    def _update_ref_motion(self):
        super()._update_ref_motion()
        if not hasattr(self,'feet_indices'):
            return
        t=self._get_motion_times()+self.dt
        pos,rot,_,_,_,_,body=self._motion_lib.calc_motion_frame(self._motion_ids,t)
        pos=pos.clone();pos[:,2]+=self.cfg.motion.height_offset
        pos[:,:2]+=self.episode_init_origin[:,:2]
        if body.shape[1]!=self._ref_body_pos.shape[1]:body=g1_body_from_38_to_52(body)
        future=convert_to_global_root_body_pos(root_pos=pos,root_rot=rot,body_pos=body)
        velocity=(future[:,self.feet_indices]-self._ref_body_pos[:,self.feet_indices])/self.dt
        height=self._ref_body_pos[:,self.feet_indices,2]-self.episode_init_origin[:,2:3]
        horizontal=(.20-velocity[:,:,:2].norm(dim=-1)).div(.15).clamp(0.,1.)
        vertical=(.12-velocity[:,:,2].abs()).div(.08).clamp(0.,1.)
        close_ground=(.15-height).div(.07).clamp(0.,1.)
        self.ref_stance_confidence=horizontal*vertical*close_ground
        self.reference_foot_velocity=velocity

    def _in_place_reference_mask(self):
        if not hasattr(self,'ref_stance_confidence'):
            return torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        return ((self._ref_root_vel[:,:2].norm(dim=-1)<.12)
                &(self._ref_root_ang_vel[:,2].abs()<.12)
                &(self.ref_stance_confidence.min(dim=-1).values>.5))

    def _reward_reference_stance_slip(self):
        contact=(self.contact_forces[:,self.feet_indices,2]>5.).float()
        v=self.rigid_body_states[:,self.feet_indices,7:9].square().sum(-1)
        return (v*self.ref_stance_confidence*contact).sum(-1)

    def _joint_ids(self,word):
        return [i for i,n in enumerate(self.dof_names) if word in n]

    @staticmethod
    def _huber(error,beta=.2):
        a=error.abs()
        return torch.where(a<beta,.5*a.square()/beta,a-.5*beta)

    def _reward_hip_yaw_tracking(self):
        ids=self._joint_ids('hip_yaw')
        return self._huber(self.dof_pos[:,ids]-self._ref_dof_pos[:,ids]).sum(-1)

    def _reward_squat_knee_tracking(self):
        ids=self._joint_ids('knee');ref=self._ref_dof_pos[:,ids]
        squat=(ref.mean(-1)>.8)&((self._ref_root_pos[:,2]-self.episode_init_origin[:,2])<.7)
        return self._huber(self.dof_pos[:,ids]-ref).sum(-1)*squat.float()

    def _reward_target_projection(self):
        target=self.default_dof_pos_all+self.actions*self.cfg.control.action_scale
        lo,hi=self._target_bounds()
        return (target-project_torch(target,lo,hi)).square().sum(-1)
