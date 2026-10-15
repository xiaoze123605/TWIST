"""Opt-in true-state/PD-target recorder; leaves the existing actor input unchanged.

Place first in the MRO of a 23-DoF P-control environment. Captures next physical
state before resets, and stores PPO commands and executed targets separately.
No existing task is enabled automatically.
"""
import torch

CONTRACT = 'physical_pd_target_v1'


class PhysicalWMTransitionMixin:
    def _physical_wm_state(self):
        if self.dof_pos.shape[-1] != 23:
            raise ValueError('physical v1 recorder requires explicit native 23-joint order')
        quat=self.root_states[:,3:7]
        xyz=quat[:,:3];w=quat[:,3:4];velocity=self.root_states[:,10:13]
        uv=torch.cross(xyz,velocity,dim=-1)
        angular=velocity-2*w*uv+2*torch.cross(xyz,uv,dim=-1)
        x,y,z,qw=quat.unbind(-1)
        roll=torch.atan2(2*(qw*x+y*z),1-2*(x*x+y*y))
        pitch=torch.asin((2*(qw*y-z*x)).clamp(-1,1))
        return torch.cat((angular,torch.stack((roll,pitch),-1),self.dof_pos,self.dof_vel),-1).detach().clone()

    def step(self, actions):
        if getattr(self.cfg.control,'control_type','P') != 'P':
            raise ValueError('physical v1 recorder requires P control')
        # Existing G1 uses identity reindex. Reject other orders, never guess.
        probe=actions.new_tensor(list(range(23))).reshape(1,23)
        if not torch.equal(self.reindex(probe),probe):
            raise ValueError('nonidentity reindex needs an explicit physical joint map')
        self._physical_wm_pending=dict(state=self._physical_wm_state(),
                                      commanded_action=actions.detach().clone())
        result=super().step(actions)
        packet=self._physical_wm_pending
        if 'next_state' not in packet or 'executed_pd_target' not in packet:
            raise RuntimeError('physical transition hooks were not called')
        packet['done']=result[3].bool().detach().clone()
        packet['valid']=~packet['done']
        packet['contract']=CONTRACT
        packet['joint_names']=tuple(self.dof_names)
        packet['control_dt']=float(self.dt)
        # Shallow-copy extras; tensor payloads already own their storage.
        infos=dict(result[4]);infos['physical_wm_transition']=packet
        self._physical_wm_pending=None
        return (*result[:4],infos)

    def _compute_torques(self, actions):
        torques=super()._compute_torques(actions)
        packet=getattr(self,'_physical_wm_pending',None)
        if packet is not None:
            target=self.default_dof_pos_all+actions*self.cfg.control.action_scale
            if bool(getattr(self.cfg.control,'target_projection',False)):
                if not hasattr(self,'_target_bounds'):
                    raise RuntimeError('projection enabled without the existing projected P controller')
                low,high=self._target_bounds()
                target=torch.maximum(torch.minimum(target,high),low)
            packet['executed_action']=actions.detach().clone()
            packet['executed_pd_target']=target.detach().clone()
        return torques

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        packet=getattr(self,'_physical_wm_pending',None)
        if packet is not None:
            packet['next_state']=self._physical_wm_state()
