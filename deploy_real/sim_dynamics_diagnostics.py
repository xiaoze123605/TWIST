"""Read-only MuJoCo snapshots and 1 ms torque aggregation for policy diagnosis."""
import numpy as np
import mujoco
import torch


def quaternion_from_euler(rpy):
    r,p,y=np.asarray(rpy)/2
    cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy])


def hull_xy(points):
    points=sorted(set(map(tuple,np.asarray(points).reshape(-1,2))))
    if len(points)<3:return np.asarray(points).reshape(-1,2)
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return np.array(lower[:-1]+upper[:-1])


def support_margin(point,polygon):
    p=hull_xy(polygon)
    if len(p)<3:return None
    e=np.roll(p,-1,axis=0)-p;v=np.asarray(point)[:2]-p
    return float(np.min((e[:,0]*v[:,1]-e[:,1]*v[:,0])/np.linalg.norm(e,axis=1)))


class DynamicsDiagnostics:
    def __init__(self,model,core_policy=None):
        self.model=model
        self.ref_data=mujoco.MjData(model)
        self.pelvis=model.body('pelvis').id
        self.torso=model.body('torso_link').id
        self.feet=[model.body(s+'_ankle_roll_link').id for s in ['left','right']]
        self.names=[model.joint(i).name for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE]
        self.body_indices=np.array([i for i,n in enumerate(self.names) if 'wrist' not in n],dtype=int)
        assert len(self.body_indices)==23
        self.sole_ids=[[i for i in range(model.ngeom) if model.geom_bodyid[i]==b and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE] for b in self.feet]
        self.core=torch.jit.load(core_policy,map_location='cpu').eval() if core_policy else None
        self.previous_action=None

    def posture(self,data):
        com=data.subtree_com[self.pelvis].copy()
        feet=data.xpos[self.feet].copy()
        soles=[data.geom_xpos[ids].copy() for ids in self.sole_ids]
        torso=data.xmat[self.torso].reshape(3,3)
        pelvis=data.xmat[self.pelvis].reshape(3,3)
        heading=np.arctan2(pelvis[1,0],pelvis[0,0])
        direction=np.array([np.cos(heading),np.sin(heading)])
        return dict(com=com.tolist(),foot_positions=feet.tolist(),sole_points=[x.tolist() for x in soles],torso_pitch=float(np.arcsin(np.clip(-torso[2,0],-1,1))),com_forward_from_feet=float((com[:2]-feet[:,:2].mean(0))@direction),com_over_both_soles_margin=support_margin(com,np.concatenate(soles)[:,:2]))

    def reference(self,data,ref,target=None):
        self.ref_data.qpos[:]=data.qpos
        self.ref_data.qvel[:]=0
        self.ref_data.qpos[2]=ref[0]
        self.ref_data.qpos[3:7]=quaternion_from_euler(ref[1:4])
        if target is None:self.ref_data.qpos[7+self.body_indices]=ref[8:31]
        else:self.ref_data.qpos[7:]=target
        mujoco.mj_forward(self.model,self.ref_data)
        return self.posture(self.ref_data)

    def contacts(self,data):
        force=np.zeros(2);points=[];cop=np.zeros(3);weight=0.
        for i in range(data.ncon):
            c=data.contact[i];b1,b2=self.model.geom_bodyid[[c.geom1,c.geom2]]
            for k,b in enumerate(self.feet):
                if (b1==b and b2==0) or (b2==b and b1==0):
                    f=np.zeros(6);mujoco.mj_contactForce(self.model,data,i,f)
                    vertical=abs(c.frame[2]*f[0])
                    if vertical>1e-6:
                        force[k]+=vertical;points.append(c.pos[:2].copy());cop+=c.pos*vertical;weight+=vertical
        return dict(foot_vertical_force=force.tolist(),foot_contact=(force>5).tolist(),contact_points=np.asarray(points).reshape(-1,2).tolist(),contact_hull_margin=support_margin(data.subtree_com[self.pelvis],points),cop=(cop/weight).tolist() if weight>0 else None)

    def begin(self,row,data,runtime,limits,target):
        self.row=row
        actual=self.posture(data)
        processed=np.asarray(row['processed']);clean=np.asarray(row['clean'])
        diag=dict(actual=actual,clean_reference=self.reference(data,clean),processed_reference=self.reference(data,processed),target_posture=self.reference(data,processed,target),**self.contacts(data))
        with torch.no_grad():
            obs=torch.from_numpy(runtime.last_policy_obs).float().unsqueeze(0)
            raw=runtime.policy(obs.to(runtime.cfg.device)).cpu().numpy()[0]
            core=self.core(obs).numpy()[0] if self.core else None
        action=np.asarray(row['final_action'])
        diag['policy_action_pre_ema']=raw.tolist()
        diag['action_rate']=((action-self.previous_action)/.02).tolist() if self.previous_action is not None else [0.]*23
        diag['heading_delta']=(raw-core).tolist() if core is not None else None
        diag['heading_context']=runtime.last_policy_obs[-2:].tolist()
        self.previous_action=action.copy()
        self.limits=limits;self.maximum=np.zeros(len(limits));self.saturated=np.zeros(len(limits));self.torque_squared=np.zeros(len(limits));self.executed_squared=np.zeros(len(limits));self.n=0
        self.row['dynamics']=diag
        if row['frame_id']==0:self.row['diagnostics_schema']=dict(joint_names=self.names,policy_joint_indices=self.body_indices.tolist(),torque_limits=limits.tolist(),state_time='start of action interval',torque_time='all 20 substeps following this action',contact_margin='static contact convex hull; not a dynamic stability proof')

    def observe(self,requested,executed):
        self.maximum=np.maximum(self.maximum,abs(requested));self.saturated+=(abs(requested)>=self.limits-1e-6);self.torque_squared+=requested**2;self.executed_squared+=executed**2;self.n+=1

    def finish(self):
        self.row['dynamics'].update(torque_requested_peak=self.maximum.tolist(),torque_requested_rms=np.sqrt(self.torque_squared/self.n).tolist(),torque_executed_rms=np.sqrt(self.executed_squared/self.n).tolist(),torque_saturation_fraction=(self.saturated/self.n).tolist(),torque_substeps=self.n)
        return self.row
