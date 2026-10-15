"""Isolated MuJoCo heading-recovery teacher, with legacy rollout parity.

The frozen 7003-D policy retains its original inputs/history. An independent
168-D correction has physical-state/goal interactions and a frozen P0 latent.
Privileged height/velocity/contact make this a simulation teacher, not a
hardware-deployable student. No production controller is modified.
"""
import copy
import math
from pathlib import Path
import sys
from collections import deque

import numpy as np
import torch
from torch import nn
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deploy_real'),str(ROOT/'rsl_rl')]
from server_low_level_g1_sim import RealTimePolicyController
from twist_anyadapter_runtime import AnyAdapterRuntime
from data_utils.rot_utils import quatToEuler,quat_rotate_inverse
from data_utils.params import DEFAULT_MIMIC_OBS
from rsl_rl.modules.physical_world_model import PhysicalWorldModel,CONTRACT as PHYSICAL_CONTRACT

POLICY=ROOT/'reports/recovery_1250/fix_stage_20261008/heading_300.pt'
PHYSICAL=ROOT/'reports/physical_wm_p0_20261013/pilot/best_physical_wm.pt'
XML=ROOT/'assets/g1/g1_sim2sim_with_wrist_roll.xml'
FEATURE_DIM=168
CONTRACT='heading_recovery_teacher_v1'
BODY_IDS=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23])


def wrap(angle):return np.arctan2(np.sin(angle),np.cos(angle))


class RecoveryActorCritic(nn.Module):
    def __init__(self,delta_scale=.1,std=.015):
        super().__init__();self.delta_scale=float(delta_scale)
        self.actor=nn.Sequential(nn.Linear(168,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,23))
        self.critic=nn.Sequential(nn.Linear(168,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight);nn.init.zeros_(self.critic[-1].bias)
        self.register_buffer('std',torch.full((23,),float(std)))

    def delta(self,features):
        if features.shape[-1]!=168:raise ValueError('recovery teacher requires physical/goal feature contract168')
        return self.delta_scale*self.actor(features).tanh()

    def mean(self,features,source_action):return source_action+self.delta(features)
    def value(self,features):return self.critic(features).squeeze(-1)


class TurnHoldReference:
    """Smooth +/-30-degree turn at 2..4s, then hold to 8s; no oracle future."""
    def __init__(self,sign):
        self.sign=int(sign);self.length=400
        full=np.asarray(DEFAULT_MIMIC_OBS['g1'],dtype=np.float32)
        self.base=full[[i for i in range(33) if i not in [27,32]]].copy()
        self.wrists=full[[27,32]].copy()

    def get(self,frame):
        t=(frame+1)*.02;u=np.clip((t-2)/2,0,1);angle=self.sign*math.radians(30)
        ref=self.base.copy();ref[3]=angle*.5*(1-np.cos(np.pi*u))
        ref[4:7]=0;ref[7]=angle*np.pi/4*np.sin(np.pi*u) if 2<t<4 else 0
        return ref,ref.copy(),self.wrists.copy()


class RecordedMotionReference:
    def __init__(self,frames):self.frames=frames;self.length=len(frames)
    def get(self,frame):
        row=self.frames[min(frame,self.length-1)]
        return np.array(row['processed'],np.float32),np.array(row['clean'],np.float32),np.array(row['pd_target_unprojected'],np.float64)[[19,24]]


class HeadingRecoveryBatch:
    def __init__(self,num_envs,seed=42):
        self.rng=np.random.default_rng(seed);self.num_envs=int(num_envs)
        self.template=RealTimePolicyController(xml_file=str(XML),policy_path=str(POLICY),device='cpu',headless=True,
            sim_duration=64.14,target_projection=True,projection_margin=.02,projection_knee_min=0.,anyadapter_ema_alpha=0.)
        self.model=self.template.model;self.source=self.template.anyadapter_runtime.policy.eval()
        for parameter in self.source.parameters():parameter.requires_grad_(False)
        self.physical_checkpoint=torch.load(PHYSICAL,map_location='cpu',weights_only=False)
        assert self.physical_checkpoint['contract']==PHYSICAL_CONTRACT
        self.physical=PhysicalWorldModel(**self.physical_checkpoint['normalizers']).eval().requires_grad_(False)
        self.physical.load_state_dict(self.physical_checkpoint['model_state_dict'])
        self.data=[mujoco.MjData(self.model) for _ in range(self.num_envs)]
        self.runtime=[]
        for _ in self.data:
            runtime=copy.copy(self.template.anyadapter_runtime)
            for k,v in vars(runtime).items():
                if isinstance(v,np.ndarray):setattr(runtime,k,v.copy())
            runtime.reset();self.runtime.append(runtime)
        self.feet=[self.model.body(side+'_ankle_roll_link').id for side in ['left','right']]
        self.refs=[None]*self.num_envs;self.frames=np.zeros(self.num_envs,dtype=int)
        self.base_history=[deque(maxlen=10) for _ in self.data]
        self.physical_history=np.zeros((self.num_envs,79,74),np.float32)
        self.history_count=np.zeros(self.num_envs,int)
        self.last_action=np.zeros((self.num_envs,23),np.float32)
        self.last_delta=np.zeros((self.num_envs,23),np.float32)
        self.ref_q_previous=[None]*self.num_envs;self.ref_dq=np.zeros((self.num_envs,23),np.float32)
        self.cache=None

    def reset_one(self,index,reference=None,yaw_degrees=None):
        data=self.data[index];mujoco.mj_resetData(self.model,data);mujoco.mj_forward(self.model,data)
        data.qpos[:]=self.template.mujoco_default_dof_pos
        yaw=math.radians(float(self.rng.uniform(-15,15) if yaw_degrees is None else yaw_degrees))
        data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        mujoco.mj_forward(self.model,data)
        self.refs[index]=reference if reference is not None else TurnHoldReference(self.rng.choice([-1,1]))
        self.frames[index]=0;self.base_history[index].clear()
        for _ in range(10):self.base_history[index].append(np.zeros(105))
        self.runtime[index].reset();self.last_action[index]=0;self.last_delta[index]=0
        self.physical_history[index]=0;self.history_count[index]=0
        self.ref_q_previous[index]=None;self.ref_dq[index]=0
        self.cache=None

    def reset(self):
        for i in range(self.num_envs):self.reset_one(i)
        return self.observe()

    def contacts(self,data):
        force=np.zeros(2)
        for i in range(data.ncon):
            contact=data.contact[i];b1,b2=self.model.geom_bodyid[[contact.geom1,contact.geom2]]
            for k,body in enumerate(self.feet):
                if (b1==body and b2==0) or (b2==body and b1==0):
                    local_force=np.zeros(6);mujoco.mj_contactForce(self.model,data,i,local_force)
                    force[k]+=abs(contact.frame[2]*local_force[0])
        return (force>5).astype(np.float32)

    def observe(self):
        if self.cache is not None:return self.cache
        observations=[];physical_states=[];contexts=[];refs=[];wrists=[];rows=[]
        for i,data in enumerate(self.data):
            processed,clean,wrist=self.refs[i].get(int(self.frames[i]));refs.append(processed);wrists.append(wrist)
            qp=data.qpos.astype(np.float32);qv=data.qvel.astype(np.float32)
            q=qp[7+BODY_IDS];dq=qv[6+BODY_IDS]
            quaternion=data.sensor('orientation').data.astype(np.float32);rpy=quatToEuler(quaternion)
            gyro=data.sensor('angular-velocity').data.astype(np.float32)
            dq_masked=dq.copy();dq_masked[[4,5,10,11]]=0
            proprio=np.concatenate((gyro*.25,rpy[:2],q-self.template.default_dof_pos,dq_masked*.05,self.last_action[i]))
            base_frame=np.concatenate((processed,proprio));base=np.concatenate((base_frame,np.array(self.base_history[i]).flatten()))
            self.base_history[i].append(base_frame)
            error=wrap(processed[3]-rpy[2]);heading=np.array([np.sin(error),1-np.cos(error)],np.float32)
            observation=self.runtime[i]._build_augmented_obs(base,adapter_context=heading)
            self.runtime[i].last_policy_obs=observation.copy();observations.append(observation)
            native_rpy=quatToEuler(data.qpos[3:7]);native_q=data.qpos[7+BODY_IDS];native_dq=data.qvel[6+BODY_IDS]
            physical=np.concatenate((data.qvel[3:6],native_rpy[:2],native_q,native_dq)).astype(np.float32)
            physical_states.append(physical)
            if self.ref_q_previous[i] is None:self.ref_dq[i]=native_dq
            else:self.ref_dq[i]=.5*self.ref_dq[i]+.5*np.clip((processed[8:]-self.ref_q_previous[i])/.02,-20,20)
            self.ref_q_previous[i]=processed[8:].copy()
            rotation=np.array([quaternion[1],quaternion[2],quaternion[3],quaternion[0]],np.float32)[None]
            linear=quat_rotate_inverse(rotation,np.array(data.qvel[:3],np.float32)[None])[0]
            true_yaw_error=wrap(processed[3]-native_rpy[2])
            context=np.concatenate(([np.sin(true_yaw_error),1-np.cos(true_yaw_error)],
                [processed[7]-data.qvel[5]],wrap(processed[1:3]-native_rpy[:2])/.2,
                (processed[8:]-native_q)/.35,(self.ref_dq[i]-native_dq)/3,
                processed[4:7]-linear,[(processed[0]-data.qpos[2])/.1],self.contacts(data),processed[4:8]))
            assert len(context)==61;contexts.append(np.clip(context,-10,10))
            rows.append(dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),joint=q.copy(),joint_velocity=dq.copy(),
                rpy=rpy.copy(),height=float(data.xpos[self.model.body('pelvis').id,2]),root_velocity=linear.copy(),
                clean=clean.copy(),processed=processed.copy(),reference_frame_index=int(self.frames[i])))
        old=torch.from_numpy(np.stack(observations)).float()
        with torch.no_grad():
            source=self.source(old)
            state=torch.from_numpy(np.stack(physical_states))
            history=torch.from_numpy(self.physical_history)
            normalized_history=torch.cat((self.physical.normalize_state(history[:,:,:51]),
                self.physical.normalize_target(history[:,:,51:])), -1)
            latent=self.physical.history_encoder(normalized_history.transpose(1,2))
            ready=torch.from_numpy((self.history_count>=79).astype(np.float32))[:,None]
            latent=torch.nn.functional.layer_norm(latent,(32,))*ready
            feature=torch.cat((self.physical.normalize_state(state).clamp(-10,10),
                torch.from_numpy(np.stack(contexts)).float(),latent.clamp(-10,10),source/3,ready),-1)
        assert feature.shape==(self.num_envs,168)
        self.cache=dict(features=feature,source=source,legacy=old,physical=state.numpy(),rows=rows,
                        references=refs,wrists=wrists)
        return self.cache

    def step(self,commands,source_actions,auto_reset=True):
        start=self.observe();commands=np.asarray(commands,dtype=np.float32)
        source_actions=np.asarray(source_actions,dtype=np.float32);targets=[];deltas=[]
        for i in range(self.num_envs):
            action=np.clip(commands[i],-10,10);runtime=self.runtime[i]
            completed=np.concatenate((runtime._pending_dyn_state,action));runtime.history[:-1]=runtime.history[1:];runtime.history[-1]=completed
            runtime._pending_dyn_state=None;runtime._history_initialized=True;runtime.prev_action=action.copy()
            self.last_action[i]=action
            body_target=action*.5+self.template.default_dof_pos
            target=np.empty(25);target[BODY_IDS]=body_target;target[[19,24]]=start['wrists'][i]
            unprojected=target.copy();target=np.maximum(np.minimum(target,self.template.target_upper),self.template.target_lower)
            delta=action-source_actions[i];deltas.append(delta.copy());targets.append(target.copy())
            self.physical_history[i,:-1]=self.physical_history[i,1:]
            self.physical_history[i,-1]=np.concatenate((start['physical'][i],target[BODY_IDS]))
            self.history_count[i]+=1
            start['rows'][i].update(final_action=action.copy(),pd_target=target.copy(),pd_target_unprojected=unprojected,
                                    correction=delta.copy())
        # Match original float32 state extraction and recomputed PD at every 1ms substep.
        for _ in range(20):
            for i,data in enumerate(self.data):
                q=data.qpos.astype(np.float32)[7:];dq=data.qvel.astype(np.float32)[6:]
                torque=(targets[i]-q)*self.template.stiffness-dq*self.template.damping
                data.ctrl[:]=np.clip(torque,-self.template.torque_limits,self.template.torque_limits)
                mujoco.mj_step(self.model,data)
        rewards=[];falls=[];timeouts=[]
        for i,data in enumerate(self.data):
            ref=start['references'][i];rpy=quatToEuler(data.qpos[3:7]);q=data.qpos[7+BODY_IDS]
            angle=float(wrap(ref[3]-rpy[2]));rate=float(ref[7]-data.qvel[5])
            vel=quat_rotate_inverse(np.array(data.qpos[[4,5,6,3]])[None],np.array(data.qvel[:3])[None])[0]
            height_error=float(data.qpos[2]-ref[0]);q_error=q[:12]-ref[8:20]
            correction_rate=(deltas[i]-self.last_delta[i])/.02
            reward=-4*angle**2-.5*rate**2-.25*(height_error/.1)**2-.1*np.mean((q_error/.35)**2)
            reward-=.1*np.mean((wrap(ref[1:3]-rpy[:2])/.2)**2)+.5*np.sum((vel-ref[4:7])**2)
            reward-=.02*np.mean((deltas[i]/.1)**2)+.001*np.mean(correction_rate**2)
            fall=bool(data.qpos[2]<.35 or max(abs(rpy[0]),abs(rpy[1]))>1.)
            if fall:reward-=20
            self.frames[i]+=1;timeout=self.frames[i]>=self.refs[i].length
            rewards.append(reward);falls.append(fall);timeouts.append(timeout);self.last_delta[i]=deltas[i]
        self.cache=None;terminal=self.observe()
        done=np.logical_or(falls,timeouts)
        terminal_features=terminal['features'].clone()
        if auto_reset and done.any():
            for i in np.flatnonzero(done):self.reset_one(int(i))
            # Observe all environments would append live histories twice. Retain
            # cached current observations for unaffected siblings and observe reset
            # siblings alone through a temporary batch view.
            surviving=terminal;reset_ids=np.flatnonzero(done)
            for i in reset_ids:
                one=self._observe_one_reset(int(i))
                for key in ['features','source','legacy']:surviving[key][i]=one[key][0]
                surviving['physical'][i]=one['physical'][0]
                for key in ['rows','references','wrists']:surviving[key][i]=one[key][0]
            self.cache=surviving
        return self.observe(),torch.tensor(rewards,dtype=torch.float32),torch.from_numpy(done),dict(
            falls=torch.tensor(falls),timeouts=torch.tensor(timeouts),terminal_features=terminal_features,
            trace_rows=start['rows'])

    def _observe_one_reset(self,index):
        # One-environment view shares only this env's runtime/state buffers.
        other=object.__new__(HeadingRecoveryBatch);other.__dict__=self.__dict__.copy();other.num_envs=1
        for key in ['data','runtime','refs','base_history','ref_q_previous']:
            setattr(other,key,[getattr(self,key)[index]])
        for key in ['frames','physical_history','history_count','last_action','last_delta','ref_dq']:
            setattr(other,key,getattr(self,key)[index:index+1])
        other.cache=None;result=other.observe();self.ref_q_previous[index]=other.ref_q_previous[0]
        return result
