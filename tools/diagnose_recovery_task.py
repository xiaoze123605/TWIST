"""Bounded, optimizer-free coherent-reference and recovery-control diagnosis."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/recovery_task_mpl')
import isaacgym  # must precede torch in this checkout
import argparse,copy,hashlib,json,sys
from collections import deque
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'deploy_real'),str(ROOT/'pose')]
from heading_recovery_core import HeadingRecoveryBatch,RecordedMotionReference,BODY_IDS,POLICY,PHYSICAL,XML,wrap
from server_high_level_motion_lib import MotionLib,build_mimic_obs,euler_from_quaternion
from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import quatToEuler,quat_rotate_inverse


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')


def snapshot(env):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data[0],state,spec)
    keys=['frames','base_history','physical_history','history_count','last_action','last_delta',
          'ref_q_previous','ref_dq','cache']
    runtime={k:copy.deepcopy(v) for k,v in vars(env.runtime[0]).items()
             if k not in ['policy','device']}
    return dict(state=state,spec=spec,sensor=env.data[0].sensordata.copy(),
                warmstart=env.data[0].qacc_warmstart.copy(),python={k:copy.deepcopy(getattr(env,k)) for k in keys},
                runtime=runtime,rng=copy.deepcopy(env.rng.bit_generator.state),reference=env.refs[0])


def restore(env,s):
    mujoco.mj_setState(env.model,env.data[0],s['state'],s['spec']);mujoco.mj_forward(env.model,env.data[0])
    env.data[0].sensordata[:]=s['sensor'];env.data[0].qacc_warmstart[:]=s['warmstart']
    for k,v in s['python'].items():setattr(env,k,copy.deepcopy(v))
    for k,v in s['runtime'].items():setattr(env.runtime[0],k,copy.deepcopy(v))
    env.rng.bit_generator.state=copy.deepcopy(s['rng']);env.refs[0]=s['reference']


def reward_terms(env,ref,delta,previous):
    d=env.data[0];rpy=quatToEuler(d.qpos[3:7]);q=d.qpos[7+BODY_IDS]
    vel=quat_rotate_inverse(np.array(d.qpos[[4,5,6,3]])[None],np.array(d.qvel[:3])[None])[0]
    return dict(heading=-4*float(wrap(ref[3]-rpy[2]))**2,
                body_angular_z=-.5*float(ref[7]-d.qvel[5])**2,
                height=-.25*((d.qpos[2]-ref[0])/.1)**2,
                leg_joint=-.1*float(np.mean(((q[:12]-ref[8:20])/.35)**2)),
                roll_pitch=-.1*float(np.mean((wrap(ref[1:3]-rpy[:2])/.2)**2)),
                linear_velocity=-.5*float(np.sum((vel-ref[4:7])**2)),
                correction=-.02*float(np.mean((delta/.1)**2)),
                correction_rate=-.001*float(np.mean(((delta-previous)/.02)**2)),
                fall=-20. if d.qpos[2]<.35 or max(abs(rpy[0]),abs(rpy[1]))>1 else 0.)


def rollout(env,count,path,delta=None,pulse=10,torque=0.):
    rows=[]
    for step in range(count):
        obs=env.observe();source=obs['source'].numpy();offset=np.zeros(23,np.float32)
        if delta is not None and step<pulse:offset[:]=delta
        previous=env.last_delta[0].copy();frame=int(env.frames[0]);ref=obs['references'][0].copy()
        before=env.data[0].qpos.copy();contact=env.contacts(env.data[0]).copy()
        env.data[0].xfrc_applied[:]=0
        if torque and step<pulse:env.data[0].xfrc_applied[env.model.body('pelvis').id,5]=torque
        new,reward,_,info=env.step(source+offset[None],source,auto_reset=False)
        row=info['trace_rows'][0];terms=reward_terms(env,ref,offset,previous)
        assert np.isclose(sum(terms.values()),float(reward[0]),rtol=1e-5,atol=2e-5)
        next_terms=reward_terms(env,env.refs[0].get(frame+1)[0],offset,previous)
        row.update(post_qpos=env.data[0].qpos.copy(),post_qvel=env.data[0].qvel.copy(),
                   native_pre_rpy=quatToEuler(before[3:7]),contact=contact,reward=float(reward[0]),
                   reward_terms=np.array(list(terms.values())),next_reference_reward=sum(next_terms.values()),
                   projected=bool(np.any(row['pd_target']!=row['pd_target_unprojected'])))
        rows.append(row)
    env.data[0].xfrc_applied[:]=0
    keys=['qpos','qvel','rpy','joint','joint_velocity','clean','processed','pd_target','final_action',
          'pd_target_unprojected','correction','post_qpos','post_qvel','native_pre_rpy','contact','reward',
          'reward_terms','next_reference_reward','projected']
    arrays={k:np.array([r[k] for r in rows]) for k in keys}
    assert all(np.isfinite(a).all() for a in arrays.values())
    if path is not None:np.savez_compressed(path,**arrays)
    return arrays


def metrics(x):
    e=wrap(x['native_pre_rpy'][:,2]-x['clean'][:,3]);post=x['post_qpos']
    tilt=np.array([quatToEuler(q[3:7])[:2] for q in post])
    return dict(frames=len(e),yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(e**2)))),
                absolute_yaw_iae_deg_seconds=float(np.degrees(abs(e)).sum()*.02),
                max_absolute_yaw_error_deg=float(np.degrees(abs(e)).max()),
                joint_rmse_rad=float(np.sqrt(np.mean((x['joint']-x['clean'][:,8:])**2))),
                min_height_m=float(post[:,2].min()),max_tilt_rad=float(abs(tilt).max()),
                fall_proxy=bool((post[:,2]<.35).any() or (abs(tilt)>1).any()),
                projected_frames=int(x['projected'].sum()),total_reward=float(x['reward'].sum()),
                reward_terms={k:float(v) for k,v in zip(['heading','body_angular_z','height','leg_joint',
                    'roll_pitch','linear_velocity','correction','correction_rate','fall'],x['reward_terms'].sum(0))},
                next_reference_total_reward=float(x['next_reference_reward'].sum()))


def paired(x,base):
    delta=wrap(x['native_pre_rpy'][:,2]-base['native_pre_rpy'][:,2]);a=abs(np.degrees(delta))
    recovery=None
    # Declared criterion: after the 0.2s pulse, <=2deg for 1s continuously.
    for start in range(10,len(a)-49):
        if (a[start:start+50]<=2).all():recovery=start*.02;break
    return dict(extra_yaw_iae_deg_seconds=float(a.sum()*.02),max_extra_yaw_deg=float(a.max()),
                recovery_time_seconds=recovery,recovery_definition='after pulse, <=2deg relative to same-policy unperturbed trajectory for1s',
                relapse_after_recovery=bool((a[int(round(recovery/.02))+50:]>2).any()) if recovery is not None else None,
                final1s_extra_yaw_rms_deg=float(np.sqrt(np.mean(a[-50:]**2))),
                baseline_absolute_yaw_rmse_deg=metrics(base)['yaw_rmse_deg'],**metrics(x))


def reference_audit(env,frames,out):
    motion=ROOT/'track_dataset/twist_motion_dataset/mocap1/77.pkl';lib=MotionLib(str(motion),'cpu')
    initial_pos,rot,*_=lib.calc_motion_frame(torch.zeros(1,dtype=torch.long),torch.zeros(1))
    offset=float(euler_from_quaternion(rot)[2].item())
    # --steps 1 is a deliberate one-control-step lookahead (20ms).
    generated=[];positions=[];rotations=[];velocities=[];angular=[];dofs=[]
    for t in range(1950,2501):
        ref,pos,quat,q,vel,omega=build_mimic_obs(lib,t,.02,torch.tensor([1]),heading_offset=offset)
        ref=DEFAULT_MIMIC_OBS['g1']+(ref-DEFAULT_MIMIC_OBS['g1'])
        pos=pos.copy()
        pos[:2]=np.array([[np.cos(offset),np.sin(offset)],[-np.sin(offset),np.cos(offset)]])@(pos[:2]-initial_pos[0,:2].numpy())
        generated.append(ref);positions.append(pos);rotations.append(quat);velocities.append(vel);angular.append(omega);dofs.append(q)
    generated=np.array(generated,dtype=np.float32);positions=np.array(positions);rotations=np.array(rotations)
    velocities=np.array(velocities);angular=np.array(angular);dofs=np.array(dofs)
    clean=np.array([f['clean'] for f in frames[1950:2501]],np.float32)
    generated31=generated[:,[i for i in range(33) if i not in [27,32]]]
    ref_error=float(abs(generated31-clean).max());assert ref_error==0,ref_error
    assert np.all(generated[:,[27,32]]==0)
    world=Rotation.from_quat(rotations).apply(velocities)
    fd_world=np.gradient(positions,.02,axis=0)
    relative=Rotation.from_quat(rotations[:-1]).inv()*Rotation.from_quat(rotations[1:])
    fd_local=relative.as_rotvec()/.02
    rpy=np.array([quatToEuler(q[[3,0,1,2]]) for q in rotations])
    yaw_fd=np.gradient(np.unwrap(rpy[:,2]),.02)
    # ZYX Euler yaw derivative uses body q,r and reference roll/pitch.
    yaw_from_omega=(np.sin(rpy[:,0])*angular[:,1]+np.cos(rpy[:,0])*angular[:,2])/np.cos(rpy[:,1])
    data=mujoco.MjData(env.model);foot_z=[];limit=[]
    for i in range(len(generated)):
        data.qpos[:]=env.template.mujoco_default_dof_pos;data.qpos[:3]=positions[i];data.qpos[3:7]=rotations[i][[3,0,1,2]]
        data.qpos[7:]=dofs[i];mujoco.mj_forward(env.model,data)
        mins=[]
        for body in env.feet:
            geoms=np.flatnonzero((env.model.geom_bodyid==body)&(env.model.geom_contype>0))
            mins.append(min(float(data.geom_xpos[g,2]-env.model.geom_size[g,0]) for g in geoms))
        foot_z.append(mins);limit.append((dofs[i]<env.template.target_lower)|(dofs[i]>env.template.target_upper))
    foot_z=np.array(foot_z);limit=np.array(limit);sl=slice(150,550)
    dq=np.gradient(dofs,.02,axis=0);body_ids=BODY_IDS
    damping_load=abs(dq*env.template.damping)
    result=dict(source_sha256=sha(motion),reference_regeneration_max_abs_error=ref_error,
                source_dof=23,simulation_dof=25,wrist_roll_definition='zero inserted by original high-level loader; no source wrist channels',
                reference_lookahead_seconds=.02,trace_segment_frames=[2100,2500],source_segment_times_seconds=[42.02,50.0],
                position_coordinate='initial root XY subtracted and initial yaw removed, same as high-level display quaternion',
                root_velocity_definition='forward position difference at source fps, 19frame box smoothing, body-frame transform',
                body_angular_velocity_definition='world quaternion difference,19frame box smoothing,body-frame transform',
                velocity_world_fd_rmse=float(np.sqrt(np.mean((world[sl]-fd_world[sl])**2))),
                angular_body_fd_rmse=float(np.sqrt(np.mean((angular[150:549]-fd_local[150:549])**2))),
                yaw_fd_vs_body_z_rmse=float(np.sqrt(np.mean((yaw_fd[sl]-angular[sl,2])**2))),
                yaw_fd_vs_euler_converted_rmse=float(np.sqrt(np.mean((yaw_fd[sl]-yaw_from_omega[sl])**2))),
                target_limit_outside_fraction=float(limit[sl].mean()),foot_sphere_min_height_m=foot_z[sl].min(0).tolist(),
                frames_any_foot_penetration_over2cm=int((foot_z[sl].min(1)<-.02).sum()),
                foot_support_heuristic='sphere bottom height <3cm; not ground-truth reference contact',
                maximum_reference_joint_speed_rad_s=float(abs(dq[sl]).max()),
                frames_any_damping_only_load_over_torque_limit=int((damping_load[sl]>env.template.torque_limits).any(1).sum()),
                torque_limitation='damping-only load is not full required torque; no reachability certification',
                reference_smoothing_samples=19,reference_smoothing_seconds=19/float(lib._motion_fps[0]))
    np.savez_compressed(out/'reference_alignment.npz',reference33=generated,root_pos=positions,root_quat=rotations,
                        root_body_vel=velocities,root_body_omega=angular,root_world_fd=fd_world,
                        yaw_fd=yaw_fd,yaw_from_omega=yaw_from_omega,foot_sphere_bottom=foot_z,dof_speed=dq,
                        reference_joint_limits=limit)
    # The reset is at source39.00s; its first commanded reference is39.02s.
    _,init_pos,init_quat,init_q,init_vel,init_omega=build_mimic_obs(lib,1949,.02,torch.tensor([1]),heading_offset=offset)
    init_pos=init_pos.copy();init_pos[:2]=np.array([[np.cos(offset),np.sin(offset)],[-np.sin(offset),np.cos(offset)]])@(init_pos[:2]-initial_pos[0,:2].numpy())
    init_dq=lib.calc_motion_frame(torch.zeros(1,dtype=torch.long),torch.tensor([39.]))[5][0].numpy()
    full_dq=np.zeros(25);full_dq[BODY_IDS]=init_dq
    result['reference_pose_reset_source_time_seconds']=39.
    result['first_commanded_reference_source_time_seconds']=39.02
    save(out/'reference_audit.json',result)
    return result,dict(position=positions,rotation=rotations,velocity=velocities,angular=angular,dofs=dofs,
                       init_position=init_pos,init_rotation=init_quat,init_dofs=init_q,
                       init_velocity=init_vel,init_angular=init_omega,init_dq=full_dq)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(42)
    source=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'
    paths=[POLICY,PHYSICAL,XML,source,Path(__file__).resolve(),ROOT/'tools/heading_recovery_core.py',ROOT/'deploy_real/server_high_level_motion_lib.py']
    manifest=dict(stage='optimizer-free coherent turn diagnosis',seed=42,segment_frames=[2100,2500],
                  fork_frames=[2100,2250,2400],probe_action_amplitude=.03,probe_pd_amplitude_rad=.015,
                  pulse_seconds=.2,probe_observation_seconds=.5,torque_pulse_Nm=1.,
                  directions=['bilateral hip yaw','opposed hip yaw','bilateral hip roll + opposed ankle roll'],
                  probe_origins=['continuous0..42s','reference pose39s plus physical3s pre-roll'],
                  optimizer_steps=0,hashes={str(f.relative_to(ROOT)):sha(f) for f in paths})
    save(out/'manifest.json',manifest)
    summary=json.loads((ROOT/'reports/heading300_sensitivity_20261013/evaluation_retry/summary.json').read_text())
    trace=next(r['trace'] for r in summary['rows'] if r['label']=='heading_300' and r['reference_mode']=='raw')
    frames=[json.loads(line) for line in open(trace)];ref=RecordedMotionReference(frames)
    env=HeadingRecoveryBatch(1);audit,source_data=reference_audit(env,frames,out)
    env.reset_one(0,ref,0);env.observe();states={};parts=[]
    for stop in [1950,2100,2250,2400,2500]:
        now=int(env.frames[0]);part=rollout(env,stop-now,None);parts.append(part)
        if stop in [2100,2250,2400]:states[stop]=snapshot(env)
    continuous={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    old=np.load(ROOT/'reports/heading_recovery_p1_20261013/zero_baseline/full77/raw.npz')
    parity={k:float(abs(continuous[k]-old[k][:2500]).max()) for k in ['qpos','qvel','rpy','joint','final_action','pd_target']}
    assert max(parity.values())==0,parity
    np.savez_compressed(out/'continuous_0_to50s.npz',**continuous)
    print('CONTINUOUS_PARITY',parity,flush=True)
    # Full restore proof against the independently generated continuous trace.
    for frame,s in states.items():
        restore(env,s);x=rollout(env,25,None)
        for k in ['qpos','qvel','rpy','final_action','pd_target']:
            assert np.array_equal(x[k],continuous[k][frame:frame+25]),(frame,k,float(abs(x[k]-continuous[k][frame:frame+25]).max()))
    save(out/'parity.json',dict(continuous_original_exact=parity,all_three_full_state_forks_exact=True))
    inits={'continuous':metrics({k:v[2100:2500] for k,v in continuous.items()})}
    # Default pose counterexample; reference pose init39s then real 3s pre-roll.
    local_states={};local_arrays=None
    for name,start in [('default_at42',2100),('reference_at39_preroll',1950)]:
        env.reset_one(0,ref,0);env.frames[0]=start;env.cache=None
        if start==1950:
            d=env.data[0];d.qpos[:3]=source_data['init_position'];d.qpos[3:7]=source_data['init_rotation'][[3,0,1,2]]
            d.qpos[7:]=source_data['init_dofs']
            q=Rotation.from_quat(source_data['init_rotation']);d.qvel[:3]=q.apply(source_data['init_velocity']);d.qvel[3:6]=source_data['init_angular']
            d.qvel[6:]=source_data['init_dq']
            mujoco.mj_forward(env.model,d);env.cache=None
        env.observe()
        if start==1950:
            chunks=[]
            for stop in [2100,2250,2400,2500]:
                chunks.append(rollout(env,stop-int(env.frames[0]),None))
                if stop in states:local_states[stop]=snapshot(env)
            x={k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]}
            np.savez_compressed(out/(name+'.npz'),**x);local_arrays=x
        else:x=rollout(env,2500-start,out/(name+'.npz'))
        inits[name]=metrics({k:v[2100-start:] for k,v in x.items()})
        inits[name]['preroll_metrics']=metrics({k:v[:2100-start] for k,v in x.items()}) if start<2100 else None
    # Initial yaw offsets from motion beginning; full physical pre-roll retained.
    for yaw in [-10,10]:
        env.reset_one(0,ref,yaw);env.observe();x=rollout(env,2500,out/f'initial_yaw{yaw}.npz')
        inits[f'initial_yaw{yaw}']=paired({k:v[2100:] for k,v in x.items()}, {k:v[2100:] for k,v in continuous.items()})
        inits[f'initial_yaw{yaw}']['recovery_time_seconds']=None
        inits[f'initial_yaw{yaw}']['recovery_definition']='Disturbance at0s; comparison observed42..50s only; no recovery-time estimate from this censored window.'
    save(out/'initializations.json',inits);print('INITIALIZATIONS',inits,flush=True)
    disturbances=[]
    for origin,pool,arrays,start in [('continuous',states,continuous,0),('reference_at39',local_states,local_arrays,1950)]:
        for sign in [-1,1]:
            restore(env,pool[2100]);x=rollout(env,400,out/f'{origin}_torque{sign}.npz',torque=sign)
            disturbances.append(dict(origin=origin,torque_Nm=sign,**paired(x,{k:v[2100-start:] for k,v in arrays.items()})))
    save(out/'disturbances.json',disturbances)
    directions=[]
    v=np.zeros(23,np.float32);v[[2,8]]=1;directions.append(('hip_yaw_same',v))
    v=np.zeros(23,np.float32);v[2]=1;v[8]=-1;directions.append(('hip_yaw_opposed',v))
    v=np.zeros(23,np.float32);v[[1,7]]=1;v[[5,11]]=-1;directions.append(('hip_ankle_roll',v))
    probes=[]
    probe_states=[(origin,frame,state,arrays,start) for origin,pool,arrays,start in
                  [('continuous',states,continuous,0),('reference_at39',local_states,local_arrays,1950)] for frame,state in pool.items()]
    for origin,frame,state,arrays,start in probe_states:
        base={k:v[frame-start:frame-start+25] for k,v in arrays.items()};bm=metrics(base)
        restore(env,state);null=rollout(env,25,None)
        for key in ['qpos','qvel','rpy','final_action','pd_target']:assert np.array_equal(null[key],base[key]),(origin,frame,key)
        for name,direction in directions:
            for sign in [-1,1]:
                restore(env,state);x=rollout(env,25,out/f'probe_{origin}_{frame}_{name}_{sign}.npz',delta=direction*.03*sign)
                row=dict(origin=origin,frame=frame,direction=name,sign=sign,**paired(x,base),
                         absolute_yaw_iae_change=metrics(x)['absolute_yaw_iae_deg_seconds']-bm['absolute_yaw_iae_deg_seconds'],
                         total_reward_change=metrics(x)['total_reward']-bm['total_reward'],
                         next_reference_reward_change=metrics(x)['next_reference_total_reward']-bm['next_reference_total_reward'],
                         max_pd_target_change_rad=float(abs(x['pd_target']-base['pd_target']).max()),
                         pulse_mean_body_angular_z_change=float(np.mean(x['post_qvel'][:10,5]-base['post_qvel'][:10,5])),
                         start_contacts=base['contact'][0].tolist())
                probes.append(row)
    save(out/'probes.json',probes)
    for path,digest in manifest['hashes'].items():assert sha(ROOT/path)==digest,path
    result=dict(reference=audit,initializations=inits,disturbances=disturbances,
                probes=probes,optimizer_steps=0,source_hashes_unchanged=True,
                limits=['Single motion and seed; no policy promotion.', 'Probe directions are finite examples, not a control-authority proof.',
                        'Reference pose initialization has empty policy history before real3s pre-roll.',
                        'Contact and torque feasibility are not certified by reference FK.'])
    save(out/'results.json',result);(out/'COMPLETE').write_text('Reference audit, exact full-state forks, initialization and bounded action/reward probes complete; zero training.\n')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
