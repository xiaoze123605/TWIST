"""Shared metrics from heading300 diagnosis; preserves phase and Welch definitions."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d,binary_dilation
DT=.02

def segments(mask):
 edges=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
 return list(zip(edges[::2],edges[1::2]))

def spectrum(signal,mask):
 ps=[];weights=[]
 for start,end in segments(mask):
  if end-start<64:continue
  f,p=welch(signal[start:end],fs=50,nperseg=64,noverlap=32,axis=0,detrend='linear')
  ps.append(p);weights.append(end-start)
 if not ps:return dict(status='No contiguous segment>=1.28s'),None
 p=np.average(np.array(ps),axis=0,weights=weights);band=(f>=5)&(f<=20);total=np.trapz(p,f,axis=0);high=np.trapz(p[band],f[band],axis=0)
 avg=p.mean(-1);idx=np.flatnonzero(band)[np.argmax(avg[band])]
 return dict(high_5_20hz_rms=float(np.sqrt(high.mean())),high_band_fraction=float(high.sum()/max(total.sum(),1e-20)),peak_high_frequency=float(f[idx]),per_joint_high_rms=np.sqrt(high).tolist(),valid_seconds=sum(weights)*DT), (f,p)

def rms(x):return float(np.sqrt(np.mean(x*x))) if np.asarray(x).size else None

def phase_masks(ref):
    height_rate=np.gradient(uniform_filter1d(ref[:,0],size=9,mode="nearest"))/DT
    knee=ref[:,[11,17]].mean(-1)
    return dict(squat=(ref[:,0]<.7)&(knee>.8),descent=(ref[:,0]<.78)&(knee>.5)&(height_rate<-.03),rise=(ref[:,0]<.78)&(knee>.5)&(height_rate>.03),walking=(ref[:,0]>.74)&(np.linalg.norm(ref[:,4:6],axis=1)>.15),turning=(abs(ref[:,7])>.5))

def analyze_trace(result,label='policy'):
    spectra={}
    frames=[json.loads(x) for x in Path(result['trace']).read_text().splitlines()]
    if len(frames)!=result['frames'] or len(frames)<2:raise ValueError('incomplete diagnostic trace')
    if [x['frame_id'] for x in frames]!=list(range(len(frames))):raise ValueError('non-contiguous reference frames')
    if any(x['dynamics']['torque_substeps']!=20 for x in frames):raise ValueError('incomplete substep torque measurements')
    ref=np.array([x['clean'] for x in frames]);proc=np.array([x['processed'] for x in frames]);q=np.array([x['joint'] for x in frames]);target=np.array([x['pd_target'] for x in frames]);idx=np.array(frames[0]['diagnostics_schema']['policy_joint_indices']);target=target[:,idx];proj=np.array([x['projection_delta'] for x in frames])[:,idx];diag=[x['dynamics'] for x in frames];raw=np.array([x['policy_action_pre_ema'] for x in diag]);act=np.array([x['final_action'] for x in frames]);torque_sat=np.array([x['torque_saturation_fraction'] for x in diag])[:,idx];actual_pitch=np.array([x['actual']['torso_pitch'] for x in diag]);ref_pitch=np.array([x['clean_reference']['torso_pitch'] for x in diag]);proc_pitch=np.array([x['processed_reference']['torso_pitch'] for x in diag]);target_pitch=np.array([x['target_posture']['torso_pitch'] for x in diag]);contact=np.array([x['foot_vertical_force'] for x in diag]);com=np.array([x['actual']['com_forward_from_feet'] for x in diag]);ref_com=np.array([x['clean_reference']['com_forward_from_feet'] for x in diag])
    velocity_array=np.array([f['joint_velocity'] for f in frames])
    for key in ['joint_rmse','root_velocity_rmse','yaw_error']:
        if not np.isfinite(result[key]):raise ValueError('nonfinite summary metric: '+key)
    for array in (ref,proc,q,target,proj,raw,act,torque_sat,actual_pitch,ref_pitch,com,ref_com,contact,velocity_array):
        if not np.isfinite(array).all():raise ValueError("nonfinite diagnostics")
    masks=phase_masks(ref)
    fall=np.array([f['height']<.35 or abs(f['rpy'][0])>1 or abs(f['rpy'][1])>1 for f in frames]);first_fall=int(np.flatnonzero(fall)[0]) if fall.any() else len(frames);pre_fall=np.arange(len(frames))<first_fall
    error=q-ref[:,8:];names=[frames[0]['diagnostics_schema']['joint_names'][i] for i in idx];stats={}
    for phase,mask in masks.items():
     if not mask.any():continue
     pitch_error=actual_pitch-ref_pitch;overshoot=mask&(pitch_error>np.deg2rad(10));tops=np.sqrt(np.mean(error[mask]**2,axis=0));topids=np.argsort(tops)[::-1][:8];support=mask&(contact.min(-1)>5);margin=[diag[i]['contact_hull_margin'] for i in np.flatnonzero(support) if diag[i]['contact_hull_margin'] is not None]
     phase_stats=dict(post_fall_frame_fraction=float((~pre_fall[mask]).mean()),seconds=float(mask.sum()*DT),intervals=[[round(a*DT,2),round(b*DT,2)] for a,b in segments(mask)],joint_rmse=rms(error[mask]),pitch_rmse_degrees=float(np.degrees(rms(pitch_error[mask]))),actual_pitch_mean_degrees=float(np.degrees(actual_pitch[mask].mean())),reference_pitch_mean_degrees=float(np.degrees(ref_pitch[mask].mean())),processed_pitch_mean_degrees=float(np.degrees(proc_pitch[mask].mean())),pitch_error_p95_degrees=float(np.degrees(np.percentile(pitch_error[mask],95))),forward_overshoot_seconds=float(overshoot.sum()*DT),actual_com_forward_cm=float(com[mask].mean()*100),reference_com_forward_cm=float(ref_com[mask].mean()*100),double_contact_fraction=float((contact[mask].min(-1)>5).mean()),static_contact_margin_mean_cm=float(np.mean(margin)*100) if margin else None,negative_static_margin_fraction=float(np.mean(np.array(margin)<0)) if margin else None,projection_frame_fraction=float((proj[mask].max(-1)>1e-7).mean()),torque_saturation_fraction=float(torque_sat[mask].mean()),joint_error_top=[dict(joint=names[i],rmse_degrees=float(np.degrees(tops[i])),bias_degrees=float(np.degrees(error[mask,i].mean())),target_minus_processed_degrees=float(np.degrees((target-proc[:,8:])[mask,i].mean())),projection_fraction=float((proj[mask,i]>1e-7).mean()),torque_saturation_fraction=float(torque_sat[mask,i].mean())) for i in topids])
     if overshoot.any():phase_stats['forward_overshoot_joint_bias_degrees']={names[i]:float(np.degrees(error[overshoot,i].mean())) for i in range(15)}
     for name,signal in [('reference',ref[:,8:20]),('processed_reference',proc[:,8:20]),('actual',q[:,:12]),('tracking_error',error[:,:12]),('policy_pre_ema',raw[:,:12]),('policy_post_ema',act[:,:12]),('pd_target',target[:,:12])]:
      phase_stats[name+'_spectrum'],spec=spectrum(signal,mask&pre_fall)
      if phase=='walking' and spec is not None:spectra[label,name]=spec
     diffmask=mask[1:]&mask[:-1]
     phase_stats['policy_action_rate_rms']=rms(np.diff(act,axis=0)[diffmask,:12]/DT)
     phase_stats['pd_target_rate_rms']=rms(np.diff(target,axis=0)[diffmask,:12]/DT)
     phase_stats['actual_joint_acceleration_rms']=rms(np.diff(np.array([x['joint_velocity'] for x in frames]),axis=0)[diffmask,:12]/DT)
     if diag[0]['heading_delta'] is not None:
      hd=np.array([x['heading_delta'] for x in diag]);phase_stats['heading_delta_leg_rms']=rms(hd[mask,:12]);phase_stats['heading_delta_rate_rms']=rms(np.diff(hd,axis=0)[diffmask,:12]/DT);phase_stats['heading_delta_spectrum'],_=spectrum(hd[:,:12],mask&pre_fall)
     phase_stats.update(hip_knee_ankle_rmse=rms(error[mask,:12]),hip_rmse=rms(error[mask][:,[0,1,2,6,7,8]]),knee_rmse=rms(error[mask][:,[3,9]]),ankle_rmse=rms(error[mask][:,[4,5,10,11]]),height_rmse=rms(np.array([f['height'] for f in frames])[mask]-ref[mask,0]))
     actual_spectrum=phase_stats['actual_spectrum']
     if 'per_joint_high_rms' in actual_spectrum:
      per=np.array(actual_spectrum['per_joint_high_rms']);phase_stats['ankle_pitch_high_5_20hz_rms']=float(np.sqrt(np.mean(per[[4,10]]**2)))
     else:phase_stats['ankle_pitch_high_5_20hz_rms']=None
     stats[phase]=phase_stats
    summary=dict(first_fall_time=first_fall*DT if first_fall<len(frames) else None,metrics={k:result[k] for k in ['joint_rmse','root_velocity_rmse','yaw_error','fall_proxy']},phases=stats)
    reference_velocity=np.gradient(ref[:,8:],DT,axis=0)
    summary['metrics']['joint_velocity_rmse_estimated']=rms(np.array([f['joint_velocity'] for f in frames])-reference_velocity)
    summary['metrics']['torso_pitch_rmse_degrees']=float(np.degrees(rms(actual_pitch-ref_pitch)))
    summary['metrics']['projection_frame_fraction']=float((proj.max(-1)>1e-7).mean())
    summary['metrics']['torque_saturation_fraction']=float(torque_sat.mean())
    contact_state=contact>5
    transitions=np.r_[False,np.any(contact_state[1:]!=contact_state[:-1],axis=1)]
    near=binary_dilation(transitions,iterations=2)&masks['walking']&pre_fall
    far=masks['walking']&pre_fall&~binary_dilation(transitions,iterations=2)
    rate=np.diff(act,axis=0,prepend=act[:1])/DT
    summary['contact_transition']=dict(window_seconds=.04,near_seconds=float(near.sum()*DT),far_seconds=float(far.sum()*DT),near_action_rate_rms=rms(rate[near,:12]),far_action_rate_rms=rms(rate[far,:12]))
    summary['reference_digest']=hashlib.sha256(ref.astype('<f8').tobytes()).hexdigest()
    summary['processed_reference_digest']=hashlib.sha256(proc.astype('<f8').tobytes()).hexdigest()
    summary['frames']=len(frames)
    cache=(ref,q,actual_pitch,ref_pitch,proc_pitch,target_pitch,com,ref_com,masks,proj,torque_sat,act)
    return summary,cache,spectra
