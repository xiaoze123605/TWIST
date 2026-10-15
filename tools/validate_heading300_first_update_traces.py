"""Validate consumed histories/JITs/logging parity and save actual WM input windows."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_world_model.runtime import resolve_checkpoint_path


def validate(root):
    summary=json.loads((root/'evaluation/summary.json').read_text())
    prior=json.loads((ROOT/'reports/heading300_closed_loop_20261012/anchor_x5_eval77/summary.json').read_text())
    checkpoint=resolve_checkpoint_path(None);wm=torch.load(checkpoint,map_location='cpu')
    mean=torch.tensor(wm['normalization_mean'],dtype=torch.float32);std=torch.tensor(wm['normalization_std'],dtype=torch.float32)
    checks=[];jit_checks=[];contacts=[];frames_by_mode={}
    for row in summary['rows']:
        frames=[json.loads(x) for x in Path(row['trace']).read_text().splitlines()]
        obs=np.load(frames[0]['actor_observation_file'],mmap_mode='r');history=obs[:,1155:7001].reshape(-1,79,74)
        state_error=max(float(abs(history[lag:,-lag,:51]-obs[:-lag,31:82]).max()) for lag in range(1,80))
        twist_error=max(float(abs(obs[lag:,105:1155].reshape(-1,10,105)[:,-lag]-obs[:-lag,:105]).max()) for lag in range(1,11))
        actions=np.array([x['final_action'] for x in frames])
        action_error=max(float(abs(history[lag:,-lag,51:]-actions[:-lag]).max()) for lag in range(1,80))
        assert state_error==twist_error==action_error==0
        if row['label']=='heading_300':
            old=next(x for x in prior['rows'] if x['role']=='baseline' and x['reference_mode']==row['reference_mode'])
            oldframes=[json.loads(x) for x in Path(old['trace']).read_text().splitlines()]
            for key in ['qpos','qvel','joint','rpy','final_action','pd_target','processed','clean']:
                assert np.array_equal(np.array([f[key] for f in frames]),np.array([f[key] for f in oldframes]))
        if row['reference_mode']=='wm':
            windows=np.array([x['reference_wm_history'] for x in frames[24:]],dtype=np.float32)
            assert windows.shape==(3183,25,31)
            np.savez_compressed(root/'analysis'/f'{row["label"]}_wm_inputs.npz',frame_ids=np.arange(24,3207),
                unnormalized_history=windows,normalized_history=((torch.from_numpy(windows)-mean)/std).numpy(),mean=mean.numpy(),std=std.numpy())
        path=ROOT/'reports/recovery_1250/fix_stage_20261008/heading_300.pt' if row['label']=='heading_300' else root/'evaluation/exports'/row['label']/'policy.pt'
        jit=torch.jit.load(str(path)).eval();comp=np.load(root/'analysis'/f'{row["label"]}_{row["reference_mode"]}_components.npz');error=0.
        with torch.no_grad():
            for i in range(0,len(obs),256):
                error=max(error,float(abs(jit(torch.from_numpy(np.array(obs[i:i+256],copy=True))).numpy()-comp['full'][i:i+256]).max()))
        assert error<1e-6
        jit_checks.append(dict(label=row['label'],mode=row['reference_mode'],actual_frame_count=len(obs),jit_eager_same_batch_max_error=error))
        checks.append(dict(label=row['label'],mode=row['reference_mode'],all79_state_history_alignment_error=state_error,
            all79_action_history_alignment_error=action_error,all10_twist_history_alignment_error=twist_error))
        if row['label'] in ['model25','model26']:frames_by_mode.setdefault(row['reference_mode'],{})[row['label']]=frames
    (root/'analysis/actual_jit_consistency.json').write_text(json.dumps(jit_checks,indent=2))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (root/'analysis/timing_and_instrumentation.json').write_text(json.dumps(dict(checks=checks,
        logging_enabled_vs_prior_baseline_trajectory_exact=True,wm_checkpoint=str(checkpoint),wm_checkpoint_sha256=sha(checkpoint),
        wm_runtime_sha256=sha(ROOT/'motion_world_model/runtime.py'),wm_ready_first_frame=24,wm_history_length=25,
        normalized_inputs_saved=True,wm_hidden_state='GRU recomputed from each complete window; no carried hidden state'),indent=2))
    for mode,records in frames_by_mode.items():
        old,new=records['model25'],records['model26'];t=np.arange(3207)*.02
        force0=np.array([f['dynamics']['foot_vertical_force'] for f in old]);force1=np.array([f['dynamics']['foot_vertical_force'] for f in new])
        first=int(np.flatnonzero(np.max(abs(force1-force0),-1)>50)[0])
        contacts.append(dict(mode=mode,event='first_foot_force_gap_gt50N',frame=first,time=first*.02,old_force=force0[first].tolist(),new_force=force1[first].tolist()))
        center=8.62 if mode=='wm' else 7.86
        fig,ax=plt.subplots(5,1,figsize=(14,12),sharex=True)
        for foot,label in [(0,'left'),(1,'right')]:
            ax[0].plot(t,force0[:,foot],label=label+'25');ax[0].plot(t,force1[:,foot],label=label+'26',ls='--')
        ax[0].set_ylabel('Vertical force (N)');ax[0].legend()
        for a,j,label in zip(ax[1:],[8,10,15,19],['right hip yaw','right ankle pitch','left shoulder pitch','right shoulder pitch']):
            for name,frames in records.items():
                a.plot(t,[f['joint'][j] for f in frames],label='actual'+name)
                a.plot(t,[f['pd_target'][old[0]['diagnostics_schema']['policy_joint_indices'][j]] for f in frames],label='PD'+name,alpha=.55)
            a.set_ylabel(label+' (rad)');a.legend()
        for a in ax:a.grid(alpha=.2);a.set_xlim(center-.3,center+.3)
        ax[-1].set_xlabel('Aligned reference time (s)');fig.tight_layout();fig.savefig(root/'analysis/plots'/f'{mode}_contact_and_targets.png',dpi=140);plt.close(fig)
        for frame in [0,1,393 if mode=='raw' else 431,1036 if mode=='raw' else 1175]:
            x,y=old[frame],new[frame];idx=x.get('diagnostics_schema',old[0]['diagnostics_schema'])['policy_joint_indices']
            contacts.append(dict(mode=mode,event='key_frame',frame=frame,time=frame*.02,
                old_force=x['dynamics']['foot_vertical_force'],new_force=y['dynamics']['foot_vertical_force'],
                old_contact_points=len(x['dynamics']['contact_points']),new_contact_points=len(y['dynamics']['contact_points']),
                joint_max_gap=float(abs(np.array(x['joint'])-y['joint']).max()),
                feet_position_max_gap=float(abs(np.array(x['dynamics']['actual']['foot_positions'])-y['dynamics']['actual']['foot_positions']).max()),
                old_projection_max=max(x['projection_delta']),new_projection_max=max(y['projection_delta']),
                executed_pd_target_delta_policy_joints=(np.array(y['pd_target'])-x['pd_target'])[idx].tolist()))
    (root/'analysis/contact_key_states.json').write_text(json.dumps(contacts,indent=2))
    print('Verified six actual JITs, full histories, prior logging parity and reference WM windows')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    validate(p.parse_args().root)
