"""Independent saved-trace checks, execution mapping and figures for diagnosis."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/recovery_task_mpl')
import isaacgym
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_recovery_task import metrics,sha
from heading_recovery_core import XML,BODY_IDS,wrap


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    manifest=json.loads((r/'manifest.json').read_text());result=json.loads((r/'results.json').read_text())
    repo=Path(__file__).resolve().parents[1]
    for name,digest in manifest['hashes'].items():assert sha(repo/name)==digest,name
    ref=np.load(r/'reference_alignment.npz');model=mujoco.MjModel.from_xml_path(str(XML))
    joint_names=[model.joint(i).name for i in range(1,model.njnt)]
    sl=slice(150,550);references=ref['reference33'][sl];limits=ref['reference_joint_limits'][sl]
    per_joint={joint_names[i]:dict(outside_target_frames=int(limits[:,i].sum())) for i in range(25) if limits[:,i].any()}
    q=references[:,8:];hard=model.jnt_range[1:]
    hard_out=(q<hard[:,0])|(q>hard[:,1])
    for i in range(25):
        if hard_out[:,i].any():per_joint.setdefault(joint_names[i],{})['outside_hard_joint_frames']=int(hard_out[:,i].sum())
    base=np.load(r/'continuous_0_to50s.npz');local=np.load(r/'reference_at39_preroll.npz')
    assert len(base['qpos'])==2500 and len(local['qpos'])==550
    all_metrics_recomputed=True
    for label,data,start in [('continuous',base,2100),('reference_at39_preroll',local,150)]:
        actual=metrics({k:data[k][start:] for k in data.files});record=result['initializations'][label]
        for key in ['yaw_rmse_deg','absolute_yaw_iae_deg_seconds','joint_rmse_rad','total_reward','fall_proxy']:
            assert actual[key]==record[key],(label,key)
    mappings=[]
    vectors={'hip_yaw_same':{2:1,8:1},'hip_yaw_opposed':{2:1,8:-1},'hip_ankle_roll':{1:1,7:1,5:-1,11:-1}}
    for row in result['probes']:
        origin=row['origin'];frame=row['frame'];name=row['direction'];sign=row['sign']
        x=np.load(r/f'probe_{origin}_{frame}_{name}_{sign}.npz')
        b=base if origin=='continuous' else local;index=frame if origin=='continuous' else frame-1950
        direct=x['pd_target'][0]-b['pd_target'][index]
        expected=np.zeros(25)
        for motor,weight in vectors[name].items():expected[BODY_IDS[motor]]=.015*sign*weight
        mismatch=direct-expected
        mappings.append(dict(origin=origin,frame=frame,direction=name,sign=sign,
                             max_first_step_pd_change_rad=float(abs(direct).max()),
                             first_step_projection_or_rounding_difference_rad=float(abs(mismatch).max()),
                             first_step_source_contacts=x['contact'][0].tolist(),
                             pulse_correction_action_rms=float(np.sqrt(np.mean(x['correction'][:10]**2)))))
        recomputed=metrics(x)
        for key in ['yaw_rmse_deg','absolute_yaw_iae_deg_seconds','total_reward','fall_proxy']:assert recomputed[key]==row[key]
    (r/'probe_execution_mapping.json').write_text(json.dumps(mappings,indent=2)+'\n')
    with (r/'probe_summary.csv').open('w') as f:
        keys=['origin','frame','direction','sign','absolute_yaw_iae_change','total_reward_change',
              'max_extra_yaw_deg','max_pd_target_change_rad','fall_proxy']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:x[k] for k in keys} for x in result['probes'])
    plt.rcParams.update({'font.size':9})
    t=42+np.arange(400)*.02;fig,ax=plt.subplots(6,1,figsize=(12,13),sharex=True)
    ax[0].plot(t,np.degrees(np.unwrap(references[:,3])),label='reference yaw (source time +20ms)')
    for label,data,start in [('continuous0..42',base,2100),('reference39 +3s physical pre-roll',local,150)]:
        ax[0].plot(t,np.degrees(np.unwrap(data['native_pre_rpy'][start:,2])),label=label)
    ax[0].set_ylabel('yaw (deg)');ax[0].legend()
    ax[1].plot(t,ref['root_body_omega'][sl,2],label='reference body omega z')
    ax[1].plot(t,ref['yaw_fd'][sl],label='Euler yaw finite difference')
    ax[1].plot(t,ref['yaw_from_omega'][sl],label='body omega converted to Euler yaw rate')
    ax[1].set_ylabel('angular rate (rad/s)');ax[1].legend()
    for j,label in enumerate(['x','y','z']):ax[2].plot(t,ref['root_body_vel'][sl,j],label='reference body v'+label)
    ax[2].set_ylabel('velocity (m/s)');ax[2].legend()
    ax[3].plot(t,references[:,0],label='reference pelvis height')
    ax[3].plot(t,local['qpos'][150:,2],label='local-init actual pelvis height')
    for j in range(2):ax[3].plot(t,ref['foot_sphere_bottom'][sl,j],label=f'reference FK foot{j} sphere bottom')
    ax[3].axhline(0,color='black',linewidth=.5);ax[3].set_ylabel('height (m)');ax[3].legend()
    for j,label in [(2,'left hip yaw'),(8,'right hip yaw'),(3,'left knee'),(9,'right knee')]:ax[4].plot(t,q[:,BODY_IDS[j]],label=label)
    ax[4].set_ylabel('reference q (rad)');ax[4].legend()
    for j in range(2):
        ax[5].step(t,local['contact'][150:,j]+j*1.5,label=f'actual contact foot{j}')
        ax[5].step(t,(ref['foot_sphere_bottom'][sl,j]<.03).astype(int)+j*1.5,linestyle='--',alpha=.6,label=f'FK height heuristic foot{j}')
    ax[5].set_ylabel('contact / heuristic');ax[5].set_xlabel('control time (s)');ax[5].legend()
    for aa in ax:aa.grid(alpha=.25)
    fig.tight_layout();fig.savefig(r/'reference_alignment.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for i,(origin,b,start) in enumerate([('continuous',base,2100),('reference_at39',local,150)]):
        refyaw=b['clean'][start:,3];native=b['native_pre_rpy'][start:,2]
        ax[i,0].plot(np.arange(400)*.02,np.degrees(wrap(native-refyaw)),label='no disturbance')
        for sign in [-1,1]:
            d=np.load(r/f'{origin}_torque{sign}.npz');extra=np.degrees(wrap(d['native_pre_rpy'][:,2]-native))
            ax[i,0].plot(np.arange(400)*.02,np.degrees(wrap(d['native_pre_rpy'][:,2]-refyaw)),label=f'{sign}Nm for0.2s')
            ax[i,1].plot(np.arange(400)*.02,extra,label=f'{sign}Nm for0.2s')
        ax[i,0].set_ylabel(origin+' absolute error (deg)');ax[i,1].set_ylabel('paired extra yaw (deg)')
        ax[i,1].axhline(2,color='grey',linestyle='--');ax[i,1].axhline(-2,color='grey',linestyle='--')
        for j in range(2):ax[i,j].legend();ax[i,j].grid(alpha=.25);ax[i,j].set_xlabel('time since pulse (s)')
    fig.tight_layout();fig.savefig(r/'disturbance_recovery.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for origin,marker in [('continuous','o'),('reference_at39','x')]:
        rows=[x for x in result['probes'] if x['origin']==origin]
        ax.scatter([-x['absolute_yaw_iae_change'] for x in rows],[x['total_reward_change'] for x in rows],marker=marker,label=origin)
    ax.axhline(0,color='grey');ax.axvline(0,color='grey');ax.grid(alpha=.25);ax.legend()
    ax.set_xlabel('0.5s yaw IAE improvement (deg*s)');ax.set_ylabel('0.5s total reward gain')
    fig.tight_layout();fig.savefig(r/'reward_sensitivity.png',dpi=160);plt.close(fig)
    receipt=dict(source_hashes_unchanged=True,saved_trace_metrics_recomputed=True,probe_count=len(mappings),
                 reference_limit_details=per_joint,reference_hard_limit_outside_fraction=float(hard_out.mean()),
                 first_step_all_probes_mapping_max_error_rad=max(x['first_step_projection_or_rounding_difference_rad'] for x in mappings),
                 all_probe_forks_no_fall=not any(x['fall_proxy'] for x in result['probes']),
                 reward_sign_agreement_count=sum((x['absolute_yaw_iae_change']<0)==(x['total_reward_change']>0) for x in result['probes']),
                 optimizer_steps=0,report_code_sha256=sha(Path(__file__).resolve()))
    (r/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
