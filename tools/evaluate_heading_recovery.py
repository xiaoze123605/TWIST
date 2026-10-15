"""Full original-reference replay and deterministic signed recovery tests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from heading_recovery_core import HeadingRecoveryBatch,RecoveryActorCritic,RecordedMotionReference,TurnHoldReference,CONTRACT,ROOT,POLICY


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,x):path.write_text(json.dumps(x,indent=2)+'\n')


def actor_from_checkpoint(path):
    actor=RecoveryActorCritic().eval()
    if path is not None:
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['contract']==CONTRACT and checkpoint['source_policy_sha256']==sha(POLICY)
        actor.load_state_dict(checkpoint['model_state_dict'])
    return actor


def run_reference(env,actor,reference,yaw_degrees,output,baseline_frames=None):
    env.reset_one(0,reference,yaw_degrees);obs=env.observe();rows=[]
    with torch.no_grad():
        for _ in range(reference.length):
            action=actor.mean(obs['features'],obs['source']).numpy()
            obs,_,_,info=env.step(action,obs['source'].numpy(),auto_reset=False)
            rows.append(info['trace_rows'][0])
    arrays={k:np.array([r[k] for r in rows]) for k in ['qpos','qvel','joint','joint_velocity','rpy','height',
        'root_velocity','clean','processed','final_action','pd_target','correction']}
    for value in arrays.values():assert np.isfinite(value).all()
    error=np.arctan2(np.sin(arrays['rpy'][:,2]-arrays['clean'][:,3]),np.cos(arrays['rpy'][:,2]-arrays['clean'][:,3]))
    fall=(arrays['height']<.35)|(abs(arrays['rpy'][:,:2]).max(-1)>1)
    result=dict(frames=len(rows),seconds=len(rows)*.02,yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(error**2)))),
        joint_rmse=float(np.sqrt(np.mean((arrays['joint']-arrays['clean'][:,8:])**2))),
        root_velocity_rmse=float(np.sqrt(np.mean((arrays['root_velocity']-arrays['clean'][:,4:7])**2))),
        fall_proxy=bool(fall.any()),first_fall_time=float(np.flatnonzero(fall)[0]*.02) if fall.any() else None,
        final2s_yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(error[-100:]**2)))),
        correction_rms=float(np.sqrt(np.mean(arrays['correction']**2))),
        initial_yaw_degrees=yaw_degrees)
    if baseline_frames is not None:
        parity={key:float(abs(arrays[key]-np.array([r[key] for r in baseline_frames])).max())
                for key in ['qpos','qvel','joint','rpy','final_action','pd_target']}
        assert max(parity.values())==0,parity
        result['exact_original_replay']=parity
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    save(output.with_suffix('.json'),result)
    return result


def full77(actor,out,assert_zero=False):
    out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((ROOT/'reports/heading300_sensitivity_20261013/evaluation_retry/summary.json').read_text())
    env=HeadingRecoveryBatch(1);results=[]
    for mode in ['raw','wm']:
        old=next(r for r in summary['rows'] if r['label']=='heading_300' and r['reference_mode']==mode)
        with open(old['trace']) as stream:frames=[json.loads(line) for line in stream]
        assert len(frames)==3207
        result=run_reference(env,actor,RecordedMotionReference(frames),0,out/mode,
                             baseline_frames=frames if assert_zero else None)
        result['mode']=mode;results.append(result)
        print('FULL77',mode,result,flush=True)
    save(out/'results.json',results);(out/'COMPLETE').write_text('Two complete 3207-frame raw/WM trajectories.\n')
    return results


def recovery_task(actor,out):
    out.mkdir(parents=True,exist_ok=False);env=HeadingRecoveryBatch(1);results=[]
    for sign in [-1,1]:
        for yaw in [-10,0,10]:
            result=run_reference(env,actor,TurnHoldReference(sign),yaw,out/f'turn{sign}_initial{yaw}')
            result['target_yaw_degrees']=sign*30;results.append(result)
    save(out/'results.json',results);(out/'COMPLETE').write_text('Six deterministic full 8-second turn-and-hold tests.\n')
    return results


def protection(results,baseline):
    reasons=[]
    for row in results:
        base=next(x for x in baseline if x['mode']==row['mode'])
        if row['frames']!=3207:reasons.append(row['mode']+': incomplete')
        for key in ['yaw_rmse_deg','joint_rmse','root_velocity_rmse']:
            if row[key]>1.05*base[key]:reasons.append(row['mode']+': '+key+' regressed >5%')
        if row['fall_proxy'] and not base['fall_proxy']:reasons.append(row['mode']+': new fall')
    return dict(passed=not reasons,reasons=reasons)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--zero-parity',action='store_true')
    p.add_argument('--task-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    actor=actor_from_checkpoint(a.checkpoint)
    if not a.task_only:full77(actor,a.output/'full77',assert_zero=a.zero_parity)
    recovery_task(actor,a.output/'task')
    (a.output/'COMPLETE').write_text('Recovery teacher evaluation complete.\n')


if __name__=='__main__':main()
