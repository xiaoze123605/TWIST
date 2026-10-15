"""Read-only exact candidate replay and declared task-reference coverage audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from heading_recovery_core import HeadingRecoveryBatch,RecordedMotionReference,TurnHoldReference,ROOT
from evaluate_heading_recovery import actor_from_checkpoint,save


BLOCKS={'physical_state':(0,51),'heading':(51,53),'yaw_rate_error':(53,54),
        'roll_pitch_error':(54,56),'joint_position_error':(56,79),'joint_velocity_error':(79,102),
        'linear_velocity_error':(102,105),'height_error':(105,106),'contact':(106,108),
        'reference_linear_velocity':(108,111),'reference_yaw_rate':(111,112),
        'physical_latent':(112,144),'source_action':(144,167),'history_ready':(167,168)}


def replay(env,actor,reference,yaw,expected):
    env.reset_one(0,reference,yaw);obs=env.observe();features=[];sources=[];deltas=[]
    with torch.no_grad():
        for _ in range(reference.length):
            features.append(obs['features'].numpy()[0].copy());sources.append(obs['source'].numpy()[0].copy())
            delta=actor.delta(obs['features']);deltas.append(delta.numpy()[0].copy())
            action=actor.mean(obs['features'],obs['source']).numpy()
            obs,_,_,info=env.step(action,obs['source'].numpy(),auto_reset=False)
            row=info['trace_rows'][0];idx=len(features)-1
            for key in ['qpos','qvel','final_action','pd_target']:
                assert np.array_equal(np.asarray(row[key]),expected[key][idx]),(idx,key)
    return np.array(features),np.array(sources),np.array(deltas)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=root/'analysis';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    candidate=actor_from_checkpoint(root/'pilot/model_15.pt');zero=actor_from_checkpoint(None)
    env=HeadingRecoveryBatch(1);pool=[]
    for sign in [-1,1]:
        for yaw in [-10,0,10]:
            expected=np.load(root/'zero_baseline/task'/f'turn{sign}_initial{yaw}.npz')
            features,source,delta=replay(env,zero,TurnHoldReference(sign),yaw,expected)
            pool.append(features)
    pool=np.concatenate(pool)
    np.save(out/'independent_task_reference_features.npy',pool)
    results=[]
    summary=json.loads((ROOT/'reports/heading300_sensitivity_20261013/evaluation_retry/summary.json').read_text())
    for mode in ['raw','wm']:
        original=next(r for r in summary['rows'] if r['label']=='heading_300' and r['reference_mode']==mode)
        with open(original['trace']) as stream:frames=[json.loads(line) for line in stream]
        expected=np.load(root/'pilot/evaluation_15/full77'/f'{mode}.npz')
        features,source,delta=replay(env,candidate,RecordedMotionReference(frames),0,expected)
        np.savez_compressed(out/f'{mode}_consumed.npz',features=features,source=source,delta=delta)
        with torch.no_grad():
            x=torch.from_numpy(features.copy());error=torch.atan2(x[:,51],1-x[:,52]);shift=error+np.deg2rad(10)
            changed=x.clone();changed[:,51]=shift.sin();changed[:,52]=1-shift.cos()
            sensitivity=(candidate.delta(changed)-candidate.delta(x)).numpy()
        blocks=[]
        for name,(lo,hi) in BLOCKS.items():
            lower=np.quantile(pool[:,lo:hi],.01,axis=0);upper=np.quantile(pool[:,lo:hi],.99,axis=0)
            outside=((features[:,lo:hi]<lower-1e-6)|(features[:,lo:hi]>upper+1e-6)).any(-1)
            blocks.append(dict(block=name,any_feature_outside_task_p1_p99_fraction=float(outside.mean()),
                reference_block_rms=float(np.sqrt(np.mean(pool[:,lo:hi]**2))),
                full77_block_rms=float(np.sqrt(np.mean(features[:,lo:hi]**2)))))
        results.append(dict(mode=mode,replay_exact=True,frames=len(features),
            correction_rms=float(np.sqrt(np.mean(delta**2))),max_abs_correction=float(abs(delta).max()),
            saturation_near_bound_fraction=float((abs(delta)>.099).any(-1).mean()),
            correction_heading_plus10deg_same_input_change_rms=float(np.sqrt(np.mean(sensitivity**2))),
            blocks=blocks))
    save(out/'results.json',dict(results=results,reference_pool_frames=len(pool),
        reference_pool='six independent zero-correction deterministic task replays, not archived historical PPO minibatches',
        scope='Descriptive per-block task-reference range audit; no proof of full training coverage or causal benefit',
        limitation='Original PPO minibatch features/advantages were not archived; exact historical conditional gradients cannot be reconstructed.',
        optimizer_steps=0))
    (out/'COMPLETE').write_text('Candidate and independent task replays exact; no learning.\n')
    print('READONLY ANALYSIS COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':main()
