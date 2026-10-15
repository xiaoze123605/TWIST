"""P0 forward-dynamics pilot on immutable MuJoCo transition traces.

Only the independent physical WM learns. No PPO actor/critic is instantiated.
Motion-level train/validation/test assignment is frozen before learning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rsl_rl'))
from rsl_rl.modules.physical_world_model import PhysicalWorldModel,fit_normalizers,CONTRACT

TRAIN={'mocap1__77','accad__General_A3___Swing_Arms_While_Stand','mocap1__1','mocap1__20','mydata2__3_seg00'}
VAL={'mocap1__59'}
TEST={'mydata2__6_seg00','mydata3__42_seg00','mydata3__47_seg01'}
assert not (TRAIN&VAL or TRAIN&TEST or VAL&TEST)
POLICY=ROOT/'reports/recovery_1250/fix_stage_20261008/heading_300.pt'
CHECKPOINT=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def trajectories():
    regression=json.loads((ROOT/'reports/heading300_sampling_20261009/regression_step50/summary.json').read_text())
    assert regression['status']=='complete'
    result=[dict(case=r['case'],mode=r['reference_mode'],path=r['trace']) for r in regression['rows'] if r['label']=='heading_300']
    full=json.loads((ROOT/'reports/heading300_sensitivity_20261013/evaluation_retry/summary.json').read_text())
    assert full['status']=='complete'
    result += [dict(case='mocap1__77',mode=r['reference_mode'],path=r['trace']) for r in full['rows'] if r['label']=='heading_300']
    assert len(result)==18
    for r in result:r['split']='train' if r['case'] in TRAIN else 'validation' if r['case'] in VAL else 'test'
    assert all(r['case'] in TRAIN|VAL|TEST for r in result)
    return result


def load_trace(row):
    with open(row['path']) as stream:frames=[json.loads(line) for line in stream]
    assert [f['frame_id'] for f in frames]==list(range(len(frames)))
    schema=frames[0]['diagnostics_schema'];joint_ids=np.array(schema['policy_joint_indices'])
    assert len(joint_ids)==23
    names=[schema['joint_names'][i] for i in joint_ids]
    assert names[4]=='left_ankle_pitch_joint' and names[10]=='right_ankle_pitch_joint'
    qpos=np.array([f['qpos'] for f in frames]);qvel=np.array([f['qvel'] for f in frames])
    w,x,y,z=qpos[:,3:7].T
    rp=np.stack((np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1))),-1)
    # MuJoCo free-joint angular qvel is already in the local body frame.
    states=np.concatenate((qvel[:,3:6],rp,qpos[:,7+joint_ids],qvel[:,6+joint_ids]),-1).astype(np.float32)
    targets=np.array([f['pd_target'] for f in frames],dtype=np.float32)[:,joint_ids]
    commands=np.array([f['final_action'] for f in frames],dtype=np.float32)
    assert np.isfinite(states).all() and np.isfinite(targets).all()
    # Preserve the unfiltered command separately; target may differ by projection.
    nominal=np.array([-.2,0,0,.4,-.2,0,-.2,0,0,.4,-.2,0,0,0,0,0,.4,0,1.2,0,-.4,0,1.2])
    unprojected=nominal+.5*commands
    source=np.array([f['pd_target_unprojected'] for f in frames])[:,joint_ids]
    assert abs(unprojected-source).max()<2e-6
    return torch.from_numpy(states),torch.from_numpy(targets),torch.from_numpy(commands),dict(
        **row,sha256=sha(row['path']),frames=len(frames),joints=names,
        real_ankle_velocity_rms=float(np.sqrt(np.mean(states[:,28+np.array([4,5,10,11])]**2))),
        projected_target_frames=int((abs(targets-source)>1e-7).any(-1).sum()),
        angular_velocity_source='instantaneous local free-joint qvel, not lagged gyro sensor',
        reset_boundaries='separate full controller rollouts; no cross-trajectory windows')


def batch(states,targets,starts,horizon):
    past=starts[:,None]+torch.arange(-79,0,device=starts.device)
    future=starts[:,None]+torch.arange(horizon,device=starts.device)
    hist=torch.cat((states[past],targets[past]),-1)
    return states[starts],hist,targets[future],states[future+1]


def rollout(model,state,history,targets):
    predictions=[]
    for t in range(targets.shape[1]):
        old_state=state
        state=model(old_state,targets[:,t],history)
        predictions.append(state)
        # A completed transition is (s_t, target_t), not (predicted s_{t+1}, target_t).
        history=torch.cat((history[:,1:],torch.cat((old_state,targets[:,t]),-1)[:,None]),1)
    return torch.stack(predictions,1)


@torch.no_grad()
def evaluate(model,states,targets,starts,horizon):
    squared=torch.zeros(2,horizon,51,dtype=torch.float64,device=states.device);count=0
    for ids in starts.split(256):
        state,history,acts,truth=batch(states,targets,ids,horizon)
        prediction=rollout(model,state,history,acts)
        persistent=state[:,None].expand_as(truth)
        for i,p in enumerate([prediction,persistent]):
            difference=p-truth
            difference[:,:,3:5]=torch.atan2(torch.sin(difference[:,:,3:5]),torch.cos(difference[:,:,3:5]))
            squared[i]+=difference.double().square().sum(0)
        count+=len(ids)
    if not count:raise ValueError('no valid physical evaluation windows')
    errors=(squared/count).sqrt().cpu().numpy();result={}
    for i,name in enumerate(['physical_wm','persistence']):
        result[name]=dict(windows=count,horizon_steps=horizon,
            one_step_joint_rmse=float(np.sqrt(np.mean(errors[i,0,5:28]**2))),
            final_step_joint_rmse=float(np.sqrt(np.mean(errors[i,-1,5:28]**2))),
            one_step_ankle_velocity_rmse=float(np.sqrt(np.mean(errors[i,0,28+np.array([4,5,10,11])]**2))),
            final_step_ankle_velocity_rmse=float(np.sqrt(np.mean(errors[i,-1,28+np.array([4,5,10,11])]**2))),
            normalized_trajectory_rmse=float(np.sqrt(np.mean((errors[i]/model.state_scale.cpu().numpy())**2))),
            per_dimension_last_rmse=errors[i,-1].tolist())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--horizon',type=int,default=20);p.add_argument('--batch-size',type=int,default=256)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('requested CUDA is unavailable')
    records=[];states=[];targets=[];commands=[];starts={k:[] for k in ['train','validation','test']};offset=0
    for row in trajectories():
        state,target,command,receipt=load_trace(row);records.append(receipt)
        ids=torch.arange(offset+79,offset+len(state)-args.horizon)
        if not len(ids):raise ValueError('trajectory too short for unpadded history')
        starts[row['split']].append(ids);states.append(state);targets.append(target);commands.append(command);offset+=len(state)
    states=torch.cat(states);targets=torch.cat(targets);commands=torch.cat(commands)
    starts={k:torch.cat(v) for k,v in starts.items()}
    train_rows=torch.cat([torch.arange(sum(r['frames'] for r in records[:i]),sum(r['frames'] for r in records[:i+1]))
                          for i,r in enumerate(records) if r['split']=='train'])
    normalizers=fit_normalizers(states[train_rows],targets[train_rows])
    torch.save(dict(contract=CONTRACT,state=states,executed_pd_target=targets,commanded_action=commands,
                    window_starts=starts,normalizers=normalizers,trajectories=records),out/'physical_transitions.pt')
    manifest=dict(contract=CONTRACT,seed=42,device=args.device,epochs=args.epochs,horizon=args.horizon,
        batch_size=args.batch_size,learning_rate=.001,train_cases=sorted(TRAIN),validation_cases=sorted(VAL),test_cases=sorted(TEST),
        trajectories=records,states=len(states),windows={k:len(v) for k,v in starts.items()},
        actor_updates=0,ppo_actions_modified=False,normalizer_fit_split='train trajectories only',
        original_model_hashes={str(f.relative_to(ROOT)):sha(f) for f in [POLICY,CHECKPOINT]},
        code={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__).resolve(),ROOT/'rsl_rl/rsl_rl/modules/physical_world_model.py']},
        limitation='Motion-held-out forward prediction pilot; not a new control policy or hardware validation.')
    save(out/'manifest.json',manifest)
    model=PhysicalWorldModel(**normalizers).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    states=states.to(args.device);targets=targets.to(args.device);starts={k:v.to(args.device) for k,v in starts.items()}
    baseline=evaluate(model,states,targets,starts['validation'],args.horizon)
    assert baseline['physical_wm']==baseline['persistence']
    save(out/'initial_validation.json',baseline)
    best=float('inf');updates=0;begin=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();order=starts['train'][torch.randperm(len(starts['train']),device=args.device)];losses=[]
        for ids in order.split(args.batch_size):
            state,history,acts,truth=batch(states,targets,ids,args.horizon)
            prediction=rollout(model,state,history,acts)
            loss=model.loss(prediction.flatten(0,1),truth.flatten(0,1))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite physical WM training loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            updates+=1;losses.append(float(loss.detach()))
        model.eval();validation=evaluate(model,states,targets,starts['validation'],args.horizon)
        score=validation['physical_wm']['normalized_trajectory_rmse']
        row=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),validation=validation)
        with (out/'training.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if score<best:
            best=score
            torch.save(dict(contract=CONTRACT,model_state_dict=model.state_dict(),normalizers=normalizers,
                            epoch=epoch,optimizer_state=optimizer.state_dict(),manifest=manifest),out/'best_physical_wm.pt')
        print('EPOCH',epoch,'loss',row['train_loss'],'validation',score,'persistence',validation['persistence']['normalized_trajectory_rmse'],flush=True)
    selected=torch.load(out/'best_physical_wm.pt',map_location=args.device);model.load_state_dict(selected['model_state_dict']);model.eval()
    test=evaluate(model,states,targets,starts['test'],args.horizon)
    per_trajectory=[]
    offset=0
    for r in records:
        if r['split']=='test':
            ids=torch.arange(offset+79,offset+r['frames']-args.horizon,device=args.device)
            per_trajectory.append(dict(case=r['case'],mode=r['mode'],**evaluate(model,states,targets,ids,args.horizon)))
        offset+=r['frames']
    gain=1-test['physical_wm']['normalized_trajectory_rmse']/test['persistence']['normalized_trajectory_rmse']
    report=dict(selected_epoch=selected['epoch'],updates=updates,wall_seconds=time.perf_counter()-begin,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
        test=test,per_test_trajectory=per_trajectory,test_normalized_error_reduction=gain,
        forward_prediction_gate_passed=gain>=.1,actor_updates=0,
        policy_performance_improvement_measured=False,next_phase='physical data/model pipeline only; policy integration requires separate closed-loop experiment')
    for path,digest in manifest['original_model_hashes'].items():assert sha(ROOT/path)==digest
    save(out/'result.json',report);(out/'COMPLETE').write_text('Physical WM pilot finished; held-out test executed once after validation selection.\n')
    print('PILOT COMPLETE',gain,flush=True)


if __name__=='__main__':main()
