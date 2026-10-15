"""Controlled prefix/fork replay using the unchanged existing MuJoCo controller."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deploy_real'),str(ROOT/'tools'),str(ROOT/'rsl_rl'),str(ROOT/'legged_gym/scripts')]
from server_low_level_g1_sim import RealTimePolicyController
from motion_world_model.runtime import reinsert_wrist_roll
from export_twist_anyadapter_opentrack_jit import build_actor,DeployActor
from heading300_first_update_verify import RUN


class RecordedReference:
    def __init__(self,frames):self.frames=frames;self.frame=-1
    def set(self,key,value):
        if key=='sim_ready_g1':self.frame=int(value)
    def mget(self,*keys):
        row=self.frames[self.frame]
        wrist=np.array(row['pd_target_unprojected'])[[19,24]]
        values=dict(action_mimic_g1=json.dumps(reinsert_wrist_roll(np.array(row['processed']),wrist).tolist()).encode(),
            action_mimic_clean_g1=json.dumps(reinsert_wrist_roll(np.array(row['clean']),wrist).tolist()).encode(),
            action_mimic_frame_g1=str(self.frame).encode(),action_mimic_trace_g1=json.dumps(row).encode())
        return [values[k] for k in keys]


class SwitchPolicy:
    def __init__(self,reference,old,new,pivot):self.reference=reference;self.old=old;self.new=new;self.pivot=pivot
    def __call__(self,x):return (self.old if self.reference.frame<self.pivot else self.new)(x)


def run(root,pivot=900,horizon=400):
    out=root/'fork';out.mkdir(exist_ok=False)
    summary=json.loads((root/'evaluation/summary.json').read_text())
    row=next(r for r in summary['rows'] if r['label']=='model25' and r['reference_mode']=='wm')
    frames=[json.loads(x) for x in Path(row['trace']).read_text().splitlines()]
    old_actor=build_actor(RUN/'model_25.pt');new_state=torch.load(RUN/'model_26.pt',map_location='cpu')['model_state_dict']
    old_path=root/'evaluation/exports/model25/policy.pt';new_path=root/'evaluation/exports/model26/policy.pt'
    policies=dict(null=old_path,full26=new_path)
    old_state=old_actor.state_dict()
    for label,core_new in [('core26',True),('head26',False)]:
        mix=dict(old_state)
        for key in mix:
            if key.startswith('layerwise_actor.adapter.') and (key.startswith('layerwise_actor.adapter.5.')!=core_new):mix[key]=new_state[key]
        actor=build_actor(RUN/'model_25.pt');actor.load_state_dict(mix)
        x=torch.zeros(1,7003);deploy=DeployActor(actor).eval();jit=torch.jit.trace(deploy,x)
        path=out/(label+'.pt');jit.save(str(path));policies[label]=path
        with torch.no_grad():assert float((jit(x)-actor.actor_mean(x)).abs().max())<1e-6
    old=torch.jit.load(str(old_path)).eval();results={}
    # Both siblings replay identical complete prefixes using original PD/physics.
    # No qpos-only restoration or missing solver-history assumptions are needed.
    os.environ.pop('TWIST_FIRST_UPDATE_DIAGNOSTICS',None)
    for label,path in policies.items():
        dest=out/label;dest.mkdir()
        np.random.seed(42);torch.manual_seed(42)
        controller=RealTimePolicyController(xml_file=str(ROOT/'assets/g1/g1_sim2sim_with_wrist_roll.xml'),
            policy_path=str(old_path),device='cpu',headless=True,sync_reference=True,
            sim_duration=(pivot+horizon)*.02,trace_out=str(dest/'frames.jsonl'),
            metrics_out=str(dest/'metrics.json'),target_projection=True,projection_margin=.02,projection_knee_min=0.,
            dynamics_diagnostics=True,anyadapter_ema_alpha=0.)
        reference=RecordedReference(frames);controller.redis_client=reference
        controller.anyadapter_runtime.policy=SwitchPolicy(reference,old,torch.jit.load(str(path)).eval(),pivot)
        controller.run()
        trace=[json.loads(x) for x in (dest/'frames.jsonl').read_text().splitlines()]
        prefix_error={key:float(np.max(abs(np.array([x[key] for x in trace[:pivot]])-np.array([x[key] for x in frames[:pivot]])))) for key in ['qpos','qvel','joint','rpy','final_action','pd_target']}
        assert max(prefix_error.values())==0,prefix_error
        if label=='null':
            assert all(np.array_equal(np.array([x[k] for x in trace]),np.array([x[k] for x in frames[:len(trace)]])) for k in prefix_error)
        post=trace[pivot:];clean=np.array([x['clean'] for x in post]);yaw=np.array([x['rpy'][2] for x in post])
        error=np.arctan2(np.sin(yaw-clean[:,3]),np.cos(yaw-clean[:,3]))
        baseline_yaw=np.array([x['rpy'][2] for x in frames[pivot:pivot+horizon]])
        gap=np.arctan2(np.sin(yaw-baseline_yaw),np.cos(yaw-baseline_yaw))
        results[label]=dict(prefix_errors=prefix_error,post_yaw_rmse_degrees=float(np.degrees(np.sqrt(np.mean(error**2)))),
            post_yaw_gap_rms_degrees=float(np.degrees(np.sqrt(np.mean(gap**2)))),post_yaw_gap_max_degrees=float(np.degrees(abs(gap).max())),
            final_yaw_gap_degrees=float(np.degrees(gap[-1])),post_frames=len(post),policy=str(path),trace=str(dest/'frames.jsonl'))
        print('Controlled fork',label,results[label],flush=True)
        (out/'results.json').write_text(json.dumps(dict(pivot_frame=pivot,pivot_seconds=pivot*.02,horizon_frames=horizon,
            horizon_seconds=horizon*.02,mode='wm',conditions='original controller, constant recorded WM reference, identical full physical/controller prefix',
            results=results,caution='local intervention at one source-policy state; not a global module ranking or proof for other motions'),indent=2))
    (out/'COMPLETE').write_text('4 controlled siblings; exact null/prefix replay')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pivot',type=int,default=900);p.add_argument('--horizon',type=int,default=400)
    a=p.parse_args();run(a.root,a.pivot,a.horizon)
