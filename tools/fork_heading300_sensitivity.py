"""Read-only full-state forks through the existing, unchanged MuJoCo controller."""
import copy,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deploy_real'),str(ROOT/'tools'),str(ROOT/'rsl_rl'),str(ROOT/'legged_gym/scripts')]
import server_low_level_g1_sim as sim
from fork_heading300_first_update import RecordedReference
import mujoco
PIVOTS=[0,400,600,900];HORIZON=250
class Clock:
    def __getattr__(self,k):return getattr(time,k)
    def sleep(self,*args):pass  # Offline pacing only; physics/control dt unchanged.

def capture(c,reference):
    attrs=['last_action','proprio_history_buf','projection_count','projection_max_delta','target_count','_debug_policy_stats_count','_last_valid_mimic_full','_last_mimic_warning_time','metrics']
    return dict(data=copy.deepcopy(c.data),controller={k:copy.deepcopy(getattr(c,k)) for k in attrs},runtime={k:copy.deepcopy(v) for k,v in vars(c.anyadapter_runtime).items() if k!='policy'},diagnostics={k:copy.deepcopy(v) for k,v in vars(c.dynamics_diagnostics).items() if k not in ['model','core']},reference_frame=reference.frame,np_rng=np.random.get_state(),torch_rng=torch.get_rng_state().clone(),python_rng=random.getstate())

def restore(c,reference,s):
    # Reconstruct the identical full prefix in each sibling, then restore the
    # integration vector into its own model-owned live data. Never exchange
    # copied native solver arenas between controller instances.
    for key in ['qpos','qvel','qacc','qacc_warmstart','ctrl','sensordata','xpos','xmat']:
        assert np.array_equal(getattr(c.data,key),getattr(s['data'],key)),('prefix mismatch',key)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(c.model,spec));mujoco.mj_getState(c.model,s['data'],state,spec)
    mujoco.mj_setState(c.model,c.data,state,spec)
    for k,v in s['controller'].items():setattr(c,k,copy.deepcopy(v))
    for k,v in s['runtime'].items():setattr(c.anyadapter_runtime,k,copy.deepcopy(v))
    for k,v in s['diagnostics'].items():setattr(c.dynamics_diagnostics,k,copy.deepcopy(v))
    reference.frame=s['reference_frame'];np.random.set_state(s['np_rng']);torch.set_rng_state(s['torch_rng']);random.setstate(s['python_rng'])
    for key in ['qpos','qvel','qacc','qacc_warmstart','ctrl','act','sensordata','qfrc_applied','xfrc_applied']:
        assert np.array_equal(getattr(c.data,key),getattr(s['data'],key)),key
    assert c.data.time==s['data'].time

class Policy:
    def __init__(self,ref,old,new,pivot,duration):self.ref=ref;self.old=old;self.new=new;self.pivot=pivot;self.duration=duration;self.inputs={}
    def __call__(self,x):
        frame=self.ref.frame
        self.inputs[frame]=x.detach().cpu().numpy()[0].copy()
        return (self.new if self.pivot<=frame<self.pivot+self.duration else self.old)(x)

def controller(dest,oldpath,frames,pivot,duration,old,new):
    np.random.seed(42);torch.manual_seed(42);random.seed(42)
    c=sim.RealTimePolicyController(xml_file=str(ROOT/'assets/g1/g1_sim2sim_with_wrist_roll.xml'),policy_path=str(oldpath),device='cpu',headless=True,sync_reference=True,sim_duration=(pivot+HORIZON)*.02,trace_out=str(dest/'frames.jsonl'),metrics_out=str(dest/'metrics.json'),target_projection=True,projection_margin=.02,projection_knee_min=0.,dynamics_diagnostics=True,anyadapter_ema_alpha=0.)
    ref=RecordedReference(frames);c.redis_client=ref;c.anyadapter_runtime.policy=Policy(ref,old,new,pivot,duration)
    return c,ref

def run(root):
    torch.set_num_threads(1);os.environ.pop('TWIST_FIRST_UPDATE_DIAGNOSTICS',None)
    summary=json.loads((root/'evaluation_retry/summary.json').read_text());assert summary['status']=='complete'
    out=root/'fork_retry';out.mkdir(exist_ok=False)
    oldpath=ROOT/'reports/recovery_1250/fix_stage_20261008/heading_300.pt';newpath=ROOT/'reports/heading300_hard_constraint_20261012/evaluation/exports/hard26/policy.pt'
    old=torch.jit.load(str(oldpath)).eval();new=torch.jit.load(str(newpath)).eval()
    savedclock=sim.time;sim.time=Clock();results=[]
    try:
      for mode in ['raw','wm']:
        row=next(r for r in summary['rows'] if r['label']=='heading_300' and r['reference_mode']==mode)
        frames=[json.loads(x) for x in Path(row['trace']).read_text().splitlines()]
        dest=out/mode/'prefix_null';dest.mkdir(parents=True)
        c,ref=controller(dest,oldpath,frames,900,0,old,new);snapshots={};extract=c.extract_data
        def wrapped_extract():
            # Called at the start of every physical substep, before observation/history updates.
            frame=round(c.data.time/.001)//20
            if round(c.data.time/.001)%20==0 and frame in PIVOTS and frame not in snapshots:
                snapshots[frame]=capture(c,ref)
            return extract()
        c.extract_data=wrapped_extract;c.run()
        trace=[json.loads(x) for x in (dest/'frames.jsonl').read_text().splitlines()]
        keys=['qpos','qvel','rpy','joint','joint_velocity','final_action','pd_target']
        assert len(trace)==1150 and len(snapshots)==4
        assert all(np.array_equal(np.array([f[k] for f in trace]),np.array([f[k] for f in frames[:1150]])) for k in keys),'full prefix null replay failed'
        for pivot in PIVOTS:
          snap=snapshots[pivot];state=np.empty(mujoco.mj_stateSize(c.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(c.model,snap['data'],state,mujoco.mjtState.mjSTATE_INTEGRATION)
          snapdir=out/mode/f'pivot_{pivot}';snapdir.mkdir()
          np.savez_compressed(snapdir/'snapshot.npz',integration_state=state,history=snap['runtime']['history'],proprio_history=np.array(snap['controller']['proprio_history_buf']),last_action=snap['controller']['last_action'],torch_rng=snap['torch_rng'].numpy())
          for label,duration in [('A',0),('B',HORIZON),('C',1),('D',10)]:
            branch=snapdir/label;branch.mkdir();bc,br=controller(branch,oldpath,frames,pivot,duration,old,new)
            original_extract=bc.extract_data;restored=[False]
            def branch_extract():
                step=round(bc.data.time/.001)
                if step==pivot*20 and not restored[0]:
                    restore(bc,br,snap);restored[0]=True
                return original_extract()
            bc.extract_data=branch_extract;bc.run();assert restored[0]
            data=[json.loads(x) for x in (branch/'frames.jsonl').read_text().splitlines()];assert len(data)==pivot+HORIZON;data=data[pivot:]
            baseline=frames[pivot:pivot+HORIZON]
            if label=='A':assert all(np.array_equal(np.array([f[k] for f in data]),np.array([f[k] for f in baseline])) for k in keys),'restored null mismatch'
            assert np.array_equal(np.array([f['clean'] for f in data]),np.array([f['clean'] for f in baseline])) and np.array_equal(np.array([f['processed'] for f in data]),np.array([f['processed'] for f in baseline]))
            # Controller's diagnostics writer starts sidecar allocation only at frame0.
            obs=branch/'actor_inputs.npy';np.save(obs,np.array([bc.anyadapter_runtime.policy.inputs[i] for i in range(pivot,pivot+HORIZON)]))
            results.append(dict(mode=mode,pivot_seconds=pivot*.02,label=label,duration_steps=duration,frames=250,window_seconds=5,trace=str(branch/'frames.jsonl'),actor_inputs=str(obs),null_exact=label=='A',snapshot_full_MjData=True,restore_method='exact complete prefix reconstruction, own live integration-state restore, full controller/runtime/RNG restore'))
            (out/'results.json').write_text(json.dumps(results,indent=2));print('FORK COMPLETE',mode,pivot,label,flush=True)
    finally:sim.time=savedclock
    (out/'COMPLETE').write_text('32 branches; eight full-state snapshots; all null branches exact\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
