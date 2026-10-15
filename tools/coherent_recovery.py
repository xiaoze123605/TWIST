"""Versioned local recovery teacher with real pre-roll and explicit activation."""
import isaacgym
import copy,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from heading_recovery_core import HeadingRecoveryBatch,RecordedMotionReference,RecoveryActorCritic,ROOT,wrap
from diagnose_recovery_task import snapshot,restore,metrics,paired,reward_terms
from data_utils.rot_utils import quatToEuler

CONTRACT='coherent_turn_squat_recovery_v2'
DIAG=ROOT/'reports/recovery_task_diagnosis_20261014_final'
ACTIVE_START=2100
ACTIVE_END=2500


def teacher_features(features):
    x=features.clone();x[:,112:144]=0;x[:,167]=0
    return x


def references(mode='raw'):
    s=json.loads((ROOT/'reports/heading300_sensitivity_20261013/evaluation_retry/summary.json').read_text())
    row=next(r for r in s['rows'] if r['label']=='heading_300' and r['reference_mode']==mode)
    return [json.loads(line) for line in open(row['trace'])]


def prepare_start():
    env=HeadingRecoveryBatch(1);ref=RecordedMotionReference(references());ref.length=ACTIVE_END
    initial=np.load(DIAG/'reference_at39_preroll.npz')
    env.reset_one(0,ref,0);env.frames[0]=1950
    env.data[0].qpos[:]=initial['qpos'][0];env.data[0].qvel[:]=initial['qvel'][0]
    mujoco.mj_forward(env.model,env.data[0]);env.cache=None;env.observe()
    for frame in range(1950,ACTIVE_START):
        obs=env.observe();assert np.array_equal(env.data[0].qpos,initial['qpos'][frame-1950]),frame
        env.step(obs['source'].numpy(),obs['source'].numpy(),auto_reset=False)
    assert np.array_equal(env.data[0].qpos,initial['qpos'][150])
    return snapshot(env)


class LocalRecoveryBatch:
    def __init__(self,n,initial,seed=42):
        self.env=HeadingRecoveryBatch(n,seed);self.initial=initial;self.rng=np.random.default_rng(seed)
        self.signs=np.zeros(n,dtype=int);self.n=n
        c=initial['python']['cache']
        self.env.cache={k:c[k].repeat(n,1).clone() for k in ['features','source','legacy']}
        self.env.cache['physical']=np.repeat(c['physical'],n,axis=0)
        for k in ['rows','references','wrists']:self.env.cache[k]=[copy.deepcopy(c[k][0]) for _ in range(n)]
        for i in range(n):self.reset_slot(i)

    def reset_slot(self,i,sign=None):
        env=self.env;s=self.initial;d=env.data[i]
        mujoco.mj_setState(env.model,d,s['state'],s['spec']);mujoco.mj_forward(env.model,d)
        d.sensordata[:]=s['sensor'];d.qacc_warmstart[:]=s['warmstart']
        for key,val in s['python'].items():
            if key=='cache':continue
            if isinstance(getattr(env,key),np.ndarray):getattr(env,key)[i]=val[0]
            else:getattr(env,key)[i]=copy.deepcopy(val[0])
        for key,val in s['runtime'].items():setattr(env.runtime[i],key,copy.deepcopy(val))
        env.refs[i]=s['reference']
        for key in ['features','source','legacy']:env.cache[key][i]=s['python']['cache'][key][0].clone()
        env.cache['physical'][i]=s['python']['cache']['physical'][0].copy()
        for key in ['rows','references','wrists']:env.cache[key][i]=copy.deepcopy(s['python']['cache'][key][0])
        self.signs[i]=int(self.rng.choice([-1,0,1]) if sign is None else sign)

    def observe(self):return self.env.observe()

    def step(self,sampled,source):
        # Keep PPO's sampled Normal action; clipping is an execution operation.
        executed=source+np.clip(sampled-source,-.1,.1)
        phases=self.env.frames.copy();signs=self.signs.copy();obs=self.observe()
        refs=[r.copy() for r in obs['references']];previous=self.env.last_delta.copy()
        for i,d in enumerate(self.env.data):
            d.xfrc_applied[:]=0
            if ACTIVE_START<=phases[i]<ACTIVE_START+10:
                d.xfrc_applied[self.env.model.body('pelvis').id,5]=signs[i]
        new,reward,done,info=self.env.step(executed,source,auto_reset=False)
        info['terminal_features']=teacher_features(info['terminal_features'])
        info.update(reference_phase=phases,disturbance_sign=signs,executed_commands=executed,
                    pd_target=np.stack([r['pd_target'] for r in info['trace_rows']]),
                    actual_action=np.stack([r['final_action'] for r in info['trace_rows']]),
                    physical_before=np.stack([np.r_[r['qpos'],r['qvel']] for r in info['trace_rows']]),
                    physical_after=np.stack([np.r_[d.qpos,d.qvel] for d in self.env.data]))
        for d in self.env.data:d.xfrc_applied[:]=0
        for i in np.flatnonzero(done):self.reset_slot(int(i))
        return self.observe(),reward,done,info


def run_case(env,initial,actor,pulse_phase,sign,output):
    restore(env,initial);rows=[]
    for _ in range(400):
        obs=env.observe();source=obs['source'];frame=int(env.frames[0]);previous=env.last_delta[0].copy()
        with torch.no_grad():command=actor.mean(teacher_features(obs['features']),source).numpy()
        command=source.numpy()+np.clip(command-source.numpy(),-.1,.1)
        env.data[0].xfrc_applied[:]=0
        if pulse_phase<=frame<pulse_phase+10:env.data[0].xfrc_applied[env.model.body('pelvis').id,5]=sign
        ref=obs['references'][0].copy();before=env.data[0].qpos.copy();contacts=env.contacts(env.data[0]).copy()
        new,reward,_,info=env.step(command,source.numpy(),auto_reset=False)
        row=info['trace_rows'][0];terms=reward_terms(env,ref,command[0]-source.numpy()[0],previous)
        assert np.isclose(sum(terms.values()),float(reward[0]),rtol=1e-5,atol=2e-5)
        row.update(post_qpos=env.data[0].qpos.copy(),post_qvel=env.data[0].qvel.copy(),
                   native_pre_rpy=np.asarray(quatToEuler(before[3:7])),
                   contact=contacts,reward=float(reward[0]),reward_terms=np.array(list(terms.values())),
                   next_reference_reward=float(reward[0]),projected=bool(np.any(row['pd_target']!=row['pd_target_unprojected'])))
        rows.append(row)
    env.data[0].xfrc_applied[:]=0
    keys=['qpos','qvel','rpy','joint','joint_velocity','clean','processed','pd_target','final_action',
          'correction','post_qpos','post_qvel','native_pre_rpy','contact','reward','reward_terms','next_reference_reward','projected']
    x={k:np.array([r[k] for r in rows]) for k in keys}
    assert all(np.isfinite(v).all() for v in x.values())
    np.savez_compressed(output,**x);return x


def evaluate_local(actor,initial,out):
    out.mkdir(parents=True,exist_ok=False);env=HeadingRecoveryBatch(1);cases=[];arrays={}
    for name,phase,sign in [('nominal',2100,0),('pulse42_negative',2100,-1),('pulse42_positive',2100,1),
                            ('pulse45_negative',2250,-1),('pulse45_positive',2250,1)]:
        x=run_case(env,initial,actor,phase,sign,out/f'{name}.npz');arrays[name]=x
        row=dict(name=name,pulse_phase=phase,sign=sign,**metrics(x))
        if sign:
            start=phase-2100
            disturbed={k:v[start:] for k,v in x.items()};nominal={k:v[start:] for k,v in arrays['nominal'].items()}
            extra=paired(disturbed,nominal)
            a=abs(np.degrees(wrap(disturbed['native_pre_rpy'][:,2]-nominal['native_pre_rpy'][:,2])))
            sustained=None
            for t in range(10,len(a)-99):
                if (a[t:]<=2).all():sustained=t*.02;break
            row['paired_recovery']=extra;row['sustained_recovery_seconds']=sustained
            row['sustained_definition']='after0.2s pulse, <=2deg for all remaining samples, with at least2s remaining'
        cases.append(row)
    (out/'results.json').write_text(json.dumps(cases,indent=2)+'\n')
    (out/'COMPLETE').write_text('Five complete local cases,400frames each; pulse45 is withheld timing.\n')
    return cases


class PhaseActivationActor:
    """Explicit local-specialist schedule; never represented as a general policy."""
    def __init__(self,actor):self.actor=actor;self.frame=0
    def mean(self,features,source):
        frame=self.frame;self.frame+=1
        return self.actor.mean(teacher_features(features),source) if ACTIVE_START<=frame<ACTIVE_END else source


def evaluate_full(actor,out):
    from evaluate_heading_recovery import full77
    # full77 uses one actor object for two modes, so reset phase at mode boundary.
    class Scheduled(PhaseActivationActor):
        def mean(self,features,source):
            if self.frame==3207:self.frame=0
            return super().mean(features,source)
    return full77(Scheduled(actor),out)
