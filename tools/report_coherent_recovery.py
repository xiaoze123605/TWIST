"""Saved-data recovery curves and reward decomposition; zero new training."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/coherent_recovery_mpl')
import isaacgym
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coherent_recovery import references
from heading_recovery_core import BODY_IDS,wrap
from data_utils.rot_utils import quatToEuler,quat_rotate_inverse


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    training=[json.loads(l) for l in (r/'pilot/training.jsonl').open()]
    base=json.loads((r/'baseline/local/results.json').read_text());summary=[]
    for update in [15,25,35]:
        e=r/'pilot'/f'evaluation_{update}';local=json.loads((e/'local/results.json').read_text());gate=json.loads((e/'receipt.json').read_text())
        pooled_base=sum(x['paired_recovery']['extra_yaw_iae_deg_seconds'] for x in base if x['sign'])
        pooled_candidate=sum(x['paired_recovery']['extra_yaw_iae_deg_seconds'] for x in local if x['sign'])
        summary.append(dict(update=update,actor_updates=update-5,nominal_yaw_rmse_deg=local[0]['yaw_rmse_deg'],
                            declared_unweighted_relative_iae_gain=gate['local_gate']['mean_paired_iae_gain'],
                            pooled_absolute_iae_gain=1-pooled_candidate/pooled_base,
                            pooled_absolute_iae_deg_seconds=pooled_candidate,full_pass=gate['full_protection']['passed'],
                            local_pass=gate['local_gate']['passed']))
    (r/'assessment.json').write_text(json.dumps(dict(evaluations=summary,
        aggregation_note='Declared gate averages per-case relative gains. Pooled absolute IAE is descriptive and does not replace the declared gate. Very small baseline errors at45s amplify relative ratios.'),indent=2)+'\n')
    plt.rcParams.update({'font.size':9})
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    cases=['pulse42_negative','pulse42_positive','pulse45_negative','pulse45_positive']
    for case,aa in zip(cases,ax.flatten()):
        phase=0 if '42' in case else 150
        for directory,label in [('baseline/local','heading300'),('pilot/evaluation_15/local','actor10'),
                                 ('pilot/evaluation_25/local','actor20'),('pilot/evaluation_35/local','actor30')]:
            pulse=np.load(r/directory/(case+'.npz'));nominal=np.load(r/directory/'nominal.npz')
            extra=np.degrees(wrap(pulse['native_pre_rpy'][phase:,2]-nominal['native_pre_rpy'][phase:,2]))
            aa.plot(np.arange(len(extra))*.02,extra,label=label)
        aa.set_title(case);aa.axhline(2,color='grey',linestyle='--');aa.axhline(-2,color='grey',linestyle='--')
        aa.set_xlabel('time since disturbance (s)');aa.set_ylabel('paired extra yaw (deg)');aa.grid(alpha=.25);aa.legend()
    fig.tight_layout();fig.savefig(r/'paired_recovery.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for mode,aa in zip(['raw','wm'],ax):
        for directory,label in [('baseline/full77','heading300'),('pilot/evaluation_35/full77','actor30 scheduled')]:
            x=np.load(r/directory/(mode+'.npz'));e=np.degrees(wrap(x['rpy'][:,2]-x['clean'][:,3]))
            aa.plot(np.arange(3207)*.02,e,label=label,linewidth=.8)
        aa.axvspan(42,50,color='grey',alpha=.15);aa.set_ylabel(mode+' yaw error (deg)');aa.legend();aa.grid(alpha=.25)
    ax[-1].set_xlabel('control time (s)');fig.tight_layout();fig.savefig(r/'full77_yaw.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
    u=[x['actor_updates'] for x in training]
    ax[0].plot(u,[x['mean_reward'] for x in training]);ax[0].set_ylabel('mean rollout reward')
    ax[1].plot(u,[x['kl'] or 0 for x in training],label='approx KL');ax[1].plot(u,[x['clip_fraction'] or 0 for x in training],label='clip fraction');ax[1].legend()
    ax[2].plot(u,[x['actor_parameter_l2'] for x in training]);ax[2].set_ylabel('actor parameter step L2');ax[2].set_xlabel('actor updates')
    for aa in ax:aa.grid(alpha=.25)
    fig.tight_layout();fig.savefig(r/'training_signals.png',dpi=160);plt.close(fig)
    # Reconstruct each reward from archived post-step physical states and exact
    # executed residuals. Previous residual resets to0 at the cached42s state.
    refs=np.array([x['processed'] for x in references()],dtype=np.float32)
    previous=np.zeros((16,23),np.float32);reward_audits=[];counts={-1:0,0:0,1:0}
    nq=np.load(r/'baseline/local/nominal.npz')['qpos'].shape[1]
    for row in training:
        update=row['update'];x=np.load(r/'pilot/rollouts'/f'rollout_{update}.npz')
        post=x['physical_after'];flat=post.reshape(-1,post.shape[-1]);qp=flat[:,:nq];qv=flat[:,nq:]
        rpy=np.array([quatToEuler(q[3:7]) for q in qp]);ref=refs[x['phase'].reshape(-1)]
        vel=quat_rotate_inverse(qp[:,[4,5,6,3]],qv[:,:3]);q=qp[:,7+BODY_IDS]
        delta=x['actual_action']-x['source'];old=[]
        for t in range(128):
            previous[x['phase'][t]==2100]=0;old.append(previous.copy());previous=delta[t].copy()
        rate=(delta-np.array(old))/.02;d=delta.reshape(-1,23);rate=rate.reshape(-1,23)
        terms=np.stack([-4*wrap(ref[:,3]-rpy[:,2])**2,-.5*(ref[:,7]-qv[:,5])**2,
                        -.25*((qp[:,2]-ref[:,0])/.1)**2,-.1*np.mean(((q[:,:12]-ref[:,8:20])/.35)**2,axis=1),
                        -.1*np.mean((wrap(ref[:,1:3]-rpy[:,:2])/.2)**2,axis=1),-.5*np.sum((vel-ref[:,4:7])**2,axis=1),
                        -.02*np.mean((d/.1)**2,axis=1),-.001*np.mean(rate**2,axis=1),
                        -20*((qp[:,2]<.35)|(abs(rpy[:,:2]).max(1)>1))],axis=1).reshape(128,16,9)
        error=float(abs(terms.sum(-1)-x['rewards']).max());assert error<2e-5,(update,error)
        np.savez_compressed(r/'pilot/rollouts'/f'reward_terms_{update}.npz',terms=terms)
        for sign in counts:counts[sign]+=int((x['disturbance']==sign).sum())
        reward_audits.append(dict(update=update,max_reward_reconstruction_error=error,mean_terms=terms.mean((0,1)).tolist()))
    (r/'reward_audit.json').write_text(json.dumps(dict(components=['heading','body_angular_z','height','leg_joint','roll_pitch',
        'linear_velocity','correction','correction_rate','fall'],updates=reward_audits,transitions_by_episode_sign=counts,
        scope='Reconstructed rewards check implementation arithmetic, not reward sufficiency or causal learning benefit.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
