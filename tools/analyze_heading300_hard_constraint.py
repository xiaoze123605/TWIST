"""Read-only reporting for the single fixed-pool hard-constraint experiment."""
import json, math, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_heading300_first_update import outputs, load_row, csv_write
from export_twist_anyadapter_opentrack_jit import build_actor
from sim_eval_metrics import phase_masks, rms
from summarize_heading300_sampling import flatten
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/heading300_hard_constraint_20261012'
RUN=ROOT/'legged_gym/logs/recovery_heading300_yaw_controls/hard1e4_adam_env512_seed42_20261012'
SOURCE=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'
OLD=ROOT/'legged_gym/logs/recovery_heading300_yaw_controls/anchor20_adamresume_original_env512_seed42_20261012/model_26.pt'

def probes():
    torch.set_num_threads(1)
    summary=json.loads((ROOT/'reports/heading300_first_actor_update_20261012/evaluation/summary.json').read_text())
    actors={label:build_actor(p) for label,p in [('heading300',SOURCE),('unconstrained26',OLD),('hard26',RUN/'model_26.pt')]}
    records=[]
    for mode in ['raw','wm']:
        row=next(r for r in summary['rows'] if r['label']=='heading_300' and r['reference_mode']==mode)
        frames,obs=load_row(row);ref=np.array([f['clean'] for f in frames]);masks={'all':np.ones(len(obs),bool),**phase_masks(ref)}
        baseline=outputs(actors['heading300'],obs)
        for label in ['unconstrained26','hard26']:
            result=outputs(actors[label],obs)
            for phase,mask in masks.items():
                delta=result['full'][mask]-baseline['full'][mask];core=result['core'][mask]-baseline['core'][mask];head=result['head'][mask]-baseline['head'][mask]
                records.append(dict(label=label,mode=mode,phase=phase,states=int(mask.sum()),max_abs=float(abs(delta).max()),mean_abs=float(abs(delta).mean()),p95=float(np.quantile(abs(delta),.95)),p99=float(np.quantile(abs(delta),.99)),full_rms=float(rms(delta)),core_rms=float(rms(core)),heading_rms=float(rms(head)),states_over_fixed_pool_bound=int((abs(delta).max(-1)>1e-4).sum())))
    csv_write(OUT/'common_77_state_drift.csv',records)
    (OUT/'common_77_state_drift.json').write_text(json.dumps(records,indent=2))

def summarize():
    summary=json.loads((OUT/'evaluation/summary.json').read_text());assert summary['status']=='complete'
    rows=[];stages=[];by={};(OUT/'plots').mkdir(exist_ok=True)
    for row in summary['rows']:
        assert row['status']=='complete',row
        flat=flatten(row,0,OUT/'evaluation');frames=[json.loads(x) for x in Path(row['trace']).read_text().splitlines()];assert len(frames)==3207
        ref=np.array([f['clean'] for f in frames]);q=np.array([f['joint'] for f in frames]);mask=phase_masks(ref)['squat']
        for side,i in [('left',3),('right',9)]:flat['squat_'+side+'_knee_rmse_degrees']=math.degrees(rms((q[:,i]-ref[:,8+i])[mask]))
        rows.append(flat);by[row['label'],row['reference_mode']]=flat
        yaw=np.arctan2(np.sin(np.array([f['rpy'][2] for f in frames])-ref[:,3]),np.cos(np.array([f['rpy'][2] for f in frames])-ref[:,3]))
        for phase,m in phase_masks(ref).items():stages.append(dict(label=row['label'],mode=row['reference_mode'],phase=phase,yaw_rmse_degrees=math.degrees(rms(yaw[m]))))
    csv_write(OUT/'comparison.csv',rows);csv_write(OUT/'phase_yaw.csv',stages)
    gates={}
    for mode,limit in [('raw',71.17),('wm',33.68)]:
        b,n=by['heading_300',mode],by['hard26',mode]
        gates[mode]=dict(yaw=n['yaw_rmse_degrees']<=limit,root_velocity=n['root_velocity_rmse']<=b['root_velocity_rmse']*1.05,no_new_falls=n['fall_proxy']<=b['fall_proxy'],yaw_limit_degrees=limit,root_velocity_ratio=n['root_velocity_rmse']/b['root_velocity_rmse'])
    passed=all(g[k] for g in gates.values() for k in ['yaw','root_velocity','no_new_falls'])
    steps=[json.loads(x) for x in (RUN/'hard_constraint_steps.jsonl').read_text().splitlines()];steps=[s for s in steps if s['actor_attempted']]
    assert len(steps)==48 and all(s['adam_state_consistent'] and s['critic_adam_step_delta']==1 and s['final']['max_abs']<=1e-4 for s in steps)
    logs=[]
    for s in steps:
        logs.append(dict({k:s[k] for k in ['minibatch_index','ppo_loss','kl','gradient_preclip','actor_accepted','accepted_ratio','backtracks','actor_parameter_l2','adam_seconds','total_step_seconds']}, **{'max_abs':s['final']['max_abs'],'mean_abs':s['final']['mean_abs'],'p95_abs':s['final']['p95_abs'],'p99_abs':s['final']['p99_abs'],'constraint_seconds':sum(t['check_seconds'] for t in s['trials'])+ (s['final']['check_seconds'] if not s['actor_accepted'] else 0)}))
    csv_write(OUT/'minibatch_updates.csv',logs)
    for mode in ['raw','wm']:
        fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
        for label in ['heading_300','unconstrained26','hard26']:
            r=next(r for r in summary['rows'] if r['label']==label and r['reference_mode']==mode);f=[json.loads(x) for x in Path(r['trace']).read_text().splitlines()];ref=np.array([x['clean'] for x in f]);t=np.arange(len(f))*.02
            yaw=np.array([x['rpy'][2] for x in f]);err=np.arctan2(np.sin(yaw-ref[:,3]),np.cos(yaw-ref[:,3]));axes[0].plot(t,np.degrees(err),label=label)
            axes[1].plot(t,np.array([x['joint'][3] for x in f])-ref[:,11],label=label)
            axes[2].plot(t,[x['dynamics']['actual']['torso_pitch']-x['dynamics']['clean_reference']['torso_pitch'] for x in f],label=label)
        for ax,title in zip(axes,['Wrapped yaw error (deg)','Left knee tracking error (rad)','Reference-relative torso pitch (rad)']):ax.set_ylabel(title);ax.legend();ax.grid(alpha=.2)
        axes[-1].set_xlabel('Reference time (s)');fig.tight_layout();fig.savefig(OUT/'plots'/f'{mode}_time_series.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6));axes[0].plot([s['max_abs'] for s in logs],label='retained full-pool max');axes[0].axhline(1e-4,color='r',ls='--');axes[1].bar(range(48),[s['accepted_ratio'] for s in logs]);axes[1].set_ylabel('Accepted Actor step ratio');axes[1].set_xlabel('Minibatch index');fig.tight_layout();fig.savefig(OUT/'plots/constraint_steps.png',dpi=150);plt.close(fig)
    receipt=json.loads((OUT/'training_receipt.json').read_text());final=json.loads((RUN/'hard_constraint_final.json').read_text())
    timing=dict(total_actor_optimizer_seconds=sum(s['total_step_seconds'] for s in logs),constraint_check_seconds=sum(s['constraint_seconds'] for s in logs),adam_candidate_seconds=sum(s['adam_seconds'] for s in logs),candidate_checks=sum(len(s['trials']) for s in steps),accepted_ratios=[s['accepted_ratio'] for s in steps if s['actor_accepted']],max_kl=max(s['kl'] for s in steps),mean_ppo_loss=float(np.mean([s['ppo_loss'] for s in steps])))
    result=dict(status='protection_passed_limited_updates' if passed else 'protection_failed_stopped',gates=gates,training=receipt,final_pool=final,timing=timing,rows=rows,no_more_training=True,full_motion_seconds=64.14,frames=3207)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    protocol=json.loads((OUT/'protocol.json').read_text());protocol['status']=result['status'];protocol['gates_result']=gates;(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    if not passed:(RUN/'STOPPED.json').write_text(json.dumps(dict(reason='Full 77 protection condition failed; no additional training authorized',gates=gates),indent=2))
    print(json.dumps(dict(status=result['status'],gates=gates,timing=timing,main_rows=[r for r in rows if r['label'] in ['heading_300','hard26','unconstrained26']]),indent=2))

if __name__=='__main__':
    import sys
    if '--probes' in sys.argv:probes()
    else:summarize()
