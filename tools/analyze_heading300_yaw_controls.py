"""Replay and critic probes plus saved simulation comparisons; no optimization."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'legged_gym/scripts')]
import numpy as np
import torch
from export_twist_anyadapter_opentrack_jit import build_actor
from heading300_training_diagnostics import residual_groups
from sim_eval_metrics import phase_masks,rms
from summarize_heading300_sampling import flatten

REPORT=ROOT/'reports/heading300_yaw_controls_20261010'
SOURCE=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'
WEIGHTED=ROOT/'legged_gym/logs/recovery_heading300_sampling/sample_only_alpha014_p1_env512_seed42_20261009/model_50.pt'
CONTROL=ROOT/'legged_gym/logs/recovery_heading300_yaw_controls/original_env512_seed42_20261010'


def write_csv(path,records):
    if not records:return
    fields=list(records[0])+sorted({k for r in records for k in r}-set(records[0]))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)


def phase_yaw(row):
    frames=[json.loads(x) for x in Path(row['trace']).read_text().splitlines()]
    reference=np.array([x['clean'] for x in frames]);actual=np.array([x['rpy'][2] for x in frames])
    error=np.arctan2(np.sin(actual-reference[:,3]),np.cos(actual-reference[:,3]))
    assert abs(rms(error)-row['metrics']['yaw_error'])<1e-10
    return {phase:math.degrees(rms(error[mask])) for phase,mask in phase_masks(reference).items() if mask.any()}


def analyze(output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    cross=json.loads((REPORT/'cross_eval/summary.json').read_text());assert cross['status']=='complete'
    cross_rows=[];phase_rows=[]
    for row in cross['rows']:
        if row['role']!='candidate':continue
        cross_rows.append(flatten(row,0,REPORT/'cross_eval'))
        phase_rows.append(dict(label=row['label'],reference_mode=row['reference_mode'],**phase_yaw(row)))
    write_csv(output/'cross_ablation.csv',cross_rows)
    by={(x['label'],x['reference_mode']):x for x in cross_rows}
    contrasts={mode:dict(core_with_old_head=by['B',mode]['yaw_rmse_degrees']-by['A',mode]['yaw_rmse_degrees'],
                        core_with_new_head=by['D',mode]['yaw_rmse_degrees']-by['C',mode]['yaw_rmse_degrees'],
                        head_with_old_core=by['C',mode]['yaw_rmse_degrees']-by['A',mode]['yaw_rmse_degrees'],
                        head_with_new_core=by['D',mode]['yaw_rmse_degrees']-by['B',mode]['yaw_rmse_degrees'],
                        interaction=by['D',mode]['yaw_rmse_degrees']-by['B',mode]['yaw_rmse_degrees']-by['C',mode]['yaw_rmse_degrees']+by['A',mode]['yaw_rmse_degrees']) for mode in ['raw','wm']}
    control_rows=[];evaluations=[]
    paths=list(CONTROL.glob('evaluations/update_*/summary.json'))+list(REPORT.glob('original*_after_crash/summary.json'))
    for path in paths:
        try:summary=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if summary['status']!='complete':continue
        manifest=json.loads((path.parent/'manifest.json').read_text())
        update=manifest['exports'][0]['iteration']
        evaluations.append(dict(update=update,report=str(path.parent),comparisons=summary['comparisons']))
        for row in summary['rows']:
            if row['role']!='candidate':continue
            flat=flatten(row,update,path.parent);flat['policy_updates']=max(0,update-25);control_rows.append(flat)
            phase_rows.append(dict(label='original_'+str(update),reference_mode=row['reference_mode'],**phase_yaw(row)))
    write_csv(output/'original_control.csv',control_rows);write_csv(output/'phase_yaw.csv',phase_rows)
    final_update=max(x['update'] for x in control_rows)
    cp=CONTROL/f'model_{final_update}.pt'
    actors={name:build_actor(path) for name,path in [('heading300',SOURCE),('weighted50',WEIGHTED),('original'+str(final_update),cp)]}
    # The same union of raw/WM replay states is used for every policy probe.
    probes=[]
    for path in [WEIGHTED,cp]:
        checkpoint=torch.load(path,map_location='cpu')
        for group,data in checkpoint['recovery1250']['replay']['groups'].items():
            probes.append(data['inputs'][:512])
    x=torch.cat(probes);base,z,context=x[:,:1155],x[:,1155:1283],x[:,-2:]
    drift=[]
    with torch.no_grad():
        old_core=actors['heading300'].layerwise_actor(base,z)
        old_head=actors['heading300'].heading_delta(z,context)
        for name,actor in actors.items():
            core=actor.layerwise_actor(base,z);head=actor.heading_delta(z,context);delta=(core-old_core)+(head-old_head)
            drift.append(dict(label=name,frames=len(x),core_rms=float((core-old_core).square().mean().sqrt()),
                              heading_rms=float((head-old_head).square().mean().sqrt()),
                              full_rms=float(delta.square().mean().sqrt()),full_mae=float(delta.abs().mean()),
                              source_heading_rms=float(old_head.square().mean().sqrt()),
                              full_leg_rms=float(delta[:,:12].square().mean().sqrt())))
    write_csv(output/'common_state_action_drift.csv',drift)
    critic_rows=[];bad_archives=[]
    for path in sorted((CONTROL/'critic_archives').glob('*.pt')):
        try:archive=torch.load(path,map_location='cpu')
        except Exception as exc:bad_archives.append(dict(path=str(path),error=repr(exc)));continue
        shape=archive['values'].shape;obs=archive['critic_obs'];gamma=archive['gamma']
        with torch.no_grad():
            for name,actor in actors.items():
                value=torch.cat([actor.evaluate(c) for c in obs.split(2048)]).reshape(shape)
                last=actor.evaluate(archive['last_critic_obs'])
                nxt=torch.cat((value[1:],last.unsqueeze(0)),0)
                td=archive['rewards_raw']+gamma*archive['timeouts']*value+gamma*(1-archive['dones'].float())*nxt-value
                gae=archive['gae_returns']-value
                for row in residual_groups(archive['context'],archive['modes'],td_residual=td,common_gae_residual=gae):
                    critic_rows.append(dict(probe_update=archive['update'],critic=name,**row))
    write_csv(output/'common_state_critic_residuals.csv',critic_rows)
    result=dict(cross=cross_rows,contrasts_degrees=contrasts,original_control=control_rows,
                control_evaluations=evaluations,phase_yaw=phase_rows,common_state_action_drift=drift,
                common_state_critic_residuals=critic_rows,bad_archives=bad_archives,
                cautions=['single seed; checkpoint-specific factorial effects, not general causal ranking',
                          'weighted50 and earlier original checkpoint have different policy-update dose',
                          'critic TD uses native timeout bootstrap; GAE targets fixed within each common probe but not ground truth',
                          'no4096-env control, so batch-size causality is not identified',
                          'crash-damaged diagnostics excluded, not imputed'])
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps(dict(final_update=final_update,contrasts=contrasts,drift=drift,bad_archives=bad_archives),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args().output)
