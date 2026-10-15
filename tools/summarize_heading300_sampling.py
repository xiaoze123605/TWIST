"""Summarize saved evaluations without rerunning or changing diagnostic metrics."""
import argparse
import csv
import json
import math
from pathlib import Path

from sim_eval_report import relative_change


def flatten(row, update, evaluation):
    metrics = row.get('metrics', {})
    analysis = row.get('analysis', {})
    phases = analysis.get('phases', {})
    squat, walk = phases.get('squat', {}), phases.get('walking', {})
    knee = squat.get('knee_rmse')
    return dict(
        update=update, label=row['label'], role=row['role'], case=row['case'],
        reference_mode=row['reference_mode'], status=row['status'],
        target_projection=row['target_projection'], seed=row['seed'], ema=row['ema'],
        joint_rmse=metrics.get('joint_rmse'),
        joint_velocity_rmse=analysis.get('metrics', {}).get('joint_velocity_rmse_estimated'),
        root_velocity_rmse=metrics.get('root_velocity_rmse'),
        yaw_rmse_degrees=math.degrees(metrics['yaw_error']) if 'yaw_error' in metrics else None,
        squat_knee_rmse_degrees=math.degrees(knee) if knee is not None else None,
        squat_coordination_rmse=squat.get('hip_knee_ankle_rmse'),
        squat_pitch_error_degrees=squat.get('pitch_rmse_degrees'),
        squat_actual_pitch_degrees=squat.get('actual_pitch_mean_degrees'),
        squat_reference_pitch_degrees=squat.get('reference_pitch_mean_degrees'),
        squat_height_rmse=squat.get('height_rmse'),
        squat_com_margin_cm=squat.get('static_contact_margin_mean_cm'),
        walking_high_frequency_rms=walk.get('actual_spectrum', {}).get('high_5_20hz_rms'),
        walking_ankle_pitch_high_frequency_rms=walk.get('ankle_pitch_high_5_20hz_rms'),
        contact_switch_action_rate_rms=analysis.get('contact_transition', {}).get('near_action_rate_rms'),
        projection_frame_fraction=analysis.get('metrics', {}).get('projection_frame_fraction'),
        torque_saturation_fraction=analysis.get('metrics', {}).get('torque_saturation_fraction'),
        fall_proxy=metrics.get('fall_proxy'), evaluation=str(evaluation), trace=row.get('trace'))


def summarize(run, output):
    run = run.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, evaluations, coverage = [], [], []
    seen_baselines = set()
    for path in sorted((run/'evaluations').glob('update_*/summary.json')):
        summary = json.loads(path.read_text())
        update = int(path.parent.name.split('_')[1])
        evaluations.append(dict(update=update, path=str(path.parent),
                                status=summary['status'], comparisons=summary['comparisons'],
                                cases=summary.get('cases'), filtered_scope=summary.get('filtered_scope')))
        for row in summary.get('rows', []):
            if 'metrics' not in row:
                continue
            if row['role'] != 'candidate':
                key = row['label'], row['case'], row['reference_mode']
                if key in seen_baselines:
                    continue
                seen_baselines.add(key)
            flat = flatten(row, update if row['role']=='candidate' else 0, path.parent)
            base = next((r for r in summary['rows'] if r['role']=='baseline' and
                         r['case']==row['case'] and r['reference_mode']==row['reference_mode']), None)
            if base and row['role']=='candidate':
                reference = flatten(base, 0, path.parent)
                flat.update({k+'_change':relative_change(flat[k], reference[k]) for k in (
                    'joint_rmse','root_velocity_rmse','yaw_rmse_degrees','squat_knee_rmse_degrees',
                    'squat_coordination_rmse','squat_pitch_error_degrees','walking_high_frequency_rms')})
            rows.append(flat)
    metrics_path = run/'anchor_metrics.jsonl'
    training = [json.loads(x) for x in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    for start in range(0, len(training), 50):
        block=training[start:start+50]
        coverage.append(dict(first_update=block[0]['update'],last_update=block[-1]['update'],
            **{key:sum(r[key] for r in block)/len(block) for key in block[0] if key.startswith('sampling_')},
            max_kl=max(r['max_rollout_kl'] for r in block),
            skipped_minibatches=sum(r['skipped_minibatches'] for r in block),
            last_source_action_mae=block[-1]['source_action_mae']))
    for name, records in [('metrics.csv', rows), ('sampling_coverage.csv', coverage)]:
        if records:
            fields=list(records[0])+sorted({k for row in records for k in row}-set(records[0]))
            with (output/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
    stopped = json.loads((run/'STOPPED.json').read_text()) if (run/'STOPPED.json').exists() else None
    result=dict(run=str(run),completed_updates=training[-1]['update'] if training else 0,
                stopped=stopped, evaluations=evaluations, training_coverage=coverage, rows=rows,
                historical_control_difference='23500: gain1, 7001D, target projection OFF; excluded from acceptance gates',
                metric_definitions='Unchanged sim_eval_metrics phases and 64-point Welch 5-20 Hz; yaw RMSE; knees combined left/right RMSE')
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    lines=['# heading300 sampling-only experiment', '',
           'Completed updates: '+str(result['completed_updates'])+'.',
           'Historical23500 uses projection OFF and is not a matched-condition acceptance baseline.', '',
           '|Update|Policy|Reference|Joint RMSE (rad)|Squat knees (deg)|Squat pitch error (deg)|Yaw RMSE (deg)|Root velocity RMSE|Walking HF (rad)|Fall|',
           '|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(x):
        return 'N/A' if x is None else f'{x:.6g}' if isinstance(x,(float,int)) else str(x)
    for row in rows:
        lines.append('|'+ '|'.join(fmt(row.get(k)) for k in ('update','label','reference_mode','joint_rmse',
             'squat_knee_rmse_degrees','squat_pitch_error_degrees','yaw_rmse_degrees','root_velocity_rmse',
             'walking_high_frequency_rms','fall_proxy'))+'|')
    for evaluation in evaluations:
        lines.extend(['', f"Update {evaluation['update']}: [{evaluation['status']}]({evaluation['path']}/report.md)"])
        for label, comparison in evaluation['comparisons'].items():
            lines.append(f"- {label}: {comparison['status']}; accepted={comparison['accepted']}; gates={comparison['gates']}")
    if stopped:
        lines.extend(['', 'Stopped: '+stopped['reason']])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(2,3,figsize=(15,8))
        for ax,key in zip(axes.flat, ('joint_rmse','squat_knee_rmse_degrees','squat_pitch_error_degrees',
                                    'yaw_rmse_degrees','root_velocity_rmse','walking_high_frequency_rms')):
            for mode in ['raw','wm']:
                selected=[r for r in rows if r['role']=='candidate' and r['reference_mode']==mode and r['case']=='mocap1__77']
                ax.plot([r['update'] for r in selected],[r[key] for r in selected],'-o',label=mode)
                base=next((r for r in rows if r['role']=='baseline' and r['reference_mode']==mode and r['case']=='mocap1__77'),None)
                if base and base[key] is not None:ax.axhline(base[key],ls='--',alpha=.6,label='heading300 '+mode)
            ax.set_title(key);ax.set_xlabel('Additional updates');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(output/'checkpoint_comparison.png',dpi=150);plt.close(fig)
    print(output/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summarize(args.run_dir,args.output)
