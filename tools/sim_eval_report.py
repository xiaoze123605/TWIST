"""Matched-condition gates and reports for the shared simulation diagnostics."""
import csv
import json
import math
from pathlib import Path


def relative_change(value, reference):
    if value is None or reference is None:
        return None
    if abs(reference) <= 1e-12:
        return 0. if abs(value-reference) <= 1e-12 else None
    return (value-reference)/abs(reference)


def phase_value(row, phase, name):
    return row.get('analysis', {}).get('phases', {}).get(phase, {}).get(name)


def aggregate(rows, phase, field, spectral=False):
    pairs = []
    for row in rows:
        stats = row.get('analysis', {}).get('phases', {}).get(phase, {})
        value = stats.get(field)
        weight = stats.get('seconds', 0.)
        if spectral:
            value = stats.get('actual_spectrum', {}).get(field)
            weight = stats.get('actual_spectrum', {}).get('valid_seconds', 0.)
        if value is not None and weight > 0:
            pairs.append((value, weight))
    if not pairs:
        return None
    return math.sqrt(sum(v*v*w for v,w in pairs)/sum(w for _,w in pairs))


def compare_models(rows, reference_modes):
    baseline = {(r['case'], r['reference_mode']): r for r in rows
                if r['role'] == 'baseline'}
    labels = sorted({r['label'] for r in rows if r['role'] == 'candidate'})
    global_failure = any(r['status'] != 'complete' for r in rows)
    comparisons = {}
    for label in labels:
        selected = [r for r in rows if r['label'] == label]
        pairs = []
        for row in selected:
            base = baseline.get((row['case'], row['reference_mode']))
            valid = (row['status'] == 'complete' and base is not None and
                     base['status'] == 'complete' and
                     row['condition_signature'] == base['condition_signature'] and
                     row['analysis']['reference_digest'] == base['analysis']['reference_digest'] and
                     row['analysis']['processed_reference_digest'] == base['analysis']['processed_reference_digest'])
            pairs.append((row, base, valid))
        matched = bool(pairs) and all(valid for _,_,valid in pairs)
        changes = []
        if matched:
            for row,base,_ in pairs:
                changes.append(dict(case=row['case'], reference_mode=row['reference_mode'],
                    yaw_change=relative_change(row['metrics']['yaw_error'], base['metrics']['yaw_error']),
                    velocity_change=relative_change(row['metrics']['root_velocity_rmse'], base['metrics']['root_velocity_rmse']),
                    joint_change=relative_change(row['metrics']['joint_rmse'], base['metrics']['joint_rmse']),
                    pitch_change=relative_change(row['analysis']['metrics']['torso_pitch_rmse_degrees'],base['analysis']['metrics']['torso_pitch_rmse_degrees']),
                    squat_coordination_change=relative_change(phase_value(row,'squat','hip_knee_ankle_rmse'),phase_value(base,'squat','hip_knee_ankle_rmse')),
                    walking_high_frequency_change=relative_change(phase_value(row,'walking','actual_spectrum').get('high_5_20hz_rms') if phase_value(row,'walking','actual_spectrum') else None,
                        phase_value(base,'walking','actual_spectrum').get('high_5_20hz_rms') if phase_value(base,'walking','actual_spectrum') else None)))
        gates = dict(complete_evaluation=not global_failure and matched,
                     no_new_falls=None, yaw_within_5_percent=None,
                     each_case_velocity_within_5_percent=None,
                     squat_coordination_improved=None, walking_high_frequency_improved=None)
        improvements = {}
        if matched:
            gates['no_new_falls'] = all(not r['metrics']['fall_proxy'] or b['metrics']['fall_proxy'] for r,b,_ in pairs)
            gates['yaw_within_5_percent'] = all(r['metrics']['yaw_error'] <= 1.05*b['metrics']['yaw_error']+1e-12 for r,b,_ in pairs)
            gates['each_case_velocity_within_5_percent'] = all(r['metrics']['root_velocity_rmse'] <= 1.05*b['metrics']['root_velocity_rmse']+1e-12 for r,b,_ in pairs)
            for mode in reference_modes:
                candidates = [r for r in selected if r['reference_mode'] == mode]
                bases = [baseline[(r['case'],mode)] for r in candidates]
                coord = aggregate(candidates,'squat','hip_knee_ankle_rmse')
                base_coord = aggregate(bases,'squat','hip_knee_ankle_rmse')
                hf = aggregate(candidates,'walking','high_5_20hz_rms',True)
                base_hf = aggregate(bases,'walking','high_5_20hz_rms',True)
                improvements[mode] = dict(squat_coordination=coord,baseline_squat_coordination=base_coord,
                    coordination_change=relative_change(coord,base_coord),walking_high_frequency=hf,
                    baseline_walking_high_frequency=base_hf,high_frequency_change=relative_change(hf,base_hf))
            for gate,field,base_field in [('squat_coordination_improved','squat_coordination','baseline_squat_coordination'),('walking_high_frequency_improved','walking_high_frequency','baseline_walking_high_frequency')]:
                values = [(x[field],x[base_field]) for x in improvements.values()]
                gates[gate] = None if any(a is None or b is None for a,b in values) else all(a < b-1e-10 for a,b in values)
        accepted = all(value is True for value in gates.values())
        tradeoffs = [dict(case=x['case'],reference_mode=x['reference_mode'],metric=k,change=v)
                    for x in changes for k,v in x.items()
                    if k.endswith('_change') and v is not None and v > 1e-6]
        if global_failure or not matched:
            status = 'failed_or_unmatched'
        elif accepted:
            status = 'meets_gates_with_tradeoffs' if tradeoffs else 'meets_gates_on_evaluated_scope'
        elif any(v is None for v in gates.values()):
            status = 'inconclusive_missing_phases'
        elif changes and all(v is None or abs(v)<1e-10 for x in changes for k,v in x.items() if k.endswith('_change')):
            status = 'unchanged'
        else:
            status = 'not_accepted'
        comparisons[label] = dict(status=status,accepted=accepted,gates=gates,
                                  by_reference_mode=improvements,case_changes=changes,tradeoffs=tradeoffs,
                                  claim='Gate result on selected cases only; not a claim of overall superiority.')
    return comparisons


def display(value):
    if value is None:
        return 'N/A'
    if isinstance(value,bool):
        return 'yes' if value else 'no'
    return f'{value:.6g}' if isinstance(value,(float,int)) else str(value)


def write_report(output, summary):
    output = Path(output)
    rows = summary['rows']
    columns = ['label','role','case','reference_mode','status','seed','target_projection','ema',
               'joint_rmse','joint_velocity_rmse_estimated','root_velocity_rmse','yaw_degrees',
               'torso_pitch_rmse_degrees','projection_frame_fraction','torque_saturation_fraction',
               'fall_proxy','error']
    with (output/'metrics.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
        writer.writeheader()
        for row in rows:
            metric = dict(row.get('metrics',{}),**row.get('analysis',{}).get('metrics',{}))
            flat = {k:row.get(k) for k in columns if k in row}
            flat.update({k:metric.get(k) for k in columns if k in metric})
            if metric.get('yaw_error') is not None:
                flat['yaw_degrees'] = math.degrees(metric['yaw_error'])
            writer.writerow(flat)
    phases = []
    for row in rows:
        for name,values in row.get('analysis',{}).get('phases',{}).items():
            flat = dict(label=row['label'],case=row['case'],reference_mode=row['reference_mode'],phase=name)
            for key,value in values.items():
                if isinstance(value,(int,float)) or value is None:
                    flat[key] = value
                elif isinstance(value,dict):
                    for k,v in value.items():
                        if isinstance(v,(int,float)):
                            flat[key+'.'+k] = v
            flat.update({'contact_transition.'+k:v for k,v in row['analysis']['contact_transition'].items()})
            phases.append(flat)
    fields = ['label','case','reference_mode','phase']+sorted({k for r in phases for k in r}-{'label','case','reference_mode','phase'})
    with (output/'phases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(phases)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines = ['# TWIST simulation evaluation', '', 'Status: **'+summary['status']+'**', '',
             'Scope: '+summary['mode']+'; '+str(len(summary['cases']))+' motions; reference modes '+', '.join(summary['reference_modes'])+'.',
             'heading_300 is the matched-condition baseline. Historical23500 has target projection OFF and is excluded from acceptance gates.',
             'No training or hardware control was performed. Missing phases/spectra are N/A, not zero.', '',
             '|Policy|Role|Motion|Reference|Joint RMSE|Root velocity RMSE|Yaw (deg)|Fall|Status|',
             '|---|---|---|---|---:|---:|---:|---|---|']
    for r in rows:
        m=r.get('metrics',{})
        lines.append('|'+ '|'.join(map(display,[r['label'],r['role'],r['case'],r['reference_mode'],m.get('joint_rmse'),m.get('root_velocity_rmse'),math.degrees(m['yaw_error']) if 'yaw_error' in m else None,m.get('fall_proxy'),r['status']]))+'|')
    for label,c in summary['comparisons'].items():
        lines += ['', '## '+label, '', '**'+c['status']+'**; accepted on this scope: '+display(c['accepted'])+'.',
                  'This result does not establish overall superiority.', '', '|Gate|Result|','|---|---|']
        lines += ['|'+k+'|'+display(v)+'|' for k,v in c['gates'].items()]
        if c.get('by_reference_mode'):
            lines += ['', '|Reference|Squat coordination baseline / candidate (rad)|Walking HF baseline / candidate (rad)|', '|---|---:|---:|']
            for mode,values in c['by_reference_mode'].items():
                lines.append('|'+mode+'|'+display(values['baseline_squat_coordination'])+' / '+display(values['squat_coordination'])+'|'+display(values['baseline_walking_high_frequency'])+' / '+display(values['walking_high_frequency'])+'|')
        if c['tradeoffs']:
            lines += ['', 'Regressions (including monitored metrics outside acceptance gates):']
            lines += ['- '+x['case']+'/'+x['reference_mode']+': '+x['metric']+' '+f"{100*x['change']:+.2f}%" for x in c['tradeoffs']]
    for r in rows:
        if r['status'] != 'complete':
            lines += ['', '- Failed '+r['label']+'/'+r['case']+'/'+r['reference_mode']+': '+r.get('error','unknown')]
    lines += ['', '## Artifacts', '', '- [Metrics CSV](metrics.csv)', '- [Phase CSV](phases.csv)', '- [Full JSON](summary.json)', '- [Conditions and source hashes](manifest.json)', '']
    for path in sorted((output/'plots').rglob('*.png')) if (output/'plots').exists() else []:
        relative=path.relative_to(output).as_posix()
        lines.append('- ['+relative+']('+relative+')')
    lines += ['', 'Joint velocity reference is estimated by numpy.gradient(clean joint angles, 0.02 s); it is not the motion-library smoothed velocity.',
              'Phase masks and 64-point Welch/5–20 Hz definitions are shared with heading300 diagnosis. COM margin is a static contact-hull diagnostic.',
              'Quick or filtered cases do not constitute full multi-motion validation. All comparisons preserve raw/WM, seed, processed reference and control signature.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def plot_case(output, case_id, mode, arrays, spectra, traces=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    specfig,specaxes=plt.subplots(1,2,figsize=(11,4))
    for label,cache in arrays.items():
        ref,q,pitch,refpitch,_,_,com,refcom,masks,_,_,act=cache
        t=np.arange(len(ref))*.02
        axes[0].plot(t,np.degrees(pitch),label=label,lw=.8)
        axes[1].plot(t,(com-refcom)*100,lw=.8)
        axes[2].plot(t,np.linalg.norm(np.diff(act[:,:12],axis=0,prepend=act[:1,:12]),axis=-1)/.02,lw=.7)
        for ax,name in zip(specaxes,['actual','policy_post_ema']):
            if (label,name) in spectra:
                f,p=spectra[label,name];ax.semilogy(f,p.mean(-1),label=label)
    if arrays:
        axes[0].plot(t,np.degrees(refpitch),'k--',label='clean reference')
    for ax,text in zip(axes,['Torso pitch (deg)','COM forward error (cm)','Leg action rate norm']):
        ax.set_ylabel(text);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[-1].set_xlabel('Reference time (s)')
    fig.suptitle(case_id+'/'+mode);fig.tight_layout();fig.savefig(output/(case_id+'__'+mode+'__curves.png'),dpi=130);plt.close(fig)
    for ax,name in zip(specaxes,['Actual leg angles','Policy leg actions']):
        ax.set_title(name);ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('PSD');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    specfig.tight_layout();specfig.savefig(output/(case_id+'__'+mode+'__spectra.png'),dpi=130);plt.close(specfig)

    # Reuse the existing detailed 23-joint/root/Yaw plot exporter. Its auxiliary
    # warmup-window summaries are visualization artifacts, never acceptance data.
    if traces:
        from plot_sim_trace import export
        seen={}
        for label,trace in traces.items():
            trace=str(Path(trace).resolve())
            if trace in seen:
                continue
            seen[trace]=label
            export(trace,output/'details'/case_id/mode/label)
