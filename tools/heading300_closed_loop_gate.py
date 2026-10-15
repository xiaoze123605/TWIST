"""User-confirmed improvement gates layered on the existing simulation report."""
import argparse
import json
from pathlib import Path


def assess(summary,label):
    comparison=summary['comparisons'][label]
    guards={k:comparison['gates'].get(k) for k in ('complete_evaluation','no_new_falls',
                'yaw_within_5_percent','each_case_velocity_within_5_percent')}
    rows=[r for r in summary['rows'] if r['label']==label]
    baseline={(r['case'],r['reference_mode']):r for r in summary['rows'] if r['role']=='baseline'}
    changes=[];improvement_gates={}
    for mode in ['raw','wm']:
        selected=[r for r in rows if r['case']=='mocap1__77' and r['reference_mode']==mode]
        if len(selected)!=1:
            improvement_gates[mode]=None;continue
        row=selected[0];base=baseline.get((row['case'],mode))
        if not base or row['status']!='complete':improvement_gates[mode]=None;continue
        values={}
        for name,phase,key,spectral in [('squat_knee','squat','knee_rmse',False),
              ('squat_coordination','squat','hip_knee_ankle_rmse',False),
              ('walking_high_frequency','walking','high_5_20hz_rms',True)]:
            def value(r):
                p=r['analysis']['phases'].get(phase,{})
                return p.get('actual_spectrum',{}).get(key) if spectral else p.get(key)
            candidate,reference=value(row),value(base)
            values[name]=dict(candidate=candidate,baseline=reference,
                 relative_change=(candidate/reference-1) if candidate is not None and reference else None,
                 improves_at_least10_percent=(candidate<=.9*reference) if candidate is not None and reference else None)
        changes.append(dict(reference_mode=mode,metrics=values))
        improvement_gates[mode]=all(x['improves_at_least10_percent'] is True for x in values.values())
    multimotion=len(summary.get('cases',[]))>=9 and not summary.get('filtered_scope',True)
    return dict(label=label,guard_pass=all(v is True for v in guards.values()),guards=guards,
                improvement_by_mode=improvement_gates,changes=changes,
                motion77_thresholds_pass=all(v is True for v in guards.values()) and all(v is True for v in improvement_gates.values()),
                multi_motion_validated=multimotion,
                accepted=all(v is True for v in guards.values()) and all(v is True for v in improvement_gates.values()) and multimotion,
                note='77 improvement alone is insufficient; multi-motion regressions must also pass. Missing/failed metrics never pass.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=assess(json.loads(a.summary.read_text()),a.label)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
