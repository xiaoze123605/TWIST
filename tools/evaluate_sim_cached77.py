"""Evaluate a new checkpoint with strictly validated existing77 baseline traces."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

from evaluate_sim import ROOT,BASELINE,HISTORY,sha,export_checkpoint,condition_signature
from diagnose_heading300_sim import run_diagnostic_case
from sim_eval_metrics import analyze_trace
from sim_eval_report import compare_models,plot_case,write_report


def evaluate(a):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    cache=json.loads((a.cache/'summary.json').read_text())
    manifest=json.loads((a.cache/'manifest.json').read_text())
    if cache['status']!='complete':raise ValueError('incomplete cache')
    if manifest['seed']!=42 or manifest['ema']!=0 or not manifest['target_projection']:
        raise ValueError('cache control conditions differ')
    for path,digest in manifest['code'].items():
        if sha(ROOT/path)!=digest:raise ValueError('cache code changed: '+path)
    cases=[c for c in manifest['cases'] if c['id']=='mocap1__77']
    if len(cases)!=1 or cases[0]['seconds']!=64.14:raise ValueError('cache requires complete77')
    case=cases[0]
    if sha(Path(case['motion']))!=case['sha256']:raise ValueError('reference changed')
    for label,path in [('heading_300',BASELINE),('historical23500',HISTORY)]:
        model=next(m for m in manifest['models'] if m['label']==label)
        if sha(path)!=model['sha256']:raise ValueError('baseline policy changed')
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        dest=a.output/'exports'/a.label/'policy.pt';dest.parent.mkdir(parents=True)
        export=export_checkpoint(a.checkpoint,dest,a.update)
        models=[m for m in manifest['models'] if m['role']!='candidate']
        models.append(dict(label=a.label,role='candidate',path=str(dest),projection=True,sha256=sha(dest)))
        receipt=dict(manifest,models=models,exports=[export],cached_baselines=True,
                     cache=str(a.cache.resolve()),cache_summary_sha256=sha(a.cache/'summary.json'),
                     cache_trace_hashes={},wrapper_sha256=sha(Path(__file__)))
        rows=[];arrays={};spectra={}
        for row in cache['rows']:
            if row['case']!='mocap1__77' or row['role']=='candidate':continue
            row=copy.deepcopy(row);analysis,data,spec=analyze_trace(row['metrics'],row['label'])
            if analysis['reference_digest']!=row['analysis']['reference_digest'] or analysis['processed_reference_digest']!=row['analysis']['processed_reference_digest']:
                raise ValueError('cached trace/reference changed')
            if json.dumps(analysis, sort_keys=True) != json.dumps(row['analysis'], sort_keys=True):
                raise ValueError('cached trace metrics changed')
            mode=row['reference_mode'];rows.append(row)
            receipt['cache_trace_hashes'][row['trace']]=sha(Path(row['trace']))
            arrays.setdefault(mode,{})[row['label']]=data;spectra.setdefault(mode,{}).update(spec)
        if len(rows)!=4:raise ValueError('requires both raw/wm baselines and historical controls')
        for mode in ['raw','wm']:
            result=run_diagnostic_case(case,mode,dest,a.output/'runs'/a.label/case['id']/mode,seed=42,ema=0,projection=True)
            analysis,data,spec=analyze_trace(result,a.label)
            row=dict(label=a.label,role='candidate',case=case['id'],reference_mode=mode,seed=42,ema=0.,
                     target_projection=True,condition_signature=condition_signature(a,case,mode,True,manifest['code']),
                     policy_sha256=sha(dest),status='complete',metrics=result,analysis=analysis,trace=result['trace'])
            rows.append(row);arrays[mode][a.label]=data;spectra[mode].update(spec)
            print('Complete',a.label,mode,flush=True)
        for path,digest in manifest['code'].items():
            if sha(ROOT/path)!=digest:raise ValueError('code changed during evaluation')
        for path,digest in receipt['cache_trace_hashes'].items():
            if sha(Path(path))!=digest:raise ValueError('cached trace changed during evaluation')
        summary=dict(schema_version=1,status='complete',mode='standard',cases=cases,filtered_scope=True,
                     reference_modes=['raw','wm'],rows=rows,comparisons=compare_models(rows,['raw','wm']),
                     cached_baselines=True)
        (a.output/'manifest.json').write_text(json.dumps(receipt,indent=2))
        for mode in ['raw','wm']:
            traces={r['label']:r['trace'] for r in rows if r['reference_mode']==mode}
            plot_case(a.output/'plots',case['id'],mode,arrays[mode],spectra[mode],traces=traces)
        write_report(a.output,summary);(a.output/'COMPLETE').write_text('6 rows; reused baseline/historical,2 new CPU simulations')
    except Exception as exc:
        (a.output/'FAILED').write_text(repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--update',type=int,required=True);p.add_argument('--label',required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.seed=42;a.ema=0.
    if a.label in ('heading_300','historical23500'):p.error('reserved label')
    evaluate(a)
