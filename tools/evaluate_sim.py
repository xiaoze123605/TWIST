"""Unified CPU-only TWIST simulation evaluation; auto evaluation is opt-in."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT/'reports/recovery_1250/fix_stage_20261008/heading_300.pt'
HISTORY = ROOT/'legged_gym/logs/recovery_1250/coordinated_1250_seed42_20260926/traced/model_23500-anyadapter-opentrack-jit.pt'
PROTOCOL = ROOT/'reports/recovery_4900/protocol.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_policy(value, index):
    label,path = value.split('=',1) if '=' in value else ('candidate_'+str(index),value)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',label) or label in ('heading_300','historical23500'):
        raise ValueError('invalid/reserved policy label: '+label)
    return label,Path(path).expanduser().resolve()


def select_cases(mode, protocol, requested=None):
    selected = [dict(c) for c in protocol['cases'] if mode == 'full' or c['phase'] == 'development']
    motion77 = dict(next(c for c in protocol['cases'] if c['id'] == 'mocap1__77'))
    motion77['seconds'] = 20. if mode == 'quick' else 64.14
    if mode == 'quick':
        selected = [motion77]
    else:
        selected = [c for c in selected if c['id'] != motion77['id']]+[motion77]
    if requested:
        unknown = set(requested)-{c['id'] for c in selected}
        if unknown:
            raise ValueError('cases outside selected mode: '+', '.join(sorted(unknown)))
        selected = [c for c in selected if c['id'] in requested]
    return sorted(selected,key=lambda c:c['id'])


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',action='append',default=[],help='JIT path or label=path; repeat for comparison')
    p.add_argument('--checkpoint',action='append',default=[],help='Training checkpoint to export on CPU, or label=path')
    p.add_argument('--mode',choices=['quick','standard','full'],default='standard')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-modes',nargs='+',choices=['raw','wm'],default=['raw','wm'])
    p.add_argument('--cases',nargs='+',help='Optional subset; explicitly reported as filtered scope')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--ema',type=float,default=0.)
    p.add_argument('--no-target-projection',action='store_true')
    p.add_argument('--workers',type=int,choices=[1,2],default=1)
    p.add_argument('--dry-run',action='store_true',help='Save a plan only; never counts as passing evaluation')
    p.add_argument('--first-update-diagnostics',action='store_true',help='Read-only actor input and reference-WM history recording')
    p.add_argument('--no-historical-control',action='store_true',help='Omit the optional unprojected23500 historical control')
    p.add_argument('--auto-eval',action='store_true',help='Explicitly enable a serial checkpoint watcher')
    p.add_argument('--watch-dir',type=Path)
    p.add_argument('--eval-interval',type=int,default=50)
    p.add_argument('--poll-seconds',type=float,default=5.)
    p.add_argument('--watch-once',action='store_true',help='Two completion scans, then wait for at most one worker')
    p.add_argument('--expected-update',type=int,help=argparse.SUPPRESS)
    p.add_argument('--auto-lock-fd',type=int,help=argparse.SUPPRESS)
    return p


def cpu_environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def export_checkpoint(source, dest, expected_update=None):
    # Import torch only in this evaluation process, never in the training hook.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    before=sha(source)
    state=torch.load(source,map_location='cpu')
    if sha(source)!=before:
        raise ValueError('checkpoint changed during complete-state validation')
    if not isinstance(state,dict) or 'model_state_dict' not in state:
        raise ValueError('checkpoint is not a complete model state')
    if expected_update is not None and state.get('iter') != expected_update:
        raise ValueError('checkpoint counter does not match completed update')
    subprocess.run([sys.executable,str(ROOT/'legged_gym/scripts/export_twist_anyadapter_opentrack_jit.py'),
                    str(source),'--output',str(dest)],env=cpu_environment(),check=True)
    if sha(source)!=before:
        raise ValueError('checkpoint replaced during export')
    return dict(source=str(source),source_sha256=before,iteration=state.get('iter'),jit=str(dest),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_cuda_initialized=torch.cuda.is_initialized())


def condition_signature(settings, case, mode, projection, code):
    values=dict(seed=settings.seed,ema=settings.ema,target_projection=projection,
                projection_margin=.02,projection_knee_min=0.,reference_mode=mode,
                case=case,control_dt=.02,sim_dt=.001,action_scale=.5,
                xml_sha256=code['assets/g1/g1_sim2sim_with_wrist_roll.xml'],code=code)
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def evaluate(args):
    if getattr(args,'first_update_diagnostics',False):
        os.environ['TWIST_FIRST_UPDATE_DIAGNOSTICS']='1'
    from diagnose_heading300_sim import run_diagnostic_case
    from sim_eval_metrics import analyze_trace
    from sim_eval_report import compare_models,plot_case,write_report
    output=args.output.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    summary=dict(schema_version=1,status='running',mode=args.mode,cases=[],
                 reference_modes=list(dict.fromkeys(args.reference_modes)),rows=[],comparisons={})
    try:
        protocol=json.loads(PROTOCOL.read_text())
        cases=select_cases(args.mode,protocol,args.cases)
        summary['cases']=cases
        summary['filtered_scope']=bool(args.cases)
        code={str(p):sha(ROOT/p) for p in [
            'tools/evaluate_sim.py','tools/diagnose_heading300_sim.py','tools/evaluate_4900_fixed.py',
            'tools/sim_eval_metrics.py','tools/sim_eval_report.py','tools/plot_sim_trace.py','deploy_real/server_low_level_g1_sim.py',
            'deploy_real/sim_dynamics_diagnostics.py','deploy_real/twist_anyadapter_runtime.py',
            'deploy_real/server_high_level_motion_lib.py','assets/g1/g1_sim2sim_with_wrist_roll.xml']}
        models=[dict(label='heading_300',role='baseline',path=BASELINE,projection=not args.no_target_projection),
                dict(label='historical23500',role='historical',path=HISTORY,projection=False)]
        if getattr(args,'no_historical_control',False):
            models=[m for m in models if m['role']!='historical']
        index=0;exports=[]
        for value in args.policy:
            index+=1;label,path=parse_policy(value,index)
            models.append(dict(label=label,role='candidate',path=path,projection=not args.no_target_projection))
        for value in args.checkpoint:
            index+=1;label,path=parse_policy(value,index)
            dest=output/'exports'/label/'policy.pt'
            dest.parent.mkdir(parents=True,exist_ok=False)
            exports.append(export_checkpoint(path,dest,args.expected_update))
            models.append(dict(label=label,role='candidate',path=dest,projection=not args.no_target_projection))
        if len({m['label'] for m in models}) != len(models):
            raise ValueError('duplicate policy labels')
        for model in models:
            model['sha256']=sha(model['path'])
        for case in cases:
            if sha(case['motion'])!=case['sha256']:
                raise ValueError('reference hash changed: '+case['id'])
        manifest=dict(mode=args.mode,cases=cases,filtered_scope=bool(args.cases),
                      reference_modes=summary['reference_modes'],seed=args.seed,ema=args.ema,
                      target_projection=not args.no_target_projection,control_dt=.02,sim_dt=.001,
                      projection_margin=.02,projection_knee_min=0.,device='cpu',synchronized=True,
                      protocol_sha256=sha(PROTOCOL),code=code,models=[dict(m,path=str(m['path'])) for m in models],exports=exports,
                      first_update_diagnostics=getattr(args,'first_update_diagnostics',False),
                      spectral_method='continuous segments>=64 samples; 64-point Welch, detrend linear, 5–20Hz',
                      joint_velocity_reference='numpy.gradient(clean q, dt=0.02)',auto_evaluation=args.auto_lock_fd is not None)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if args.dry_run:
            summary['status']='planned';write_report(output,summary)
            print('Plan only; no acceptance decision:',output)
            return 0
        # Deduplicate identical policies only when every execution condition matches.
        jobs={}
        for model in models:
            for case in cases:
                for mode in summary['reference_modes']:
                    sig=condition_signature(args,case,mode,model['projection'],code)
                    key=(model['sha256'],sig)
                    jobs.setdefault(key,dict(model=model,case=case,mode=mode,signature=sig))
        completed={};plot_arrays={};plot_spectra={}
        def run(job):
            model,case,mode=job['model'],job['case'],job['mode']
            dest=output/'runs'/model['label']/case['id']/mode
            row=run_diagnostic_case(case,mode,model['path'],dest,seed=args.seed,
                                    ema=args.ema,projection=model['projection'])
            analysis,cache,spectra=analyze_trace(row,model['label'])
            return row,analysis,cache,spectra
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run,job):key for key,job in jobs.items()}
            for future in as_completed(futures):
                key=futures[future];job=jobs[key]
                try:
                    completed[key]=dict(status='complete',result=future.result())
                    print('Complete',job['model']['label'],job['case']['id'],job['mode'],flush=True)
                except Exception as exc:
                    completed[key]=dict(status='failed',error=type(exc).__name__+': '+str(exc))
                    print('Failed',job['model']['label'],job['case']['id'],job['mode'],repr(exc),flush=True)
        for model in models:
            for case in cases:
                for mode in summary['reference_modes']:
                    sig=condition_signature(args,case,mode,model['projection'],code)
                    item=completed[model['sha256'],sig]
                    row=dict(label=model['label'],role=model['role'],case=case['id'],reference_mode=mode,
                             seed=args.seed,ema=args.ema,target_projection=model['projection'],
                             condition_signature=sig,policy_sha256=model['sha256'],status=item['status'])
                    if item['status']=='complete':
                        metrics,analysis,cache,spec=item['result']
                        row.update(metrics=metrics,analysis=analysis,trace=metrics['trace'])
                        group=(case['id'],mode)
                        plot_arrays.setdefault(group,{})[model['label']]=cache
                        renamed={(model['label'],name):v for (_,name),v in spec.items()}
                        plot_spectra.setdefault(group,{}).update(renamed)
                    else:
                        row['error']=item['error']
                    summary['rows'].append(row)
        if any(sha(ROOT/p)!=digest for p,digest in code.items()):
            raise ValueError('evaluation code/control configuration changed during run')
        if any(sha(m['path'])!=m['sha256'] for m in models):
            raise ValueError('policy changed during evaluation')
        summary['status']='failed' if any(r['status']=='failed' for r in summary['rows']) else 'complete'
        summary['comparisons']=compare_models(summary['rows'],summary['reference_modes'])
        for (case_id,mode),arrays in plot_arrays.items():
            traces={r['label']:r['trace'] for r in summary['rows'] if r['status']=='complete' and r['case']==case_id and r['reference_mode']==mode}
            plot_case(output/'plots',case_id,mode,arrays,plot_spectra[case_id,mode],traces=traces)
        write_report(output,summary)
        (output/('COMPLETE' if summary['status']=='complete' else 'FAILED')).write_text(str(len(summary['rows'])))
        print('Report:',output/'report.md')
        return 0 if summary['status']=='complete' else 1
    except Exception as exc:
        summary['status']='failed';summary['error']=type(exc).__name__+': '+str(exc)
        summary['rows'].append(dict(label='evaluation',role='candidate',case='preflight_or_report',
                                   reference_mode='unknown',status='failed',error=type(exc).__name__+': '+str(exc)))
        summary['comparisons']={'evaluation':dict(status='failed',accepted=False,gates={},tradeoffs=[])}
        (output/'failure.txt').write_text(traceback.format_exc())
        if not (output/'manifest.json').exists():
            (output/'manifest.json').write_text(json.dumps(dict(status='failed',mode=args.mode),indent=2))
        write_report(output,summary);(output/'FAILED').write_text(repr(exc))
        print('Evaluation failed:',repr(exc),file=sys.stderr)
        return 1


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if not 0<=args.ema<1 or args.eval_interval<1 or args.poll_seconds<=0:
        p.error('EMA must be in [0,1), interval/poll must be positive')
    if args.auto_eval:
        if not args.watch_dir or args.policy or args.checkpoint:
            p.error('--auto-eval requires --watch-dir and no manual policy/checkpoint')
        from sim_auto_eval import watch_checkpoints
        return watch_checkpoints(args)
    if args.watch_dir or args.watch_once:
        p.error('watching requires explicit --auto-eval')
    if not args.policy and not args.checkpoint:
        p.error('specify --policy or --checkpoint')
    if args.expected_update is not None and len(args.checkpoint)!=1:
        p.error('expected-update requires one checkpoint')
    if args.auto_lock_fd is not None:
        try:os.nice(10)
        except OSError:pass
    return evaluate(args)


if __name__=='__main__':
    sys.exit(main())
