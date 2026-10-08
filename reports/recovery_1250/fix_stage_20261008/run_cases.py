"""Run paired high/low-level simulation with full traces for mocap1/77."""
import argparse,inspect,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'tools'))
import evaluate_4900_fixed as fixed
p=argparse.ArgumentParser();p.add_argument('--policy',action='append',required=True,help='label=JIT path');p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--target-projection',action='store_true');p.add_argument('--duration',type=float,default=64.14);p.add_argument('--phase',choices=['motion77','development'],default='motion77');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
# Same physics, reference protocol and metric path; additionally record frames.
source=inspect.getsource(fixed.run_case)
old="'--metrics_out',str(out/'metrics.json'),"
new="'--metrics_out',str(out/'metrics.json'),'--trace_out',str(out/'frames.jsonl'),"
if a.target_projection:
 new += "'--target-projection',"
assert old in source
scope=dict(fixed.__dict__);exec(source.replace(old,new),scope);run_case=scope['run_case']
protocol=ROOT/'reports/recovery_4900/protocol.json';case=next(x for x in json.loads(protocol.read_text())['cases'] if x['id']=='mocap1__77')
assert fixed.sha(case['motion'])==case['sha256']
case = dict(case, seconds=a.duration)
cases = ([case] if a.phase=='motion77' else [c for c in json.loads(protocol.read_text())['cases'] if c['phase']=='development'])
def run(label,policy,mode,case):
 for attempt in range(1,5):
  dest=a.output/label/f'attempt_{attempt}';dest.mkdir(parents=True,exist_ok=True)
  try:
   row=run_case(case,mode,policy.resolve(),dest.resolve())
   assert sum(1 for _ in (dest/case['id']/mode/'frames.jsonl').open())==row['frames']
   return dict(label=label,policy=str(policy.resolve()),policy_sha256=fixed.sha(policy),trace=str((dest/case['id']/mode/'frames.jsonl').resolve()),**row)
  except Exception as e:print('RETRY',label,mode,attempt,repr(e),flush=True)
 raise RuntimeError(label+'/'+mode)
policies=[x.split('=',1) for x in a.policy]
receipt=dict(policies={label:dict(path=str(Path(path).resolve()),sha256=fixed.sha(path)) for label,path in policies},protocol_sha256=fixed.sha(protocol),cases=cases,code={str(x):fixed.sha(ROOT/x) for x in ['tools/evaluate_4900_fixed.py','deploy_real/server_low_level_g1_sim.py','deploy_real/server_high_level_motion_lib.py']},driver_sha256=fixed.sha(__file__),seed=42,headless=True,synchronized=True,action_ema_alpha=0,target_projection=a.target_projection,projection_margin=.02,projection_knee_min=0.)
(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2));rows=[]
with ThreadPoolExecutor(max_workers=a.workers) as pool:
 for f in as_completed([pool.submit(run,label,Path(path),mode,case) for label,path in policies for mode in ['raw','wm'] for case in cases]):
  row=f.result();rows.append(row);rows.sort(key=lambda x:(x['label'],x['case'],x['mode']));(a.output/'results.json').write_text(json.dumps(rows,indent=2));print(row['label'],row['mode'],{k:row[k] for k in ['joint_rmse','root_velocity_rmse','yaw_error','fall_proxy']},flush=True)
(a.output/'COMPLETE').write_text(str(len(rows)))
