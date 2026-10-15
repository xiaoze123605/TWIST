"""Export and evaluate saved pilot checkpoints without launching further training."""
import argparse,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--max-attempts',type=int,default=3);a=p.parse_args();out=ROOT/'reports/recovery_4900'
for step in (50,100):
 ckpt=ROOT/'legged_gym/logs/recovery_4900'/a.run/f'model_{step}.pt'
 deadline=time.monotonic()+900
 while not ckpt.exists():
  if time.monotonic()>deadline:raise TimeoutError(str(ckpt))
  time.sleep(3)
 time.sleep(2)
 jit=out/f'{a.run}_{step}.pt'
 subprocess.run([sys.executable,str(ROOT/'legged_gym/scripts/export_twist_anyadapter_opentrack_jit.py'),str(ckpt),'--output',str(jit)],check=True)
 evaluation=out/f'{a.run}_{step}_dev'
 import shutil
 for attempt in range(1,a.max_attempts+1):
  if evaluation.exists():shutil.rmtree(evaluation)
  try:
   subprocess.run([sys.executable,str(ROOT/'tools/evaluate_4900_fixed.py'),'--policy',str(jit),'--output',str(evaluation)],check=True)
   break
  except subprocess.CalledProcessError:
   if attempt==a.max_attempts:raise
   print(f'evaluate attempt {attempt} failed (transient sim SIGSEGV); retrying in 5s',flush=True)
   time.sleep(5)
 subprocess.run([sys.executable,str(ROOT/'tools/compare_4900_fixed.py'),'--reference',str(out/'incumbent_dev_v1'),'--candidate',str(evaluation),'--output',str(out/f'{a.run}_{step}_gate.json')],check=True)
