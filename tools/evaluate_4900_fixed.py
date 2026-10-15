"""Synchronous MuJoCo comparison using a predeclared motion protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run_case(case, mode, policy, output, *, seed=42, low_args=()):
    out = output / case['id'] / mode
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    logs = [open(out / name, 'w') for name in ('redis.log', 'high.log', 'low.log')]
    redis = subprocess.Popen(['redis-server','--bind','127.0.0.1','--port',str(port),'--save','','--appendonly','no'],stdout=logs[0],stderr=subprocess.STDOUT)
    high = None
    try:
        import redis as redis_module
        client = redis_module.Redis(port=port)
        for _ in range(100):
            try:
                if client.ping(): break
            except redis_module.RedisError: time.sleep(.05)
        else: raise RuntimeError('Redis startup failed')
        high = subprocess.Popen([sys.executable,'-u',str(ROOT/'deploy_real/server_high_level_motion_lib.py'),
            '--motion_file',case['motion'],'--device','cpu','--steps','1',
            '--reference-mode',mode,'--motion-speed',str(case.get('speed',1)),
            '--wait-for-sim-ready','--sim-ready-timeout','60','--redis-port',str(port),'--seed',str(seed)],
            cwd=ROOT,env=env,stdout=logs[1],stderr=subprocess.STDOUT)
        subprocess.run([sys.executable,'-u',str(ROOT/'deploy_real/server_low_level_g1_sim.py'),
            '--policy_path',str(policy),'--device','cpu','--headless','--sync-reference',
            '--sim_duration',str(case['seconds']),'--metrics_out',str(out/'metrics.json'),
            '--redis-port',str(port),'--seed',str(seed),*low_args],cwd=ROOT,env=env,stdout=logs[2],stderr=subprocess.STDOUT,
            check=True,timeout=case['seconds']+120)
        metrics=json.loads((out/'metrics.json').read_text())
        expected=round(case['seconds']*50)
        if metrics['frames'] != expected: raise ValueError(f"incomplete: {metrics['frames']} != {expected}")
        return dict(case=case['id'],mode=mode,phase=case['phase'],motion=case['motion'],seconds=case['seconds'],
                    fall_proxy=metrics['minimum_pelvis_height']<.35 or metrics['max_tilt']>1.0,**metrics)
    finally:
        for proc in (high,redis):
            if proc is not None:
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        for log in logs: log.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'reports/recovery_4900/protocol.json')
    p.add_argument('--phase',choices=['development','confirmation','all'],default='development')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text());cases=[c for c in protocol['cases'] if a.phase=='all' or c['phase']==a.phase]
    (a.output/'receipt.json').write_text(json.dumps(dict(policy=str(a.policy.resolve()),policy_sha256=sha(a.policy),
        protocol_sha256=sha(a.protocol),cases=cases,code={str(f):sha(ROOT/f) for f in [
        'tools/evaluate_4900_fixed.py','deploy_real/server_low_level_g1_sim.py','deploy_real/server_high_level_motion_lib.py']}),indent=2))
    rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run_case,c,m,a.policy.resolve(),a.output):(c['id'],m) for c in cases for m in protocol['modes']}
        for future in as_completed(pending):
            row=future.result();rows.append(row);rows.sort(key=lambda x:(x['case'],x['mode']))
            (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(row['case'],row['mode'],'joint',round(row['joint_rmse'],5),'vel',round(row['root_velocity_rmse'],5),'yaw',round(row['yaw_error'],5),'fall',row['fall_proxy'],flush=True)
    (a.output/'COMPLETE').write_text(str(len(rows)))
if __name__=='__main__':main()
