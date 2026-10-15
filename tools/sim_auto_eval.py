"""Opt-in, nonblocking, globally serialized checkpoint evaluation hook.

Importing/constructing a disabled hook does not import torch, spawn a process,
create files or alter CUDA settings in the training process.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]


class AutoEvalHook:
    def __init__(self, *, enabled=False, interval=50, output_root=None,
                 mode='quick', reference_modes=('raw','wm'), seed=42, ema=0.,
                 target_projection=True, dry_run=False, cases=None, lock_path=None,
                 python_executable=None):
        if interval<1 or mode not in ('quick','standard','full') or not 0<=ema<1:
            raise ValueError('invalid auto-evaluation settings')
        if not reference_modes or set(reference_modes)-{'raw','wm'}:
            raise ValueError('invalid reference modes')
        self.enabled=bool(enabled);self.interval=interval
        self.output_root=Path(output_root or ROOT/'reports/auto_sim_eval').expanduser().resolve()
        self.mode=mode;self.reference_modes=tuple(reference_modes)
        self.seed=seed;self.ema=ema;self.target_projection=target_projection
        self.dry_run=dry_run;self.cases=cases;self.lock_path=Path(lock_path or ROOT/'reports/.sim_auto_eval.lock').expanduser().resolve()
        self.python=python_executable or sys.executable
        self.worker=None;self.lock_file=None;self.job=None;self.seen=set()
        self.mutex=threading.RLock()

    def _event(self,status,**values):
        event=dict(status=status,time_unix=time.time(),**values)
        try:
            self.output_root.mkdir(parents=True,exist_ok=True)
            with (self.output_root/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(event,ensure_ascii=False)+'\n')
        except OSError as exc:
            event['log_error']=repr(exc)
        return event

    def poll(self):
        if not self.enabled or not self.mutex.acquire(blocking=False):
            return None
        try:
            return self._poll()
        finally:
            self.mutex.release()

    def _poll(self):
        if not self.enabled or self.worker is None:
            return None
        code=self.worker.poll()
        if code is None:
            return None
        summary_path=self.job/'summary.json'
        try:
            summary=json.loads(summary_path.read_text())
            if not isinstance(summary,dict) or not isinstance(summary.get('comparisons',{}),dict):
                raise ValueError('invalid worker report schema')
            state=summary['status']
        except (OSError,ValueError,KeyError,TypeError):
            summary={};state='missing_report'
        status='finished' if code==0 and state in ('complete','planned') else 'failed'
        if status=='failed':
            # A nonzero exit must invalidate even a previously-written pass.
            summary['status']='failed'
            summary['worker_exit_code']=code
            for label,comparison in list(summary.get('comparisons',{}).items()):
                if not isinstance(comparison,dict):
                    comparison={};summary['comparisons'][label]=comparison
                if not isinstance(comparison.get('gates',{}),dict):comparison['gates']={}
                comparison['accepted']=False;comparison['status']='failed_worker'
                comparison.setdefault('gates',{})['complete_evaluation']=False
            try:
                self.job.mkdir(parents=True,exist_ok=True)
                (self.job/'summary.json').write_text(json.dumps(summary,indent=2))
                (self.job/'FAILED').write_text('evaluation worker failed: '+str(code))
                if (self.job/'COMPLETE').exists():(self.job/'COMPLETE').unlink()
                report=self.job/'report.md'
                if report.exists():report.rename(self.job/'report.before_worker_failure.md')
                report.write_text('# Failed evaluation worker\n\nExit code: '+str(code)+'. No acceptance result is valid. See summary.json and the worker log.\n')
            except OSError:
                pass
        event=self._event(status,output=str(self.job),exit_code=code,
                          report_status=summary.get('status',state),evaluation_mode=summary.get('mode',self.mode),comparisons=summary.get('comparisons',{}))
        self.worker=None;self.job=None
        if self.lock_file is not None:
            self.lock_file.close();self.lock_file=None
        return event

    def on_checkpoint_saved(self,path,update):
        """Call AFTER torch.save(temp) and os.replace(temp, final).

        No checkpoint is queued when another worker is active. Complete ZIP/state
        validation and CPU export are performed in the worker before simulation.
        """
        if not self.enabled:
            return dict(status='disabled')
        if not self.mutex.acquire(blocking=False):
            return dict(status='skipped_busy')
        try:
            self.poll()
            if update<=0 or update%self.interval:
                return dict(status='skipped_interval',update=update)
            path=Path(path).resolve()
            if not path.is_file() or path.suffix not in ('.pt','.pth') or path.stat().st_size==0:
                return self._event('failed_checkpoint',checkpoint=str(path),update=update,
                                   error='expected nonempty final .pt/.pth after atomic save')
            fingerprint=(str(path),update,path.stat().st_size,path.stat().st_mtime_ns)
            if fingerprint in self.seen:
                return dict(status='skipped_duplicate',update=update)
            if self.worker is not None:
                return self._event('skipped_busy',checkpoint=str(path),update=update)
            self.lock_path.parent.mkdir(parents=True,exist_ok=True)
            lock=self.lock_path.open('a+')
            try:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                return self._event('skipped_busy',checkpoint=str(path),update=update)
            self.output_root.mkdir(parents=True,exist_ok=True)
            token='update_'+f'{update:08d}'+'_'+str(time.time_ns())+'_'+uuid.uuid4().hex[:8]
            output=self.output_root/token
            command=[self.python,str(ROOT/'tools/evaluate_sim.py'),
                     '--checkpoint','checkpoint='+str(path),'--expected-update',str(update),
                     '--mode',self.mode,'--output',str(output),'--workers','1',
                     '--seed',str(self.seed),'--ema',str(self.ema),
                     '--reference-modes',*self.reference_modes,'--auto-lock-fd',str(lock.fileno())]
            if not self.target_projection:command+=['--no-target-projection']
            if self.dry_run:command+=['--dry-run']
            if self.cases:command+=['--cases',*self.cases]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
                     MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            try:
                with (self.output_root/(token+'.log')).open('w') as stream:
                    worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,
                                            stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),
                                            start_new_session=True)
            except Exception as exc:
                lock.close()
                return self._event('failed_spawn',checkpoint=str(path),update=update,error=repr(exc))
            self.worker=worker;self.lock_file=lock;self.job=output;self.seen.add(fingerprint)
            return self._event('started',checkpoint=str(path),update=update,output=str(output),
                               pid=worker.pid,device='cpu',workers=1,dry_run=self.dry_run,evaluation_mode=self.mode)
        except Exception as exc:
            return self._event('failed_hook',checkpoint=str(path),update=update,error=repr(exc))
        finally:
            self.mutex.release()

    def wait(self,timeout=None):
        """Optional shutdown/test helper; never called by the save callback."""
        if self.worker is None:
            return None
        try:self.worker.wait(timeout=timeout)
        except subprocess.TimeoutExpired:return dict(status='running')
        return self.poll()


class CompletedCheckpointScanner:
    def __init__(self,interval=50):
        self.interval=interval;self.previous={};self.latest_dispatched=-1;self.dispatched={}

    def mark_dispatched(self,update,path):
        stat=Path(path).stat();self.latest_dispatched=update
        self.dispatched[Path(path)]=(stat.st_size,stat.st_mtime_ns)

    def scan(self,directory):
        stable=[];current={}
        for path in Path(directory).glob('model_*.pt'):
            match=re.fullmatch(r'model_(\d+)\.pt',path.name)
            if not match:continue
            update=int(match[1])
            if update<self.latest_dispatched or update<=0 or update%self.interval:continue
            try:stat=path.stat()
            except OSError:continue
            signature=(stat.st_size,stat.st_mtime_ns)
            current[path]=signature
            if self.dispatched.get(path)==signature:continue
            if stat.st_size>0 and self.previous.get(path)==signature:
                stable.append((update,path))
        self.previous=current
        return max(stable,key=lambda x:x[0]) if stable else None


def watch_checkpoints(args):
    if not args.watch_dir.is_dir():
        raise ValueError('watch directory does not exist')
    hook=AutoEvalHook(enabled=True,interval=args.eval_interval,output_root=args.output,
                     mode=args.mode,reference_modes=args.reference_modes,seed=args.seed,
                     ema=args.ema,target_projection=not args.no_target_projection,dry_run=args.dry_run,cases=args.cases)
    scanner=CompletedCheckpointScanner(args.eval_interval)
    scans=0
    try:
        while True:
            hook.poll();candidate=scanner.scan(args.watch_dir);scans+=1
            if candidate:
                update,path=candidate;event=hook.on_checkpoint_saved(path,update)
                print(json.dumps(event),flush=True)
                if event['status']=='started':scanner.mark_dispatched(update,path)
            if args.watch_once and scans>=2:
                result=hook.wait()
                return 1 if result and result['status']=='failed' else 0
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print('Watcher stopped; an already-running CPU worker may finish independently.')
        return 0
