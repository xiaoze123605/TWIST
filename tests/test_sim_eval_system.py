import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import evaluate_sim as entry
from sim_auto_eval import AutoEvalHook,CompletedCheckpointScanner
from sim_eval_report import compare_models,write_report


def row(label,role,mode='wm',velocity=.1,yaw=.5,coord=.2,hf=.01,fall=False):
    return dict(label=label,role=role,case='motion',reference_mode=mode,status='complete',
        condition_signature='same',seed=42,ema=0.,target_projection=True,
        metrics=dict(root_velocity_rmse=velocity,yaw_error=yaw,joint_rmse=.15,fall_proxy=fall),
        analysis=dict(reference_digest='clean',processed_reference_digest='processed',
            metrics=dict(torso_pitch_rmse_degrees=5),contact_transition={},phases=dict(
                squat=dict(seconds=4.,hip_knee_ankle_rmse=coord),
                walking=dict(seconds=10.,actual_spectrum=dict(high_5_20hz_rms=hf,valid_seconds=8.)))))


class EvaluationTests(unittest.TestCase):
    def test_levels_and_mode_parser(self):
        protocol=json.loads(entry.PROTOCOL.read_text())
        for mode in ['quick','standard','full']:
            a=entry.parser().parse_args(['--policy','x.pt','--mode',mode,'--output','x'])
            self.assertEqual(a.mode,mode);self.assertFalse(a.auto_eval)
        q=entry.select_cases('quick',protocol);s=entry.select_cases('standard',protocol);f=entry.select_cases('full',protocol)
        self.assertEqual(q[0]['seconds'],20.)
        self.assertEqual(next(c for c in s if c['id']=='mocap1__77')['seconds'],64.14)
        self.assertGreater(len(f),len(s))
        with self.assertRaises(ValueError):entry.select_cases('quick',protocol,['bad'])
        with self.assertRaises(SystemExit):entry.main(['--policy','x','--output','x','--watch-dir','x'])

    def test_all_gates_per_case_and_reference_mode(self):
        bases=[row('heading_300','baseline',m) for m in ['raw','wm']]
        candidates=[row('candidate','candidate',m,coord=.18,hf=.009) for m in ['raw','wm']]
        result=compare_models(bases+candidates,['raw','wm'])['candidate']
        self.assertTrue(result['accepted'])
        candidates[0]['metrics']['root_velocity_rmse']=.106
        self.assertFalse(compare_models(bases+candidates,['raw','wm'])['candidate']['accepted'])
        candidates[0]['metrics']['root_velocity_rmse']=.1
        candidates[1]['metrics']['fall_proxy']=True
        self.assertFalse(compare_models(bases+candidates,['raw','wm'])['candidate']['accepted'])
        candidates[1]['metrics']['fall_proxy']=False
        candidates[1]['condition_signature']='projection changed'
        self.assertEqual(compare_models(bases+candidates,['raw','wm'])['candidate']['status'],'failed_or_unmatched')

    def test_missing_spectra_failure_and_history_not_pass(self):
        base=row('heading_300','baseline');candidate=row('c','candidate',coord=.18,hf=.009)
        del candidate['analysis']['phases']['walking']['actual_spectrum']['high_5_20hz_rms']
        result=compare_models([base,candidate],['wm'])['c']
        self.assertIsNone(result['gates']['walking_high_frequency_improved']);self.assertFalse(result['accepted'])
        candidate=row('c','candidate',coord=.18,hf=.009)
        history=row('historical23500','historical');history['condition_signature']='historical projection off'
        self.assertTrue(compare_models([base,candidate,history],['wm'])['c']['accepted'])
        history['status']='failed'
        self.assertFalse(compare_models([base,candidate,history],['wm'])['c']['accepted'])

    def test_zero_reference_and_processed_reference_mismatch(self):
        base=row('heading_300','baseline',velocity=0,yaw=0)
        candidate=row('c','candidate',velocity=.001,yaw=.001,coord=.18,hf=.009)
        self.assertFalse(compare_models([base,candidate],['wm'])['c']['accepted'])
        candidate['analysis']['processed_reference_digest']='different'
        self.assertEqual(compare_models([base,candidate],['wm'])['c']['status'],'failed_or_unmatched')

    def test_reports_and_failed_preflight_for_all_levels(self):
        for mode in ['quick','standard','full']:
            with tempfile.TemporaryDirectory() as folder:
                dest=Path(folder)/'evaluation'
                with patch.object(entry,'BASELINE',Path(folder)/'missing-baseline'):
                    code=entry.main(['--policy','bad.pt','--mode',mode,'--output',str(dest),'--cases','mocap1__77'])
                self.assertEqual(code,1)
                for filename in ['summary.json','metrics.csv','phases.csv','report.md','FAILED','manifest.json']:
                    self.assertTrue((dest/filename).exists(),filename)
                self.assertFalse(any(v['accepted'] for v in json.loads((dest/'summary.json').read_text())['comparisons'].values()))
                with self.assertRaises(FileExistsError):entry.main(['--policy','bad','--output',str(dest)])

    def test_full_success_and_runtime_failure_report_transport(self):
        artifact=ROOT/'reports/heading300_diagnosis_20261008/heading_wm/results.json'
        if not artifact.exists():self.skipTest('recorded diagnostic fixture absent')
        fixture=json.loads(artifact.read_text())
        motion=next(c for c in json.loads(entry.PROTOCOL.read_text())['cases'] if c['id']=='mocap1__77')
        protocol={'cases':[dict(motion,seconds=64.14),dict(motion,id='recorded_fixture_alias',phase='development',seconds=64.14)]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'protocol.json';path.write_text(json.dumps(protocol))
            base=root/'base.pt';base.write_bytes(b'baseline fixture')
            history=root/'history.pt';history.write_bytes(b'history fixture')
            candidate=root/'candidate.pt';candidate.write_bytes(b'candidate fixture')
            def replay(case,mode,policy,output,**kwargs):
                return dict(fixture,case=case['id'],mode=mode,phase=case['phase'])
            with patch.object(entry,'PROTOCOL',path),patch.object(entry,'BASELINE',base),patch.object(entry,'HISTORY',history),patch('diagnose_heading300_sim.run_diagnostic_case',side_effect=replay),patch('sim_eval_report.plot_case'):
                code=entry.main(['--policy',str(candidate),'--mode','full','--reference-modes','wm','--output',str(root/'success')])
            self.assertEqual(code,0)
            result=json.loads((root/'success/summary.json').read_text())
            self.assertEqual(len(result['rows']),6);self.assertTrue((root/'success/COMPLETE').exists())
            self.assertFalse(result['comparisons']['candidate_1']['accepted'])
            with patch.object(entry,'PROTOCOL',path),patch.object(entry,'BASELINE',base),patch.object(entry,'HISTORY',history),patch('diagnose_heading300_sim.run_diagnostic_case',side_effect=RuntimeError('fixture runner crashed')):
                code=entry.main(['--policy',str(candidate),'--mode','full','--reference-modes','wm','--output',str(root/'failure')])
            self.assertEqual(code,1)
            self.assertTrue((root/'failure/FAILED').exists());self.assertFalse((root/'failure/COMPLETE').exists())
            self.assertFalse(json.loads((root/'failure/summary.json').read_text())['comparisons']['candidate_1']['accepted'])

    def test_invalid_diagnostic_velocity_and_substeps_are_rejected(self):
        from sim_eval_metrics import analyze_trace
        artifact=ROOT/'reports/heading300_diagnosis_20261008/heading_wm/results.json'
        if not artifact.exists():self.skipTest('diagnostic fixture absent')
        result=json.loads(artifact.read_text())
        with Path(result['trace']).open() as stream:
            frames=[json.loads(next(stream)),json.loads(next(stream))]
        with tempfile.TemporaryDirectory() as folder:
            trace=Path(folder)/'frames.jsonl';result=dict(result,frames=2,trace=str(trace))
            frames[1]['joint_velocity'][0]=float('nan')
            trace.write_text(''.join(json.dumps(x)+'\n' for x in frames))
            with self.assertRaisesRegex(ValueError,'nonfinite'):analyze_trace(result)
            frames[1]['joint_velocity'][0]=0.;frames[1]['dynamics']['torque_substeps']=19
            trace.write_text(''.join(json.dumps(x)+'\n' for x in frames))
            with self.assertRaisesRegex(ValueError,'substep'):analyze_trace(result)

    def test_shared_metrics_preserve_diagnosis(self):
        from sim_eval_metrics import analyze_trace,spectrum
        import numpy as np
        artifact=ROOT/'reports/heading300_diagnosis_20261008/heading_wm/results.json'
        if not artifact.exists():self.skipTest('local diagnostic trace absent')
        result=json.loads(artifact.read_text());summary,_,_=analyze_trace(result)
        original=json.loads((artifact.parents[1]/'diagnostic_summary.json').read_text())['heading_wm']
        for phase in ['squat','descent','rise','walking']:
            for metric in ['pitch_rmse_degrees','joint_rmse','policy_action_rate_rms']:
                self.assertEqual(summary['phases'][phase][metric],original['phases'][phase][metric])
        data=np.sin(np.arange(200)*2*np.pi*10/50)[:,None]
        stats,_=spectrum(data,np.ones(200,dtype=bool));self.assertGreater(stats['high_5_20hz_rms'],.6)
        self.assertIn('status',spectrum(data,np.zeros(200,dtype=bool))[0])


class FakeWorker:
    def __init__(self):self.returncode=None;self.pid=123
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode


class AutoHookTests(unittest.TestCase):
    def test_disabled_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'not-created'
            with patch('sim_auto_eval.subprocess.Popen') as spawn:
                hook=AutoEvalHook(output_root=out)
                self.assertEqual(hook.on_checkpoint_saved('does-not-exist',50)['status'],'disabled')
                spawn.assert_not_called();self.assertFalse(out.exists())

    def test_interval_tmp_and_complete_scanner(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);hook=AutoEvalHook(enabled=True,output_root=root/'out',lock_path=root/'lock')
            self.assertEqual(hook.on_checkpoint_saved(root/'model_1.pt',1)['status'],'skipped_interval')
            temp=root/'model_50.pt.tmp';temp.write_bytes(b'x')
            self.assertEqual(hook.on_checkpoint_saved(temp,50)['status'],'failed_checkpoint')
            scan=CompletedCheckpointScanner(50)
            self.assertIsNone(scan.scan(root))
            final=root/'model_50.pt';temp.rename(final)
            self.assertIsNone(scan.scan(root));self.assertEqual(scan.scan(root),(50,final))
            final.write_bytes(b'longer')
            self.assertIsNone(scan.scan(root))

    def test_replaced_checkpoint_can_be_checked_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model_50.pt';path.write_bytes(b'partial')
            scanner=CompletedCheckpointScanner(50)
            scanner.scan(folder);self.assertEqual(scanner.scan(folder),(50,path))
            scanner.mark_dispatched(50,path);self.assertIsNone(scanner.scan(folder))
            path.write_bytes(b'complete replacement')
            self.assertIsNone(scanner.scan(folder));self.assertEqual(scanner.scan(folder),(50,path))

    def test_serial_cpu_worker_unique_output_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'model_50.pt';checkpoint.write_bytes(b'x')
            hook=AutoEvalHook(enabled=True,output_root=root/'out',lock_path=root/'lock')
            second=AutoEvalHook(enabled=True,output_root=root/'second',lock_path=root/'lock')
            worker=FakeWorker()
            with patch('sim_auto_eval.subprocess.Popen',return_value=worker) as spawn:
                event=hook.on_checkpoint_saved(checkpoint,50);self.assertEqual(event['status'],'started')
                command=spawn.call_args[0][0];env=spawn.call_args[1]['env']
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(command[command.index('--workers')+1],'1')
                self.assertEqual(second.on_checkpoint_saved(checkpoint,50)['status'],'skipped_busy')
                next_checkpoint=root/'model_100.pt';next_checkpoint.write_bytes(b'y')
                self.assertEqual(hook.on_checkpoint_saved(next_checkpoint,100)['status'],'skipped_busy')
                self.assertEqual(spawn.call_count,1)
                worker.returncode=1
                self.assertEqual(hook.poll()['status'],'failed')
                other=FakeWorker();spawn.return_value=other
                event2=hook.on_checkpoint_saved(next_checkpoint,100)
                self.assertEqual(event2['status'],'started');self.assertNotEqual(event['output'],event2['output'])
                other.returncode=1;hook.poll()

    def test_worker_failure_invalidates_a_written_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'model_50.pt';checkpoint.write_bytes(b'x')
            hook=AutoEvalHook(enabled=True,output_root=root/'out',lock_path=root/'lock')
            worker=FakeWorker()
            with patch('sim_auto_eval.subprocess.Popen',return_value=worker):
                event=hook.on_checkpoint_saved(checkpoint,50)
            job=Path(event['output']);job.mkdir()
            (job/'summary.json').write_text(json.dumps({'status':'complete','comparisons':{'c':{'accepted':True,'status':'meets_gates'}}}))
            (job/'COMPLETE').write_text('1');worker.returncode=1
            result=hook.poll();self.assertEqual(result['status'],'failed')
            self.assertFalse(result['comparisons']['c']['accepted'])
            self.assertFalse((job/'COMPLETE').exists());self.assertTrue((job/'FAILED').exists())
            self.assertFalse(json.loads((job/'summary.json').read_text())['comparisons']['c']['accepted'])

    def test_invalid_worker_report_does_not_interrupt_caller(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'model_50.pt';checkpoint.write_bytes(b'x')
            hook=AutoEvalHook(enabled=True,output_root=root/'out',lock_path=root/'lock')
            worker=FakeWorker()
            with patch('sim_auto_eval.subprocess.Popen',return_value=worker):event=hook.on_checkpoint_saved(checkpoint,50)
            job=Path(event['output']);job.mkdir();(job/'summary.json').write_text('[]')
            worker.returncode=0;result=hook.poll()
            self.assertEqual(result['status'],'failed');self.assertTrue((job/'FAILED').exists())

    def test_hook_errors_do_not_interrupt_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'model_50.pt';checkpoint.write_bytes(b'x')
            hook=AutoEvalHook(enabled=True,output_root=root/'out',lock_path=root/'lock')
            with patch('sim_auto_eval.subprocess.Popen',side_effect=OSError('cannot launch')):
                self.assertEqual(hook.on_checkpoint_saved(checkpoint,50)['status'],'failed_spawn')
            with patch.object(Path,'stat',side_effect=OSError('deleted')):
                self.assertEqual(hook.on_checkpoint_saved(checkpoint,100)['status'],'failed_hook')


if __name__=='__main__':unittest.main()
