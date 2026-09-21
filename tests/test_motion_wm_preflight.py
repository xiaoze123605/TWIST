import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'pose'), str(ROOT/'rsl_rl')]
from pose.utils.motion_validation import validate_motion_data, normalized_continuous_quaternions
from motion_world_model.utils import ensure_quaternion_continuity
from tools.prepare_motion_wm_training import split_membership, verify_prepared_training_yaml


class PreflightTests(unittest.TestCase):
    def test_rsi_keeps_temporal_warmup_tail(self):
        path = ROOT/'pose/pose/utils/motion_lib_pkl.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'MotionLib')
        namespace = {'torch': torch, 'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        MotionLib = namespace['MotionLib']
        library = MotionLib.__new__(MotionLib)
        library._device = 'cpu'
        library._motion_lengths = torch.tensor([.30, .68, 10.0])
        ids = torch.tensor([0, 1, 2]).repeat(200)
        times = library.sample_time(ids, minimum_remaining_time=.52)
        limits = torch.clamp(library._motion_lengths[ids] - .52, min=0)
        self.assertTrue(torch.all(times >= 0))
        self.assertTrue(torch.all(times <= limits))
        self.assertTrue(torch.all(times[ids == 0] == 0))
        with self.assertRaises(ValueError):
            library.sample_time(ids, minimum_remaining_time=-1)

    def test_training_yaml_requires_matching_successful_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root/'train.yaml'
            train.write_text('root_path: /tmp\nmotions:\n- file: a.pkl\n  weight: 1\n')
            digest = __import__('hashlib').sha256(train.read_bytes()).hexdigest()
            audit = dict(passed=True, errors=[], split_yaml_sha256={'train': digest},
                         splits={'train': {'motions': 1}})
            (root/'audit.json').write_text(json.dumps(audit))
            self.assertEqual(verify_prepared_training_yaml(train), root/'audit.json')
            train.write_text(train.read_text() + '# changed\n')
            with self.assertRaisesRegex(ValueError, 'stale'):
                verify_prepared_training_yaml(train)
            with self.assertRaisesRegex(ValueError, 'audited train.yaml'):
                verify_prepared_training_yaml(root/'twist_dataset.yaml')
            self.assertIsNone(verify_prepared_training_yaml(root/'one.pkl'))

    def test_resume_cannot_mix_checkpoints_or_overwrite_results(self):
        from tools.run_motion_wm_dtera import check_resume_manifest
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            check_resume_manifest(out, {'checkpoint_sha256': 'one'}, False)
            (out/'manifest.json').write_text(json.dumps({'checkpoint_sha256': 'one'}))
            check_resume_manifest(out, {'checkpoint_sha256': 'one'}, True)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                check_resume_manifest(out, {'checkpoint_sha256': 'two'}, True)
            with self.assertRaises(FileExistsError):
                check_resume_manifest(out, {'checkpoint_sha256': 'one'}, False)

    def test_quaternion_preprocessing_matches_frozen_wm(self):
        rng = np.random.default_rng(42)
        q = rng.normal(size=(100,4)) * rng.uniform(.2, 2., size=(100,1))
        actual = normalized_continuous_quaternions(q)
        np.testing.assert_allclose(actual, ensure_quaternion_continuity(q), atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), 1., atol=1e-7)
        with self.assertRaises(ValueError):
            normalized_continuous_quaternions(np.zeros((2,4)))

    def test_schema_rejects_bad_targets(self):
        data = dict(fps=50., root_pos=np.zeros((4,3)), root_rot=np.tile([0,0,0,1.],(4,1)),
                    dof_pos=np.zeros((4,23)), local_body_pos=np.zeros((4,2,3)),
                    link_body_list=['pelvis','foot'])
        self.assertEqual(validate_motion_data(data, expected_dofs=23)[0], 23)
        for key,value in [('fps',0), ('dof_pos',np.zeros((3,23))),
                          ('root_rot',np.zeros((4,4))), ('root_pos',np.full((4,3),np.nan))]:
            with self.assertRaises(ValueError):
                validate_motion_data(dict(data, **{key:value}), expected_dofs=23)

    def test_motion_segments_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            for split,name in [('train','walk_seg00.pkl'), ('val','walk_seg01.pkl'), ('test','jump.pkl')]:
                group = 'walk' if split != 'test' else 'jump'
                Path(tmp,f'{split}_manifest.json').write_text(json.dumps(dict(motions=[dict(motion_id=name,group_id=group)])))
            with self.assertRaisesRegex(ValueError,'leakage'):
                split_membership(tmp)

    def test_rsi_boundary_is_timeout_but_failure_stays_failure(self):
        path = ROOT/'legged_gym/legged_gym/envs/g1/g1_motion_wm_dtera.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='G1MotionWMDTERA')
        class Base:
            def check_termination(self):
                pass
        ns = dict(G1MimicDistill=Base)
        exec(compile(ast.Module(body=[node], type_ignores=[]),str(path),'exec'), ns)
        env = ns['G1MotionWMDTERA']()
        env._motion_ids = torch.arange(3)
        env._motion_lib = SimpleNamespace(get_motion_length=lambda ids: torch.full((3,),10.))
        env.dt = .02
        env.episode_length_buf = torch.tensor([100,100,100])
        env._motion_time_offsets = torch.tensor([1.,8.,8.])
        env.time_out_buf = torch.zeros(3,dtype=torch.bool)
        env.reset_buf = torch.tensor([False,False,True])
        env.check_termination()
        self.assertEqual(env.reset_buf.tolist(),[False,True,True])
        self.assertEqual(env.time_out_buf.tolist(),[False,True,False])

    def test_export_rejects_wrong_preset_but_records_explicit_ablation(self):
        spec=importlib.util.spec_from_file_location('wm_export',ROOT/'legged_gym/scripts/export_twist_dtera_jit.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        checkpoint=dict(training_config=dict(policy=dict(tracking_branch_gain=.25)))
        model=SimpleNamespace(tracking_branch_gain=1.)
        with self.assertRaisesRegex(ValueError,'mismatch'):
            module.check_saved_policy_config(checkpoint,model,set())
        overrides=module.check_saved_policy_config(checkpoint,model,{'--tracking_branch_gain'})
        self.assertEqual(overrides,[dict(parameter='tracking_branch_gain',trained=.25,exported=1.)])


if __name__ == '__main__':
    unittest.main()
