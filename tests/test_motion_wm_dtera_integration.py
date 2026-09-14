import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deploy_real'))
from twist_anyadapter_runtime import AnyAdapterRuntime, AnyAdapterRuntimeConfig
from motion_world_model.model import MotionGRU
from motion_world_model.runtime import MotionReferenceRefiner, RuntimeReferenceCorruptor


class ContractPolicy(torch.nn.Module):
    def forward(self, observations):
        return observations.reshape(-1, 3695)[:, :23]


class MotionDTERAIntegrationTest(unittest.TestCase):
    def test_warmup_histories_follow_processed_reference_and_reset_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.jit.trace(ContractPolicy(), torch.zeros(1, 3695)).save(str(root/'policy.pt'))
            torch.manual_seed(42)
            model = MotionGRU(hidden_dim=8)
            torch.save(dict(model_state=model.state_dict(),
                            model_config=dict(hidden_dim=8, num_layers=1, dropout=0.0),
                            normalization_mean=np.zeros(31, np.float32),
                            normalization_std=np.ones(31, np.float32)), root/'wm.pt')
            wm = MotionReferenceRefiner(root/'wm.pt', device='cpu')
            corruptor = RuntimeReferenceCorruptor.from_preset('formal', seed=42)
            runtime = AnyAdapterRuntime(AnyAdapterRuntimeConfig(
                base_obs_dim=1155, num_actions=23, history_len=20, state_indices=list(range(31,82)),
                policy_path=str(root/'policy.pt'), tracking_error_history_len=20,
                fill_history_on_first_observation=True))
            sequences = []
            for repeat in range(2):
                wm.reset()
                corruptor.reset()
                runtime.reset()
                saved = []
                for i in range(30):
                    clean = np.full(31, i*.01, np.float32)
                    corrupt = corruptor.corrupt(clean)
                    processed = wm.refine(corrupt)
                    base = np.zeros(1155, np.float32)
                    base[:31] = processed
                    base[31:82] = i*.02
                    runtime.act(base, tracking_reference=processed, dof_pos=np.zeros(23),
                                dof_vel=np.zeros(23), root_linear_velocity=np.zeros(3),
                                root_yaw_velocity=0, roll_pitch=np.zeros(2))
                    np.testing.assert_allclose(runtime.tracking_error_history[-1,:23], processed[8:])
                    np.testing.assert_allclose(runtime.history[-1,:51], i*.02)
                    if i < 24:
                        np.testing.assert_allclose(processed, corrupt, atol=1e-6)
                    if i:
                        np.testing.assert_array_equal(runtime.tracking_error_history[:-1], saved[-1][1][1:])
                    saved.append((runtime.history.copy(), runtime.tracking_error_history.copy()))
                sequences.append(saved)
            for first, second in zip(*sequences):
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[1], second[1])


if __name__ == '__main__':
    unittest.main()
