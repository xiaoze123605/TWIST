import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from motion_world_model.model import MotionGRU
from motion_world_model.runtime import MotionReferenceRefiner, RuntimeReferenceCorruptor
from motion_world_model.training_reference import TrainingReferencePipeline


class TrainingReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name)/'wm.pt'
        torch.manual_seed(5)
        model = MotionGRU(hidden_dim=8)
        torch.save(dict(model_state=model.state_dict(), model_config=dict(hidden_dim=8, num_layers=1, dropout=0.),
                        normalization_mean=np.linspace(-.1,.1,31).astype(np.float32),
                        normalization_std=np.linspace(.5,1.5,31).astype(np.float32)), self.checkpoint)

    def pipeline(self, **kwargs):
        return TrainingReferencePipeline(self.checkpoint, 2, mode_probabilities=(0,0,1), **kwargs)

    def test_batched_matches_deployment_including_wrap_and_warmup(self):
        batch = self.pipeline(seed=42)
        refiners = [MotionReferenceRefiner(self.checkpoint, 'cpu') for _ in range(2)]
        corruptors = [RuntimeReferenceCorruptor.from_preset('formal', seed=42+9973*i) for i in range(2)]
        for frame in range(40):
            clean = np.full((2,31), frame*.01, np.float32)
            clean[:,3] = np.arctan2(np.sin(3.1+frame*.02),np.cos(3.1+frame*.02))
            expected = np.stack([refiners[i].refine(corruptors[i].corrupt(clean[i])) for i in range(2)])
            source = torch.from_numpy(clean)
            saved = source.clone()
            result = batch.process(source)
            torch.testing.assert_close(source,saved)
            np.testing.assert_allclose(result.numpy(), expected, atol=2e-6, rtol=2e-5)
            if frame < 24:
                torch.testing.assert_close(result, batch.corrupted)

    def test_partial_reset_does_not_change_other_environment(self):
        first, control = self.pipeline(), self.pipeline()
        clean = torch.zeros(2,31)
        for _ in range(30):
            first.process(clean)
            control.process(clean)
        first.reset(torch.tensor([0]))
        self.assertEqual(first.count.tolist(),[0,25])
        self.assertEqual(first.history[0].count_nonzero().item(),0)
        for i in range(25):
            output = first.process(clean)
            other = control.process(clean)
            # Ready-mask changes the GRU batch size; BLAS roundoff may differ.
            torch.testing.assert_close(output[1],other[1],rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(first.history[1],control.history[1],rtol=0,atol=0)
            if i < 24:
                torch.testing.assert_close(output[0],first.corrupted[0])

    def test_deployment_aligned_mode_refines_raw_input_without_corruption(self):
        pipeline = self.pipeline(seed=42, artificial_corruption=False)
        refiners = [MotionReferenceRefiner(self.checkpoint, 'cpu') for _ in range(2)]
        for frame in range(40):
            raw = np.full((2, 31), frame * .01, np.float32)
            raw[:, 3] = np.arctan2(np.sin(3.1 + frame * .02),
                                   np.cos(3.1 + frame * .02))
            expected = np.stack([refiners[i].refine(raw[i]) for i in range(2)])
            result = pipeline.process(torch.from_numpy(raw))
            np.testing.assert_allclose(pipeline.corrupted.cpu().numpy(), raw,
                                       atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(result.cpu().numpy(), expected,
                                       atol=2e-6, rtol=2e-5)
        self.assertTrue(all(value is None for value in pipeline.corruptors))

    def test_mixture_output_selection_and_frozen_parameters(self):
        pipeline = self.pipeline()
        source = torch.ones(2,31,requires_grad=True)
        pipeline.mode[:] = torch.tensor([0,1])
        for _ in range(30):
            output = pipeline.process(source)
        torch.testing.assert_close(output[0],source[0])
        torch.testing.assert_close(output[1],pipeline.corrupted[1])
        self.assertFalse(output.requires_grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in pipeline.model.parameters()))
        self.assertFalse(pipeline.model.training)

    def test_reset_schedule_is_reproducible(self):
        a = TrainingReferencePipeline(self.checkpoint,2,seed=7)
        b = TrainingReferencePipeline(self.checkpoint,2,seed=7)
        for i in range(35):
            if i%5 == 0:
                a.reset([i%2]); b.reset([i%2])
            torch.testing.assert_close(a.mode,b.mode)
            torch.testing.assert_close(a.process(torch.ones(2,31)),b.process(torch.ones(2,31)),rtol=0,atol=0)

    def test_invalid_contract_fails(self):
        with self.assertRaises(ValueError):
            self.pipeline(control_dt=.01)
        with self.assertRaises(ValueError):
            TrainingReferencePipeline(self.checkpoint,2,mode_probabilities=(1,1,1))
        pipeline = self.pipeline()
        with self.assertRaises(ValueError):
            pipeline.process(torch.zeros(2,33))
        with self.assertRaises(ValueError):
            pipeline.process(torch.full((2,31),float('nan')))


if __name__ == '__main__':
    unittest.main()
