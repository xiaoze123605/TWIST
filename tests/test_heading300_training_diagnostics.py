import sys
import unittest
from pathlib import Path
import torch
import tempfile
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from heading300_training_diagnostics import residual_groups,gradient_groups,objective_gradients


class DiagnosticTests(unittest.TestCase):
    def test_heading_bins_count_and_no_rng(self):
        angles=torch.deg2rad(torch.tensor([0.,14.,31.,70.,130.,-45.]))
        context=torch.stack((angles.sin(),1-angles.cos()),-1)
        modes=torch.tensor([0,0,0,2,2,2]);state=torch.get_rng_state().clone()
        records=residual_groups(context,modes,residual=torch.arange(6.))
        self.assertEqual(sum(r['frames'] for r in records),6)
        first=next(r for r in records if r['reference_mode']=='raw' and r['heading_low_degrees']==0)
        self.assertEqual(first['frames'],2)
        self.assertAlmostEqual(first['residual_rmse'],2**-.5,places=6)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_gradient_observation_does_not_change_gradients(self):
        actor=torch.nn.Module();actor.adapter=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(6)])
        actor.critic=torch.nn.Linear(1,1)
        for p in actor.parameters():p.grad=torch.ones_like(p)
        before=[p.grad.clone() for p in actor.parameters()];result=gradient_groups(actor)
        self.assertAlmostEqual(result['core'],10**.5,places=6)
        for expected,p in zip(before,actor.parameters()):self.assertTrue(torch.equal(expected,p.grad))
        for p in actor.adapter.parameters():p.grad=None
        result=gradient_groups(actor);self.assertEqual(result['core'],0);self.assertEqual(result['heading'],0)

    def test_objective_probes_do_not_accumulate_or_change_backward(self):
        actor=torch.nn.Module();actor.adapter=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(6)])
        output=sum(layer(torch.ones(2,1)).sum() for layer in actor.adapter)
        surrogate=output.square();anchor=.1*(output-1).square()
        expected=torch.autograd.grad(surrogate+anchor,list(actor.parameters()),retain_graph=True)
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            alg=SimpleNamespace(actor_critic=actor,count=25,audit_path=Path(directory)/'metrics.jsonl')
            objective_gradients(alg,surrogate,anchor)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        (surrogate+anchor).backward()
        for grad,p in zip(expected,actor.parameters()):torch.testing.assert_close(grad,p.grad,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
