import sys
from pathlib import Path
import unittest
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from heading_recovery_core import RecoveryActorCritic,TurnHoldReference,HeadingRecoveryBatch
from train_heading_recovery_pilot import compute_gae


class RecoveryTests(unittest.TestCase):
    def test_zero_correction_preserves_arbitrary_source_actions(self):
        model=RecoveryActorCritic();features=torch.randn(8,168);source=torch.randn(8,23)
        self.assertTrue(torch.equal(model.mean(features,source),source))
        self.assertTrue(torch.equal(model.delta(features),torch.zeros_like(source)))

    def test_reference_yaw_rate_matches_smooth_goal_and_hold(self):
        reference=TurnHoldReference(1)
        for frame in [105,125,150,175,195]:
            prev=reference.get(frame-1)[0][3];nxt=reference.get(frame+1)[0][3]
            self.assertAlmostEqual(float((nxt-prev)/.04),float(reference.get(frame)[0][7]),places=3)
        self.assertEqual(float(reference.get(399)[0][7]),0.)
        self.assertAlmostEqual(float(reference.get(399)[0][3]),np.pi/6,places=6)

    def test_reset_sibling_does_not_update_survivor_histories_twice(self):
        torch.set_num_threads(1);env=HeadingRecoveryBatch(2);obs=env.reset()
        env.frames[0]=399;env.cache=None;obs=env.observe();before=env.runtime[1].history.copy()
        new,_,done,info=env.step(obs['source'].numpy(),obs['source'].numpy())
        self.assertTrue(bool(done[0]));self.assertFalse(bool(done[1]))
        self.assertTrue(np.array_equal(env.runtime[1].history[:-1],before[1:]))
        self.assertEqual(len(env.base_history[1]),10);self.assertEqual(env.history_count[1],1)
        self.assertEqual(env.history_count[0],0);self.assertEqual(new['legacy'].shape,(2,7003))
        self.assertEqual(info['terminal_features'].shape,(2,168))

    def test_timeout_bootstrap_uses_terminal_state_and_gae_does_not_cross_reset(self):
        r=torch.tensor([[1.],[100.]]);v=torch.tensor([[2.],[0.]]);nv=torch.tensor([[5.],[0.]])
        done=torch.tensor([[True],[False]]);falls=torch.zeros_like(done)
        advantages,returns=compute_gae(r,v,nv,done,falls,gamma=.9,lam=.8)
        self.assertAlmostEqual(float(advantages[0,0]),3.5)
        self.assertAlmostEqual(float(returns[0,0]),5.5)
        self.assertEqual(float(advantages[1,0]),100.)

    def test_fall_has_no_bootstrap_from_reset_or_terminal_value(self):
        r=torch.tensor([[1.]]);v=torch.tensor([[2.]]);nv=torch.tensor([[999.]])
        done=torch.tensor([[True]]);adv,ret=compute_gae(r,v,nv,done,done,gamma=.9,lam=.8)
        self.assertEqual(float(adv[0,0]),-1.);self.assertEqual(float(ret[0,0]),1.)

    def test_correction_bound_and_explicit_heading_feature_interaction(self):
        model=RecoveryActorCritic(delta_scale=.1)
        with torch.no_grad():
            model.actor[-1].weight.fill_(1.)
        feature=torch.randn(4,168);self.assertLessEqual(float(model.delta(feature).abs().max()),.1)
        self.assertNotEqual(float(model.delta(feature)[0,0]),float(model.delta(feature+1)[0,0]))


if __name__=='__main__':unittest.main()
