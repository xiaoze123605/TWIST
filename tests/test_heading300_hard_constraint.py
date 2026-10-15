"""Meaningful acceptance/rejection tests for shared Adam, plus the real full pool."""
from copy import deepcopy
from pathlib import Path
import sys
import os
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'rsl_rl'),str(ROOT/'legged_gym/scripts')]
from heading300_hard_constraint import BoundedAdamStep,FixedCompleteTeacherPool,BOUND
from export_twist_anyadapter_opentrack_jit import build_actor
DEVICE=os.environ.get('HARD_GUARD_TEST_DEVICE','cpu')


class TransactionTests(unittest.TestCase):
    def fixture(self,rate):
        p=[torch.nn.Parameter(torch.zeros(23,dtype=torch.float64,device=DEVICE)),torch.nn.Parameter(torch.zeros(1,dtype=torch.float64,device=DEVICE)),torch.nn.Parameter(torch.zeros(23,dtype=torch.float64,device=DEVICE))]
        opt=torch.optim.Adam([dict(params=[p[0]],lr=rate),dict(params=[p[1]],lr=.001),dict(params=[p[2]],lr=rate)])
        for x in p:x.grad=torch.ones_like(x)
        opt.step()  # Exercise restoration of existing moments, not just empty state.
        old=(p[0]+p[2]).detach().clone()
        return p,opt,old

    def compare(self,guard,reference):
        self.assertTrue(guard.equal_snapshot(reference))

    def test_accept_recomputes_once_with_scaled_actor_rates_and_original_critic(self):
        p,opt,old=self.fixture(.00024)
        guard=BoundedAdamStep(opt,lambda:dict(max_abs=float(abs(p[0]+p[2]-old).max())))
        before=guard.snapshot(p)
        ref_p=[torch.nn.Parameter(x.detach().clone()) for x in p]
        ref=torch.optim.Adam([dict(params=[ref_p[0]],lr=.00024),dict(params=[ref_p[1]],lr=.001),dict(params=[ref_p[2]],lr=.00024)])
        ref.load_state_dict(deepcopy(opt.state_dict()));ref.param_groups[0]['lr']*=.125;ref.param_groups[2]['lr']*=.125
        for x in ref_p:x.grad=torch.ones_like(x)
        ref.step();guard.step();record=guard.records[-1]
        self.assertEqual(record['accepted_ratio'],.125)
        for x,y in zip(p,ref_p):self.assertTrue(torch.equal(x,y))
        for x,y in zip(p,ref_p):
            for key in opt.state[x]:self.assertTrue(torch.equal(opt.state[x][key],ref.state[y][key]))
        self.assertEqual([g['lr'] for g in opt.param_groups],[.00024,.001,.00024])
        self.assertLessEqual(record['final']['max_abs'],BOUND)
        self.assertEqual(record['actor_adam_step_delta'],1)

    def test_rejection_restores_actor_moments_steps_and_keeps_one_critic_update(self):
        p,opt,old=self.fixture(.1)
        guard=BoundedAdamStep(opt,lambda:dict(max_abs=float(abs(p[0]+p[2]-old).max())))
        before=guard.snapshot(p);critic_old=float(opt.state[p[1]]['step'])
        guard.step();record=guard.records[-1]
        self.assertFalse(record['actor_accepted']);self.assertEqual(len(record['trials']),6)
        self.compare(guard,{x:before[x] for x in [p[0],p[2]]})
        self.assertEqual(float(opt.state[p[1]]['step']),critic_old+1)
        self.assertEqual(record['actor_parameter_l2'],0.)

    def test_joint_compensation_checks_full_action_not_individual_modules(self):
        p,opt,old=self.fixture(.00024)
        # Matching opposite Adam histories cause exactly compensating updates.
        opt.state[p[2]]['exp_avg'].neg_();p[2].grad=-torch.ones_like(p[2])
        guard=BoundedAdamStep(opt,lambda:dict(max_abs=float(abs(p[0]+p[2]-old).max())))
        guard.step();r=guard.records[-1]
        self.assertEqual(r['accepted_ratio'],1.);self.assertGreater(r['actor_parameter_l2'],0)
        self.assertLessEqual(r['final']['max_abs'],BOUND)

    def test_checker_exception_restores_all_parameters_and_optimizer_states(self):
        p,opt,_=self.fixture(.00024)
        def failed():raise RuntimeError('diagnostic check failed')
        guard=BoundedAdamStep(opt,failed);before=guard.snapshot(p)
        with self.assertRaises(RuntimeError):guard.step()
        self.compare(guard,before)

    def test_warmup_does_not_advance_actor(self):
        p,opt,_=self.fixture(.00024)
        def forbidden():raise AssertionError('Actor guard should not be needed during critic-only warmup')
        guard=BoundedAdamStep(opt,forbidden);before=guard.snapshot([p[0],p[2]])
        p[0].grad=p[2].grad=None;guard.step()
        self.compare(guard,before);self.assertEqual(guard.records[-1]['status'],'critic_only_warmup')


class RealPoolTests(unittest.TestCase):
    def test_real_complete_teacher_pool_and_numerical_integrity(self):
        source=ROOT/'legged_gym/logs/recovery_23500_heading/heading23500_gain05_pilot100_20261008/model_300.pt'
        anchor=ROOT/'legged_gym/logs/recovery_heading300_yaw_controls/anchor20_adamresume_original_env512_seed42_20261012/model_25.pt'
        actor=build_actor(source).to(DEVICE);teacher=build_actor(source).to(DEVICE).requires_grad_(False)
        pool=FixedCompleteTeacherPool(actor,teacher,anchor)
        rng=torch.get_rng_state().clone();baseline=pool.check()
        self.assertEqual(baseline['states_checked'],19200);self.assertEqual(baseline['max_abs'],0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with torch.no_grad():actor.adapter[5].bias[-1]+=.001
        changed=pool.check();self.assertGreater(changed['max_abs'],BOUND)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(changed['actions_checked'],23)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))


if __name__=='__main__':unittest.main()
