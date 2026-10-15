import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rsl_rl'))
from rsl_rl.modules.physical_world_model import PhysicalWorldModel,physical_state,validate_transition,CONTRACT
spec=importlib.util.spec_from_file_location('physical_mixin',ROOT/'legged_gym/legged_gym/envs/g1/physical_wm_transition_mixin.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class FakeEnv:
    def __init__(self):
        self.root_states=torch.zeros(2,13);self.root_states[:,6]=1
        self.dof_pos=torch.zeros(2,23);self.dof_vel=torch.ones(2,23)
        self.default_dof_pos_all=torch.zeros(2,23);self.dof_names=[str(i) for i in range(23)]
        self.cfg=SimpleNamespace(control=SimpleNamespace(control_type='P',action_scale=.5,target_projection=True))
        self.dt=.02;self.extras={'preserved':True}
    def reindex(self,x):return x
    def _target_bounds(self):return torch.full((23,),-.1),torch.full((23,),.1)
    def _compute_torques(self,x):return x.clone()
    def _post_physics_step_callback(self):pass
    def step(self,action):
        # Delay selects previous command. State crosses a terminal transition.
        executed=torch.full_like(action,-.5);self._compute_torques(executed)
        self.dof_pos+=.02*self.dof_vel;self._post_physics_step_callback()
        self.dof_pos[0]=99;self.dof_vel[0]=0  # terminal reset after the callback
        return torch.zeros(2,7003),None,torch.ones(2),torch.tensor([True,False]),self.extras


class RecordedEnv(module.PhysicalWMTransitionMixin,FakeEnv):pass


class PhysicalTests(unittest.TestCase):
    def test_delayed_projected_action_and_pre_reset_next_state_are_separate(self):
        env=RecordedEnv();command=torch.full((2,23),.5)
        result=env.step(command);p=result[4]['physical_wm_transition'];validate_transition(p)
        self.assertTrue(torch.equal(p['commanded_action'],command))
        self.assertTrue(torch.equal(p['executed_action'],torch.full_like(command,-.5)))
        self.assertTrue(torch.equal(p['executed_pd_target'],torch.full_like(command,-.1)))
        torch.testing.assert_close(p['next_state'][:,5:28],torch.full_like(command,.02))
        self.assertEqual(p['valid'].tolist(),[False,True]);self.assertNotIn('physical_wm_transition',env.extras)
        env.dof_vel.fill_(7);self.assertTrue(torch.equal(p['state'][:,28:],torch.ones(2,23)))

    def test_collector_preserves_environment_output_and_rng(self):
        plain=FakeEnv();wrapped=RecordedEnv();cmd=torch.randn(2,23)
        rng=torch.get_rng_state().clone();a=plain.step(cmd);b=wrapped.step(cmd)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for index in [0,2,3]:self.assertTrue(torch.equal(a[index],b[index]))
        self.assertTrue(torch.equal(plain.dof_pos,wrapped.dof_pos))
        self.assertTrue(torch.equal(plain.dof_vel,wrapped.dof_vel))

    def test_unmasked_velocity_and_invalid_values(self):
        q=torch.zeros(2,23);dq=torch.zeros_like(q);dq[:,[4,5,10,11]]=3
        state=physical_state(torch.zeros(2,3),torch.zeros(2,2),q,dq)
        self.assertEqual(state[:,28:][:,[4,5,10,11]].unique().tolist(),[3.])
        with self.assertRaises(ValueError):physical_state(torch.zeros(2,3),torch.zeros(2,2),q,dq*float('nan'))

    def test_model_can_represent_q_change_with_zero_endpoint_velocity(self):
        model=PhysicalWorldModel(torch.zeros(51),torch.ones(51),torch.zeros(23),torch.ones(23))
        with torch.no_grad():model.transition[-1].bias[9]=.02  # ankle q changes; endpoint dq is zero
        state=torch.zeros(2,51);prediction=model(state,torch.zeros(2,23),torch.zeros(2,79,74))
        self.assertEqual(float(prediction[:,32].abs().max()),0.)
        torch.testing.assert_close(prediction[:,9],torch.full((2,),.02))
        self.assertEqual(float(model.loss(prediction,prediction)),0.)

    def test_history_and_model_gradient_ownership(self):
        model=PhysicalWorldModel(torch.zeros(51),torch.ones(51),torch.zeros(23),torch.ones(23))
        with torch.no_grad():model.transition[-1].weight.fill_(.01)
        x=torch.randn(3,51);h=torch.randn(3,79,74)
        loss=model.loss(model(x,torch.randn(3,23),h),torch.randn(3,51));loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.history_encoder.parameters()),0.)

    def test_autoregressive_history_pairs_pre_step_state_with_target(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from train_physical_wm_pilot import rollout
        class Recorder:
            def __init__(self):self.histories=[]
            def __call__(self,state,target,history):
                self.histories.append(history.clone());return state+1
        recorder=Recorder();state=torch.full((1,51),2.)
        history=torch.zeros(1,79,74);targets=torch.stack((torch.full((1,23),3.),torch.full((1,23),4.)),1)
        prediction=rollout(recorder,state,history,targets)
        self.assertTrue(torch.equal(recorder.histories[1][:,-1,:51],state))
        self.assertTrue(torch.equal(recorder.histories[1][:,-1,51:],targets[:,0]))
        self.assertTrue(torch.equal(prediction[:,1],state+2))

    def test_incompatible_joint_order_fails_before_physics(self):
        env=RecordedEnv();env.reindex=lambda x:x.flip(-1)
        with self.assertRaises(ValueError):env.step(torch.zeros(2,23))
        self.assertTrue(torch.equal(env.dof_pos,torch.zeros(2,23)))


if __name__=='__main__':unittest.main()
