import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from coherent_recovery import teacher_features,prepare_start,LocalRecoveryBatch,PhaseActivationActor
from heading_recovery_core import RecoveryActorCritic
import numpy as np
import torch


class CoherentRecoveryTests(unittest.TestCase):
    def test_physical_latent_cannot_affect_teacher_input(self):
        x=torch.randn(3,168);y=x.clone();y[:,112:144]+=100;y[:,167]+=100
        self.assertTrue(torch.equal(teacher_features(x),teacher_features(y)))
        self.assertTrue(torch.equal(teacher_features(x)[:,:112],x[:,:112]))

    def test_specialist_activation_is_explicitly_limited(self):
        class Constant:
            def mean(self,features,source):return source+.01
        wrapper=PhaseActivationActor(Constant());x=torch.zeros(1,168);source=torch.zeros(1,23)
        for frame in range(3207):
            actual=wrapper.mean(x,source)
            self.assertTrue(torch.equal(actual,source+.01 if 2100<=frame<2500 else source),frame)

    def test_reset_preserves_sibling_and_terminal_data_and_clips_only_execution(self):
        torch.set_num_threads(1);start=prepare_start();env=LocalRecoveryBatch(2,start)
        env.signs[:]=0;env.env.refs[0]=copy.deepcopy(env.env.refs[0]);env.env.refs[0].length=2101
        obs=env.observe();source=obs['source'].clone();legacy=obs['legacy'].clone()
        sibling_history=env.env.runtime[1].history.copy();sampled=source.numpy()+.2;before=sampled.copy()
        new,reward,done,info=env.step(sampled,source.numpy())
        self.assertTrue(bool(done[0]));self.assertFalse(bool(done[1]))
        self.assertEqual(int(env.env.frames[0]),2100);self.assertEqual(int(env.env.frames[1]),2101)
        self.assertTrue(np.array_equal(sampled,before))
        self.assertLessEqual(float(abs(info['actual_action']-source.numpy()).max()),.100001)
        self.assertTrue(np.array_equal(env.env.runtime[1].history[:-1],sibling_history[1:]))
        self.assertFalse(np.array_equal(info['physical_after'][0,:env.env.model.nq],env.env.data[0].qpos))
        self.assertFalse(torch.equal(info['terminal_features'][0],teacher_features(new['features'])[0]))
        self.assertTrue(torch.equal(legacy[0],start['python']['cache']['legacy'][0]))


if __name__=='__main__':unittest.main()
