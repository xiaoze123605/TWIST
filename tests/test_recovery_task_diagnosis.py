import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_recovery_task import snapshot,restore,rollout,paired
from heading_recovery_core import HeadingRecoveryBatch,TurnHoldReference
import numpy as np
import torch


class RecoveryDiagnosisTests(unittest.TestCase):
    def test_complete_fork_restores_physics_history_cache_and_applied_force(self):
        torch.set_num_threads(1);env=HeadingRecoveryBatch(1)
        env.reset_one(0,TurnHoldReference(1),0);env.observe();rollout(env,85,None)
        state=snapshot(env);first=rollout(env,30,None)
        env.data[0].xfrc_applied[env.model.body('pelvis').id,5]=10
        env.runtime[0].history[:]=9;env.physical_history[:]=99;env.rng.random()
        restore(env,state)
        self.assertTrue(np.array_equal(env.data[0].xfrc_applied,np.zeros_like(env.data[0].xfrc_applied)))
        self.assertTrue(np.array_equal(env.cache['legacy'].numpy(),state['python']['cache']['legacy'].numpy()))
        second=rollout(env,30,None)
        for key in first:self.assertTrue(np.array_equal(first[key],second[key]),key)

    def test_early_recovery_and_later_relapse_are_distinguished(self):
        torch.set_num_threads(1);env=HeadingRecoveryBatch(1)
        env.reset_one(0,TurnHoldReference(1),0);env.observe();base=rollout(env,110,None)
        disturbed={k:v.copy() for k,v in base.items()}
        disturbed['native_pre_rpy'][:10,2]+=np.radians(5)
        disturbed['native_pre_rpy'][90:,2]+=np.radians(4)
        row=paired(disturbed,base)
        self.assertAlmostEqual(row['recovery_time_seconds'],.2)
        self.assertTrue(row['relapse_after_recovery'])
        self.assertGreater(row['final1s_extra_yaw_rms_deg'],2)
        self.assertGreater(row['absolute_yaw_iae_deg_seconds'],0)


if __name__=='__main__':unittest.main()
