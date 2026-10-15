import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from heading300_closed_loop_gate import assess


def report():
    def row(label, role, mode, value):
        return dict(label=label,role=role,case='mocap1__77',reference_mode=mode,status='complete',
                    analysis=dict(phases=dict(squat=dict(knee_rmse=value,hip_knee_ankle_rmse=value),
                    walking=dict(actual_spectrum=dict(high_5_20hz_rms=value)))))
    return dict(rows=[row(label,role,mode,value) for mode in ['raw','wm']
                     for label,role,value in [('baseline','baseline',1.),('new','candidate',.89)]],
                comparisons=dict(new=dict(gates={k:True for k in ['complete_evaluation','no_new_falls',
                    'yaw_within_5_percent','each_case_velocity_within_5_percent']})),
                cases=list(range(9)),filtered_scope=False)


class ClosedLoopGateTest(unittest.TestCase):
    def test_requires_every_improvement_and_multi_motion(self):
        s=report()
        self.assertTrue(assess(s,'new')['accepted'])
        s['filtered_scope']=True
        self.assertFalse(assess(s,'new')['accepted'])
        s['filtered_scope']=False
        s['rows'][-1]['analysis']['phases']['walking']['actual_spectrum']['high_5_20hz_rms']=.95
        self.assertFalse(assess(s,'new')['accepted'])

    def test_missing_failure_or_regression_never_passes(self):
        for mutation in ['missing','failed','regression']:
            s=report()
            if mutation=='missing':s['rows'].pop()
            elif mutation=='failed':s['rows'][-1]['status']='failed'
            else:s['comparisons']['new']['gates']['yaw_within_5_percent']=False
            self.assertFalse(assess(s,'new')['accepted'])


if __name__=='__main__':
    unittest.main()
