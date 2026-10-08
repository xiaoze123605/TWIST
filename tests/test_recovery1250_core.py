import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from recovery1250_core import StratifiedReplay, anchor_error

class Recovery1250Tests(unittest.TestCase):
    def test_anchor_fixed_units_and_gradient(self):
        action = torch.ones(4, 23, requires_grad=True)
        error = anchor_error(action, torch.zeros_like(action), .1, torch.ones(4))
        self.assertTrue(torch.allclose(error, torch.full((4,), 1150.)))
        error.mean().backward()
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertTrue((action.grad > 0).all())

    def test_reservoir_retains_history_and_balances_rare_strata(self):
        torch.manual_seed(17)
        bank = StratifiedReplay(capacity=100, insertion=100)
        for iteration in range(100):
            groups = torch.tensor([0]*100 + ([5]*20 if iteration == 0 else []))
            inputs = torch.full((len(groups), 3), float(iteration))
            bank.add(inputs, inputs[:, :1], torch.ones(len(groups)), groups)
        self.assertEqual(bank.counts()[5], 20)
        self.assertEqual(bank.counts()[0], 100)
        self.assertTrue((bank.groups[0]['inputs'][:, 0] < 50).any())
        x, y, w = bank.sample(per_group=8)
        self.assertEqual(len(x), 16)
        self.assertTrue((x[8:] == 0).all())
        restored = StratifiedReplay(capacity=100, insertion=100)
        restored.load_state_dict(bank.state_dict(), 'cpu')
        torch.manual_seed(42)
        a = bank.sample()
        torch.manual_seed(42)
        b = restored.sample()
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

if __name__ == '__main__':
    unittest.main()
