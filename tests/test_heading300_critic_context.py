"""Verify the approved critic adaptation and preservation of policy behavior."""
import sys
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'rsl_rl'), str(ROOT/'legged_gym/scripts'), str(ROOT)]
from heading300_critic_context import append_heading_context, TwistHeadingCriticContextActorCritic
from heading23500_actor import TwistHeadingSupportActorCritic
from export_twist_anyadapter_opentrack_jit import BASE_JIT, DeployActor


class CriticContextTest(unittest.TestCase):
    def test_alignment_and_reject_wrong_layout(self):
        observations = torch.randn(3, 7003)
        privileged = torch.randn(3, 1318)
        combined = append_heading_context(privileged, observations)
        self.assertTrue(torch.equal(combined[:, :1318], privileged))
        self.assertTrue(torch.equal(combined[:, -2:], observations[:, -2:]))
        with self.assertRaises(ValueError):
            append_heading_context(combined, observations)

    def test_constructor_rng_zero_columns_and_jit(self):
        kwargs = dict(num_actions=23, base_actor_jit_path=BASE_JIT, base_obs_dim=1155,
                      history_len=79, history_frame_dim=74, hist_state_dim=51,
                      wm_target_indices=list(range(31,82)), latent_dim=128,
                      world_model_hidden_dims=(512,512,256,256,256,128),
                      critic_hidden_dims=(512,256,128), activation='silu', freeze_base=True)
        torch.manual_seed(42)
        original = TwistHeadingSupportActorCritic(num_critic_observations=1318, **kwargs).eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        expanded = TwistHeadingCriticContextActorCritic(**kwargs).eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        expanded.load_state_dict(original.state_dict())
        original.layerwise_actor.adapter_gain = expanded.layerwise_actor.adapter_gain = .5
        self.assertEqual(int(expanded.critic[0].weight[:,1318:].count_nonzero()), 0)
        policy_input = torch.randn(4,7003)
        old_value_input = torch.randn(4,1318)
        new_value_input = append_heading_context(old_value_input, policy_input)
        with torch.no_grad():
            self.assertTrue(torch.equal(original.actor_mean(policy_input), expanded.actor_mean(policy_input)))
            torch.testing.assert_close(original.evaluate(old_value_input), expanded.evaluate(new_value_input), atol=1e-6, rtol=1e-5)
            traced = torch.jit.trace(DeployActor(expanded), policy_input)
            self.assertTrue(torch.equal(traced(policy_input), DeployActor(original)(policy_input)))


if __name__ == '__main__':
    unittest.main()
