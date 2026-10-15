"""Initialization receipts for the independently authorized critic experiment."""
import json
import torch
from heading300_teacher import complete_teacher_action


def verify_initialization(alg, source, env, out):
    stored = torch.load(source, map_location=env.device)['model_state_dict']
    current = alg.actor_critic.state_dict()
    assert all(torch.equal(current[k], v) for k, v in stored.items() if k != 'critic.0.weight')
    assert torch.equal(current['critic.0.weight'][:, :1318], stored['critic.0.weight'])
    assert torch.count_nonzero(current['critic.0.weight'][:, 1318:]) == 0
    obs = env.get_observations()
    critic = env.get_privileged_observations()
    assert obs.shape[-1] == 7003 and critic.shape[-1] == 1320, (obs.shape, critic.shape)
    assert torch.equal(critic[:, -2:], obs[:, -2:])
    with torch.no_grad():
        base, hist = alg.actor_critic.split_obs(obs)
        z = alg.actor_critic.history_encoder(hist)
        actual = alg.actor_critic.actor_mean(obs)
        target = complete_teacher_action(alg.source, base, z, obs[:, -2:])
        actor_error = float((actual - target).abs().max())
        # Calculate the original critic directly from the source tensors.
        reference = torch.nn.functional.linear(critic[:, :1318], stored['critic.0.weight'], stored['critic.0.bias'])
        for layer in alg.source.critic[1:]:
            reference = layer(reference)
        value_error = float((alg.actor_critic.evaluate(critic) - reference).abs().max())
    assert actor_error == 0 and actual.shape[-1] == 23
    assert value_error <= 1e-5, value_error
    assert alg.actor_critic.layerwise_actor.adapter_gain == .5
    assert all(not p.requires_grad for p in alg.source.parameters())
    receipt = dict(input_dim=7003, critic_input_dim=1320, original_critic_input_dim=1318,
                   output_dim=23, adapter_gain=.5, teacher_max_abs_error=actor_error,
                   initial_value_max_abs_error=value_error, original_columns_exact=True,
                   added_columns_zero=True, actor_state_exact=True, teacher_frozen=True,
                   critic_context_matches_actor=True,
                   optimizer_group_learning_rates=[g['lr'] for g in alg.ppo_optimizer.param_groups])
    (out/'initialization.json').write_text(json.dumps(receipt, indent=2))
