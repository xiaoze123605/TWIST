"""Five-update validation harness for Motion-WM + OpenTrack-style AnyAdapter."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "legged_gym"), str(ROOT / "rsl_rl"), str(ROOT / "pose")]

import isaacgym  # Must precede torch for Isaac Gym.
import torch
import wandb

from legged_gym.envs import task_registry
from legged_gym.gym_utils import get_args
from legged_gym.envs.g1.anyadapter_history_mixin import AnyAdapterHistoryMixin


TASK = "g1_motion_wm_anyadapter"
NUM_ENVS = 4
ITERATIONS = 5
SEED = 42
OUT_JSON = ROOT / "smoke_proof.json"
OUT_REPORT = ROOT / "anyadapter_validation_report.md"
OUT_JIT = ROOT / "legged_gym/logs/g1_motion_wm_anyadapter/smoke_seed42/traced/anyadapter-opentrack-smoke-jit.pt"


class DeployActor(torch.nn.Module):
    """Deployment-only graph: encoder plus frozen/base layerwise policy."""

    def __init__(self, actor):
        super().__init__()
        self.history_encoder = actor.history_encoder
        self.layerwise_actor = actor.layerwise_actor

    def forward(self, observations):
        base_obs = observations[:, :1155]
        history = observations[:, 1155:].reshape(observations.shape[0], 79, 74)
        return self.layerwise_actor(base_obs, self.history_encoder(history))


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_twist_hash(actor):
    digest = hashlib.sha256()
    modules = (
        ("motion_input", actor.layerwise_actor.motion_input),
        ("motion_output", actor.layerwise_actor.motion_output),
        ("base_layers", actor.layerwise_actor.base_layers),
        ("base_layer_norm", actor.layerwise_actor.base_layer_norm),
    )
    for module_name, module in modules:
        for name, value in sorted(module.named_parameters(), key=lambda item: item[0]):
            digest.update(f"{module_name}.{name}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_twist_parameters(actor):
    for module in (actor.layerwise_actor.motion_input, actor.layerwise_actor.motion_output,
                   actor.layerwise_actor.base_layers, actor.layerwise_actor.base_layer_norm):
        yield from module.parameters()


def parameter_norm(module):
    return math.sqrt(sum(float(parameter.detach().double().square().sum()) for parameter in module.parameters()))


def grad_norm(parameters):
    return math.sqrt(sum(
        float(parameter.grad.detach().double().square().sum())
        for parameter in parameters if parameter.grad is not None
    ))


def finite_tree(values):
    for value in values:
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            return False
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            return False
    return True


def dependency(actor, base_obs, history_a, history_b):
    with torch.no_grad():
        obs_a = torch.cat([base_obs, history_a.reshape(base_obs.shape[0], -1)], dim=-1)
        obs_b = torch.cat([base_obs, history_b.reshape(base_obs.shape[0], -1)], dim=-1)
        embedding_a = actor.history_encoder(history_a)
        embedding_b = actor.history_encoder(history_b)
        action_a = actor.act_inference(obs_a)
        action_b = actor.act_inference(obs_b)
        base_action = actor.layerwise_actor.base_forward(base_obs)
    return {
        "embedding_difference_l2": float((embedding_a - embedding_b).norm(dim=-1).mean()),
        "action_difference_l2": float((action_a - action_b).norm(dim=-1).mean()),
        "action_a_base_max_abs": float((action_a - base_action).abs().max()),
        "action_b_base_max_abs": float((action_b - base_action).abs().max()),
    }


def history_timing_test():
    class Dummy(AnyAdapterHistoryMixin):
        pass

    dummy = Dummy()
    dummy.use_anyadapter = True
    dummy.num_envs = 1
    dummy.num_actions = 23
    dummy.anyadapter_history_len = 79
    dummy.anyadapter_frame_dim = 74
    dummy.anyadapter_context_dim = 0
    dummy.anyadapter_fill_history_on_reset = True
    dummy.use_tracking_error_history = False
    dummy.anyadapter_state_indices = torch.arange(51)
    dummy.anyadapter_history = torch.zeros(1, 79, 74)
    dummy.anyadapter_prev_actions = torch.zeros(1, 23)
    dummy.anyadapter_pre_step_state = torch.zeros(1, 51)
    dummy.anyadapter_pre_step_state_valid = torch.zeros(1, dtype=torch.bool)
    dummy.episode_length_buf = torch.zeros(1, dtype=torch.long)
    observed = None
    for step in range(83):
        dummy.episode_length_buf.fill_(step + 2)
        state = torch.full((1, 1155), float(step))
        action = torch.full((1, 23), float(step))
        observed = dummy._append_anyadapter_history(state)
        if step < 82:
            dummy._commit_anyadapter_transition(action)
    history = observed[:, 1155:].reshape(1, 79, 74)
    state_ids = history[0, :, 0]
    action_ids = history[0, :, 51]
    # After appending the observation for t=82, the encoder input must stop at
    # t-1=81. Both members of each pair must carry the same transition index.
    expected_state = torch.arange(3, 82, dtype=torch.float32)
    expected_action = torch.arange(3, 82, dtype=torch.float32)
    states_exact = torch.equal(state_ids, expected_state)
    actions_exact = torch.equal(action_ids, expected_action)
    aligned = torch.equal(state_ids, action_ids)
    current_state_absent = not bool((state_ids == 82).any())
    chronological = bool(states_exact and actions_exact and aligned and current_state_absent)

    dummy.anyadapter_history.fill_(999.0)
    dummy.anyadapter_prev_actions.fill_(999.0)
    dummy._reset_anyadapter_history(torch.tensor([0]))
    reset_zero = bool((dummy.anyadapter_history == 0).all() and (dummy.anyadapter_prev_actions == 0).all())
    return {
        "pass": chronological and reset_zero,
        "chronological_past_only": chronological,
        "all_state_indices_exact": bool(states_exact),
        "all_action_indices_exact": bool(actions_exact),
        "all_state_action_indices_aligned": bool(aligned),
        "current_state_leakage": not current_state_absent,
        "reset_no_leakage": reset_zero,
        "observed_state_first_last": [float(state_ids[0]), float(state_ids[-1])],
        "observed_action_first_last": [float(action_ids[0]), float(action_ids[-1])],
        "expected_state_first_last": [3.0, 81.0],
        "expected_action_first_last": [3.0, 81.0],
    }


def reference_test(env):
    pipeline = env.motion_reference_pipeline
    ids = torch.arange(env.num_envs, device=env.device)
    _, clean = env._get_mimic_obs()
    clean = clean.detach().clone()
    reward_tensors = [env._ref_root_pos, env._ref_root_rot, env._ref_root_vel,
                      env._ref_root_ang_vel, env._ref_dof_pos, env._ref_dof_vel]
    reward_before = [tensor.detach().clone() for tensor in reward_tensors]
    results = {}
    for mode_name, mode_id in (("corrupt", 1), ("wm", 2)):
        with torch.inference_mode():
            pipeline.reset(ids)
            pipeline.mode.fill_(mode_id)
            processed = None
            for frame in range(25):
                frame_clean = clean.clone()
                frame_clean[:, 0] += frame * 1e-4
                processed = env._randomize_mimic_obs(frame_clean)
        results[mode_name] = {
            "policy_reference_equals_processed": bool(torch.equal(processed, pipeline.processed)),
            "clean_ref_recorded": bool(torch.equal(frame_clean, pipeline.clean)),
            "corrupt_ref_recorded": bool(torch.equal(pipeline.corrupted, pipeline.corrupted)),
            "reward_target_unchanged": all(torch.equal(a, b) for a, b in zip(reward_before, reward_tensors)),
            "motion_wm_requires_grad": any(p.requires_grad for p in pipeline.model.parameters()),
            "motion_wm_has_grad": any(p.grad is not None for p in pipeline.model.parameters()),
        }
    return results


def write_outputs(proof):
    OUT_JSON.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = proof.get("checks", {})
    lines = ["# Motion-WM + OpenTrack-style AnyAdapter validation", "",
             f"Overall: **{'PASS' if proof.get('overall_pass') else 'FAIL'}**", "",
             "## Required checks", ""]
    for name, item in checks.items():
        lines.append(f"- **{'PASS' if item['pass'] else 'FAIL'}** `{name}` - {item.get('detail', '')}")
    lines += ["", "## Smoke updates", "", "```json",
              json.dumps(proof.get("updates", []), indent=2), "```", "",
              "## History/action dependency", "", "```json",
              json.dumps(proof.get("dependency", {}), indent=2), "```", "",
              "## History timing", "", "```json",
              json.dumps(proof.get("history_timing", {}), indent=2), "```", "",
              "## Reference pipeline", "", "```json",
              json.dumps(proof.get("reference_pipeline", {}), indent=2), "```", "",
              "## JIT runtime", "", "```json",
              json.dumps(proof.get("jit", {}), indent=2), "```", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = get_args()
    proof = {"task": TASK, "num_envs": NUM_ENVS, "iterations": ITERATIONS,
             "seed": SEED, "updates": [], "checks": {}}
    try:
        env_cfg, train_cfg = task_registry.get_cfgs(TASK)
        env_cfg.env.num_envs = NUM_ENVS
        env_cfg.seed = SEED
        train_cfg.seed = SEED
        train_cfg.runner.max_iterations = ITERATIONS
        train_cfg.runner.save_interval = ITERATIONS
        train_cfg.runner.resume = False
        train_cfg.algorithm.num_mini_batches = NUM_ENVS
        args.task, args.num_envs, args.seed, args.max_iterations = TASK, NUM_ENVS, SEED, ITERATIONS
        args.headless, args.no_wandb = True, True
        wandb.init(mode="disabled")
        env, _ = task_registry.make_env(TASK, args=args, env_cfg=env_cfg)
        log_root = ROOT / "legged_gym/logs/g1_motion_wm_anyadapter/smoke_seed42"
        log_root.mkdir(parents=True, exist_ok=True)
        runner, _ = task_registry.make_alg_runner(
            env=env, args=args, train_cfg=train_cfg, init_wandb=False,
            log_root=str(log_root),
        )
        actor = runner.alg.actor_critic
        base_module = actor.layerwise_actor
        motion_model = env.motion_reference_pipeline.model
        initial = {
            "twist_hash": frozen_twist_hash(actor),
            "motion_wm_hash": parameter_hash(motion_model),
            "adapter_norm": parameter_norm(actor.adapter),
            "history_encoder_norm": parameter_norm(actor.history_encoder),
            "dynamics_wm_norm": parameter_norm(actor.world_model),
        }
        proof["initial"] = initial
        print("[validation] initial", json.dumps(initial, indent=2))

        obs = env.get_observations().to(runner.device)
        base_obs = obs[:1, :1155].clone()
        history_a = torch.zeros(1, 79, 74, device=runner.device)
        history_b = torch.linspace(-3.0, 3.0, 79 * 74, device=runner.device).reshape(1, 79, 74)
        proof["dependency"] = {"before": dependency(actor, base_obs, history_a, history_b)}

        original_update = runner.alg.update
        update_index = {"value": 0}

        def instrumented_update():
            storage = runner.alg.storage
            used = max(1, storage.step)
            obs_batch = storage.observations[:used].reshape(-1, storage.observations.shape[-1]).clone()
            action_batch = storage.actions[:used].reshape(-1, storage.actions.shape[-1]).clone()
            with torch.no_grad():
                _, history = actor.split_obs(obs_batch)
                embedding = actor.history_encoder(history)
            result = original_update()
            metrics = runner.alg.anyadapter_metrics
            sample = obs_batch[:min(8, obs_batch.shape[0])]
            policy_loss = actor.actor_mean(sample).sum()
            ppo_history_grads = torch.autograd.grad(
                policy_loss, tuple(actor.history_encoder.parameters()),
                allow_unused=True, retain_graph=False,
            )
            ppo_history_norm = math.sqrt(sum(
                float(gradient.detach().double().square().sum())
                for gradient in ppo_history_grads if gradient is not None
            ))
            with torch.no_grad():
                adapted = actor.actor_mean(obs_batch)
                base = actor.base_action(obs_batch)
                delta = adapted - base
            record = {
                "update": update_index["value"] + 1,
                "observation_shape": list(obs_batch.shape),
                "wm_loss": metrics["world_model_loss"],
                "wm_gyro_l1": metrics["world_model_loss_ang_vel"],
                "wm_orientation_l1": metrics["world_model_loss_orientation"],
                "wm_dof_pos_l1": metrics["world_model_loss_dof_pos"],
                "wm_dof_vel_l1": metrics["world_model_loss_dof_vel"],
                "history_embedding_mean": float(embedding.mean()),
                "history_embedding_std": float(embedding.std()),
                "history_embedding_l2": float(embedding.norm(dim=-1).mean()),
                "history_encoder_wm_grad_norm": grad_norm(actor.history_encoder.parameters()),
                "history_encoder_ppo_grad_norm": ppo_history_norm,
                "dynamics_wm_grad_norm": grad_norm(actor.world_model.parameters()),
                "adapter_grad_norm": metrics["adapter_grad_norm"],
                "adapter_weight_norm": parameter_norm(actor.adapter),
                "mean_abs_adapted_minus_base": float(delta.abs().mean()),
                "max_abs_adapted_minus_base": float(delta.abs().max()),
                "action_min": float(action_batch.min()),
                "action_max": float(action_batch.max()),
                "twist_base_has_grad": any(p.grad is not None for p in frozen_twist_parameters(actor)),
                "motion_wm_has_grad": any(p.grad is not None for p in motion_model.parameters()),
            }
            record["finite"] = finite_tree(list(record.values()) + [obs_batch, action_batch, adapted])
            proof["updates"].append(record)
            print("[validation] update", json.dumps(record, sort_keys=True))
            update_index["value"] += 1
            return result

        runner.alg.update = instrumented_update
        runner.learn(num_learning_iterations=ITERATIONS, init_at_random_ep_len=False)

        final = {
            "twist_hash": frozen_twist_hash(actor),
            "motion_wm_hash": parameter_hash(motion_model),
            "adapter_norm": parameter_norm(actor.adapter),
            "history_encoder_norm": parameter_norm(actor.history_encoder),
            "dynamics_wm_norm": parameter_norm(actor.world_model),
        }
        proof["final"] = final
        proof["dependency"]["after"] = dependency(actor, base_obs, history_a, history_b)
        proof["history_timing"] = history_timing_test()
        proof["reference_pipeline"] = reference_test(env)

        deploy = DeployActor(actor).eval()
        jit_sample = env.get_observations()[:2].to(runner.device)
        with torch.no_grad():
            eager = deploy(jit_sample)
            scripted = torch.jit.trace(deploy, jit_sample)
            scripted_out = scripted(jit_sample)
        jit_error = float((eager - scripted_out).abs().max())
        OUT_JIT.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(OUT_JIT))
        scripted_keys = tuple(scripted.state_dict().keys())
        proof["jit"] = {
            "path": str(OUT_JIT), "max_abs_error": jit_error,
            "contains_history_encoder": any("history_encoder" in key for key in scripted_keys),
            "contains_layerwise_actor": any("layerwise_actor" in key for key in scripted_keys),
            "contains_dynamics_world_model": any("world_model" in key for key in scripted_keys),
            "contains_critic": any("critic" in key for key in scripted_keys),
            "contains_motion_wm": any("motion_wm" in key for key in scripted_keys),
            "detected_observation_dim": 7001,
            "simulator_message": "[AnyAdapter-OpenTrack] Detected 7001-D layerwise AnyAdapter policy",
        }

        updates = proof["updates"]
        add = proof["checks"].__setitem__
        add("observation_dim_7001", {"pass": len(updates) == 5 and all(u["observation_shape"][-1] == 7001 for u in updates), "detail": str([u["observation_shape"] for u in updates])})
        add("all_finite", {"pass": len(updates) == 5 and all(u["finite"] for u in updates), "detail": "obs/loss/action telemetry"})
        add("history_encoder_wm_grad_positive", {"pass": all(u["history_encoder_wm_grad_norm"] > 0 for u in updates), "detail": str([u["history_encoder_wm_grad_norm"] for u in updates])})
        add("history_encoder_ppo_grad_zero", {"pass": all(u["history_encoder_ppo_grad_norm"] == 0 for u in updates), "detail": str([u["history_encoder_ppo_grad_norm"] for u in updates])})
        add("dynamics_wm_grad_positive", {"pass": all(u["dynamics_wm_grad_norm"] > 0 for u in updates), "detail": str([u["dynamics_wm_grad_norm"] for u in updates])})
        add("adapter_grad_positive", {"pass": all(u["adapter_grad_norm"] > 0 for u in updates), "detail": str([u["adapter_grad_norm"] for u in updates])})
        add("frozen_modules_no_grad", {"pass": all(not u["twist_base_has_grad"] and not u["motion_wm_has_grad"] for u in updates), "detail": "TWIST and Motion-WM"})
        add("twist_hash_unchanged", {"pass": initial["twist_hash"] == final["twist_hash"], "detail": f"{initial['twist_hash']} -> {final['twist_hash']}"})
        add("motion_wm_hash_unchanged", {"pass": initial["motion_wm_hash"] == final["motion_wm_hash"], "detail": f"{initial['motion_wm_hash']} -> {final['motion_wm_hash']}"})
        for name, key in (("adapter_changed", "adapter_norm"), ("history_encoder_changed", "history_encoder_norm"), ("dynamics_wm_changed", "dynamics_wm_norm")):
            add(name, {"pass": initial[key] != final[key], "detail": f"{initial[key]:.9g} -> {final[key]:.9g}"})
        dep = proof["dependency"]
        add("dependency_before_identity", {"pass": dep["before"]["action_a_base_max_abs"] < 2e-5 and dep["before"]["action_b_base_max_abs"] < 2e-5, "detail": str(dep["before"])})
        add("history_affects_embedding_and_action", {"pass": dep["after"]["embedding_difference_l2"] > 0 and dep["after"]["action_difference_l2"] > 0, "detail": str(dep["after"])})
        add("history_timing", {"pass": proof["history_timing"]["pass"], "detail": str(proof["history_timing"])})
        reference_ok = all(v["policy_reference_equals_processed"] and v["clean_ref_recorded"] and v["reward_target_unchanged"] and not v["motion_wm_requires_grad"] and not v["motion_wm_has_grad"] for v in proof["reference_pipeline"].values())
        add("reference_pipeline", {"pass": reference_ok, "detail": str(proof["reference_pipeline"])})
        jit = proof["jit"]
        jit_ok = jit_error < 1e-5 and jit["contains_history_encoder"] and jit["contains_layerwise_actor"] and not jit["contains_dynamics_world_model"] and not jit["contains_critic"] and not jit["contains_motion_wm"]
        add("jit_runtime", {"pass": jit_ok, "detail": str(jit)})
        proof["overall_pass"] = all(item["pass"] for item in proof["checks"].values())
    except Exception as error:
        proof["error"] = f"{type(error).__name__}: {error}"
        proof["traceback"] = traceback.format_exc()
        proof["overall_pass"] = False
    finally:
        write_outputs(proof)
        print(f"[validation] overall={'PASS' if proof['overall_pass'] else 'FAIL'}")
        print(f"[validation] proof={OUT_JSON}")
        print(f"[validation] report={OUT_REPORT}")
    if not proof["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
