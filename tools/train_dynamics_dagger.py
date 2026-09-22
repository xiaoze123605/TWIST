"""Train the existing DynamicsTracker actor from cumulative stratified replay."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "legged_gym"), str(ROOT / "rsl_rl"), str(ROOT / "pose")]


def error_metrics(error):
    import torch
    absolute = error.abs().flatten()
    return dict(mean_abs_rad=float(absolute.mean()), rmse_rad=float(error.square().mean().sqrt()),
                p95_abs_rad=float(torch.quantile(absolute, .95)), max_abs_rad=float(absolute.max()))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--template-checkpoint", required=True,
                        help="Defines the existing actor/deployment contract; weights are ignored unless also used as --init-checkpoint")
    parser.add_argument("--dynamics-checkpoint",
                        help="Optional trained History Encoder/WM weights; actor weights are never loaded from it")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    custom, remaining = parser.parse_known_args()
    if custom.epochs < 1 or custom.batch_size < 1 or custom.learning_rate <= 0:
        raise ValueError("invalid DAgger optimization setting")
    for path in (custom.output_checkpoint, custom.report):
        if Path(path).exists():
            raise FileExistsError(path)
    import torch
    from torch.nn import functional as F
    from rsl_rl.datasets.dynamics_dagger_buffer import DynamicsDaggerBuffer
    from rsl_rl.modules.dynamics_tracker import DynamicsTrackerActorCritic, STATE_DIM

    if remaining:
        raise ValueError("unknown DAgger trainer arguments: " + " ".join(remaining))
    if custom.task not in ("g1_dynamics_tracker_wide", "g1_dynamics_tracker_adaptive_wide"):
        raise ValueError("DAgger training requires a wide DynamicsTracker task")
    torch.manual_seed(custom.seed)
    template = torch.load(custom.template_checkpoint, map_location="cpu")
    train_cfg = copy.deepcopy(template["train_cfg"])
    use_latent = custom.task == "g1_dynamics_tracker_adaptive_wide"
    train_cfg["policy"]["use_dynamics_latent"] = use_latent
    deployment_spec = copy.deepcopy(template["deployment_spec"])
    deployment_spec["use_dynamics_latent"] = use_latent
    actor_critic = DynamicsTrackerActorCritic(
        deployment_spec["observation_dim"], template["critic_dim"], 23,
        **train_cfg["policy"]).to(custom.rl_device)
    if custom.init_checkpoint:
        initial = torch.load(custom.init_checkpoint, map_location=custom.rl_device)
        old_spec = dict(initial["deployment_spec"])
        old_spec.pop("use_dynamics_latent", None)
        new_spec = dict(deployment_spec)
        new_spec.pop("use_dynamics_latent", None)
        if old_spec != new_spec:
            raise ValueError("initial checkpoint deployment contract differs")
        actor_critic.load_state_dict(initial["model_state_dict"], strict=True)
        if use_latent and not initial["train_cfg"]["policy"]["use_dynamics_latent"]:
            with torch.no_grad():
                actor_critic.actor[0].weight[:, -actor_critic.latent_dim:] = 0
    if custom.dynamics_checkpoint:
        dynamics = torch.load(custom.dynamics_checkpoint, map_location=custom.rl_device)
        state = dynamics["model_state_dict"]
        actor_critic.history_encoder.load_state_dict({
            key[len("history_encoder."):]: value for key, value in state.items()
            if key.startswith("history_encoder.")})
        actor_critic.world_model.load_state_dict({
            key[len("world_model."):]: value for key, value in state.items()
            if key.startswith("world_model.")})
    actor_critic.train()
    replay = DynamicsDaggerBuffer.load(custom.dataset)
    train_ids, validation_ids = replay.split_train_validation(custom.validation_fraction)
    if train_ids.numel() < 1 or validation_ids.numel() < 1:
        raise ValueError("DAgger replay split is empty")
    latest_round = int(replay.tensors["collection_round"].max())
    samples_per_epoch = custom.samples_per_epoch or train_ids.numel()
    optimizer = torch.optim.Adam(actor_critic.actor.parameters(), lr=custom.learning_rate)
    generator = torch.Generator().manual_seed(int(custom.seed) + latest_round * 1009)
    sampled_source = torch.zeros(3, dtype=torch.long)
    epoch_losses = []
    scales = torch.tensor(deployment_spec["target_scales"], device=custom.rl_device)
    lower = torch.tensor(deployment_spec["joint_lower"], device=custom.rl_device)
    upper = torch.tensor(deployment_spec["joint_upper"], device=custom.rl_device)
    joint_names = deployment_spec["joint_names"]

    def batch_values(ids):
        inputs = replay.tensors["actor_input"][ids].to(custom.rl_device)
        teacher_target = replay.tensors["teacher_target"][ids].to(custom.rl_device)
        reference_q = inputs[:, STATE_DIM+8:STATE_DIM+31]
        normalized = ((teacher_target-reference_q)/scales).clamp(-.999, .999)
        return inputs, teacher_target, reference_q, normalized

    for _ in range(custom.epochs):
        ids = replay.stratified_indices(train_ids, samples_per_epoch, latest_round,
                                        generator=generator)
        sampled_source += torch.bincount(replay.tensors["source"][ids], minlength=3)
        losses = []
        for start in range(0, ids.numel(), custom.batch_size):
            batch = ids[start:start+custom.batch_size]
            inputs, _, _, normalized = batch_values(batch)
            prediction = actor_critic.actor(inputs)
            loss = F.smooth_l1_loss(torch.tanh(prediction), normalized)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.actor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(sum(losses)/len(losses))

    actor_critic.eval()

    def evaluate(ids):
        total_loss = total_count = 0
        errors = []
        with torch.no_grad():
            for start in range(0, ids.numel(), custom.batch_size):
                batch = ids[start:start+custom.batch_size]
                inputs, teacher_target, reference_q, normalized = batch_values(batch)
                raw = actor_critic.actor(inputs)
                total_loss += float(F.smooth_l1_loss(torch.tanh(raw), normalized,
                                                    reduction="sum"))
                total_count += normalized.numel()
                predicted_target = torch.maximum(torch.minimum(
                    reference_q + scales*torch.tanh(raw), upper), lower)
                errors.append((predicted_target-teacher_target).cpu())
        error = torch.cat(errors)
        groups = {"all_joints": list(range(23)),
                  "hips": [i for i,n in enumerate(joint_names) if "hip" in n],
                  "knees": [i for i,n in enumerate(joint_names) if "knee" in n],
                  "ankles": [i for i,n in enumerate(joint_names) if "ankle" in n],
                  "upper_body": [i for i,n in enumerate(joint_names)
                                 if any(part in n for part in ("waist", "shoulder", "elbow"))]}
        for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint",
                     "left_hip_roll_joint", "right_hip_roll_joint",
                     "left_hip_yaw_joint", "right_hip_yaw_joint"):
            if name in joint_names:
                groups[name] = [joint_names.index(name)]
        return total_loss/total_count, {name: error_metrics(error[:, indices])
                                       for name, indices in groups.items() if indices}

    train_loss, train_errors = evaluate(train_ids)
    validation_loss, validation_errors = evaluate(validation_ids)

    # Approximate local ambiguity: nearby held-out observations should not ask
    # for incompatible teacher targets if the Stage-A observation is sufficient.
    ambiguity_generator = torch.Generator().manual_seed(99173)
    ambiguity_order = torch.randperm(validation_ids.numel(), generator=ambiguity_generator)
    ambiguity_ids = validation_ids[ambiguity_order[:min(2048, validation_ids.numel())]]
    ambiguity_input = replay.tensors["actor_input"][ambiguity_ids].float()
    projection_generator = torch.Generator().manual_seed(99174)
    projection = torch.randn(ambiguity_input.shape[1], min(32, ambiguity_input.shape[1]),
                             generator=projection_generator) / math.sqrt(ambiguity_input.shape[1])
    projected = ambiguity_input @ projection
    projected = (projected-projected.mean(0))/projected.std(0).clamp_min(1e-5)
    distances = torch.cdist(projected, projected)
    distances.fill_diagonal_(float("inf"))
    nearest_distance, nearest = distances.min(1)
    target = replay.tensors["teacher_target"][ambiguity_ids]
    neighbor_error = target-target[nearest]
    ambiguity = dict(samples=int(ambiguity_ids.numel()),
                     nearest_observation_distance_mean=float(nearest_distance.mean()),
                     nearest_teacher_target_rmse_rad=float(neighbor_error.square().mean().sqrt()),
                     nearest_teacher_target_p95_abs_rad=float(torch.quantile(neighbor_error.abs(), .95)))

    validation_source_error = {}
    for source_id, source_name in enumerate(("teacher", "old_student", "current_student")):
        ids = validation_ids[replay.tensors["source"][validation_ids] == source_id]
        if ids.numel():
            _, metrics = evaluate(ids)
            validation_source_error[source_name] = dict(samples=int(ids.numel()),
                                                         all_joints=metrics["all_joints"])

    dataset_file_sha = DynamicsDaggerBuffer.file_sha256(custom.dataset)
    checkpoint_metadata = dict(
        dagger_dataset_sha256=dataset_file_sha,
        dagger_dataset_content_sha256=replay.content_sha256(),
        dagger_dataset_sample_count=len(replay), dagger_collection_round=latest_round,
        dynamics_checkpoint=(str(Path(custom.dynamics_checkpoint).resolve())
                             if custom.dynamics_checkpoint else None))
    output = Path(custom.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict=actor_critic.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), wm_optimizer_state_dict={},
                    iter=0, train_cfg=train_cfg, deployment_spec=deployment_spec,
                    critic_dim=template["critic_dim"], metadata=checkpoint_metadata), output)
    sampled_total = int(sampled_source.sum())
    report = dict(dataset=str(Path(custom.dataset).resolve()), dataset_sha256=dataset_file_sha,
                  dataset_sample_count=len(replay), collection_round=latest_round,
                  dataset_source_counts=replay.source_counts(), train_samples=int(train_ids.numel()),
                  validation_samples=int(validation_ids.numel()), epochs=custom.epochs,
                  sampled_source_fraction=dict(
                      teacher=float(sampled_source[0]/sampled_total),
                      old_student=float(sampled_source[1]/sampled_total),
                      current_student=float(sampled_source[2]/sampled_total)),
                  final_epoch_imitation_loss=epoch_losses[-1],
                  train_imitation_loss=train_loss, validation_imitation_loss=validation_loss,
                  imitation_gap=validation_loss-train_loss,
                  train_target_error=train_errors, validation_target_error=validation_errors,
                  validation_source_target_error=validation_source_error,
                  observation_ambiguity=ambiguity,
                  checkpoint=str(output.resolve()), checkpoint_metadata=checkpoint_metadata)
    report_path = Path(custom.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
