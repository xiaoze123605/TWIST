"""Persistent, cumulative and stratified replay for DynamicsTracker DAgger."""
import hashlib
import os
from pathlib import Path

import torch

from rsl_rl.modules.dynamics_tracker import CURRENT_DIM, STATE_DIM, TASK_DIM


SOURCE_TEACHER = 0
SOURCE_OLD_STUDENT = 1
SOURCE_CURRENT_STUDENT = 2
SOURCE_NAMES = ["teacher", "old_student", "current_student"]
REQUIRED_FIELDS = (
    "actor_input", "teacher_target", "motion_id", "motion_time", "motion_phase",
    "episode_id", "source", "steps_to_failure", "failure_margin_m", "seed",
    "action_mode", "target_scale_version", "collection_round",
)


def build_actor_input(observation, actor_critic, latent_mode="normal"):
    """Build the existing actor input without adding or changing a network branch."""
    current, history = actor_critic.split_obs(observation)
    latent = actor_critic.history_encoder(history).detach()
    if not actor_critic.use_dynamics_latent or latent_mode == "zero":
        latent = torch.zeros_like(latent)
    elif latent_mode == "shuffled":
        latent = latent[torch.randperm(latent.shape[0], device=latent.device)]
    elif latent_mode != "normal":
        raise ValueError("latent_mode must be normal, zero or shuffled")
    state = current[:, :STATE_DIM]
    task = current[:, STATE_DIM:STATE_DIM + TASK_DIM]
    q_error = task[:, 8:31] - state[:, 6:29]
    dq_error = task[:, 31:] - state[:, 29:52]
    return torch.cat((current, q_error, dq_error, latent), dim=-1)


class DynamicsDaggerBuffer:
    version = 1

    def __init__(self, tensors=None, metadata=None):
        self.tensors = tensors or {}
        self.metadata = metadata or {}
        self.validate()

    def __len__(self):
        return 0 if not self.tensors else int(self.tensors["actor_input"].shape[0])

    def validate(self):
        if not self.tensors:
            return
        missing = set(REQUIRED_FIELDS) - set(self.tensors)
        if missing:
            raise ValueError("missing replay fields: " + ", ".join(sorted(missing)))
        count = self.tensors["actor_input"].shape[0]
        if count < 1 or any(value.shape[0] != count for value in self.tensors.values()):
            raise ValueError("replay tensors have inconsistent sample counts")
        if self.tensors["actor_input"].ndim != 2 or self.tensors["teacher_target"].shape != (count, 23):
            raise ValueError("invalid DAgger actor input or teacher target shape")
        if not all(torch.isfinite(self.tensors[name]).all() for name in
                   ("actor_input", "teacher_target", "motion_time", "motion_phase",
                    "failure_margin_m")):
            raise ValueError("non-finite value in DAgger replay")
        if not torch.all((self.tensors["motion_phase"] >= 0) &
                         (self.tensors["motion_phase"] <= 1.00001)):
            raise ValueError("motion phase outside [0, 1]")

    @staticmethod
    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("version") != cls.version:
            raise ValueError("unsupported DAgger replay version")
        replay = cls(payload["tensors"], payload["metadata"])
        expected = payload.get("content_sha256")
        if expected and expected != replay.content_sha256():
            raise ValueError("DAgger replay content hash mismatch")
        return replay

    def content_sha256(self):
        digest = hashlib.sha256()
        for name in sorted(self.tensors):
            value = self.tensors[name].contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(version=self.version, tensors=self.tensors, metadata=self.metadata,
                        content_sha256=self.content_sha256()), temporary)
        os.replace(temporary, path)
        return self.file_sha256(path)

    def append_round(self, tensors, round_metadata):
        tensors = {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
        incoming = DynamicsDaggerBuffer(tensors, {})
        new_round = int(tensors["collection_round"][0])
        if torch.any(tensors["collection_round"] != new_round):
            raise ValueError("one append call must contain exactly one collection round")
        rounds = self.metadata.setdefault("rounds", [])
        if rounds and new_round <= max(int(item["round"]) for item in rounds):
            raise ValueError("collection round must increase; replay cannot be overwritten")
        if self.tensors:
            # Once a newer student is collected, older current-student rows
            # become historical rows. Teacher rows retain their identity.
            old_source = self.tensors["source"].clone()
            old_source[old_source == SOURCE_CURRENT_STUDENT] = SOURCE_OLD_STUDENT
            self.tensors["source"] = old_source
            for name in REQUIRED_FIELDS:
                if self.tensors[name].shape[1:] != incoming.tensors[name].shape[1:]:
                    raise ValueError("replay schema changed for " + name)
                self.tensors[name] = torch.cat((self.tensors[name], incoming.tensors[name]), 0)
        else:
            self.tensors = incoming.tensors
        rounds.append(dict(round_metadata))
        self.validate()

    def source_counts(self):
        if not self.tensors:
            return {name: 0 for name in SOURCE_NAMES}
        source = self.tensors["source"]
        return {name: int((source == index).sum()) for index, name in enumerate(SOURCE_NAMES)}

    def split_train_validation(self, validation_fraction=0.2):
        """Split by episode, preventing adjacent frames from leaking across splits."""
        episode = self.tensors["episode_id"].to(torch.int64)
        # Stable integer hash; include seed so separate collections do not alias.
        hashed = (episode * 1103515245 + self.tensors["seed"].to(torch.int64) * 12345) & 0x7fffffff
        denominator = max(2, round(1.0 / validation_fraction))
        validation = hashed.remainder(denominator) == 0
        if not validation.any() or validation.all():
            order = torch.arange(len(self))
            validation = order.remainder(denominator) == 0
        return (~validation).nonzero(as_tuple=False).flatten(), validation.nonzero(as_tuple=False).flatten()

    def stratified_indices(self, candidate_indices, count, latest_round=None,
                           ratios=(0.4, 0.3, 0.3), phase_bins=20,
                           recovery_steps=(25, 50), recovery_weight=2.0,
                           motion_balance=True, generator=None):
        """Sample teacher/history/latest strata with motion/phase balancing.

        Failure-adjacent samples receive a modest weight within each phase bin;
        source quotas prevent them from dominating the full replay.
        """
        candidate_indices = candidate_indices.to(torch.long).cpu()
        if candidate_indices.numel() == 0 or count < 1:
            raise ValueError("cannot sample an empty replay")
        source = self.tensors["source"][candidate_indices]
        rounds = self.tensors["collection_round"][candidate_indices]
        if latest_round is None:
            latest_round = int(rounds.max())
        masks = (
            source == SOURCE_TEACHER,
            (source != SOURCE_TEACHER) & (rounds < latest_round),
            (source != SOURCE_TEACHER) & (rounds == latest_round),
        )
        available = torch.tensor([bool(mask.any()) for mask in masks])
        requested = torch.tensor(ratios, dtype=torch.float64) * available
        if requested.sum() == 0:
            raise ValueError("no requested replay stratum is available")
        requested /= requested.sum()
        quotas = torch.floor(requested * count).to(torch.long)
        for index in torch.argsort(requested * count - quotas, descending=True)[:count-int(quotas.sum())]:
            quotas[index] += 1
        sampled = []
        phase = self.tensors["motion_phase"]
        motion = self.tensors["motion_id"].to(torch.long)
        steps = self.tensors["steps_to_failure"]
        for mask, quota in zip(masks, quotas.tolist()):
            if quota == 0:
                continue
            pool = candidate_indices[mask]
            bins = (phase[pool] * phase_bins).floor().clamp_max(phase_bins - 1).long()
            if motion_balance:
                # Remap arbitrary stable motion IDs to a compact range, then
                # balance each (motion, phase) cell. This first equalizes
                # motions and then phase coverage inside every motion.
                _, compact_motion = torch.unique(motion[pool], sorted=True,
                                                 return_inverse=True)
                cells = compact_motion * phase_bins + bins
                cell_count = torch.bincount(cells).clamp_min(1)
                weights = 1.0 / cell_count[cells].float()
                # The cell term alone gives motions with more occupied bins
                # more mass. Normalize that residual difference explicitly.
                unique_cells = torch.unique(cells)
                occupied = torch.bincount(
                    torch.div(unique_cells, phase_bins, rounding_mode="floor"),
                    minlength=int(compact_motion.max()) + 1).clamp_min(1)
                weights /= occupied[compact_motion].float().clamp_min(1.)
            else:
                bin_count = torch.bincount(bins, minlength=phase_bins).clamp_min(1)
                weights = 1.0 / bin_count[bins].float()
            near_failure = (steps[pool] >= recovery_steps[0]) & (steps[pool] <= recovery_steps[1])
            weights *= torch.where(near_failure, recovery_weight, 1.0)
            draw = torch.multinomial(weights, quota, replacement=quota > pool.numel(),
                                     generator=generator)
            sampled.append(pool[draw])
        result = torch.cat(sampled)
        return result[torch.randperm(result.numel(), generator=generator)]
