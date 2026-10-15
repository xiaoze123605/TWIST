"""Promote a stable adapter only after paired short and complete-motion screens."""

import argparse
import json
from pathlib import Path


WEIGHTS = {"joint_rmse": 0.4, "root_velocity_rmse": 0.3, "yaw_error": 0.3}


def index(rows, labels):
    result = {}
    for row in rows:
        if row["label"] in labels:
            key = (row["motion"], row["label"])
            if key in result:
                raise ValueError(f"duplicate result: {key}")
            result[key] = row
    return result


def compare(reference, candidate, long_reference, long_candidate,
            incumbent="baseline_raw_150"):
    label_set = {r["label"] for r in candidate}
    if len(label_set) != 1:
        raise ValueError("candidate results require exactly one label")
    label = label_set.pop()
    reference = index(reference, {"base", incumbent})
    candidate = index(candidate, {label})
    motions = sorted({motion for motion, _ in candidate})
    if len(motions) != 12 or any((m, "base") not in reference or
                                 (m, incumbent) not in reference for m in motions):
        raise ValueError("candidate requires the same twelve paired reference clips")
    means = {}
    for policy, rows in (("base", reference), (incumbent, reference), (label, candidate)):
        means[policy] = {k: sum(rows[m, policy][k] for m in motions) / len(motions)
                         for k in WEIGHTS}
    base = means["base"]
    scores = {name: sum(weight * values[k] / base[k]
                        for k, weight in WEIGHTS.items())
              for name, values in means.items()}
    regression = {k: means[label][k] / means[incumbent][k]
                  for k in WEIGHTS}
    worst_clip = {k: max(candidate[m, label][k] /
                         max(reference[m, incumbent][k], 1e-9) for m in motions)
                  for k in ("joint_rmse", "root_velocity_rmse")}
    new_falls = [m for m in motions if candidate[m, label].get("fall_proxy", False)
                 and not reference[m, incumbent].get("fall_proxy", False)]
    long_reference = index(long_reference, {"base", incumbent})
    long_candidate = index(long_candidate, {label})
    long_motions = sorted({m for m, _ in long_candidate})
    if not long_motions or any((m, "base") not in long_reference or
                               (m, incumbent) not in long_reference for m in long_motions):
        raise ValueError("full-motion coverage is incomplete")
    long_ratios = {}
    for m in long_motions:
        current, previous = long_candidate[m, label], long_reference[m, incumbent]
        long_ratios[m] = {k: current[k] / max(previous[k], 1e-9) for k in WEIGHTS}
        if current.get("fall_proxy", False) and not previous.get("fall_proxy", False):
            new_falls.append(m)
    reasons = []
    if scores[label] > 0.99 * scores[incumbent]:
        reasons.append("12-clip score did not improve by 1%")
    for k in ("joint_rmse", "root_velocity_rmse"):
        if regression[k] > 1.01:
            reasons.append(f"mean {k} regressed by more than 1%")
        if worst_clip[k] > 1.05:
            reasons.append(f"one clip's {k} regressed by more than 5%")
    if new_falls:
        reasons.append("new fall proxy")
    for m, ratios in long_ratios.items():
        for k in ("joint_rmse", "root_velocity_rmse"):
            if ratios[k] > 1.01:
                reasons.append(f"full {Path(m).name} {k} regressed by more than 1%")
        if ratios["yaw_error"] > 1.03:
            reasons.append(f"full {Path(m).name} yaw regressed by more than 3%")
    return dict(candidate=label, incumbent=incumbent, accepted=not reasons,
                reasons=reasons, scores=scores, means=means,
                mean_ratios_to_incumbent=regression, worst_clip_ratios=worst_clip,
                new_falls=new_falls, full_motion_ratios=long_ratios)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--long-reference", type=Path, required=True)
    p.add_argument("--long-candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(*(json.loads(path.read_text()) for path in
                       (args.reference, args.candidate,
                        args.long_reference, args.long_candidate)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
