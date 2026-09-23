"""Select a TWIST adapter checkpoint from paired held-out MuJoCo results."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("joint_rmse", "root_error", "yaw_error", "root_velocity_rmse")


def select(rows, baseline_label="base", max_joint_regression=0.02,
           max_velocity_regression=0.02):
    grouped = defaultdict(dict)
    for row in rows:
        motion, label = row["motion"], row["label"]
        if label in grouped[motion]:
            raise ValueError(f"duplicate {label} result for {motion}")
        if any(not math.isfinite(float(row[key])) or float(row[key]) < 0 for key in METRICS):
            raise ValueError(f"non-finite or negative metric for {label} on {motion}")
        grouped[motion][label] = row
    if not grouped or any(baseline_label not in labels for labels in grouped.values()):
        raise ValueError("every motion needs a baseline result")
    labels = set.intersection(*(set(motion_rows) for motion_rows in grouped.values()))
    partial = set.union(*(set(motion_rows) for motion_rows in grouped.values())) - labels
    if partial:
        raise ValueError(f"incomplete paired motion coverage: {sorted(partial)}")
    motions = sorted(grouped)
    baseline = {key: sum(grouped[m][baseline_label][key] for m in motions) / len(motions)
                for key in METRICS}
    if any(value <= 0 for value in baseline.values()):
        raise ValueError("baseline metric must be positive")
    candidates = []
    for label in sorted(labels - {baseline_label}):
        means = {key: sum(grouped[m][label][key] for m in motions) / len(motions)
                 for key in METRICS}
        falls = sum(bool(grouped[m][label].get("fall_proxy", False)) for m in motions)
        base_falls = sum(bool(grouped[m][baseline_label].get("fall_proxy", False))
                         for m in motions)
        max_joint_ratio = max(grouped[m][label]["joint_rmse"] /
                              max(grouped[m][baseline_label]["joint_rmse"], 1e-9)
                              for m in motions)
        ratios = {key: means[key] / baseline[key] for key in METRICS}
        accepted = (falls <= base_falls and max_joint_ratio <= 1.05 and
                    ratios["joint_rmse"] <= 1 + max_joint_regression and
                    ratios["root_velocity_rmse"] <= 1 + max_velocity_regression)
        score = sum(ratios.values()) / len(ratios)
        candidates.append(dict(label=label, accepted=accepted, score=score,
                               means=means, ratios=ratios, falls=falls,
                               max_joint_ratio=max_joint_ratio))
    accepted = [row for row in candidates if row["accepted"]]
    winner = min(accepted, key=lambda row: row["score"])["label"] if accepted else None
    return dict(motions=motions, baseline=baseline, candidates=candidates,
                selected_label=winner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-label", default="base")
    args = parser.parse_args()
    result = select(json.loads(args.results.read_text()), args.baseline_label)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    if result["selected_label"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
