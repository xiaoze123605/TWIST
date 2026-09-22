"""Select and materialize the lexicographically best fixed-phase checkpoint."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", nargs=3,
                        metavar=("NAME", "CHECKPOINT", "EVALUATION"), required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    output, manifest_path = Path(args.output_checkpoint), Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("best output or manifest already exists")
    candidates = []
    for name, checkpoint, evaluation in args.candidate:
        report = json.loads(Path(evaluation).read_text())
        durations = report["first_episode_duration_s"]
        metrics = dict(completion_rate=float(report["first_episode_completion_rate"]),
                       mean_duration_s=sum(durations)/len(durations),
                       root_height_rmse_m=float(report["root_height_rmse_m"]),
                       joint_rmse_rad=float(report["joint_rmse_rad"]))
        # Higher completion/duration, then lower height/joint RMSE.
        key = (metrics["completion_rate"], metrics["mean_duration_s"],
               -metrics["root_height_rmse_m"], -metrics["joint_rmse_rad"])
        candidates.append(dict(name=name, checkpoint=str(Path(checkpoint).resolve()),
                               evaluation=str(Path(evaluation).resolve()), metrics=metrics,
                               selection_key=key))
    best = max(candidates, key=lambda item: tuple(item["selection_key"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["checkpoint"], output)
    manifest = dict(priority=["completion_rate", "mean_duration_s",
                              "min_root_height_rmse_m", "min_joint_rmse_rad"],
                    selected=best, candidates=candidates,
                    output_checkpoint=str(output.resolve()))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
