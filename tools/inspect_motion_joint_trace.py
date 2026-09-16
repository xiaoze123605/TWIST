"""Per-joint clean-reference errors for a completed paired evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np


def inspect(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()][24:]
    actual = np.asarray([r['joint'] for r in rows])
    clean = np.asarray([r['clean'][8:] for r in rows])
    processed = np.asarray([r['processed'][8:] for r in rows])
    result = dict(frames=len(rows), joint_rmse=float(np.sqrt(np.mean((actual-clean)**2))),
                  per_joint_rmse=np.sqrt(np.mean((actual-clean)**2, axis=0)).tolist(),
                  reference_rmse=float(np.sqrt(np.mean((processed-clean)**2))),
                  lower_body_rmse=float(np.sqrt(np.mean((actual[:, :12]-clean[:, :12])**2))),
                  upper_body_rmse=float(np.sqrt(np.mean((actual[:, 15:]-clean[:, 15:])**2))),
                  actual_range=(np.percentile(actual,95,axis=0)-np.percentile(actual,5,axis=0)).tolist(),
                  clean_range=(np.percentile(clean,95,axis=0)-np.percentile(clean,5,axis=0)).tolist())
    # Descriptive lag scan only: primary RMSE above always uses the same frame.
    scores = []
    for lag in range(-15,16):
        a = actual[15:-15]
        b = clean[15-lag:len(clean)-15-lag]
        scores.append(float(np.mean((a-b)**2)))
    result['best_descriptive_lag_ms'] = (int(np.argmin(scores))-15)*20
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    results = {str(p.parent.relative_to(args.directory)): inspect(p)
               for p in sorted(args.directory.glob('*/*/*/frames.jsonl'))
               if (p.parent/'summary.json').exists()}
    (args.directory/'joint_diagnostics.json').write_text(json.dumps(results, indent=2)+'\n')
    for name, r in results.items():
        print(name, 'joint',round(r['joint_rmse'],6), 'legs',round(r['lower_body_rmse'],6),
              'arms',round(r['upper_body_rmse'],6), 'lag_ms',r['best_descriptive_lag_ms'])
