import argparse
import json
from pathlib import Path
import sys

import isaacgym

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twist_supermimic.data import prepare


def main():
    parser = argparse.ArgumentParser(description='仅转换已有TWIST动作，不下载新数据')
    parser.add_argument('--source', type=Path, default=ROOT/'legged_gym/motion_data_configs/twist_dataset.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/twist_supermimic_local/motions.pt')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.limit), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
