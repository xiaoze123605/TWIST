from pathlib import Path
import sys

__version__ = '0.1.0'

_root = Path(__file__).resolve().parents[1]
for _directory in ('pose', 'legged_gym', 'rsl_rl'):
    sys.path.insert(0, str(_root / _directory))
