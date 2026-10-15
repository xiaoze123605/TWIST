"""Terminal live dashboard for a PPO4900Improved training run.

Plain text, redraws in place in the terminal (ANSI), unicode sparklines.
No images, no browser. Ctrl+C to stop. Pure read-only — safe for the run.

Usage: python tools/live_4900_term.py [exptid] [--interval 5] [--tail 80]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPT_DIR = ROOT / 'legged_gym/logs/recovery_4900'
BARS = '▁▂▃▄▅▆▇█'


def read_jsonl(path):
    rows = []
    try:
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def spark(vals, width=60):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ''
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1
    return ''.join(BARS[int((v - lo) / (hi - lo) * (len(BARS) - 1))] for v in vals)[-width:]


def fmt(v, nd=4):
    return f'{v:.{nd}f}' if isinstance(v, (int, float)) else str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exptid', nargs='?', default='transanchor_norm_seed42_20260925')
    p.add_argument('--interval', type=float, default=5)
    p.add_argument('--tail', type=int, default=80)
    a = p.parse_args()

    while True:
        anchor = read_jsonl(EXPT_DIR / a.exptid / 'anchor_metrics.jsonl')[-a.tail:]
        train = read_jsonl(EXPT_DIR / a.exptid / 'train_metrics.jsonl')
        lines = []
        lines.append(f'\x1b[2J\x1b[H=== {a.exptid}  ({time.strftime("%F %T")}) ===')
        if not anchor:
            lines.append('no anchor_metrics.jsonl yet')
            sys.stdout.write('\n'.join(lines) + '\n')
            sys.stdout.flush()
            time.sleep(a.interval)
            continue
        d = anchor[-1]
        upd = [r['update'] for r in anchor]

        def series(*names):
            out = {}
            for n in names:
                out[n] = [r.get(n) for r in anchor]
            return out

        s = series('source_action_mae', 'source_anchor_kl', 'max_rollout_kl',
                   'anchor_coef_effective', 'transition_anchor_weight_mean',
                   'transition_fraction', 'v_p50', 'v_p90', 'dev_p50', 'dev_p90',
                   'turn_p90', 'std_mean', 'actor_lr', 'bank_size')
        n = len(upd)
        # iteration rate from the last two metric rows
        rate = ''
        if len(upd) >= 2:
            # anchor_metrics is written once per update; use wall clock spacing
            rate = ''
        lines.append(f'update {upd[-1]}    anchor_coef {fmt(d["anchor_coef_effective"])}    '
                     f'std {fmt(d["std_mean"])}    actor_lr {d["actor_lr"]:.2e}')
        rows = [
            ('action_mae   ', 'source_action_mae', 5, None),
            ('anchor_kl    ', 'source_anchor_kl', 5, None),
            ('rollout_kl   ', 'max_rollout_kl', 5, None),
            ('w_mean(raw)  ', 'transition_anchor_weight_mean', 2, None),
            ('frac w>2     ', 'transition_fraction', 3, None),
            ('v_p50 / p90  ', 'v_p90', 3, 'v_p50'),
            ('dev_p50 /p90 ', 'dev_p90', 3, 'dev_p50'),
            ('turn_p90     ', 'turn_p90', 2, None),
            ('std_mean     ', 'std_mean', 4, None),
            ('actor_lr     ', 'actor_lr', 0, None),
            ('bank_size    ', 'bank_size', 0, None),
        ]
        for label, key, nd, key2 in rows:
            v = s[key]
            if key == 'actor_lr' and key in d:
                head = f'{d[key]:.2e}'
            elif key == 'bank_size' and key in d:
                head = str(int(d[key]))
            else:
                head = fmt(d[key], nd) if key in d else '—'
            if key2 and key2 in s and key2 in d:
                head = f'{fmt(d[key2], 3)} / {head}'
            if key == 'actor_lr':
                sp = spark([x if x is None else __import__('math').log10(max(x, 1e-12)) for x in v])
            else:
                sp = spark(v)
            lines.append(f'{label} {head:>14}  {sp}')
        if train:
            it = [r.get('iteration', i) for i, r in enumerate(train)]
            for key in ('Total/Rewards/mean', 'Mean reward'):
                if all(key in r for r in train):
                    rw = [r[key] for r in train]
                    lines.append(f'reward        {fmt(rw[-1], 3):>14}  {spark(rw[-a.tail:])}')
                    break
        lines.append('(Ctrl+C 退出；纯读取，不影响训练)')
        sys.stdout.write('\n'.join(lines) + '\n')
        sys.stdout.flush()
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
