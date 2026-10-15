"""Live-updating PNG dashboard for a PPO4900Improved training run.

Reads the run's anchor_metrics.jsonl (adapter/anchor diagnostics) and
train_metrics.jsonl (reward metrics) and re-renders a 2x2 panel to a PNG
every --interval seconds. Open the PNG in the VSCode image viewer; it
refreshes in place. Headless-safe (Agg backend). Ctrl+C to stop.

Usage: python tools/live_4900.py [exptid] [--interval 30] [--tail 600]
"""
import argparse, json, sys, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPT_DIR = ROOT / 'legged_gym/logs/recovery_4900'

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

def plot(exptid, tail, png):
    anchor = read_jsonl(EXPT_DIR / exptid / 'anchor_metrics.jsonl')
    train = read_jsonl(EXPT_DIR / exptid / 'train_metrics.jsonl')
    if not anchor:
        print(f'no anchor_metrics.jsonl yet for {exptid}', flush=True)
        return
    anchor = anchor[-tail:]
    upd = [d['update'] for d in anchor]

    def series(*names):
        return {n: [d[n] for d in anchor] for n in names if all(n in d for d in anchor)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'{exptid}  (latest update {upd[-1]})', fontsize=12)

    ax = axes[0][0]
    ax.set_title('Adapter deviation from source')
    s = series('source_action_mae', 'source_anchor_kl', 'max_rollout_kl')
    for n in ('source_action_mae', 'max_rollout_kl'):
        if n in s:
            ax.plot(upd, s[n], label=n, linewidth=1)
    if 'source_anchor_kl' in s:
        ax2 = ax.twinx(); ax2.plot(upd, s['source_anchor_kl'], 'g-', linewidth=1, alpha=.8, label='source_anchor_kl')
        ax2.set_ylabel('KL', color='g'); ax2.tick_params(axis='y', labelcolor='g')
    ax.legend(loc='upper left', fontsize=7); ax.grid(alpha=.3)

    ax = axes[0][1]
    ax.set_title('Anchor pressure')
    s = series('anchor_coef_effective', 'transition_anchor_weight_mean', 'transition_fraction')
    if 'anchor_coef_effective' in s: ax.plot(upd, s['anchor_coef_effective'], label='anchor_coef_effective', linewidth=1)
    if 'transition_anchor_weight_mean' in s: ax.plot(upd, s['transition_anchor_weight_mean'], label='w_mean (raw)', linewidth=1)
    if 'transition_fraction' in s: ax.plot(upd, s['transition_fraction'], label='frac w>2', linewidth=1)
    ax.legend(loc='upper left', fontsize=7); ax.grid(alpha=.3)

    ax = axes[1][0]
    ax.set_title('Transition detector inputs')
    s = series('v_p50', 'v_p90', 'dev_p50', 'dev_p90', 'turn_p90')
    for n in ('v_p50', 'v_p90', 'dev_p50', 'dev_p90', 'turn_p90'):
        if n in s: ax.plot(upd, s[n], label=n, linewidth=1)
    ax.legend(loc='upper left', fontsize=7); ax.grid(alpha=.3)

    ax = axes[1][1]
    ax.set_title('Exploration / LR')
    s = series('std_mean', 'actor_lr', 'bank_size')
    if 'std_mean' in s: ax.plot(upd, s['std_mean'], label='std_mean', linewidth=1)
    ax2 = ax.twinx()
    if 'actor_lr' in s:
        ax2.semilogy(upd, s['actor_lr'], 'r-', linewidth=1, alpha=.8, label='actor_lr')
        ax2.set_ylabel('LR (log)', color='r'); ax2.tick_params(axis='y', labelcolor='r')
    if 'bank_size' in s: ax.plot(upd, s['bank_size'], label='bank_size', linewidth=1)
    ax.legend(loc='upper left', fontsize=7); ax.grid(alpha=.3)

    if train:
        it = [d.get('iteration', i) for i, d in enumerate(train)]
        for key in ('Total/Rewards/mean', 'Mean reward', 'Mean_episode_rew_tracking_keybody_pos'):
            if all(key in d for d in train):
                axes[0][0].plot(it, [d[key] for d in train], 'k--', linewidth=.7, alpha=.5, label='total reward')
                axes[0][0].legend(loc='upper left', fontsize=7)
                break
    fig.tight_layout(rect=[0, 0, 1, .96])
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=90)
    plt.close(fig)
    print(f'{time.strftime("%T")} rendered {png}  (updates {upd[0]}..{upd[-1]}, {len(anchor)} points)', flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('exptid', nargs='?', default='transanchor_norm_seed42_20260925')
    p.add_argument('--interval', type=float, default=30)
    p.add_argument('--tail', type=int, default=600)
    a = p.parse_args()
    png = ROOT / 'reports/recovery_4900' / f'{a.exptid}_live.png'
    while True:
        try:
            plot(a.exptid, a.tail, png)
        except Exception as e:
            print(f'plot error: {e}', flush=True)
        time.sleep(a.interval)

if __name__ == '__main__':
    main()
