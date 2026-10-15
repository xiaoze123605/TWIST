"""Offline probe of the training reference root-speed / yaw-rate statistics,
mirroring the detector inputs used by _transition_weights (no simulator)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'legged_gym'), str(ROOT/'rsl_rl'), str(ROOT/'pose')]
import isaacgym  # must precede torch
import torch
from pose.utils.motion_lib_pkl import MotionLib

YAML = ROOT/'legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml'
lib = MotionLib(motion_file=str(YAML), device='cpu')
n = lib.num_motions()
print('motions:', n, 'total time:', round(lib.get_total_length(), 1), 's')

ids = lib.sample_motions(n=4096)
times = lib.sample_time(ids, minimum_remaining_time=5.0)
_, root_rot, root_vel, root_ang_vel, _, _, _ = lib.calc_motion_frame(ids, times)
v = root_vel.norm(dim=-1)          # (4096, 1) ref speed at random frame
turn = root_ang_vel[..., 2:3].abs()
print('v   : p10 %.3f p50 %.3f p90 %.3f max %.3f' % (
    v.quantile(.1).item(), v.median().item(), v.quantile(.9).item(), v.max().item()))
print('turn: p50 %.4f p90 %.4f p99 %.4f max %.4f rad/s' % (
    turn.median().item(), turn.quantile(.9).item(), turn.quantile(.99).item(), turn.max().item()))

# Simulate the EMA detector along random 10s trajectories (50 Hz)
print('\nEMA detector simulation (tau=1s, alpha=0.02) over 10s trajectories:')
T = 500
frac = []
all_w = []
walk_devs = []
for trial in range(12):
    mid = int(torch.randint(n, (1,)))
    t0 = float(torch.rand(1) * (lib.get_motion_length(torch.tensor([mid])) - 10.5))
    ts = torch.full((T,), mid, dtype=torch.int64)
    tt = t0 + torch.arange(T, dtype=torch.float32) / 50.
    _, _, rv, rav, _, _, _ = lib.calc_motion_frame(ts, tt)
    vv = rv.norm(dim=-1)
    ema = vv[0].clone()
    devs = []
    for s in range(T):
        ema = ema + 0.02 * (vv[s] - ema)
        devs.append((vv[s] - ema).abs().item())
    d = torch.tensor(devs)
    # final detector: dev^2 + turn^2 only (no speed band)
    score = torch.maximum((d / .4).square(), (rav[:, 2].abs() / 1.5).square()).clamp(max=1.)
    w = 1 + 9 * score
    frac.append((w > 2).float().mean().item())
    all_w.append(w)
    is_walk = vv.mean() > 0.5
    if is_walk:
        walk_devs.append(d)
    print('trial %d: v_mean %.2f v_std %.2f dev_p90 %.3f | w_mean %.2f frac>2 %.2f %s' % (
        trial, vv.mean().item(), vv.std().item(), d.quantile(.9).item(), w.mean().item(), frac[-1],
        '[walking]' if is_walk else ''))
print('overall frac>2: %.2f' % (sum(frac)/len(frac)))
aw = torch.cat(all_w)
print('w dist: p50 %.2f p90 %.2f p95 %.2f | frac>2 %.2f frac>5 %.2f frac>8 %.2f' % (
    aw.median().item(), aw.quantile(.9).item(), aw.quantile(.95).item(),
    (aw > 2).float().mean().item(), (aw > 5).float().mean().item(), (aw > 8).float().mean().item()))
if walk_devs:
    wd = torch.cat(walk_devs)
    print('walking-only dev: p50 %.3f p90 %.3f p95 %.3f' % (
        wd.median().item(), wd.quantile(.9).item(), wd.quantile(.95).item()))

