# TWIST baseline anchored adapter

The `g1_twist_baseline_adapter` task keeps `G1MimicDistill` and its TWIST
student observation, reward, reset, curriculum, and PD control. The frozen
0529 student JIT supplies the tracking policy. A zero-initialized layerwise
AnyAdapter and a separately optimized action world model use an additional
79-frame dynamics history. The motion file is the full prepared train split;
validation and test motions stay outside training.

Run the CPU-only contract check before training:

```bash
LD_LIBRARY_PATH=/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=2 /home/hank/anaconda3/envs/twist/bin/python \
tools/check_twist_baseline_adapter.py
```

After the current GPU training has ended, start this as a **separate** run:

```bash
LD_LIBRARY_PATH=/home/hank/anaconda3/envs/twist/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=2 /home/hank/anaconda3/envs/twist/bin/python -u \
legged_gym/legged_gym/scripts/train.py \
  --task g1_twist_baseline_adapter \
  --proj_name g1_twist_baseline_adapter \
  --exptid anchored_full_v1 \
  --num_envs 4096 --max_iterations 30000 --seed 42 --no_wandb
```

Checkpoints are saved every 500 iterations. Export a selected checkpoint with
`legged_gym/scripts/export_twist_anyadapter_opentrack_jit.py`, then compare it
against the base JIT on the same held-out motions, motion server settings,
MuJoCo model, and seed. Include easy, locomotion, and dynamic motions. A
checkpoint must improve the aggregate result without a material increase in
falls or loss of easy-motion stability; iteration count alone is not a
selection criterion. First assess raw references. The existing high-level
`--reference-mode wm` path can then be tested as a separate ablation; do not
assume the reference model improves clean mocap inputs.

The initial adapter was checked against the original TWIST JIT: CPU actor
output max difference was zero, and a 380-frame MuJoCo walk gave exactly the
same tracking metrics. This is an initialization property, not a guarantee
about later trained checkpoints.
