# Lower-body training experiment

Task: `g1_motion_wm_dtera_legs`. Inherits the V2 policy and rewards, adding
clean-reference per-joint position and velocity rewards for twelve hip, knee
and ankle joints. Name-based indexing validates all twelve names and applies
identical left/right weights. Position reward is mean exp(-error^2/0.20^2),
weight 0.4. Velocity reward uses sigma 2 rad/s, weight 0.1. These values are
experimental initial settings, not validated optimal hyperparameters.

Existing whole-body, root and slip rewards remain active. No symmetry-of-pose
penalty is imposed: the motion may legitimately be asymmetric. PD gains,
torque limits and residual bounds are unchanged. Policy observations remain
3695-D; export with `--preset motion_wm_v2`. Existing JITs do not acquire these
training changes until a new checkpoint is trained and exported.

Fresh training (choose a new experiment directory):

```bash
cd '/home/hank/TWIST（anyadapter）'
OMP_NUM_THREADS=1 \
PYTHONPATH='/home/hank/TWIST（anyadapter）/pose:/home/hank/TWIST（anyadapter）/legged_gym:/home/hank/TWIST（anyadapter）/rsl_rl' \
/home/hank/anaconda3/envs/twist/bin/python \
  legged_gym/legged_gym/scripts/train.py \
  --task g1_motion_wm_dtera_legs \
  --proj_name g1_motion_wm_dtera_legs --exptid legs_fresh_seed42_v1 \
  --motion_file '/home/hank/TWIST（anyadapter）/legged_gym/motion_data_configs/twist_dataset.yaml' \
  --num_envs 64 --max_iterations 1500 --seed 42 \
  --device cuda:0 --rl_device cuda:0 --no_wandb --fix_action_std
```

Evaluate checkpoints at 500, 1000 and 1500 on fixed motions and seeds against
V2-1400 and frozen TWIST. Report twelve individual joint errors, lower-body
RMSE, root velocity/orientation, max tilt and action rate. Do not select by
raw total reward across tasks because reward weights changed. Keep the
mydata3 motion as a development case, not the only test motion. Foot contact
timing and slip causality are not established by the new joint rewards.

Validation: two unit tests passed; actual two-environment, three-update CUDA
smoke completed on mydata3/24_seg00.pkl. 98 restored frames, frozen WM/base
weights, changed adapter weights, finite 3695-D observations. Evidence:
`legged_gym/logs/g1_motion_wm_dtera_legs/legs_smoke_seed42/smoke_proof.json`.
No long training run or improved-checkpoint claim is made by this smoke.
