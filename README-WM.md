# Motion-WM + DTERA 运行指令

以下命令在 `/home/hank/TWIST（anyadapter）` 下执行。示例动作统一使用 `mydata3/24_seg00.pkl`，污染使用训练一致的 `formal`、seed 42，motion speed/scale/hip-yaw scale 均为默认的 1.0。Motion-WM checkpoint 属于**高层 reference 修复**；DTERA JIT 属于**低层动作策略**，两者不能互换。

先确认 Redis 可用：`redis-cli ping` 应返回 `PONG`。高层与低层在两个终端运行，Redis 端口默认均为 6379。每次对比换一个新的 `--plot-dir`，重新启动两层服务；不要在同步单次实验里加 `--loop`。

## 1. 高层：发布 motion reference

终端 A（WM + formal corruption）：

```bash
cd '/home/hank/TWIST（anyadapter）/deploy_real'
/home/hank/anaconda3/envs/twist/bin/python server_high_level_motion_lib.py \
  --motion_file '/home/hank/TWIST（anyadapter）/track_dataset/twist_motion_dataset/mydata3/24_seg00.pkl' \
  --device cpu --reference-mode wm \
  --wm-checkpoint '/home/hank/TWIST（anyadapter）/legged_gym/logs/motion_world_model/full_stable_v2/best.pt' \
  --corruption-preset formal --seed 42 \
  --motion-speed 1.0 --motion-scale 1.0 --hip-yaw-scale 1.0 \
  --wait-for-sim-ready --sim-ready-timeout 120
```

比较时仅替换 `--reference-mode`：`clean` 为原始 clean reference，`corrupt` 为污染 reference，`wm` 为 WM 修复后的 reference。三个模式使用同一个 motion 与 seed；clean ground truth 仍用于三组统一的 tracking 指标。`--vis` 可选，仅打开高层可视化。

## 2. 低层：运行 MuJoCo 与曲线诊断

终端 B（V2-1400 DTERA JIT，需先执行第 3 节导出）：

```bash
cd '/home/hank/TWIST（anyadapter）/deploy_real'
/home/hank/anaconda3/envs/twist/bin/python server_low_level_g1_sim.py \
  --policy_path '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_dtera_v2/fresh_v2_seed42/traced/fresh_v2_seed42-1400-motion_wm_v2-jit.pt' \
  --device cpu --require-dtera --seed 42 \
  --sim_duration 67.4 --headless \
  --plot-dir '/home/hank/TWIST（anyadapter）/sim_analysis/wm_dtera_v2_1400_seed42'
```

`--plot-dir` 自动开启逐帧同步，并在新目录中保存 `frames.jsonl`、`summary.json`、CSV 和诊断 PNG 曲线；该目录运行前**不能已存在**。如果要保存视频，追加 `--record_video --video_path '/home/hank/TWIST（anyadapter）/sim_analysis/wm_dtera_v2_1400_seed42.mp4'`。DTERA 正确加载时必须打印 `[DTERA] Detected 3695-D dual-history policy`；1155-D 原始 TWIST JIT 不应加 `--require-dtera`。

原始 TWIST 对照只需将低层 `--policy_path` 换为：

```text
/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt
```

同时删去 `--require-dtera`，换一个新的 `--plot-dir`。这样可以分别运行 corrupt + TWIST、WM + TWIST、corrupt + DTERA、WM + DTERA、clean + DTERA；每组都重新启动高层和低层，并保持 motion、seed、duration 一致。

## 3. 将 DTERA checkpoint 转为 JIT

立即可用的 V2-1400（导出器参数与 V2 训练配置匹配）：

```bash
cd '/home/hank/TWIST（anyadapter）'
/home/hank/anaconda3/envs/twist/bin/python legged_gym/scripts/export_twist_dtera_jit.py \
  --preset motion_wm_v2 \
  --ckpt '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_dtera_v2/fresh_v2_seed42/model_1400.pt' \
  --out '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_dtera_v2/fresh_v2_seed42/traced/fresh_v2_seed42-1400-motion_wm_v2-jit.pt' \
  --device cpu
```

第 4 节的新训练达到 1500 轮后，导出新下肢策略（若评估选择 500 或 1000 轮，则相应替换文件名）：

```bash
cd '/home/hank/TWIST（anyadapter）'
/home/hank/anaconda3/envs/twist/bin/python legged_gym/scripts/export_twist_dtera_jit.py \
  --preset motion_wm_v2 \
  --ckpt '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_dtera_legs/legs_fresh_seed42_v1/model_1500.pt' \
  --out '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_dtera_legs/legs_fresh_seed42_v1/traced/legs_fresh_seed42_v1-1500-motion_wm_v2-jit.pt' \
  --device cpu
```

低层运行新策略时，将 `--policy_path` 改成这个新 JIT 路径。不要将旧 V2-1400 checkpoint 当成新下肢任务的训练结果，也不要用默认的 `legacy` 导出 preset。

## 4. 从零训练下肢优化版 WM + DTERA

任务 `g1_motion_wm_dtera_legs` 在 V2 基础上增加左右对称权重的髋、膝、踝 clean-reference 跟踪奖励。以下命令不带 resume 参数，**不从旧 4800 或 V2-1400 checkpoint 接着训练**。`legs_fresh_seed42_v1` 仅用于一个全新实验目录；如目录已存在，请换新的 `--exptid`。

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

训练后优先检查 500、1000、1500 轮 checkpoint，在固定的多个未见 motion 与 seed 上和 V2-1400、原始 TWIST 做同条件比较。现有 V2-1400 JIT 不会自动获得新任务的下肢优化效果；必须重新训练并导出新 checkpoint。
训练环境可能安装了 `/home/hank/TWIST/pose` 的旧版本；上述 `PYTHONPATH` 确保读取当前项目中已修复 GPU 拼接峰值的 `pose/pose/utils/motion_lib_pkl.py`。如 `legs_fresh_seed42_v1` 目录已被使用，换一个新的 `--exptid`。
