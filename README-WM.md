# Motion-WM + DTERA 运行指令

2026-09-16 训练前审计见 [TRAINING-PREFLIGHT.md](TRAINING-PREFLIGHT.md)。新训练请使用第 4 节的隔离清单，不再将原始全集 YAML 直接作为训练集。完整清单已通过 64 环境、3 次更新的短预检；这不是性能提升或实机安全证明。

以下命令在 `/home/hank/TWIST（anyadapter）` 下执行。示例动作统一使用 `mydata3/24_seg00.pkl`，不人工添加任何扰动：`raw` 直接使用 PKL 参考，`wm` 直接将同一 PKL 参考送入 Motion-WM 修复。seed 42 仅用于运行可复现，motion speed/scale/hip-yaw scale 均为 1.0。Motion-WM checkpoint 属于**高层 reference 修复**；DTERA JIT 属于**低层动作策略**，两者不能互换。

先确认 Redis 可用：`redis-cli ping` 应返回 `PONG`。高层与低层在两个终端运行，Redis 端口默认均为 6379。每次对比换一个新的 `--plot-dir`，重新启动两层服务；不要在同步单次实验里加 `--loop`。

## 1. 高层：发布 motion reference

终端 A（原始 PKL reference 直接由 WM 修复）：

```bash
cd '/home/hank/TWIST（anyadapter）/deploy_real'
/home/hank/anaconda3/envs/twist/bin/python server_high_level_motion_lib.py \
  --motion_file '/home/hank/TWIST（anyadapter）/track_dataset/twist_motion_dataset/mydata3/24_seg00.pkl' \
  --device cpu --reference-mode wm \
  --wm-checkpoint '/home/hank/TWIST（anyadapter）/legged_gym/logs/motion_world_model/full_stable_v2/best.pt' \
  --seed 42 \
  --motion-speed 1.0 --motion-scale 1.0 --hip-yaw-scale 1.0 \
  --wait-for-sim-ready --sim-ready-timeout 120
```

比较时仅替换 `--reference-mode`：`raw` 为原始 PKL reference 直通，`wm` 为 WM 对原始 PKL reference 的修复结果。部署端不再提供 `corrupt`、`formal` 或 `demo_stress` 模式。`--vis` 可选，仅打开高层可视化。

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

同时删去 `--require-dtera`，换一个新的 `--plot-dir`。四组对比为 raw + TWIST、WM(raw) + TWIST、raw + DTERA、WM(raw) + DTERA；每组都重新启动高层和低层，并保持 motion、seed、duration 一致。

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
  --ckpt '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_dtera_legs/legs_audited_seed42_v2/model_1500.pt' \
  --out '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_dtera_legs/legs_audited_seed42_v2/traced/legs_audited_seed42_v2-1500-motion_wm_v2-jit.pt' \
  --device cpu
```

低层运行新策略时，将 `--policy_path` 改成这个新 JIT 路径。不要将旧 V2-1400 checkpoint 当成新下肢任务的训练结果，也不要用默认的 `legacy` 导出 preset。

无桌面环境保存视频时，在低层 Python 命令前设置 `MUJOCO_GL=egl`，不要依赖 `xvfb-run`。本机已验证 EGL 可生成 MuJoCo 离屏视频。同步模式会在等待期间重复发布 frame request，避免高低层几乎同时启动时丢失 frame 0。失败运行产生的空 `frames.jsonl` 不会再触发曲线绘制或空视频渲染。

## 4. 从零训练下肢优化版 WM + DTERA

### 当前推荐：deploy V4（harm-aware DTERA）

V3-5000 在固定未见动作 `mydata3/24_seg00.pkl` 上的四组对照显示：Motion-WM 单独将 joint RMSE 从 `0.15215` 降到 `0.15073`，但 DTERA 的 dynamics gate 在 98.95% 帧上高于 0.95，使 WM+DTERA 的 joint RMSE 回升到 `0.15261`。V4 因此不再继续盲目放大残差，而是：

- 将总残差上限从 `0.09` 收到 `0.055 action`（仍是旧 V2 `0.0225` 的 2.44 倍）；
- 提高 demand gate 阈值，避免 dynamics branch 几乎全程满开；
- 使用 action-conditioned error predictor 对每条支路试算 `0/25/50/75/100%` 五档残差，按下一帧相对跟踪误差软选最优比例；
- 取消固定 `0.02` 绝对改善阈值，避免实际 `10^-5`级改善被错误关闭；
- 前 500 轮旁路 improvement gate，500–1000 轮平滑接管，避免零初始化残差造成梯度死锁；
- 保留下肢全权限，腰部限制为 75%，上肢限制为 50%。

必须新训，不得从 V3-5000 续训：

```bash
cd '/home/hank/TWIST（anyadapter）'

OMP_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH='/home/hank/IsaacGym_Preview_4_Package/isaacgym/python:/home/hank/TWIST（anyadapter）/pose:/home/hank/TWIST（anyadapter）/legged_gym:/home/hank/TWIST（anyadapter）/rsl_rl' \
/home/hank/anaconda3/envs/twist/bin/python \
  tools/auto_resume_motion_wm_dtera_legs.py \
  --task g1_motion_wm_dtera_deploy_v4 \
  --project g1_motion_wm_dtera_deploy_v4 \
  --root-exptid deploy_v4_feedback_seed42_fresh_20260919 \
  --target-iterations 3000 \
  --motion-file '/home/hank/TWIST（anyadapter）/legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml' \
  --num-envs 4096 --seed 42 --device cuda:0 \
  --python /home/hank/anaconda3/envs/twist/bin/python
```

在 500 轮只检查训练健康度；由于 improvement gate 到 1000 轮才完全接管，正式四组评估优先使用 1000、1500、2000 和 3000 轮。导出命令：

```bash
python legged_gym/scripts/export_twist_dtera_jit.py \
  --preset motion_wm_deploy_v4 \
  --ckpt '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_dtera_deploy_v4/deploy_v4_feedback_seed42_fresh_20260919/model_1000.pt' \
  --device cpu
```

### deploy V3（仅用于已有 5000 轮对照）

`g1_motion_wm_dtera_deploy_v3` 不在训练 reference 前添加人工扰动，固定使用“原始 PKL → 冻结 Motion-WM → DTERA”。它从 motion 第 0 帧开始，避免 RSI 与 motion 尾部终止叠加造成 episode 只覆盖剩余半段。DTERA 理论修正上限由 `0.0225` 提高到 `0.09 action`（4 倍），同时保留 tanh 限幅和更强的前期饱和正则。请使用全新实验目录，不要续训 2500 轮旧 checkpoint：

```bash
cd '/home/hank/TWIST（anyadapter）'
OMP_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH='/home/hank/IsaacGym_Preview_4_Package/isaacgym/python:/home/hank/TWIST（anyadapter）/pose:/home/hank/TWIST（anyadapter）/legged_gym:/home/hank/TWIST（anyadapter）/rsl_rl' \
/home/hank/anaconda3/envs/twist/bin/python \
  legged_gym/legged_gym/scripts/train.py \
  --task g1_motion_wm_dtera_deploy_v3 \
  --proj_name g1_motion_wm_dtera_deploy_v3 \
  --exptid deploy_v3_seed42_fresh \
  --motion_file '/home/hank/TWIST（anyadapter）/legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml' \
  --num_envs 4096 --max_iterations 3000 --seed 42 \
  --device cuda:0 --rl_device cuda:0 --no_wandb --fix_action_std
```

新 checkpoint 导出时必须使用 `--preset motion_wm_deploy_v3`，不能使用旧 `motion_wm_v2` preset。先在 500、1000、1500 轮做固定 raw/WM 四组评估，不应在没有 held-out 改善证据时盲目训练到 30000 轮。

推荐正式训练使用自动断点接续脚本；它会校验 V3 的支路增益、残差尺度和 warmup 配置，不会误接旧 legs/V2 checkpoint：

```bash
cd '/home/hank/TWIST（anyadapter）'
OMP_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH='/home/hank/IsaacGym_Preview_4_Package/isaacgym/python:/home/hank/TWIST（anyadapter）/pose:/home/hank/TWIST（anyadapter）/legged_gym:/home/hank/TWIST（anyadapter）/rsl_rl' \
/home/hank/anaconda3/envs/twist/bin/python \
  tools/auto_resume_motion_wm_dtera_legs.py \
  --task g1_motion_wm_dtera_deploy_v3 \
  --project g1_motion_wm_dtera_deploy_v3 \
  --root-exptid deploy_v3_seed42_fresh_20260918 \
  --target-iterations 3000 \
  --motion-file '/home/hank/TWIST（anyadapter）/legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml' \
  --num-envs 4096 --seed 42 --device cuda:0 \
  --python /home/hank/anaconda3/envs/twist/bin/python
```

V3 会额外记录 `Termination/catastrophic_count`、`Termination/motion_end_count` 和 `Termination/motion_end_share`。因此不要再只根据 `Mean episode length` 判断稳定性：短动作正常播完也会缩短该均值。正式训练时同时检查 catastrophic 是否下降、motion-end share 是否上升，并检查 `candidate_saturation_fraction`，避免为了增大修正量而让残差长期撞限幅。

### 旧 legs V2（仅用于复现）

任务 `g1_motion_wm_dtera_legs` 在 V2 基础上增加左右对称权重的髋、膝、踝 clean-reference 跟踪奖励。以下命令不带 resume 参数，**不从旧 4800 或 V2-1400 checkpoint 接着训练**。`legs_audited_seed42_v2` 仅用于一个全新实验目录；如目录已存在，请换新的 `--exptid`。原始 TWIST 和 Motion-WM 仍加载各自的冻结预训练模型；“从零”指 DTERA 新支路。

```bash
cd '/home/hank/TWIST（anyadapter）'
OMP_NUM_THREADS=1 \
PYTHONPATH='/home/hank/TWIST（anyadapter）/pose:/home/hank/TWIST（anyadapter）/legged_gym:/home/hank/TWIST（anyadapter）/rsl_rl' \
/home/hank/anaconda3/envs/twist/bin/python \
  legged_gym/legged_gym/scripts/train.py \
  --task g1_motion_wm_dtera_legs \
  --proj_name g1_motion_wm_dtera_legs --exptid legs_audited_seed42_v2 \
  --motion_file '/home/hank/TWIST（anyadapter）/legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml' \
  --num_envs 64 --max_iterations 1500 --seed 42 \
  --device cuda:0 --rl_device cuda:0 --no_wandb --fix_action_std
```

训练后优先检查 500、1000、1500 轮 checkpoint，在固定的多个未见 motion 与 seed 上和 V2-1400、原始 TWIST 做同条件比较。现有 V2-1400 JIT 不会自动获得新任务的下肢优化效果；必须重新训练并导出新 checkpoint。
训练入口和高层服务已强制优先加载本 checkout 的 MotionLib，避免误用 `/home/hank/TWIST/pose` 的旧安装。保留 `PYTHONPATH` 便于其他工具使用同一版本。新 run 保存 `run_manifest.json`（输入哈希及完整训练配置），正常结束后生成 `training_completion.json`。

Deploy V3 前 500 轮残差按线性 warmup 放开，250 轮为半幅；旧 legs/V2 任务仍是 1000 轮。不要把早期与 base 相似直接当成训练失败，也不要仅凭总 reward 上升继续加训。固定验证集及停止条件见审计文档；test.yaml 不用于挑 checkpoint 或调参数。
训练入口会拒绝直接使用 `twist_dataset.yaml` 或被修改过但未重新审计的 `train.yaml`。Deploy V3 每个 motion 从第 0 帧开始；旧任务的随机初始相位则至少保留 26 个控制帧。两者都保留真实跌倒等提前终止。

### 训练中断自动续训

`tools/auto_resume_motion_wm_dtera_legs.py` 会校验最新 checkpoint 的完整性、数值有限性和 DTERA 配置，并以“总目标轮数 - checkpoint 轮数”计算续训轮数。每次重启写入新的 `__resume_*` 目录，不覆盖任何已有 run。以下命令会从 `legs_audited_seed42_v2` 的最新有效 checkpoint 自动接续，直到总轮数 30000：

```bash
cd '/home/hank/TWIST（anyadapter）'
/home/hank/anaconda3/envs/twist/bin/python \
  tools/auto_resume_motion_wm_dtera_legs.py \
  --root-exptid legs_audited_seed42_v2 \
  --target-iterations 30000 \
  --num-envs 4096 \
  --device cuda:0
```

异常退出后默认等待 30 秒并自动重启，最多重启 100 次。手动 `Ctrl+C` 会同时停止子进程和守护脚本。机器或终端意外中断后，重新执行同一条命令即可继续。守护日志保存在 `tools/auto_resume_legs_audited_seed42_v2.log`。
