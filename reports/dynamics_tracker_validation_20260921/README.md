# 验证范围与结果

最终代码使用当前参考帧 `tar_obs_steps=[0]`。下列证据均来自这一版本，不含前期提前一帧参考的调试结果。

- `tests.txt`：21 项通过，包含新策略的时序、终止遮罩、梯度归属、timeout bootstrap、阶段切换输出一致性、两种动作空间的 TorchScript/runtime 一致性、真实 GPU 延迟/reset/名义子集检查，以及原 AnyAdapter 的已有回归测试。805 条 warning 主要来自旧 terrain 的 SciPy 接口、旧 tracing 和 checkpoint 加载提示，没有失败项。
- `baseline_source_verification.json`：列出的 10 个旧环境、配置、PPO、模型和部署源码与 `4806b2b` 逐字节相同。
- `nominal_training.jsonl`、`robust_training.jsonl`、`direct_training.jsonl`：各 4 环境 × 24 步 × 3 updates，共各 288 条环境转移；PPO、20 步自回归 WM、checkpoint 保存正常。鲁棒任务从名义任务 checkpoint 接续。
- `robust_evaluation.json`：4 环境 × 100 步，共 400 个环境步；逐步校验训练网络与 scripted 输出一致。8 次物理失败、0 次完整 motion 完成，因此这里的 RMSE 不能当作跟踪成功证据。
- `summary.json`：短训练配置与 checkpoint 位置。权重位于本地忽略的 `legged_gym/logs/dynamics_tracker/causal_smoke_20260921/`，不随源码提交。

测试入口（现有 twist 环境中需要 pytest）：

```bash
export PYTHONPATH=".:rsl_rl:legged_gym:pose:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
RUN_DYNAMICS_SIM_TEST=1 OMP_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q \
  tests/test_dynamics_tracker.py tests/test_dynamics_tracker_sim.py \
  tests/test_twist_anyadapter_opentrack.py tests/test_twist_anyadapter_runtime.py \
  tests/test_twist_anyadapter_wm_encoder_grad.py
```

本次 pytest 8.3.5 及其依赖放在 `/tmp/twist-dynamics-test-deps`，测试时额外加入 PYTHONPATH，没有更改现有 conda 环境依赖。

完整数据训练、baseline 同预算多 seed 对比、MuJoCo 迁移和实机实验尚未进行。当前结果验证实现能运行，不能验证新策略更优，也不能证明旧 baseline 的物理表现逐 bit 不变。
