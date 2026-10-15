# TWIST仿真自动监测评估系统

2026-10-09。基于`5e45fe8`与[heading300诊断](heading300_diagnosis_20261008.md)，仅实现评估系统；未修改网络、模型权重、奖励、训练流程或实机代码，未启动训练/实机测试。

## 统一入口与范围

```bash
python tools/evaluate_sim.py --policy <JIT路径> --mode standard --output <新输出目录>
```

在项目根目录、twist环境运行，Redis可执行程序需可用；评估自行创建隔离Redis，不依赖6379已有服务。默认CPU、seed42、EMA0、目标投影开启、raw/WM分别评估。主要基线固定为已有heading_300 JIT；历史对照固定为原23500无投影。模型权重和动作数据仍需在本机存在。

|级别|默认动作集合|用途|
|---|---|---|
|quick|77前20秒，raw/WM|首次深蹲及有效连续行走窗口的筛查，不覆盖完整转向/多动作|
|standard|完整77（64.14秒）+原协议8个开发集短动作，共9动作|常规回归|
|full|原协议全部16动作，含确认集长动作；77仍完整64.14秒|更广动作覆盖，每次运行仍是指定的一个seed|

`--cases`可筛选该级别内动作，报告明确记录filtered_scope。full不自动代表多seed或实机验证。quick早期12秒方案缺少>=1.28秒的连续行走窗口，已调整为20秒；保留旧测试产物以记录问题。

手动复现完整77（本轮已实际运行同等命令）：

```bash
python tools/evaluate_sim.py \
  --policy heading_reproduction=reports/recovery_1250/fix_stage_20261008/heading_300.pt \
  --mode standard --cases mocap1__77 --workers 2 \
  --output reports/sim_eval_manual/heading77
```

常规评估不筛选动作：

```bash
python tools/evaluate_sim.py \
  --policy candidate=reports/recovery_1250/fix_stage_20261008/heading_300.pt \
  --mode standard --output reports/sim_eval_manual/standard
```

多模型/多checkpoint比较：

```bash
python tools/evaluate_sim.py \
  --policy candidateA=/path/to/A.pt \
  --policy candidateB=/path/to/B.pt \
  --mode standard --output reports/sim_eval_manual/compare

# 训练checkpoint通过已有exporter在CPU导出，而非直接交给低层。
python tools/evaluate_sim.py \
  --checkpoint candidate=/path/to/model_300.pt \
  --mode quick --output reports/sim_eval_manual/checkpoint
```

支持`--seed`、`--ema`、`--reference-modes raw wm`、`--no-target-projection`、`--workers 1或2`。这些设置同时用于候选与heading_300，不能用不同控制条件的缓存冒充基线。输出目录必须新建，禁止覆盖既有结果。`--dry-run`只保存计划，status=planned，没有验收结论。

## 指标及定义

原诊断计算移动到共享`sim_eval_metrics.py`，旧诊断绘图程序与新入口使用同一函数：

- 原23关节RMSE、根速度RMSE及周期yaw误差直接沿用低层原指标。
- 躯干Pitch比较相同XML重建的clean参考，不要求Pitch归零。
- 蹲/下降/起身/行走/转向条件和9点高度平滑沿用诊断定义，不改变阈值。
- 深蹲高度RMSE、髋/膝/踝各组及12下肢协调RMSE：在相同深蹲mask内计算实际减clean参考。
- 双支撑COM使用竖向足力>5N且接触凸包有效的静态裕量；无有效凸包为N/A，不以零替代，也不把静态裕量当成动态稳定证明。
- 行走频谱沿用连续片段>=64采样、50Hz、64点Welch、32点重叠、线性去趋势、5–20Hz积分。摔倒后信号不参与频谱。踝Pitch高频RMS按左右踝Pitch同一积分计算。
- 接触切换±40ms附近与远离切换的腿动作变化率，保留原诊断的计算方法。
- 投影比例、控制器力矩饱和保留25/23关节映射；饱和聚合每动作后全部20个1ms物理子步。
- **新增关节速度RMSE**：trace没有clean参考速度，因此明确使用`numpy.gradient(clean关节角, dt=0.02)`估计，再与实际23关节速度比较；不冒称MotionLib的19点平滑参考速度，不替换已有位置/频谱指标。

JSON中保留各阶段的原详细指标；CSV列出全程与分阶段标量。详细绘图复用`plot_sim_trace.py`，其`curve_summary.json`中的warmup窗口仅作可视化辅助，不参与本系统验收；门槛只读取主summary与共享阶段指标。相同JIT字节哈希且所有执行条件相同可复用本次结果，仍分别保留候选和基线角色；不读取条件不明的旧缓存。

## 比较与自动验收

每动作、每raw/WM分别验证：seed、EMA、投影、动作文件哈希、长度/速度、XML和控制程序哈希，以及逐帧clean/processed参考摘要一致。任何不匹配都不能通过。

门槛：无新增跌倒、每动作yaw<=基线105%、每动作根速度误差<=基线105%；蹲协调与行走高频在各reference mode内均有改善。阶段汇总按有效秒数加权RMS，raw与WM不混算。不足的阶段/频谱标记N/A并判inconclusive；任何评估异常均为failed、accepted=false。

历史23500无投影单列展示，不参与以上模型验收。报告逐项列出关节、Pitch、协调、高频等退化，即使满足声明门槛，也可能标记`meets_gates_with_tradeoffs`，不宣布整体更优。相同模型对照为`unchanged`，并非“改善通过”。quick或filtered通过只适用于所评范围。

## 自动启用方式（默认关闭）

本轮**没有修改已有训练脚本**。提供两种供后续接入的方式。

### 保存回调接入

训练脚本可按以下模式添加开关；回调必须放在已有原子保存完成之后：

```python
from tools.sim_auto_eval import AutoEvalHook

parser.add_argument('--auto-eval', action='store_true', default=False)
parser.add_argument('--eval-interval', type=int, default=50)
# args为后续训练脚本自己解析的参数。
auto_eval = AutoEvalHook(
    enabled=args.auto_eval,
    interval=args.eval_interval,
    output_root='reports/auto_sim_eval',
    mode='quick',
)

# 放在已有保存函数成功完成之后，不要重复实现训练保存流程：
# torch.save(checkpoint, temporary_path)
# os.replace(temporary_path, final_path)
auto_eval.on_checkpoint_saved(final_path, completed_updates)
# 可在训练日志更新点调用auto_eval.poll()记录已完成/失败状态。
```

未加`--auto-eval`时，hook不生成文件、不启动进程、不读取模型、不导入torch。启用后保存回调立即返回；忙时记录skipped_busy，不排队。实际CPU worker先验证完整模型状态、计数与保存文件哈希，再调用已有导出程序；验证失败不导出JIT、不启动仿真。

单项目文件锁同时覆盖导出和评估，跨hook/跨进程不并发堆积；锁FD由worker继承，训练父进程退出不会释放正在执行worker的锁。worker隐藏所有CUDA设备，BLAS/OMP单线程、仿真workers=1、较低CPU调度优先级。不使用训练GPU，不改变训练父进程CUDA环境。回调启动/日志失败以事件记录，不抛异常打断训练。每任务独立目录，不覆盖旧结果；interval只是触发机会，繁忙时会跳过，并不保证每个保存点都被评估。

### 独立目录监测

无需现在修改训练代码，可在另一个终端监测后续训练的最终checkpoint目录：

```bash
python tools/evaluate_sim.py \
  --auto-eval \
  --watch-dir legged_gym/logs/<后续训练目录> \
  --eval-interval 50 --mode quick \
  --output reports/auto_sim_eval
```

只扫描`model_<数字>.pt`，忽略`.tmp`、更新0和非interval保存点；连续两次文件大小/mtime稳定后交给worker完整性验证。忙时只保留检查最新保存点的机会，不积压历史任务。未完整文件若后来被替换，允许重新验证。只有显式`--auto-eval`才允许watch；`--watch-once`用于接口测试，扫描两次并等待至多一个worker。

`events.jsonl`区分started、finished、failed、skipped_busy、failed_checkpoint等；finished仅表示评估完成，是否验收通过要读取summary.comparisons，planned绝不算通过。

## 输出结构与示例

每次手动输出根目录：

```text
manifest.json           # 模型/协议/控制哈希与条件
summary.json            # 状态、逐动作指标、阶段和验收
metrics.csv             # 全程指标，含历史角色及执行条件
phases.csv              # 蹲起/行走等分项
report.md               # 自动结论、退化和曲线链接
plots/*curves.png       # 参考相对Pitch、COM和变化率
plots/*spectra.png      # 同定义频谱
plots/details/...       # 复用既有23关节、root/Yaw、蹲深与tracking曲线
runs/...                # 每组高/低层/Redis日志、metrics和逐帧trace
exports/...             # 仅checkpoint输入时生成JIT与来源侧车
COMPLETE 或 FAILED       # 整个评估的完成/异常状态，不等于验收通过
```

已生成的完整77示例：[报告](../reports/sim_eval_system_20261009/report_example_final/report.md)、[JSON](../reports/sim_eval_system_20261009/heading77_standard/summary.json)、[CSV](../reports/sim_eval_system_20261009/heading77_standard/metrics.csv)。同模型候选结果为unchanged、accepted=false。WM复现：joint=0.1393055067 rad、root velocity=0.1038641779 m/s、yaw=0.5599072897 rad（32.0803°），与诊断记录完全一致，raw也完全一致。

[包含详细关节/root/Yaw图的重绘验证](../reports/sim_eval_system_20261009/curve_report_validation/report.md)，复用上述真实完整77记录，没有重新训练或更改指标。

[quick20多模型示例](../reports/sim_eval_system_20261009/quick20_multiple/report.md)：同时比较heading副本和仅关闭heading头的模型；有有效行走频谱、历史对照独立标注，副本unchanged，head_off不满足门槛。该示例不替代完整动作/其他动作验收。

## 验证范围

15项unittest已通过；运行命令`python tests/test_sim_eval_system.py`。

- 统一入口实际跑完整77 raw/WM、heading基线及原23500历史对照；joint/velocity/yaw逐值一致。
- quick20真实多模型回放、CSV/JSON/Markdown及关键曲线生成。
- full全部16动作的计划解析/文件保存；full成功和运行失败的报告链路用已记录trace的fixture验证，不冒称已实际回放全部16动作。
- 分层模式、未覆盖阶段、+5%保护、不同执行/processed参考、零基数、无新增跌倒、历史角色隔离、失败报告和禁止覆盖测试。
- 自动默认关闭/interval/.tmp/稳定扫描/互斥/唯一目录/CPU隔离/启动失败不中断调用方测试；实际完整checkpoint CPU导出计划与不完整checkpoint失败验证。
- 实际自动保存回调→CPU导出→quick物理仿真→报告链路，使用已有model_300，优化器/训练更新0。

未验证：full16动作的完整物理回放、接入真实训练后的长期运行/资源竞争、多seed整体验收及实机。系统没有自动连接实机或修改训练脚本。

## 本轮文件

新增：`tools/evaluate_sim.py`、`tools/sim_eval_metrics.py`、`tools/sim_eval_report.py`、`tools/sim_auto_eval.py`、`tests/test_sim_eval_system.py`、本文。

修改：`tools/evaluate_4900_fixed.py`（可选seed/低层诊断参数，CPU隔离）、`tools/diagnose_heading300_sim.py`（可调用函数，复用runner）、`tools/analyze_heading300_diagnostics.py`（复用共享计算，保留旧绘图）、`docs/README.md`（入口链接）。已有低层诊断与实机代码本轮未改动。
