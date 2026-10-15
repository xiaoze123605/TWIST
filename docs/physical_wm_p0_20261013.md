# 物理世界模型P0优化：新数据通道、独立训练与验收

2026-10-13。本阶段已实施上一轮建议中的物理数据与监督修复。新分支独立运行，不覆盖heading300，不修改现有任务的默认策略观测、PPO采样动作或部署代码。**尚未产生仿真控制效果更好的新策略。**

## 本次实际完成

- `rsl_rl/rsl_rl/modules/physical_world_model.py`：独立物理状态、最终PD目标与世界模型。51维状态是局部角速度3、roll/pitch2、绝对关节角23、真实关节速度23，SI单位；四踝速度保留。模型用自己的可训练history encoder，不替换旧策略的encoder。
- `legged_gym/legged_gym/envs/g1/physical_wm_transition_mixin.py`：可选环境包装。分别记录采样指令、延迟/裁剪后的执行动作和投影后的PD目标；在reset前捕获真实next state。旧7003维actor输入保持原定义。对非23关节、非P控制或未知关节重排拒绝运行。
- 新世界模型消费投影后的实际PD目标；预测关节角变化与端点速度，不强制把端点速度当成整个20ms区间的平均速度。自回归历史严格追加`(s_t, target_t)`，与下一状态分开。
- 正常数据采集2048条＋重置专项512条真实PhysX转移；控制策略冻结，没有创建策略优化器。
- 使用已保存的18条MuJoCo轨迹，完成20 epochs、660次独立WM优化；按验证集选择第5轮，然后仅在冻结选择后执行留出测试。没有根据测试结果扫参数。

现有`PPOAnyAdapterOpenTrack`及旧checkpoint的默认路径仍保留，作为原行为对照。新监督不是对旧输入数值的原地替换，也没有自动把新模型启用到已有任务。这样避免旧encoder/actor突然消费不同含义的7003维数据。新接口需在新的环境/训练分支明确启用。

## 真实环境数据检查

16环境×128步：

|检查|结果|
|---|---:|
|真实转移|2048|
|采样指令与执行动作不同|928|
|PD目标被投影|706|
|真实踝速度最大绝对值|30.229 rad/s|
|旧actor消费的踝速度最大绝对值|0|
|actor权重变化／优化器更新|0／0|

另做16环境×32步检查，在预声明的第16步令一个环境超时。确认该转移done=true、valid=false，packet保存的是reset前物理状态，返回环境已是RSI reset后的状态。两份记录没有被混在一起，其他环境仍正常记录。本轮不把训练环境中的速度峰值解释为电机或实机测量。

## 留出预测结果

所有轨迹使用原MuJoCo部署资产；每条文件单独构建窗口，79步真实过去历史，无跨文件历史或padding。训练、验证、测试按motion划分，在学习前固定：

- 训练：77、站立摆臂、mocap1/1、mocap1/20、mydata2/3，raw/WM各一条。
- 验证：mocap1/59，raw/WM。
- 测试：mydata2/6、mydata3/42、mydata3/47_seg01，raw/WM。

归一化仅拟合训练轨迹。上述“留出”只针对本次新WM训练；不证明生成轨迹的旧策略从未见过这些动作，也不构成新的控制策略测试集。20步窗口高度重叠，不能把1344个窗口解释为1344个独立实验。

MuJoCo的free-joint角速度采用局部body frame；本数据读取即时qvel真值，未把可能滞后一个积分步的gyro sensor缓存当作同一时刻的物理速度。部署传感器估计及其噪声/延迟仍须后续匹配。[MuJoCo官方说明](https://mujoco.readthedocs.io/en/3.1.6/overview.html)。

|测试量|新物理WM|状态保持基线|误差降低|
|---|---:|---:|---:|
|20步轨迹归一化RMSE|0.63872|1.01971|37.36%|
|1步关节RMSE(rad)|0.01254|0.02108|40.50%|
|第20步关节RMSE(rad)|0.11612|0.23037|49.59%|
|1步四踝速度RMSE(rad/s)|0.68223|0.76536|10.86%|
|第20步四踝速度RMSE(rad/s)|0.87956|1.37064|35.83%|

6条测试轨迹分别降低归一化误差31.74%至43.79%，均超过预设的10%预测保护条件。这个基线是保持初始状态不变的预测器，**不是原TWIST控制策略，也不是旧WM的成对重新训练对照**。单步与多步预测改善不能换算成Yaw、深蹲、抖动或闭环恢复能力的改善。

训练实测峰值Torch GPU allocation约227.6MiB，训练/验证/测试部分约8.06秒。已有数据准备不计入这一时间。没有终止其他GPU工作。

## 检查与局限

新增7项有针对性的测试通过：未屏蔽速度、延迟/投影动作分离、reset前状态、保持环境输出和RNG、独立q/端点dq表示、历史梯度、自回归时间配对及不兼容顺序拒绝。另有24项原评估/runtime/reference回归通过，共31项。

独立验证脚本核对source hash、motion划分、窗口不跨轨迹、验证集选模、六条测试门和真实reset收据。旧heading300 checkpoint/JIT哈希保持不变。

当前模型只预测51维物理状态，尚无root height、线速度、接触与摩擦/载荷监督。训练数据主要来自同一部署资产、旧策略和固定条件。没有证明其对新动力学的泛化，也没有把新history embedding注入actor。PhysX采集用于验证数据流；没有将不同资产的PhysX样本混入这次MuJoCo预测训练。

## 下一阶段的技术条件

P0数据/预测通道已具备进入下一阶段的条件。应另建与当前MuJoCo资产一致的纠偏训练分支，给核心跟踪模块显式相对heading、yaw-rate error、关节/目标误差及新物理历史，而保留旧策略作为初始化与保护参考。先验证一个转向→稳定保持任务及正负扰动恢复，再扩展专科teacher和蒸馏。

接入时必须版本化新的policy observation/JIT/runtime，并证明零初始化修正时完整控制轨迹与heading300一致。不能把`best_physical_wm.pt`交给现有低层程序：它是动力学预测器，**不是可部署控制policy**。新actor学习后仍需完整77 raw/WM与独立动作验收，不能只通过WM loss或动作MAE。

现在不需要用户手动安装依赖、修复驱动、启动GPU任务或更换模型。宿主机GPU已验证可用。查看当前仿真仍使用此前的heading300高/低层命令。

## 产物与复现

结果：`reports/physical_wm_p0_20261013/pilot/result.json`；模型：该目录`best_physical_wm.pt`；数据：`physical_transitions.pt`；配置/来源：`manifest.json`；真实环境：`physics_smoke/receipt.json`与`physics_reset_smoke/receipt.json`；逐测试轨迹：`test_comparison.csv`；独立收据：`verification.json`。

```bash
cd '/home/hank/TWIST（anyadapter）'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /home/hank/anaconda3/envs/twist/bin/python tools/train_physical_wm_pilot.py \
  --output reports/physical_wm_p0_20261013/repeat_new \
  --device cuda:0 --epochs 20 --horizon 20 --batch-size 256
```

输出必须选新目录；已有目录会拒绝覆盖。GPU须在能访问设备的宿主执行环境运行，沙盒内的CUDA不可用不代表宿主机驱动损坏。
