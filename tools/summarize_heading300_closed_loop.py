"""Collect the existing closed-loop reports without changing any metric definitions."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def summarize(root):
    ledger=json.loads((root/'round_ledger.json').read_text())
    records=[]
    for item in ledger['completed']:
        summary=json.loads((Path(item['eval'])/'summary.json').read_text())
        for row in summary['rows']:
            if row['role']!='candidate':continue
            baseline=next(r for r in summary['rows'] if r['role']=='baseline' and r['reference_mode']==row['reference_mode'])
            phase=row['analysis']['phases'];bp=baseline['analysis']['phases']
            record=dict(round=item['round'],mode=row['reference_mode'],
                joint_rmse=row['metrics']['joint_rmse'],
                joint_velocity_rmse=row['analysis']['metrics']['joint_velocity_rmse_estimated'],
                root_velocity_rmse=row['metrics']['root_velocity_rmse'],
                root_velocity_change_percent=100*(row['metrics']['root_velocity_rmse']/baseline['metrics']['root_velocity_rmse']-1),
                yaw_degrees=math.degrees(row['metrics']['yaw_error']),
                yaw_change_percent=100*(row['metrics']['yaw_error']/baseline['metrics']['yaw_error']-1),
                squat_knee_degrees=math.degrees(phase['squat']['knee_rmse']),
                squat_knee_change_percent=100*(phase['squat']['knee_rmse']/bp['squat']['knee_rmse']-1),
                squat_coordination_rmse=phase['squat']['hip_knee_ankle_rmse'],
                squat_coordination_change_percent=100*(phase['squat']['hip_knee_ankle_rmse']/bp['squat']['hip_knee_ankle_rmse']-1),
                squat_pitch_rmse_degrees=phase['squat']['pitch_rmse_degrees'],
                walking_hf=phase['walking']['actual_spectrum']['high_5_20hz_rms'],
                walking_hf_change_percent=100*(phase['walking']['actual_spectrum']['high_5_20hz_rms']/bp['walking']['actual_spectrum']['high_5_20hz_rms']-1),
                first_fall_time=row['analysis']['first_fall_time'],guard_pass=item['guard_pass'],accepted=item['accepted'])
            records.append(record)
    with (root/'round_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (root/'round_comparison.json').write_text(json.dumps(records,indent=2))
    lines=['# heading300闭环实验记录（2026-10-12）','',
       '当前最优参考仍为heading300；尚未达到“明显优于baseline”的验收标准。原23500与heading300均未覆盖，全部新训练/仿真独立保存。', '',
       '验收：raw/WM的深蹲膝与髋膝踝协调误差、行走5–20Hz振荡均降低至少10%；Yaw和逐动作根速度误差不退化超过5%；无新增跌倒；通过多动作回归。未通过完整77保护门的候选不扩展训练、不晋级多动作验收。', '',
       '所有正式短训从heading300初始化；原始采样、512环境、seed42、24步rollout、16 minibatches、3 epochs、25次critic warmup，最多30次更新（实际只有5次actor更新）。TWIST主干、历史编码器、WM、actor结构7003→23、Adapter增益0.5、奖励、PD、投影和EMA保持原设置。仅criticctx分支按用户授权增加两维训练critic输入。', '',
       '|对照|模式|Yaw RMSE(°)|Yaw变化|膝误差变化|协调误差变化|行走高频变化|朝向/速度/跌倒保护门|',
       '|---|---|---:|---:|---:|---:|---:|---|',
       '|heading300|raw|67.777|基线|基线|基线|基线|基线|',
       '|heading300|wm|32.080|基线|基线|基线|基线|基线|']
    for r in records:
        lines.append('|{}|{}|{:.3f}|{:+.2f}%|{:+.2f}%|{:+.2f}%|{:+.2f}%|{}|'.format(r['round'],r['mode'],r['yaw_degrees'],r['yaw_change_percent'],r['squat_knee_change_percent'],r['squat_coordination_change_percent'],r['walking_hf_change_percent'],'通过' if r['guard_pass'] else '未通过'))
    lines+=['','x5锚定0.4/0.2，x20为1.6/0.8，x50为4.0/2.0；headlr01仅将heading头学习率降为5e-7；frozenhead仅冻结heading头。后续三项均相对x20条件：criticctx仅追加两维critic上下文，criticlr005恢复原critic并仅降低critic学习率至5e-6，adamresume在criticlr005条件上仅恢复源Adam状态。各分支未组合修改，失败后均未延长训练。', '',
        '## 实测与机制边界','',
        '- criticctx保留原1318列，新增两列零初始化。第0次完整教师误差、初始价值误差均为0；随机64组输入的新JIT与heading300误差0。512环境1次更新冒烟通过，新列确实得到更新。正式训练后所有冻结权重仍逐元素等于源checkpoint。',
        '- 新critic输入来自actor已经使用的同一sin(heading误差)、1−cos(heading误差)，没有重算参考或推进WM。前两次冒烟因初始化缓存仍是1318维而在训练前停止；已适配初始缓存，失败目录保留。',
        '- 原heading300保存的critic optimizer学习率为5e-6，fresh续训constructor为1e-4。降低这一项以及进一步恢复20项Adam状态均已独立对照。结果不支持仅通过上述单项就能解决退化，不能据此认定critic因素完全无关。',
        '- 无新增跌倒仍不足以通过验收：Yaw退化及下蹲/波动未达到10%改善是真正限制。低KL、动作MAE、价值残差或JIT一致均不能替代闭环检查。',
        '- 同seed的GPU训练不保证相同接触轨迹；跨分支差值未被当作精确因果效应。固定状态价值残差使用原分支GAE目标，不能视为真实价值标签或因果证据。', '',
        '## 检查与证据','',
        '- `tests/test_heading300_critic_context.py`：2项通过（输入对齐/错误布局；构造RNG保持/零列/actor/JIT保持）。',
        '- `tests/test_heading300_closed_loop_gate.py`：2项通过（必须满足全部10%指标和多动作回归；缺失、失败、退化均不通过）。',
        '- 缓存指标篡改测试按预期失败，保存FAILED，未生成COMPLETE；真实评估复算旧trace指标并核验源模型、代码、参考及条件。',
        '- 各项正式评估均使用完整77的64.14秒/3207帧，raw/WM控制条件相同，CPU串行评估；原23500无投影只作历史对照，不参与性能保护门。',
        '- CSV/JSON：`reports/heading300_closed_loop_20261012/round_comparison.*`；逐实验`*_eval77/summary.json`、Markdown、plots、trace保留。',
        '- `critic_round_phase_yaw.csv`沿用已有phase_yaw计算；`criticctx_fixed_state_diagnostics.json`为只读价值残差探针；`critic_round_state_checks.json`为冻结权重检查。',
        '- 尚未验证多动作回归和实机效果；失败候选未晋级，因此没有宣称整体更优。', '',
        '## 后续最小诊断方向','',
        '优先定位第一个actor更新是否已突破Yaw保护门。使用已保存的26–30次checkpoint和逐帧trace，检查相同源策略观测上的核心Adapter、heading修正与接触切换前后动作差异；先验证现有全动作锚定在朝向敏感状态上的覆盖，不再单纯提高全局系数或扩大critic网络。未得到通过raw/WM保护门的候选前，不追加长训或恢复蹲起采样加权。', '',
        '当前活动：'+ledger['current_round']+'.']
    (ROOT/'docs/heading300_closed_loop_20261012.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'reports/heading300_closed_loop_20261012')
    summarize(parser.parse_args().root)
