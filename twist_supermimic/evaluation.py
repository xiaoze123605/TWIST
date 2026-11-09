from dataclasses import replace

import numpy as np
import torch

from tools.sim_eval_metrics import spectrum
from .data import reference
from .environment import GymEnvironment
from .geometry import increments
from .mdp import Task


@torch.no_grad()
def prediction_errors(learner, batch=32):
    results = {}
    for horizon in (1, 8, 16, 32):
        if learner.replay.size < horizon:
            raise ValueError('真实缓存不足32步预测验证')
        data = learner.replay.windows(batch, horizon)
        state = data['state'][:, 0]
        base = state.clone()
        errors, constants = [], []
        for t in range(horizon):
            state = learner.world(state, data['executed'][:, t], data['context'][:, t])
            errors.append(increments(state, data['next_state'][:, t]).square().mean(-1)*data['mask'][:, t])
            constants.append(increments(base, data['next_state'][:, t]).square().mean(-1)*data['mask'][:, t])
        denominator = data['mask'].sum().clamp_min(1)
        results[str(horizon)] = dict(mse=float(torch.stack(errors, 1).sum()/denominator),
                                     hold_state_mse=float(torch.stack(constants, 1).sum()/denominator),
                                     source='当前真实训练回合的连续窗口')
    return results


@torch.no_grad()
def evaluate(cfg, actor, robot, library, role='dev', count=None, max_frames=None, names=None, world=None, baseline=False):
    if names is None:
        ids = library.roles[role]
        if count is not None:
            ids = ids[:count]
    else:
        lookup = {row['name']: i for i, row in enumerate(library.rows)}
        ids = torch.tensor([lookup[library.aliases.get(n, n)] for n in names], device=cfg.device)
        if any(library.rows[int(i)]['role'] != role for i in ids):
            raise ValueError('指定动作不属于请求的评价集合')
    if not len(ids):
        raise ValueError('请求的评价动作为空')
    eval_cfg = replace(cfg, num_envs=min(cfg.evaluation_envs, len(ids)), domain_randomization=False, observation_noise=False)
    task = Task(robot, library, noisy=False)
    env = GymEnvironment(eval_cfg, robot, library, task, evaluation=True)
    actor.eval()
    results = []
    world_errors = {str(h): [] for h in (1, 8, 16, 32)}
    try:
        for begin in range(0, len(ids), env.n):
            real_ids = ids[begin:begin+env.n]
            padding = torch.cat((real_ids, real_ids[:1].expand(env.n-len(real_ids))))
            env.reset(torch.arange(env.n, device=cfg.device), padding, torch.zeros(env.n, device=cfg.device, dtype=torch.long))
            if baseline:
                actor.reset()
            lengths = library.lengths[padding]+32
            if max_frames is not None:
                lengths = lengths.clamp_max(max_frames)
            fallen = torch.zeros(env.n, dtype=torch.bool, device=cfg.device)
            predicted = env.state()
            unchanged = predicted.clone()
            traces = {k: [] for k in ('state', 'ref', 'action', 'previous_action', 'contact')}
            for t in range(int(lengths.max())):
                request = actor(env.obs, env.state(), library.state(padding, env.frame)) if baseline else actor(env.obs)
                row, failed, _ = env.step(request, auto_reset=False, calculate_reward=False)
                state = row['next_state']
                ref = library.state(padding, env.frame)
                fallen |= failed & (t < lengths)
                for key, value in (('state', state), ('ref', ref), ('action', row['executed']),
                                   ('previous_action', row['previous_action']), ('contact', row['contact'])):
                    traces[key].append(value)
                if world is not None and t < 32:
                    predicted = world(predicted, row['executed'], row['context'])
                    if t+1 in (1, 8, 16, 32):
                        err, base_err = increments(predicted, state), increments(unchanged, state)
                        predicted_reward = task.reward(predicted, ref, row['executed'], row['previous_action'])
                        true_reward = task.reward(state, ref, row['executed'], row['previous_action'])
                        for i in range(len(real_ids)):
                            world_errors[str(t+1)].append(dict(joint_position=float(err[i, 12:35].square().mean().sqrt()),
                                joint_velocity=float(err[i, 35:].square().mean().sqrt()),
                                root_velocity=float(err[i, 6:9].square().mean().sqrt()), root_orientation=float(err[i, 3:6].norm()),
                                mse=float(err[i].square().mean()), hold_state_mse=float(base_err[i].square().mean()),
                                reward_absolute_error=float((predicted_reward[i]-true_reward[i]).abs())))
                if (t+1) % 500 == 0:
                    print('完整动作评价进度：%d/%d帧' % (t+1, int(lengths.max())), flush=True)
            traces = {k: torch.stack(v) for k, v in traces.items()}
            steps = len(traces['state'])
            flat = {k: v.flatten(0, 1) for k, v in traces.items()}
            metrics_parts = {}
            # 真实物理完成后批量计算运动学和指标，减少小批量GPU调用。
            for start in range(0, len(flat['state']), 2048):
                stop = start+2048
                reward, values = task.reward(flat['state'][start:stop], flat['ref'][start:stop],
                                             flat['action'][start:stop], flat['previous_action'][start:stop], details=True)
                values['reward'] = reward
                for key, value in values.items():
                    metrics_parts.setdefault(key, []).append(value)
            metrics = {k: torch.cat(v).reshape(steps, env.n) for k, v in metrics_parts.items()}
            contact = traces['contact']
            metrics['unsupported'] = (~contact.bool().any(-1)).float()
            metrics['contact_foot_speed'] = (metrics['left_foot_speed_squared']*contact[..., 0]
                                              +metrics['right_foot_speed_squared']*contact[..., 1])/contact.sum(-1).clamp_min(1)
            clean = reference(flat['ref']).reshape(steps, env.n, 31)
            knee = clean[:, :, [11, 17]].mean(-1)
            height_rate = torch.cat((torch.zeros_like(clean[:1, :, 0]), torch.diff(clean[:, :, 0], dim=0)/.02), 0)
            frame_ids = torch.arange(1, steps+1, device=cfg.device)[:, None]
            valid = frame_ids <= lengths[None]
            masks = dict(full=valid, walking=valid & (clean[:, :, 0] > .74) & (clean[:, :, 4:6].norm(dim=-1) > .15),
                         squat=valid & (clean[:, :, 0] < .7) & (knee > .8),
                         rise=valid & (clean[:, :, 0] < .78) & (knee > .5) & (height_rate > .03),
                         descent=valid & (clean[:, :, 0] < .78) & (knee > .5) & (height_rate < -.03),
                         tail=valid & (frame_ids >= (library.lengths[padding]-250).clamp_min(0)[None]))
            q = traces['state'][:, :, 13:25].cpu().numpy()
            error = (traces['state'][:, :, 13:25]-traces['ref'][:, :, 13:25]).cpu().numpy()
            for i, mid in enumerate(real_ids):
                phases = {}
                for phase, mask in masks.items():
                    selected = mask[:, i]
                    if not selected.any():
                        continue
                    values = {k: float(v[:, i][selected].mean()) for k, v in metrics.items()}
                    values['frames'] = int(selected.sum())
                    selected_np = selected.cpu().numpy()
                    values['jitter_5_20hz_rms'] = spectrum(q[:, i], selected_np)[0].get('high_5_20hz_rms')
                    values['tracking_jitter_5_20hz_rms'] = spectrum(error[:, i], selected_np)[0].get('high_5_20hz_rms')
                    phases[phase] = values
                full = phases.pop('full')
                cost = .35*full['torso']**.5+.25*full['velocity']**.5+.2*full['torso_rate']**.5+.1*full['joint']**.5+.1*full['action_change']**.5
                results.append(dict(name=library.rows[int(mid)]['name'], frames=full['frames'], failed=bool(fallen[i]),
                                    metrics=full, phases=phases, stability_cost=cost+100*int(fallen[i])))
        controls = env.physical_controls
    finally:
        env.close()
        actor.train()
    return dict(role=role, rows=results, clips=len(results), physical_controls=controls,
                policy_family='heading300' if baseline else 'native_twist_supermimic',
                model_input_dimension=7003 if baseline else 1155,
                world_prediction=dict(source='独立完整动作回合，使用真实执行动作预测未来状态', horizons=world_errors),
                full_clips=max_frames is None, fall_fraction=sum(r['failed'] for r in results)/len(results),
                stability_cost=sum(r['stability_cost'] for r in results)/len(results))
