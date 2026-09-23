"""Human-readable DynamicsTracker metrics in the original TWIST log style."""


def format_iteration(metrics, total_iterations=None, num_envs=None, steps_per_env=None):
    width, pad = 80, 35
    iteration = int(metrics['iteration'])
    goal = total_iterations if total_iterations is not None else '?'
    header = f' Learning iteration {iteration}/{goal} '

    def row(label, value):
        return f'{label:>{pad}} {value}'

    ended = sum(int(metrics.get(key, 0)) for key in
                ('physical_failures', 'motion_completions', 'timeouts'))
    completed = int(metrics.get('motion_completions', 0))
    lines = ['#' * width, header.center(width), '']
    if 'fps' in metrics:
        lines.append(row('Computation:', f"{metrics['fps']:.0f} steps/s "
                                      f"(collection: {metrics['collection_time_s']:.3f}s, "
                                      f"learning: {metrics['learning_time_s']:.3f}s)"))
    lines.extend([
        row('Value function loss:', f"{metrics['value_loss']:.4f}"),
        row('Surrogate loss:', f"{metrics['policy_loss']:.4f}"),
        row('Mean action noise std:', f"{metrics['mean_action_std']:.4f}"),
        row('Mean reward/step:', f"{metrics['mean_reward']:.5f}"),
        '-' * width,
        row('Physical failures:', int(metrics.get('physical_failures', 0))),
        row('Motion completions:', completed),
        row('Episode timeouts:', int(metrics.get('timeouts', 0))),
        row('Completion / ended episodes:', f'{completed / ended:.1%}' if ended else 'n/a'),
        row('Motions seen this iteration:', int(metrics.get('motions_seen_iteration', 0))),
        row('Motions seen total:', f"{int(metrics.get('motions_seen_total', 0))}/"
                                   f"{int(metrics.get('motion_count', 0))}"),
        row('Mean motion difficulty:', f"{metrics['mean_motion_difficulty']:.4f}"),
    ])
    if 'world_model_loss' in metrics and metrics['world_model_valid_steps']:
        lines.append(row('World model loss:', f"{metrics['world_model_loss']:.4f}"))
    if num_envs is not None and steps_per_env is not None:
        lines.extend(['-' * width,
                      row('Total timesteps:', f'{iteration * num_envs * steps_per_env:,}')])
    if 'iteration_time_s' in metrics:
        lines.append(row('Iteration time:', f"{metrics['iteration_time_s']:.2f}s"))
    if 'elapsed_time_s' in metrics:
        lines.append(row('Total time:', f"{metrics['elapsed_time_s'] / 3600:.2f}h"))
    if 'eta_s' in metrics:
        lines.append(row('ETA:', f"{metrics['eta_s'] / 3600:.2f}h"))
    return '\n'.join(lines) + '\n'
