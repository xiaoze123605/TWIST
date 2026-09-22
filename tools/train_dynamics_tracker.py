"""Train the independent tracker; every invocation writes a fresh run directory."""
import argparse
import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'legged_gym'), str(ROOT / 'rsl_rl'), str(ROOT / 'pose')]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--warm-start')
    group.add_argument('--resume-checkpoint')
    parser.add_argument('--smoke-test', action='store_true')
    custom, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    import isaacgym  # Must precede torch.
    from legged_gym.envs import task_registry
    from legged_gym.gym_utils import get_args, class_to_dict
    args = get_args()
    if not args.task.startswith('g1_dynamics_tracker'):
        raise ValueError('Select a g1_dynamics_tracker task')
    if args.resume or args.resumeid:
        raise ValueError('Use --resume-checkpoint with an explicit file')
    args.headless = True
    cfg, training = task_registry.get_cfgs(args.task)
    if custom.smoke_test:
        if not args.motion_file or not args.num_envs or args.num_envs > 16:
            raise ValueError('Smoke test requires --motion_file and --num_envs <= 16')
        if not args.max_iterations or args.max_iterations > 10:
            raise ValueError('Smoke test requires --max_iterations <= 10')
        training.algorithm.horizon_curriculum_updates = 0
    else:
        from tools.prepare_motion_wm_training import verify_prepared_training_yaml
        verify_prepared_training_yaml(args.motion_file or cfg.motion.motion_file)
    output = Path(custom.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env, cfg = task_registry.make_env(args.task, args=args)
    runner, training = task_registry.make_alg_runner(env, name=args.task, args=args,
                                                     init_wandb=False, log_root=str(output))
    checkpoint = custom.warm_start or custom.resume_checkpoint
    if checkpoint:
        runner.load(checkpoint, warm_start=bool(custom.warm_start))
    (output / 'manifest.json').write_text(json.dumps(dict(
        task=args.task, smoke_test=custom.smoke_test, initialization=checkpoint,
        environment=class_to_dict(cfg), training=class_to_dict(training),
        input_sha256={str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for path in [cfg.motion.motion_file] + ([checkpoint] if checkpoint else [])},
        deployment=runner.spec), indent=2, default=lambda value: value.tolist()))
    runner.save(output / 'initial_model.pt')
    runner.learn(training.runner.max_iterations)


if __name__ == '__main__':
    main()
