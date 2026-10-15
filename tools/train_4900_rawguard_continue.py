"""Long continuation of the 4900 rawguard pilot in a new experiment directory.
Reuses the bounded pilot's algorithm unchanged; only the run-length gate is lifted.
Fresh from model_4900 by default, or resume any 4900-family pilot checkpoint via --resume-from.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'legged_gym'),str(ROOT/'rsl_rl'),str(ROOT/'pose')]
import isaacgym
import torch
import wandb
from legged_gym.envs import task_registry
from legged_gym.gym_utils import get_args, class_to_dict
import rsl_rl.runners.on_policy_runner_mimic as runner_module
from train_4900_rawguard import PPO4900Guarded, SOURCE
PILOT_SCRIPT=ROOT/'tools/train_4900_rawguard.py'

class PPO4900GuardedContinue(PPO4900Guarded):
    """Same source-anchored update as the pilot; identical training semantics."""

runner_module.PPO4900GuardedContinue=PPO4900GuardedContinue

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--actor-rate',type=float,default=2e-6)
    p.add_argument('--anchor-coef',type=float,default=.1)
    p.add_argument('--exploration-std',type=float,default=.1)
    p.add_argument('--last-layer-only',action='store_true')
    p.add_argument('--critic-warmup-updates',type=int,default=20)
    p.add_argument('--resume-from',type=str,default=None,help='4900-family pilot checkpoint .pt to continue from (fresh from model_4900 if omitted)')
    custom,remaining=p.parse_known_args();sys.argv=[sys.argv[0]]+remaining
    args=get_args()
    if args.resume or args.resumeid:raise ValueError('use --resume-from <checkpoint.pt> instead of --resume/--resumeid')
    if not 1<=args.max_iterations<=50000:raise ValueError('max_iterations must be in [1,50000]')
    resume_path=Path(custom.resume_from) if custom.resume_from else None
    if resume_path is not None and not resume_path.is_file():raise FileNotFoundError(f'--resume-from not found: {resume_path}')
    args.task='g1_motion_wm_anyadapter';args.headless=True;args.no_wandb=True
    cfg,train=task_registry.get_cfgs(args.task)
    cfg.motion.motion_file=str(ROOT/'legged_gym/motion_data_configs/wm_dtera_prepared_20260916_local/train.yaml')
    cfg.motion.motion_curriculum=False
    cfg.motion_wm.mode_probabilities=(.8,0.,.2);cfg.motion_wm.artificial_corruption=False
    cfg.env.num_envs=args.num_envs;cfg.seed=args.seed
    train.seed=args.seed;train.policy.fix_action_std=True;args.fix_action_std=True
    train.runner.algorithm_class_name='PPO4900GuardedContinue';train.runner.save_interval=50;train.runner.constant_save_interval=True;train.runner.resume=False
    train.algorithm.source_checkpoint=str(SOURCE);train.algorithm.actor_rate=custom.actor_rate;train.algorithm.anchor_coef=custom.anchor_coef;train.algorithm.exploration_std=custom.exploration_std;train.algorithm.last_layer_only=custom.last_layer_only;train.algorithm.critic_warmup_updates=custom.critic_warmup_updates
    train.algorithm.num_learning_epochs=3;train.algorithm.num_mini_batches=16;train.algorithm.world_model_loss_coef=0.;train.algorithm.entropy_coef=0.
    out=ROOT/'legged_gym/logs/recovery_4900'/args.exptid;out.mkdir(parents=True,exist_ok=False)
    wandb.init(mode='disabled')
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner,train=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=str(out))
    runner.alg.audit_path=out/"anchor_metrics.jsonl"
    if resume_path is not None:
        runner.load(str(resume_path))
        # Pilot checkpoints do not persist alg.count (used only to gate critic warmup); re-derive it from the loaded iteration.
        runner.alg.count=runner.current_learning_iteration
    else:
        runner.save(str(out/'model_0.pt'))
    manifest=dict(training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  pilot_script_sha256=hashlib.sha256(PILOT_SCRIPT.read_bytes()).hexdigest(),
                  source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  resume_from=str(resume_path) if resume_path is not None else None,
                  resume_sha256=hashlib.sha256(resume_path.read_bytes()).hexdigest() if resume_path is not None else None,
                  resumed_iteration=runner.current_learning_iteration,
                  protocol_sha256=hashlib.sha256((ROOT/'reports/recovery_4900/protocol.json').read_bytes()).hexdigest(),
                  environment=class_to_dict(cfg),training=class_to_dict(train),custom=vars(custom),seed=args.seed)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    runner.learn(num_learning_iterations=args.max_iterations,init_at_random_ep_len=False)
if __name__=='__main__':main()
