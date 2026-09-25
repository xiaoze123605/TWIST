# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# Prefer this checkout over editable installs pointing at another TWIST tree.
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'legged_gym'), str(ROOT / 'rsl_rl'), str(ROOT / 'pose')]

import isaacgym
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry, class_to_dict

import torch
import wandb

def train(args, warm_start_checkpoint=None):
    args.headless = True
    dataset_receipt = None
    anchored_adapter = args.task.startswith('g1_twist_baseline_adapter')
    clean_motion_adapter = args.task.startswith('g1_motion_wm_anyadapter_clean')
    guarded_clean_pilot = args.task.startswith('g1_motion_wm_anyadapter_clean_guarded')
    baseline_motion_pilot = args.task.startswith('g1_motion_wm_anyadapter_baseline_pilot')
    baseline_motion_continue = args.task == 'g1_motion_wm_anyadapter_baseline_continue'
    stable_motion_adapter = args.task.startswith('g1_motion_wm_anyadapter_stable')
    audited_training = (args.task.startswith('g1_motion_wm_dtera') or anchored_adapter or
                        clean_motion_adapter or baseline_motion_pilot or
                        baseline_motion_continue or stable_motion_adapter)
    if warm_start_checkpoint is not None:
        if not (anchored_adapter or guarded_clean_pilot or baseline_motion_pilot or
                baseline_motion_continue or stable_motion_adapter) or args.resume or args.resumeid:
            raise ValueError('--warm-start-checkpoint requires a fresh guarded adapter run')
        warm_start_checkpoint = Path(warm_start_checkpoint).resolve()
        if not warm_start_checkpoint.is_file():
            raise FileNotFoundError(warm_start_checkpoint)
    if guarded_clean_pilot or baseline_motion_pilot or baseline_motion_continue:
        if warm_start_checkpoint is None:
            raise ValueError('Motion-WM adapter pilot requires --warm-start-checkpoint')
        pilot_env_cfg, pilot_train_cfg = task_registry.get_cfgs(args.task)
        anchor = Path(pilot_train_cfg.algorithm.policy_anchor_checkpoint).resolve()
        if warm_start_checkpoint != anchor:
            raise ValueError('Motion-WM adapter pilot must warm-start from its policy anchor')
        source_manifest = warm_start_checkpoint.parent / 'run_manifest.json'
        if not source_manifest.is_file():
            raise ValueError('Motion-WM adapter pilot requires an audited source run')
        source = json.loads(source_manifest.read_text())
        source_motion = source.get('environment', {}).get('motion', {}).get('motion_file')
        target_motion = pilot_env_cfg.motion.motion_file
        required_source_task = (
            'g1_motion_wm_anyadapter_clean' if guarded_clean_pilot else
            'g1_motion_wm_anyadapter_baseline_pilot_raw' if baseline_motion_continue else
            'g1_twist_baseline_adapter_refine'
        )
        if (source.get('task') != required_source_task or
                not source_motion or
                Path(source_motion).resolve() != Path(target_motion).resolve()):
            raise ValueError('Motion-WM adapter pilot requires the matching audited source dataset')
    if stable_motion_adapter:
        stable_env_cfg, stable_train_cfg = task_registry.get_cfgs(args.task)
        if warm_start_checkpoint is not None:
            anchor = Path(stable_train_cfg.algorithm.policy_anchor_checkpoint).resolve()
            if warm_start_checkpoint != anchor:
                raise ValueError('Stable run must warm-start from raw150 anchor')
            source_manifest = warm_start_checkpoint.parent / 'run_manifest.json'
            if not source_manifest.is_file():
                raise ValueError('Stable run requires audited raw150 source')
            source = json.loads(source_manifest.read_text())
            if source.get('task') != 'g1_motion_wm_anyadapter_baseline_pilot_raw':
                raise ValueError('Stable run source task must be baseline raw pilot')
        elif args.resumeid:
            source_manifest = (Path(LEGGED_GYM_ROOT_DIR) / 'logs' /
                               args.proj_name / args.resumeid / 'run_manifest.json')
            if not source_manifest.is_file():
                raise ValueError('Stable resume requires an audited source run')
            source = json.loads(source_manifest.read_text())
            if source.get('task') != args.task:
                raise ValueError('Stable resume must use the same task')
        else:
            raise ValueError('Stable run requires --warm-start-checkpoint or --resumeid')
        source_motion = source.get('environment', {}).get('motion', {}).get('motion_file')
        if not source_motion or Path(source_motion).resolve() != Path(stable_env_cfg.motion.motion_file).resolve():
            raise ValueError('Stable run source motion dataset mismatch')
    if args.task == 'g1_twist_baseline_adapter_refine':
        if warm_start_checkpoint is None and not (args.resume or args.resumeid):
            raise ValueError('refinement requires --warm-start-checkpoint or --resumeid')
        if warm_start_checkpoint is not None:
            _, refine_cfg = task_registry.get_cfgs(args.task)
            anchor = Path(refine_cfg.algorithm.policy_anchor_checkpoint).resolve()
            if warm_start_checkpoint != anchor:
                raise ValueError('refinement warm start must match the policy anchor checkpoint')
    if audited_training:
        from tools.prepare_motion_wm_training import verify_prepared_training_yaml
        env_cfg, _ = task_registry.get_cfgs(args.task)
        selected_motion = args.motion_file or env_cfg.motion.motion_file
        dataset_receipt = verify_prepared_training_yaml(selected_motion)
    if clean_motion_adapter and (args.resume or args.resumeid):
        if not args.resumeid:
            raise ValueError('clean Motion-WM adapter resume requires --resumeid')
        source_manifest = (Path(LEGGED_GYM_ROOT_DIR) / 'logs' /
                           args.proj_name / args.resumeid / 'run_manifest.json')
        if not source_manifest.is_file():
            raise ValueError('clean Motion-WM adapter may resume only its own audited runs')
        source = json.loads(source_manifest.read_text())
        source_motion = source.get('environment', {}).get('motion', {}).get('motion_file')
        if (source.get('task') != args.task or
                not source_motion or
                Path(source_motion).resolve() != Path(selected_motion).resolve()):
            raise ValueError('clean Motion-WM adapter may resume only its own audited runs')
    
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    if audited_training and os.path.isdir(log_pth):
        raise FileExistsError('Use a new --exptid; preserving existing run: ' + log_pth)
    os.makedirs(log_pth, exist_ok=True)
    
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 5
        args.num_envs = 32
        args.headless = False
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"
        
    robot_type = args.task.split("_")[0]
    
    wandb_project = f"{robot_type}_mimic"
    wandb_dir = (log_pth if anchored_adapter or clean_motion_adapter or
                 baseline_motion_pilot or baseline_motion_continue or
                 stable_motion_adapter else "../../logs")
    wandb.init(project=wandb_project, name=args.exptid, mode=mode, dir=wandb_dir)
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid.py", policy="now")
    if robot_type == "g1":
        wandb.save(LEGGED_GYM_ENVS_DIR + "/g1/g1_mimic_distill_config.py", policy="now")
    
    env, _ = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args)
    if warm_start_checkpoint is not None:
        ppo_runner.load(str(warm_start_checkpoint), load_optimizer=False)
        ppo_runner.current_learning_iteration = 0
        ppo_runner.alg.counter = 0
        env.global_counter = 0
        env.total_env_steps_counter = 0
    if hasattr(env, 'motion_reference_pipeline') or anchored_adapter or stable_motion_adapter:
        import pose.utils.motion_lib_pkl as motion_module
        paths = [env.cfg.motion.motion_file, train_cfg.policy.base_actor_jit_path]
        if hasattr(env.cfg, 'motion_wm'):
            paths.append(env.cfg.motion_wm.checkpoint)
        if dataset_receipt is not None:
            paths.append(dataset_receipt)
        if warm_start_checkpoint is not None:
            paths.append(warm_start_checkpoint)
        resume_checkpoint = None
        if args.resumeid and args.checkpoint is not None and args.checkpoint >= 0:
            resume_checkpoint = (Path(LEGGED_GYM_ROOT_DIR) / 'logs' /
                                 args.proj_name / args.resumeid /
                                 f'model_{args.checkpoint}.pt')
            paths.append(resume_checkpoint)
        anchor_checkpoint = getattr(train_cfg.algorithm, 'policy_anchor_checkpoint', None)
        if anchor_checkpoint:
            paths.append(anchor_checkpoint)
        manifest = dict(task=args.task, seed=args.seed, num_envs=env.num_envs,
                        warm_start_checkpoint=str(warm_start_checkpoint)
                        if warm_start_checkpoint is not None else None,
                        resume_checkpoint=str(resume_checkpoint)
                        if resume_checkpoint is not None else None,
                        motion_library=motion_module.__file__, control_dt=env.dt,
                        loaded_motions=env._motion_lib.num_motions(),
                        minimum_reference_remaining_time=getattr(
                            env, '_minimum_reference_remaining_time', 0.0),
                        environment=class_to_dict(env.cfg),
                        inputs={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in paths},
                        policy=ppo_runner.policy_cfg, algorithm=ppo_runner.alg_cfg,
                        runner=ppo_runner.cfg)
        Path(log_pth, 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    remaining_iterations = train_cfg.runner.max_iterations
    if (anchored_adapter or clean_motion_adapter or baseline_motion_pilot or
            baseline_motion_continue or stable_motion_adapter):
        # The command-line limit is absolute even when resuming from a checkpoint.
        remaining_iterations = max(0, remaining_iterations - ppo_runner.current_learning_iteration)
    ppo_runner.learn(num_learning_iterations=remaining_iterations,
                     init_at_random_ep_len=getattr(train_cfg.runner, 'init_at_random_ep_len', True))
    if hasattr(env, 'motion_reference_pipeline') or anchored_adapter or stable_motion_adapter:
        completion = dict(completed_iterations=ppo_runner.current_learning_iteration,
                          peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(env.device)
                          if str(env.device).startswith('cuda') else 0,
                          peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(env.device)
                          if str(env.device).startswith('cuda') else 0)
        Path(log_pth, 'training_completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    

if __name__ == "__main__":
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument('--warm-start-checkpoint')
    custom_args, remaining_args = custom_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    args = get_args()
    train(args, warm_start_checkpoint=custom_args.warm_start_checkpoint)
