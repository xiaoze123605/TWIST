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

def train(args):
    args.headless = True
    dataset_receipt = None
    if args.task.startswith('g1_motion_wm_dtera'):
        from tools.prepare_motion_wm_training import verify_prepared_training_yaml
        env_cfg, _ = task_registry.get_cfgs(args.task)
        selected_motion = args.motion_file or env_cfg.motion.motion_file
        dataset_receipt = verify_prepared_training_yaml(selected_motion)
    
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    if args.task.startswith('g1_motion_wm_dtera') and os.path.isdir(log_pth):
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
    wandb.init(project=wandb_project, name=args.exptid, mode=mode, dir="../../logs")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid.py", policy="now")
    if robot_type == "g1":
        wandb.save(LEGGED_GYM_ENVS_DIR + "/g1/g1_mimic_distill_config.py", policy="now")
    
    env, _ = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args)
    if hasattr(env, 'motion_reference_pipeline'):
        import pose.utils.motion_lib_pkl as motion_module
        paths = [env.cfg.motion.motion_file, env.cfg.motion_wm.checkpoint,
                 train_cfg.policy.base_actor_jit_path]
        if dataset_receipt is not None:
            paths.append(dataset_receipt)
        manifest = dict(task=args.task, seed=args.seed, num_envs=env.num_envs,
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
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=getattr(train_cfg.runner, 'init_at_random_ep_len', True))
    if hasattr(env, 'motion_reference_pipeline'):
        completion = dict(completed_iterations=ppo_runner.current_learning_iteration,
                          peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(env.device)
                          if str(env.device).startswith('cuda') else 0,
                          peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(env.device)
                          if str(env.device).startswith('cuda') else 0)
        Path(log_pth, 'training_completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    

if __name__ == "__main__":
    args = get_args()
    train(args)
