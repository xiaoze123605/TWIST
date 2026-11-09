import math

from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from .data import reference
from .geometry import finite


class GymEnvironment:
    def __init__(self, config, robot, library, task, evaluation=False):
        self.config, self.robot, self.library, self.task = config, robot, library, task
        self.device, self.n = config.device, config.num_envs
        self.evaluation = evaluation
        self.gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.dt = robot.cfg.sim.dt
        params.substeps = robot.cfg.sim.substeps
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, -9.81)
        params.use_gpu_pipeline = True
        for name in ('solver_type', 'num_position_iterations', 'num_velocity_iterations',
                     'contact_offset', 'rest_offset', 'bounce_threshold_velocity',
                     'max_depenetration_velocity', 'default_buffer_size_multiplier'):
            setattr(params.physx, name, getattr(robot.cfg.sim.physx, name))
        params.physx.use_gpu = True
        params.physx.num_threads = 4
        params.physx.max_gpu_contact_pairs = max(2**18, self.n*4096)
        params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
        index = torch.device(config.device).index or 0
        self.sim = self.gym.create_sim(index, -1, gymapi.SIM_PHYSX, params)
        if self.sim is None:
            raise RuntimeError('Isaac Gym GPU物理环境创建失败')
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane)
        options = gymapi.AssetOptions()
        for name in ('collapse_fixed_joints', 'replace_cylinder_with_capsule', 'flip_visual_attachments',
                     'fix_base_link', 'disable_gravity', 'default_dof_drive_mode', 'density',
                     'angular_damping', 'linear_damping', 'max_angular_velocity', 'max_linear_velocity',
                     'armature', 'thickness'):
            setattr(options, name, getattr(robot.cfg.asset, name))
        asset = self.gym.load_asset(self.sim, str(robot.path.parent), robot.path.name, options)
        self.asset = asset
        if self.gym.get_asset_dof_names(asset) != robot.names:
            raise ValueError('GPU物理关节顺序与TWIST动作顺序不同')
        properties = self.gym.get_asset_dof_properties(asset)
        self.torque_limits = torch.tensor(properties['effort'].copy(), device=self.device)*robot.cfg.rewards.torque_safety_limit
        properties['driveMode'][:] = gymapi.DOF_MODE_EFFORT
        properties['stiffness'][:] = 0
        properties['damping'][:] = 0
        properties['armature'][:] = robot.cfg.asset.dof_armature
        self.envs, self.actors, origins, contexts = [], [], [], []
        rand = np.random.default_rng(config.seed)
        body_names = self.gym.get_asset_rigid_body_names(asset)
        torso_id = body_names.index('torso_link')
        self.feet = [body_names.index(name) for name in robot.cfg.asset.feet_bodies]
        self.key_ids = [body_names.index(name) for name in robot.key_names]
        for i in range(self.n):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 3), int(math.ceil(math.sqrt(self.n))))
            pose = gymapi.Transform()
            pose.p.z = .8
            actor = self.gym.create_actor(env, asset, pose, 'g1', i, robot.cfg.asset.self_collisions, 0)
            self.gym.set_actor_dof_properties(env, actor, properties)
            randomized = config.domain_randomization and not evaluation
            friction = rand.uniform(.3, 1.2) if randomized else 1.
            com = rand.uniform([-0.025, -.05, -.05], [.025, .05, .05]) if randomized else np.zeros(3)
            shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
            for shape in shapes:
                shape.friction = float(friction)
            self.gym.set_actor_rigid_shape_properties(env, actor, shapes)
            bodies = self.gym.get_actor_rigid_body_properties(env, actor)
            bodies[torso_id].com.x += float(com[0])
            bodies[torso_id].com.y += float(com[1])
            bodies[torso_id].com.z += float(com[2])
            self.gym.set_actor_rigid_body_properties(env, actor, bodies, recomputeInertia=True)
            origin = self.gym.get_env_origin(env)
            origins.append([origin.x, origin.y, origin.z])
            contexts.append([friction]+[0.]*23+list(com)+[0.])
            self.envs.append(env)
            self.actors.append(actor)
        self.gym.prepare_sim(self.sim)
        self.origins = torch.tensor(origins, device=self.device)
        self.context = torch.tensor(contexts, device=self.device, dtype=torch.float32)
        self.root = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.n, 23, 2)
        self.contacts = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.n, len(body_names), 3)
        self.rigid = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.n, len(body_names), 13)
        self.episode = torch.zeros(self.n, 1, device=self.device, dtype=torch.long)
        self.motion = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.frame = torch.zeros_like(self.motion)
        self.previous_request = torch.zeros(self.n, 23, device=self.device)
        self.previous_action = torch.zeros_like(self.previous_request)
        self.history = torch.zeros(self.n, 10, 105, device=self.device)
        self.processor = torch.zeros(self.n, 155, device=self.device)
        self.obs = torch.zeros(self.n, 1155, device=self.device)
        self.physical_controls = 0
        ids = torch.arange(self.n, device=self.device)
        if evaluation:
            first = library.roles['dev']
            if not len(first):
                raise ValueError('评价动作集合为空')
            self.reset(ids, first[torch.zeros(self.n, device=self.device, dtype=torch.long)], torch.zeros(self.n, device=self.device, dtype=torch.long))
        else:
            self.reset(ids)

    def state(self):
        return torch.cat((self.root[:, :3]-self.origins, self.root[:, 3:13], self.dof[:, :, 0], self.dof[:, :, 1]), -1).clone()

    def refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    @torch.no_grad()
    def reset(self, ids, motion=None, frame=None):
        if len(ids) == 0:
            return
        if motion is None:
            if self.evaluation:
                raise ValueError('评价环境重置必须指定完整动作')
            motion, frame = self.library.sample(len(ids))
        self.motion[ids], self.frame[ids] = motion, frame
        state = self.library.state(motion, frame)
        self.root[ids, :3] = state[:, :3]+self.origins[ids]
        self.root[ids, 3:13] = state[:, 3:13]
        self.dof[ids, :, 0] = state[:, 13:36]
        self.dof[ids, :, 1] = state[:, 36:59]
        actor_ids = ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root), gymtorch.unwrap_tensor(actor_ids), len(ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof), gymtorch.unwrap_tensor(actor_ids), len(ids))
        self.previous_action[ids] = 0
        self.previous_request[ids] = 0
        if self.config.domain_randomization and not self.evaluation:
            self.context[ids, 1:24] = torch.empty(len(ids), 23, device=self.device).uniform_(-.01, .01)
            self.context[ids, 27] = torch.randint(2, (len(ids),), device=self.device).float()
        else:
            self.context[ids, 1:24] = 0
            self.context[ids, 27] = 0
        self.episode[ids] += 1
        obs, hist, processor = self.task.observe(state, self.previous_request[ids], reference(state), None, None)
        self.obs[ids], self.history[ids], self.processor[ids] = obs, hist, processor

    @torch.no_grad()
    def step(self, request, auto_reset=True, calculate_reward=True):
        if not calculate_reward and not self.evaluation:
            raise ValueError('训练采样必须计算真实奖励')
        before = self.state()
        before_motion, before_frame = self.motion.clone(), self.frame.clone()
        executed = self.task.action(request, self.previous_request, self.context)
        target = self.task.pd_target(executed, self.context)
        old_action = self.previous_action.clone()
        row = dict(state=before, request=request.clone(), executed=executed.clone(), previous_request=self.previous_request.clone(),
                   previous_action=old_action, pd_target=target.clone(), context=self.context.clone(),
                   obs=self.obs[:, :105].clone(), processor=self.processor.clone(),
                   motion=before_motion[:, None], frame=before_frame[:, None], episode=self.episode.clone())
        for _ in range(self.robot.cfg.control.decimation):
            torque = ((target-self.dof[:, :, 0])*self.robot.kp-self.dof[:, :, 1]*self.robot.kd).clamp(-self.torque_limits, self.torque_limits)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torque.contiguous()))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.refresh()
        sn = self.state()
        finite('GPU物理状态', sn)
        self.frame += 1
        ref = self.library.state(self.motion, self.frame)
        reward = self.task.reward(sn, ref, executed, old_action) if calculate_reward else None
        failed = self.task.termination(sn, ref)
        ended = self.frame >= self.library.lengths[self.motion]+32
        done = failed | ended
        row.update(next_state=sn.clone(), done=done[:, None],
                   contact=(self.contacts[:, self.feet, 2] > 5).float())
        if calculate_reward:
            row['reward'] = reward[:, None]
        self.previous_request.copy_(request)
        self.previous_action.copy_(executed)
        obs, hist, processor = self.task.observe(sn, request, reference(ref), self.history, self.processor)
        self.obs, self.history, self.processor = obs, hist, processor
        self.physical_controls += self.n
        if not self.evaluation:
            self.library.record(before_motion, before_frame, reward)
        if auto_reset:
            self.reset(done.nonzero().flatten())
        return row, failed, ended

    def close(self):
        self.gym.destroy_sim(self.sim)
