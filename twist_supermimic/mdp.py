import torch

from .data import reference
from .geometry import quat_inv, quat_rotate, rpy, quat_log, quat_mul


class Task:
    context_dim = 28
    processor_dim = 155

    def __init__(self, robot, library, noisy=True):
        self.robot, self.library, self.noisy = robot, library, noisy
        device = robot.default.device
        cfg = robot.cfg
        scales = cfg.normalization.obs_scales
        self.scales = (scales.ang_vel, scales.dof_pos, scales.dof_vel)
        ns = cfg.noise.noise_scales
        self.noise = torch.tensor([ns.ang_vel]*3+[ns.imu]*2+[ns.dof_pos]*23+[ns.dof_vel]*23+[0.]*23, device=device)
        self.soft_lower = .5*(robot.lower+robot.upper)-.475*(robot.upper-robot.lower)
        self.soft_upper = .5*(robot.lower+robot.upper)+.475*(robot.upper-robot.lower)

    def clean_frame(self, state, previous_request, ref):
        gyro = quat_rotate(quat_inv(state[:, 3:7]), state[:, 10:13])*self.scales[0]
        dq = state[:, 36:59]*self.scales[2]
        mask = torch.ones(23, device=state.device)
        mask[[4, 5, 10, 11]] = 0
        proprio = torch.cat((gyro, rpy(state[:, 3:7])[:, :2],
                             (state[:, 13:36]-self.robot.default)*self.scales[1], dq*mask, previous_request), -1)
        return torch.cat((ref, proprio), -1)

    def observe(self, state, previous_request, ref, history, processor, reset=None):
        n = len(state)
        if processor is None:
            processor = ref[:, None].expand(-1, 5, -1).reshape(n, 155).clone()
        queue, filtered = processor[:, :124].reshape(n, 4, 31), processor[:, 124:]
        if self.noisy:
            noisy = ref+.02*torch.randn_like(ref)
            noisy = torch.where(torch.rand(n, 1, device=ref.device)<.03, queue[:, -1], noisy)
            queue = torch.cat((queue[:, 1:], noisy[:, None]), 1)
            delay = torch.randint(4, (n,), device=ref.device)
            delayed = queue[torch.arange(n, device=ref.device), 3-delay]
            alpha = torch.empty(n, 1, device=ref.device).uniform_(.6, .9)
            filtered = torch.where(torch.rand(n, 1, device=ref.device)<.5,
                                   alpha*filtered+(1-alpha)*delayed, delayed)
        else:
            queue = torch.cat((queue[:, 1:], ref[:, None]), 1)
            filtered = ref
        frame = self.clean_frame(state, previous_request, filtered)
        if self.noisy:
            noise = (2*torch.rand_like(frame[:, 31:])-1)*self.noise
            mask = torch.ones(74, device=state.device)
            mask[torch.tensor([32, 33, 38, 39], device=state.device)] = 0
            frame = torch.cat((frame[:, :31], frame[:, 31:]+noise*mask), -1)
        if history is None:
            history = frame[:, None].expand(-1, 10, -1).clone()
        if reset is not None:
            history = torch.where(reset[:, None, None], frame[:, None].expand(-1, 10, -1), history)
        obs = torch.cat((frame, history.flatten(1)), -1).clamp(-100, 100)
        next_history = torch.cat((history[:, 1:], frame[:, None]), 1)
        next_processor = torch.cat((queue.flatten(1), filtered), -1)
        return obs, next_history, next_processor

    def action(self, request, previous_request, context):
        delayed = torch.where(context[:, 27:28] > .5, previous_request, request)
        clip = self.robot.cfg.normalization.clip_actions/self.robot.cfg.control.action_scale
        return delayed.clamp(-clip, clip)

    def pd_target(self, executed, context):
        return self.robot.default+self.robot.cfg.control.action_scale*executed+context[:, 1:24]

    def reward(self, state, ref_state, action, previous_action, details=False):
        p, r, v, w = self.robot.bodies(state)
        rp, rr, rv, rw = self.robot.bodies(ref_state)
        # 身体位置采用机器人当前水平位置和航向，根部单独监督全局轨迹。
        yaw = rpy(state[:, 3:7])[:, 2]-rpy(ref_state[:, 3:7])[:, 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        z, one = torch.zeros_like(c), torch.ones_like(c)
        rotation = torch.stack((c, -s, z, s, c, z, z, z, one), -1).reshape(-1, 3, 3)
        anchor = torch.cat((state[:, :2], ref_state[:, 2:3]), -1)
        rp = anchor[:, None]+((rotation[:, None] @ (rp-ref_state[:, None, :3]).unsqueeze(-1)).squeeze(-1))
        rr = rotation[:, None] @ rr
        count = self.robot.tracking_count
        pos_error = (p[:, :count]-rp[:, :count]).square().sum(-1).mean(-1)
        relative = rr.transpose(-1, -2) @ r
        cos_angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)*.5).clamp(-1+1e-6, 1-1e-6)
        ori_error = torch.acos(cos_angle[:, :count]).square().mean(-1)
        velocity_error = (v[:, :count]-rv[:, :count]).square().sum(-1).mean(-1)
        rate_error = (w[:, :count]-rw[:, :count]).square().sum(-1).mean(-1)
        root_error = (state[:, :3]-ref_state[:, :3]).square().sum(-1)
        root_ori = quat_log(quat_mul(quat_inv(ref_state[:, 3:7]), state[:, 3:7])).square().sum(-1)
        action_change = (action-previous_action).square().sum(-1)
        limit = ((state[:, 13:36]-self.soft_upper).relu()+(self.soft_lower-state[:, 13:36]).relu()).sum(-1)
        reward = .02*(torch.exp(-pos_error/.3**2)+torch.exp(-ori_error/.4**2)
                      +torch.exp(-velocity_error)+torch.exp(-rate_error/3.14**2)
                      +.5*torch.exp(-root_error/.3**2)+.5*torch.exp(-root_ori/.4**2)
                      -.1*action_change-2*limit)
        if details:
            yaw_error = rpy(state[:, 3:7])[:, 2]-rpy(ref_state[:, 3:7])[:, 2]
            torso_r, torso_ref_r = r[:, -1], (rotation.transpose(-1, -2)[:, None] @ rr)[:, -1]
            def tilt(matrix):
                return torch.stack((torch.atan2(matrix[:, 2, 1], matrix[:, 2, 2]),
                                    torch.asin((-matrix[:, 2, 0]).clamp(-.999999, .999999))), -1)
            torso_error = tilt(torso_r)-tilt(torso_ref_r)
            torso_error = torch.atan2(torch.sin(torso_error), torch.cos(torso_error))
            torso_w = (torso_r.transpose(-1, -2) @ w[:, -1, :, None]).squeeze(-1)
            torso_ref_w = (torso_ref_r.transpose(-1, -2) @ rw[:, -1, :, None]).squeeze(-1)
            return reward, dict(joint=(state[:, 13:36]-ref_state[:, 13:36]).square().mean(-1),
                                velocity=(state[:, 7:10]-ref_state[:, 7:10]).square().mean(-1),
                                torso=torso_error.square().mean(-1), torso_rate=(torso_w[:, :2]-torso_ref_w[:, :2]).square().mean(-1),
                                yaw=torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error)).square(),
                                action_change=action_change, body_position=pos_error,
                                left_foot_speed_squared=v[:, 2, :2].square().sum(-1),
                                right_foot_speed_squared=v[:, 3, :2].square().sum(-1))
        return reward

    def termination(self, state, ref_state, soft=False):
        height = (state[:, 2]-ref_state[:, 2]).abs()
        gravity_z = 1-2*state[:, 3:5].square().sum(-1)
        reference_z = 1-2*ref_state[:, 3:5].square().sum(-1)
        tilt = (gravity_z-reference_z).abs()
        if soft:
            a = torch.sigmoid((height-.25)/.02)
            b = torch.sigmoid((tilt-.8)/.05)
            return 1-(1-a)*(1-b)
        return (height > .25) | (tilt > .8)

    def critic_features(self, state, ref, obs, context, previous_action):
        return torch.cat((state, ref, obs, context, previous_action), -1)
