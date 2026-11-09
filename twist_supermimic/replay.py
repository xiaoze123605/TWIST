import torch


class Replay:
    def __init__(self, steps, environments, device):
        self.steps, self.environments, self.device = steps, environments, device
        fields = dict(state=59, next_state=59, request=23, executed=23, previous_request=23,
                      previous_action=23, pd_target=23, context=28, obs=1155,
                      history=1050, processor=155, motion=1, frame=1, done=1, reward=1,
                      contact=2, episode=1)
        # 历史通过当前及先前帧重建，避免每个时间步重复保存十份观测。
        fields.pop('history')
        fields['obs'] = 105
        bytes_per_transition = sum(width*(8 if k in ('motion', 'frame', 'episode') else 1 if k == 'done' else 4)
                                   for k, width in fields.items())
        required = steps*environments*bytes_per_transition
        if torch.device(device).type == 'cuda':
            available, _ = torch.cuda.mem_get_info(device)
            if required > available:
                raise RuntimeError('经验缓存需要%.2f GiB，当前可用显存只有%.2f GiB' % (required/2**30, available/2**30))
        self.data = {k: torch.empty((steps, environments, width), device=device,
                                   dtype=torch.long if k in ('motion', 'frame', 'episode') else torch.bool if k == 'done' else torch.float32)
                     for k, width in fields.items()}
        self.ptr = self.size = 0

    def add(self, row):
        if set(row) != set(self.data):
            raise ValueError('经验缓存字段不一致：'+repr(set(row)^set(self.data)))
        for key, dst in self.data.items():
            src = row[key]
            if src.shape != dst[self.ptr].shape:
                raise ValueError(key+'缓存形状不一致')
            dst[self.ptr].copy_(src)
        self.ptr = (self.ptr+1) % self.steps
        self.size = min(self.size+1, self.steps)

    def rows(self, ages):
        return (self.ptr-self.size+ages) % self.steps

    def windows(self, batch, length):
        if self.size < length:
            raise ValueError('经验缓存不足一个训练窗口')
        age = torch.randint(self.size-length+1, (batch,), device=self.device)
        env = torch.randint(self.environments, (batch,), device=self.device)
        rows = self.rows(age[:, None]+torch.arange(length, device=self.device))
        out = {k: self.data[k][rows, env[:, None]] for k in ('state', 'next_state', 'executed', 'context', 'done')}
        done = out['done'].squeeze(-1).float()
        out['mask'] = torch.cumprod(torch.cat((torch.ones_like(done[:, :1]), 1-done[:, :-1]), 1), 1)
        return out

    def starts(self, batch):
        if self.size == 0:
            raise ValueError('经验缓存为空')
        age = torch.randint(self.size, (batch,), device=self.device)
        env = torch.randint(self.environments, (batch,), device=self.device)
        row = self.rows(age)
        out = {k: x[row, env] for k, x in self.data.items()}
        current_episode = out['episode']
        history = []
        earliest = out['obs']
        # 从最近帧向前遇到回合边界后，使用该回合最早的真实观测补齐。
        for back in range(1, 11):
            prior_age = (age-back).clamp_min(0)
            prior_row = self.rows(prior_age)
            valid = (age >= back) & (self.data['episode'][prior_row, env].squeeze(-1) == current_episode.squeeze(-1))
            earliest = torch.where(valid[:, None], self.data['obs'][prior_row, env], earliest)
            history.append(earliest)
        history = torch.stack(list(reversed(history)), 1)
        out['history'] = history
        out['full_obs'] = torch.cat((out['obs'], history.flatten(1)), -1)
        out['next_history'] = torch.cat((history[:, 1:], out['obs'][:, None]), 1)
        return out
