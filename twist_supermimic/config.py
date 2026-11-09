from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    library: str = str(ROOT / 'artifacts/twist_supermimic_local/motions.pt')
    output: str = str(ROOT / 'artifacts/twist_supermimic_local/seed0')
    device: str = 'cuda:0'
    seed: int = 0
    num_envs: int = 4096
    iterations: int = 86000
    buffer_steps: int = 1000
    warmup_steps: int = 64
    wm_only_iters: int = 500
    wm_window: int = 8
    horizon: int = 32
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 2048
    micro_batch: int = 512
    wm_width: int = 2048
    gamma: float = .99
    lam: float = .95
    lr: float = 2e-4
    lr_decay_iters: int = 69000
    critic_epochs: int = 5
    critic_minibatches: int = 4
    action_noise: float = .1
    amp: bool = True
    observation_noise: bool = True
    domain_randomization: bool = True
    checkpoint_every: int = 2000
    evaluate_every: int = 2000
    evaluation_clips: int = 128
    evaluation_envs: int = 32
    evaluation_motions: tuple = ()
    max_gpu_hours: float = 16.

    def validate(self):
        for name in ('num_envs', 'buffer_steps', 'wm_window', 'horizon', 'wm_batches',
                     'wm_batch', 'pi_batch', 'micro_batch', 'wm_width', 'iterations',
                     'checkpoint_every', 'evaluate_every', 'evaluation_clips', 'evaluation_envs'):
            if getattr(self, name) <= 0:
                raise ValueError(name + '必须大于零')
        if self.buffer_steps < max(11, self.wm_window):
            raise ValueError('经验缓存必须能够保存历史观测与完整训练窗口')
        if self.micro_batch > min(self.pi_batch, self.wm_batch):
            raise ValueError('分批大小不能超过有效批量')
        if not 0 < self.gamma < 1 or not 0 <= self.lam <= 1 or self.lr <= 0:
            raise ValueError('训练参数无效')
        if not self.device.startswith('cuda'):
            raise ValueError('该训练入口使用GPU物理和GPU优化')
        if self.warmup_steps < max(11, self.wm_window) or self.max_gpu_hours <= 0:
            raise ValueError('热身长度或时间预算无效')
        return self

    def to_dict(self):
        return asdict(self)
