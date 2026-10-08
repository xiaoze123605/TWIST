"""Torch-only objectives and stratified historical replay for 1250 continuation."""
import torch


def anchor_error(action, target, scale, weight):
    """Fixed action-unit scale, deliberately independent of exploration std."""
    return .5 * ((action - target) / scale).square().sum(-1) * weight


class StratifiedReplay:
    """Uniform priority reservoir within each of raw/wm x three motion regimes.

    Insert once per rollout, not once per PPO minibatch. Random persistent
    priorities retain a uniform sample of the historical insertion stream.
    Sampling gives each nonempty stratum equal representation.
    """
    def __init__(self, capacity=4096, insertion=128):
        if capacity < 1 or insertion < 1:
            raise ValueError('replay capacity and insertion must be positive')
        self.capacity, self.insertion = int(capacity), int(insertion)
        self.groups = {}

    @torch.no_grad()
    def add(self, inputs, targets, weights, strata):
        for group in range(6):
            idx = (strata == group).nonzero(as_tuple=True)[0]
            if not len(idx):
                continue
            idx = idx[torch.randperm(len(idx), device=idx.device)[:self.insertion]]
            new = dict(inputs=inputs[idx].clone(), targets=targets[idx].clone(),
                       weights=weights[idx].clone(),
                       priority=torch.rand(len(idx), device=idx.device))
            if group in self.groups:
                new = {k: torch.cat((self.groups[group][k], v)) for k, v in new.items()}
            keep = new['priority'].topk(min(self.capacity, len(new['priority']))).indices
            self.groups[group] = {k: v[keep] for k, v in new.items()}

    def sample(self, per_group=128):
        if not self.groups:
            raise RuntimeError('empty replay')
        batches = []
        for group in sorted(self.groups):
            data = self.groups[group]
            idx = torch.randint(len(data['weights']), (per_group,), device=data['weights'].device)
            batches.append({k: data[k][idx] for k in ('inputs', 'targets', 'weights')})
        return tuple(torch.cat([b[k] for b in batches]) for k in ('inputs', 'targets', 'weights'))

    def counts(self):
        return {g: len(self.groups[g]['weights']) if g in self.groups else 0 for g in range(6)}

    def state_dict(self):
        return dict(capacity=self.capacity, insertion=self.insertion,
                    groups={g: {k: v.cpu() for k, v in d.items()} for g, d in self.groups.items()})

    def load_state_dict(self, state, device):
        if (state['capacity'], state['insertion']) != (self.capacity, self.insertion):
            raise ValueError('resume replay configuration differs')
        self.groups = {int(g): {k: v.to(device) for k, v in d.items()} for g, d in state['groups'].items()}
