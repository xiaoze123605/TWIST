import torch
from torch import nn

from .geometry import rpy


class Heading300(nn.Module):
    def __init__(self, path, device):
        super().__init__()
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.history = None

    def reset(self):
        self.history = None

    def forward(self, observation, state, reference_state):
        state_frame = observation[:, 31:82]
        if self.history is None:
            initial = torch.cat((state_frame, torch.zeros_like(observation[:, 82:105])), -1)
            self.history = initial[:, None].expand(-1, 79, -1).clone()
        error = rpy(reference_state[:, 3:7])[:, 2]-rpy(state[:, 3:7])[:, 2]
        heading = torch.stack((torch.sin(error), 1-torch.cos(error)), -1)
        extended = torch.cat((observation, self.history.flatten(1), heading), -1)
        action = self.model(extended)
        transition = torch.cat((state_frame, action.clamp(-10, 10)), -1)
        self.history = torch.cat((self.history[:, 1:], transition[:, None]), 1)
        return action
