"""Append the existing actor heading channels to training critic observations."""
from heading23500_env import Heading23500Env
from heading300_critic_context import append_heading_context


class Heading300CriticContextEnv(Heading23500Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_privileged_obs != 1318:
            raise ValueError('expected original critic layout')
        self.num_privileged_obs = 1320
        # Base construction resets without computing the first observations.
        # Preserve those original values; do not advance physics or the WM.
        if self.privileged_obs_buf.shape[-1] == 1318:
            self.privileged_obs_buf = append_heading_context(self.privileged_obs_buf, self.obs_buf)

    def compute_observations(self):
        super().compute_observations()
        self.privileged_obs_buf = append_heading_context(self.privileged_obs_buf, self.obs_buf)
