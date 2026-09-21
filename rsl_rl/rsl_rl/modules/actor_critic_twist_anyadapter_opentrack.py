"""Paper-style AnyAdapter for a frozen TWIST actor.

This intentionally contains no DTERA tracking-error path or action-level
residual.  The implementation is shared with the reviewed Any2Track module:
the frozen TWIST backbone is reconstructed from its JIT artifact and receives
only zero-initialized, layer-wise adapter additions.
"""

from .actor_critic_twist_any2track import (
    Any2TrackHistoryEncoder,
    Any2TrackWorldModel,
    TwistLayerwiseAdapterActor,
    TwistAny2TrackActorCritic,
)


class TwistAnyAdapterOpenTrackActorCritic(TwistAny2TrackActorCritic):
    """OpenTrack-faithful naming and strict 7001-D policy observation contract."""

    EXPECTED_BASE_OBS_DIM = 1155
    EXPECTED_HISTORY_LEN = 79
    EXPECTED_HISTORY_FRAME_DIM = 74
    EXPECTED_POLICY_OBS_DIM = 7001

    def __init__(self, *args, **kwargs):
        base_obs_dim = int(kwargs.get("base_obs_dim", self.EXPECTED_BASE_OBS_DIM))
        history_len = int(kwargs.get("history_len", self.EXPECTED_HISTORY_LEN))
        history_frame_dim = int(
            kwargs.get("history_frame_dim", self.EXPECTED_HISTORY_FRAME_DIM)
        )
        if (base_obs_dim, history_len, history_frame_dim) != (
            self.EXPECTED_BASE_OBS_DIM,
            self.EXPECTED_HISTORY_LEN,
            self.EXPECTED_HISTORY_FRAME_DIM,
        ):
            raise ValueError(
                "AnyAdapter-OpenTrack requires 1155 base + 79x74 history = 7001 "
                f"dims, got {base_obs_dim} + {history_len}x{history_frame_dim}."
            )
        kwargs.update(
            base_obs_dim=base_obs_dim,
            history_len=history_len,
            history_frame_dim=history_frame_dim,
        )
        super().__init__(*args, **kwargs)

    def split_obs(self, obs):
        if obs.shape[-1] != self.EXPECTED_POLICY_OBS_DIM:
            raise RuntimeError(
                "AnyAdapter-OpenTrack policy expects exactly 7001 observation "
                f"dims, got {obs.shape[-1]}."
            )
        return super().split_obs(obs)
