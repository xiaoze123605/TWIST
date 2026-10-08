"""7003-D23500 actor with a zero-initialized, bounded heading correction."""
import torch
from torch import nn
from rsl_rl.modules.actor_critic_twist_anyadapter_opentrack import TwistAnyAdapterOpenTrackActorCritic

class TwistHeadingSupportActorCritic(TwistAnyAdapterOpenTrackActorCritic):
    EXPECTED_POLICY_OBS_DIM=7003
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        head=nn.Linear(130,23)
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.layerwise_actor.adapter.append(head)

    def load_state_dict(self,state_dict,strict=True):
        state=dict(state_dict)
        for key in ['layerwise_actor.adapter.5.weight','layerwise_actor.adapter.5.bias']:
            if key not in state:state[key]=self.state_dict()[key]
        return super().load_state_dict(state,strict=strict)

    def heading_delta(self,z,context):
        return .2*torch.tanh(self.layerwise_actor.adapter[5](torch.cat((z,context),-1)))

    def actor_mean(self,observations):
        base,history=self.split_obs(observations)
        z=self.encode_history_for_policy(history)
        return self.layerwise_actor(base,z)+self.heading_delta(z,observations[:,-2:])
