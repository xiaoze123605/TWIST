"""Export a trusted local tracker checkpoint with its embedded deployment schema."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rsl_rl'))
import torch
from rsl_rl.modules.dynamics_tracker import DynamicsTrackerActorCritic
from rsl_rl.modules.dynamics_tracker_runtime import DeploymentPolicy


def export(checkpoint_path, output_path):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    spec = checkpoint['deployment_spec']
    actor = DynamicsTrackerActorCritic(spec['observation_dim'], checkpoint['critic_dim'],
                                      **checkpoint['train_cfg']['policy'])
    actor.load_state_dict(checkpoint['model_state_dict'], strict=True)
    policy = torch.jit.script(DeploymentPolicy(actor.eval(), spec).eval())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(policy, str(output), _extra_files={'deployment.json': json.dumps(spec)})
    return policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
