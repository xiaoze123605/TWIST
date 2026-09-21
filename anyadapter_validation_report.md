# Motion-WM + OpenTrack-style AnyAdapter validation

Overall: **PASS**

## Required checks

- **PASS** `observation_dim_7001` - [[96, 7001], [96, 7001], [96, 7001], [96, 7001], [96, 7001]]
- **PASS** `all_finite` - obs/loss/action telemetry
- **PASS** `history_encoder_wm_grad_positive` - [0.00026633263733733507, 0.0003760375805857886, 0.0008474069250696899, 0.000299437769794284, 0.0004440220146074781]
- **PASS** `history_encoder_ppo_grad_zero` - [0.0, 0.0, 0.0, 0.0, 0.0]
- **PASS** `dynamics_wm_grad_positive` - [1.0000000171311585, 0.999999991643849, 0.9999996505808614, 0.9999999193725848, 0.9999999132642606]
- **PASS** `adapter_grad_positive` - [45.9197889553383, 151.4946934700012, 187.12650680541992, 155.37458057403563, 177.58934650421142]
- **PASS** `frozen_modules_no_grad` - TWIST and Motion-WM
- **PASS** `twist_hash_unchanged` - 45005a8e66097fc45b1b11a3b9d94ef3c5b17f61060b2ae5d9fcd468ccc4e0ef -> 45005a8e66097fc45b1b11a3b9d94ef3c5b17f61060b2ae5d9fcd468ccc4e0ef
- **PASS** `motion_wm_hash_unchanged` - 40c92737a6693050f17d136a31deedb202e4e940da328bbb07ef2885bc25feb6 -> 40c92737a6693050f17d136a31deedb202e4e940da328bbb07ef2885bc25feb6
- **PASS** `adapter_changed` - 0 -> 0.10324248
- **PASS** `history_encoder_changed` - 9.2570257 -> 9.26403429
- **PASS** `dynamics_wm_changed` - 25.5140492 -> 25.5453568
- **PASS** `dependency_before_identity` - {'embedding_difference_l2': 0.8538138270378113, 'action_difference_l2': 0.0, 'action_a_base_max_abs': 0.0, 'action_b_base_max_abs': 0.0}
- **PASS** `history_affects_embedding_and_action` - {'embedding_difference_l2': 0.8596292734146118, 'action_difference_l2': 0.0010465114610269666, 'action_a_base_max_abs': 0.22807466983795166, 'action_b_base_max_abs': 0.22794580459594727}
- **PASS** `history_timing` - {'pass': True, 'chronological_past_only': True, 'all_state_indices_exact': True, 'all_action_indices_exact': True, 'all_state_action_indices_aligned': True, 'current_state_leakage': False, 'reset_no_leakage': True, 'observed_state_first_last': [3.0, 81.0], 'observed_action_first_last': [3.0, 81.0], 'expected_state_first_last': [3.0, 81.0], 'expected_action_first_last': [3.0, 81.0]}
- **PASS** `reference_pipeline` - {'corrupt': {'policy_reference_equals_processed': True, 'clean_ref_recorded': True, 'corrupt_ref_recorded': True, 'reward_target_unchanged': True, 'motion_wm_requires_grad': False, 'motion_wm_has_grad': False}, 'wm': {'policy_reference_equals_processed': True, 'clean_ref_recorded': True, 'corrupt_ref_recorded': True, 'reward_target_unchanged': True, 'motion_wm_requires_grad': False, 'motion_wm_has_grad': False}}
- **PASS** `jit_runtime` - {'path': '/home/hank/TWIST（anyadapter）/legged_gym/logs/g1_motion_wm_anyadapter/smoke_seed42/traced/anyadapter-opentrack-smoke-jit.pt', 'max_abs_error': 0.0, 'contains_history_encoder': True, 'contains_layerwise_actor': True, 'contains_dynamics_world_model': False, 'contains_critic': False, 'contains_motion_wm': False, 'detected_observation_dim': 7001, 'simulator_message': '[AnyAdapter-OpenTrack] Detected 7001-D layerwise AnyAdapter policy'}

## Smoke updates

```json
[
  {
    "update": 1,
    "observation_shape": [
      96,
      7001
    ],
    "wm_loss": 5.707610130310059,
    "wm_gyro_l1": 0.5549619570374489,
    "wm_orientation_l1": 0.30414559692144394,
    "wm_dof_pos_l1": 1.20551335811615,
    "wm_dof_vel_l1": 0.4131181761622429,
    "history_embedding_mean": 0.0004832979175262153,
    "history_embedding_std": 0.03475366160273552,
    "history_embedding_l2": 0.3931756913661957,
    "history_encoder_wm_grad_norm": 0.00026633263733733507,
    "history_encoder_ppo_grad_norm": 0.0,
    "dynamics_wm_grad_norm": 1.0000000171311585,
    "adapter_grad_norm": 45.9197889553383,
    "adapter_weight_norm": 0.08907158652966786,
    "mean_abs_adapted_minus_base": 0.045781854540109634,
    "max_abs_adapted_minus_base": 0.3513067960739136,
    "action_min": -3.3425021171569824,
    "action_max": 3.005718469619751,
    "twist_base_has_grad": false,
    "motion_wm_has_grad": false,
    "finite": true
  },
  {
    "update": 2,
    "observation_shape": [
      96,
      7001
    ],
    "wm_loss": 4.941298007965088,
    "wm_gyro_l1": 0.5817119777202606,
    "wm_orientation_l1": 0.15332337841391563,
    "wm_dof_pos_l1": 1.0782033205032349,
    "wm_dof_vel_l1": 0.37583601474761963,
    "history_embedding_mean": 0.00014090663171373308,
    "history_embedding_std": 0.03705587983131409,
    "history_embedding_l2": 0.41894015669822693,
    "history_encoder_wm_grad_norm": 0.0003760375805857886,
    "history_encoder_ppo_grad_norm": 0.0,
    "dynamics_wm_grad_norm": 0.999999991643849,
    "adapter_grad_norm": 151.4946934700012,
    "adapter_weight_norm": 0.09307389840457031,
    "mean_abs_adapted_minus_base": 0.03759729117155075,
    "max_abs_adapted_minus_base": 0.3105314373970032,
    "action_min": -2.583156108856201,
    "action_max": 3.742361307144165,
    "twist_base_has_grad": false,
    "motion_wm_has_grad": false,
    "finite": true
  },
  {
    "update": 3,
    "observation_shape": [
      96,
      7001
    ],
    "wm_loss": 4.542367577552795,
    "wm_gyro_l1": 0.4916266053915024,
    "wm_orientation_l1": 0.15864421799778938,
    "wm_dof_pos_l1": 1.1039285361766815,
    "wm_dof_vel_l1": 0.37416972219944,
    "history_embedding_mean": 0.00031874034903012216,
    "history_embedding_std": 0.039889149367809296,
    "history_embedding_l2": 0.4505351483821869,
    "history_encoder_wm_grad_norm": 0.0008474069250696899,
    "history_encoder_ppo_grad_norm": 0.0,
    "dynamics_wm_grad_norm": 0.9999996505808614,
    "adapter_grad_norm": 187.12650680541992,
    "adapter_weight_norm": 0.0966332853606672,
    "mean_abs_adapted_minus_base": 0.02968386746942997,
    "max_abs_adapted_minus_base": 0.217523455619812,
    "action_min": -3.3829922676086426,
    "action_max": 4.235774993896484,
    "twist_base_has_grad": false,
    "motion_wm_has_grad": false,
    "finite": true
  },
  {
    "update": 4,
    "observation_shape": [
      96,
      7001
    ],
    "wm_loss": 3.766058921813965,
    "wm_gyro_l1": 0.4324806183576584,
    "wm_orientation_l1": 0.08256415650248528,
    "wm_dof_pos_l1": 1.0146721452474594,
    "wm_dof_vel_l1": 0.3523256704211235,
    "history_embedding_mean": -0.0007168941665440798,
    "history_embedding_std": 0.043628495186567307,
    "history_embedding_l2": 0.4907204508781433,
    "history_encoder_wm_grad_norm": 0.000299437769794284,
    "history_encoder_ppo_grad_norm": 0.0,
    "dynamics_wm_grad_norm": 0.9999999193725848,
    "adapter_grad_norm": 155.37458057403563,
    "adapter_weight_norm": 0.10024615406606173,
    "mean_abs_adapted_minus_base": 0.043891921639442444,
    "max_abs_adapted_minus_base": 0.36268794536590576,
    "action_min": -3.090981960296631,
    "action_max": 3.91390323638916,
    "twist_base_has_grad": false,
    "motion_wm_has_grad": false,
    "finite": true
  },
  {
    "update": 5,
    "observation_shape": [
      96,
      7001
    ],
    "wm_loss": 3.4372439980506897,
    "wm_gyro_l1": 0.36493542045354843,
    "wm_orientation_l1": 0.08904510829597712,
    "wm_dof_pos_l1": 0.9960609674453735,
    "wm_dof_vel_l1": 0.3425608277320862,
    "history_embedding_mean": -0.0015756412176415324,
    "history_embedding_std": 0.04896650090813637,
    "history_embedding_l2": 0.5438809990882874,
    "history_encoder_wm_grad_norm": 0.0004440220146074781,
    "history_encoder_ppo_grad_norm": 0.0,
    "dynamics_wm_grad_norm": 0.9999999132642606,
    "adapter_grad_norm": 177.58934650421142,
    "adapter_weight_norm": 0.10324248008788742,
    "mean_abs_adapted_minus_base": 0.031849347054958344,
    "max_abs_adapted_minus_base": 0.25177139043807983,
    "action_min": -3.2180521488189697,
    "action_max": 3.721498727798462,
    "twist_base_has_grad": false,
    "motion_wm_has_grad": false,
    "finite": true
  }
]
```

## History/action dependency

```json
{
  "before": {
    "embedding_difference_l2": 0.8538138270378113,
    "action_difference_l2": 0.0,
    "action_a_base_max_abs": 0.0,
    "action_b_base_max_abs": 0.0
  },
  "after": {
    "embedding_difference_l2": 0.8596292734146118,
    "action_difference_l2": 0.0010465114610269666,
    "action_a_base_max_abs": 0.22807466983795166,
    "action_b_base_max_abs": 0.22794580459594727
  }
}
```

## History timing

```json
{
  "pass": true,
  "chronological_past_only": true,
  "all_state_indices_exact": true,
  "all_action_indices_exact": true,
  "all_state_action_indices_aligned": true,
  "current_state_leakage": false,
  "reset_no_leakage": true,
  "observed_state_first_last": [
    3.0,
    81.0
  ],
  "observed_action_first_last": [
    3.0,
    81.0
  ],
  "expected_state_first_last": [
    3.0,
    81.0
  ],
  "expected_action_first_last": [
    3.0,
    81.0
  ]
}
```

## Reference pipeline

```json
{
  "corrupt": {
    "policy_reference_equals_processed": true,
    "clean_ref_recorded": true,
    "corrupt_ref_recorded": true,
    "reward_target_unchanged": true,
    "motion_wm_requires_grad": false,
    "motion_wm_has_grad": false
  },
  "wm": {
    "policy_reference_equals_processed": true,
    "clean_ref_recorded": true,
    "corrupt_ref_recorded": true,
    "reward_target_unchanged": true,
    "motion_wm_requires_grad": false,
    "motion_wm_has_grad": false
  }
}
```

## JIT runtime

```json
{
  "path": "/home/hank/TWIST\uff08anyadapter\uff09/legged_gym/logs/g1_motion_wm_anyadapter/smoke_seed42/traced/anyadapter-opentrack-smoke-jit.pt",
  "max_abs_error": 0.0,
  "contains_history_encoder": true,
  "contains_layerwise_actor": true,
  "contains_dynamics_world_model": false,
  "contains_critic": false,
  "contains_motion_wm": false,
  "detected_observation_dim": 7001,
  "simulator_message": "[AnyAdapter-OpenTrack] Detected 7001-D layerwise AnyAdapter policy"
}
```
