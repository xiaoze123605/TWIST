# DynamicsTracker cumulative DAgger nominal report

## Scope and acceptance gate

This run only transfers the legacy TWIST teacher's stable-control behavior into the existing DynamicsTracker actor. The network, History Encoder and Dynamics-WM structure were not changed. No gate, residual, FiLM, Motion-WM, PPO fine-tuning, domain randomization or real-robot work was added.

Training and evaluation use only `accad/B3___walk1.pkl`. Collection randomizes the initial phase across the clip; validation always uses phase 0, seed 123, 8 environments and 400 control steps. The acceptance gate is at least 7/8 full-motion completions.

## Cumulative replay implementation

Replay is persisted in one versioned dataset and appended atomically. Each sample stores actor input, teacher sent target, motion id/time/phase, episode id, source policy, steps-to-failure, root-height failure margin, seed, action mode, target-scale version and collection round. Every checkpoint records the dataset file/content SHA256, sample count and round.

Sampling uses source quotas and phase-balanced weights. From round 2 onward the realized source sampling ratio is 40% teacher stable states, 30% historical student states and 30% latest student states. States 0.5–1.0 seconds before a physical failure receive 2x weight inside their source/phase stratum. Episode control is sampled independently at each reset using the round beta; the teacher labels every state regardless of which policy controls it.

## Stage-A0 rounds

| Round | beta | cumulative samples | dataset source counts T/O/C | sampled T/O/C | held-out target MAE/RMSE/p95 (rad) | completion | duration (s) | joint RMSE (rad) | height RMSE (m) | failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.00 | 25,600 | 25,600/0/0 | 1.00/0/0 | .0280/.0456/.0883 | 0/8 | 1.115 | .1558 | .0721 | height 56 |
| 1 | .70 | 51,200 | 48,499/0/2,701 | .571/0/.429 | .0285/.0477/.0928 | 0/8 | .855 | .1914 | .0649 | height 72 |
| 2 | .50 | 76,800 | 69,384/2,701/4,715 | .40/.30/.30 | .0291/.0502/.0954 | 0/8 | .880 | .2265 | .0656 | height 72, tilt 3 |
| 3 | .25 | 102,400 | 82,316/7,416/12,668 | .40/.30/.30 | .0314/.0560/.1073 | 0/8 | 1.433 | .1783 | .0550 | height 34, tilt 6 |
| 4 | 0 | 128,000 | 82,316/20,084/25,600 | .40/.30/.30 | .0298/.0537/.1018 | 0/8 | 1.733 | .1513 | .0612 | height 32 |
| 5 | 0 | 153,600 | 82,316/45,684/25,600 | .40/.30/.30 | .0241/.0469/.0842 | **8/8** | **7.620** | **.1253** | **.0243** | **none** |

Round 1 and round 2 regressed, but previous checkpoints and replay remained intact. The fixed validation rule selected round 5 lexicographically by completion rate, duration, root-height RMSE and joint RMSE. This is the first cumulative run to reach the nominal gate. Its torque-saturation fraction is 0, no NaN/Inf occurred, and scripted/runtime action parity passed.

The Stage-A held-out imitation gap remained small while early closed-loop policies still failed, supporting covariate shift rather than simple train-set overfitting. On the final mixed dataset, approximate neighboring observations have teacher-target RMSE .092 rad and p95 absolute disagreement .182 rad. Teacher, historical-student and latest-student validation RMSEs are .0301, .0559 and .0231 rad respectively. Historical recovery states remain the hardest part of the dataset.

## Stage B latent ablation

The round-5 actor was combined with the already-trained wide Dynamics-WM/History Encoder. Actor parameters were unchanged exactly, and the first-layer latent columns were zero-initialized. Round 6 then appended 25,600 all-student samples with normal causal latent and trained the actor once on the full 179,200-sample replay.

| Latent mode | completion | duration (s) | joint RMSE (rad) | height RMSE (m) | torque saturation | failures | scripted parity |
|---|---:|---:|---:|---:|---:|---|---|
| normal | 2/8 | 3.550 | .1495 | .0425 | .00038 | height 11, tilt 1 | pass |
| zero | **8/8** | **7.620** | **.1305** | **.0253** | 0 | none | pass |
| shuffled | 3/8 | 4.235 | .1393 | .0322 | .00008 | height 5 | not applicable |

Normal and shuffled latent both underperform zero latent. The latent therefore does not yet carry control-useful information that the actor can safely exploit, even though its world model was trained. Stage B is rejected and is not selected as best. No PPO preservation experiment is started because the accepted nominal controller is the zero-latent Stage-A model and enabling latent fails the preservation requirement.

## Outcome

Catastrophic replay replacement has been removed: sample count grows monotonically from 25,600 to 179,200 and older student rows are relabeled as historical rather than overwritten. The nominal acceptance gate is reached by Stage-A round 5 at 8/8 completions. The selected artifact is `dagger_b3_cumulative/best.pt`, which is a copy of round 5; `best_manifest.json` contains every candidate and the exact selection key.

Work stops here as required. The result has not entered multi-motion training, PPO, robust domain randomization or real-robot deployment.
