#!/usr/bin/env python3
"""
Enhanced real-robot low-level policy controller (v2).

Features over the original:
  Step  2: CSV deployment logging
  Step  3: Redis stale-check with timestamp/frame_id + hold-last-valid
  Step  4: Action ramp on startup
  Step  5: Target dof_pos rate-of-change limiting
  Step  6: Risk detection (risk_score)
  Step  7: Hierarchical control modes: TRACKING / RECOVERY / SAFE_STAND
  Step  8: Recovery behaviour (reduce amplitude, smooth return to default)
  Step 10: Test modes: --stand-test, --replay-mimic, --dry-run, --recovery-test

Usage:
  # Normal operation
  python server_low_level_g1_real_v2.py --policy_path <path> --config_path <path>

  # Stand test (no GMR needed)
  python server_low_level_g1_real_v2.py --stand-test --policy_path <path>

  # Dry run (no motor commands)
  python server_low_level_g1_real_v2.py --dry-run --policy_path <path>

  # Replay recorded mimic
  python server_low_level_g1_real_v2.py --replay-mimic <file.npy> --policy_path <path>

  # Recovery test
  python server_low_level_g1_real_v2.py --recovery-test --policy_path <path>
"""

import argparse, csv, json, os, time, traceback
from collections import deque
from datetime import datetime

import numpy as np
import torch
import redis
from tqdm import tqdm

from robot_control.common.remote_controller import KeyMap
from robot_control.g1_wrapper import G1RealWorldEnv
from robot_control.config import Config
from data_utils.rot_utils import quatToEuler

# ═══════════════════════════════════════════════════════════════════════════════
# Configurable constants (tune these for your robot)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Action ramp ──
RAMP_TIME = 5.0                # seconds to ramp from 0 → full action

# ── Rate limiting ──
MAX_DELTA_PER_STEP = 0.15      # max radians change per 0.02s step (≈ 7.5 rad/s)

# ── Redis stale ──
REDIS_STALE_THRESHOLD = 0.10   # seconds — if older, consider stale
REDIS_MAX_MISS_FRAMES = 5      # consecutive misses before recovery

# ── Risk detection thresholds ──
RISK_ROLL_THRESHOLD = 0.6      # rad (~34 deg)
RISK_PITCH_THRESHOLD = 0.6     # rad
RISK_ROLL_RATE_THRESHOLD = 6.0 # rad/s
RISK_PITCH_RATE_THRESHOLD = 6.0
RISK_TRACKING_ERROR = 0.5      # rad RMS
RISK_TORQUE_RATIO = 0.85       # fraction of torque limit
RISK_LOOP_DT_RATIO = 1.5       # control_dt multiplier

# ── Control mode hysteresis ──
RISK_HIGH = 0.6                # risk_score ≥ this → RECOVERY
RISK_LOW = 0.2                 # risk_score ≤ this → TRACKING
RISK_STABLE_FRAMES = 15        # frames below RISK_LOW before switching back
RECOVERY_MIN_DURATION = 0.3    # seconds minimum recovery

# ── Recovery ──
RECOVERY_ACTION_SCALE = 0.1    # reduced action during recovery
RECOVERY_RAMP_TIME = 1.0       # seconds to blend target toward default


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_mimic_obs_to_body_and_wrist(mimic_obs):
    """Extract 31-dof policy input and 2-dof wrist from 33-dof mimic_obs."""
    wrist_ids = [27, 32]
    other_ids = [f for f in range(33) if f not in wrist_ids]
    return mimic_obs[other_ids], mimic_obs[wrist_ids]


def compute_risk_score(roll, pitch, ang_vel, dof_pos, target_dof_pos,
                       tau_est, torque_limits, redis_age, loop_dt, control_dt):
    """Compute a 0-1 risk score; higher = more dangerous."""
    risk_items = {}

    # Roll / pitch
    r = abs(roll) / RISK_ROLL_THRESHOLD
    p = abs(pitch) / RISK_PITCH_THRESHOLD
    risk_items['roll'] = min(r, 1.0)
    risk_items['pitch'] = min(p, 1.0)

    # Angular velocity
    risk_items['roll_rate'] = min(abs(ang_vel[0]) / RISK_ROLL_RATE_THRESHOLD, 1.0)
    risk_items['pitch_rate'] = min(abs(ang_vel[1]) / RISK_PITCH_RATE_THRESHOLD, 1.0)

    # Joint tracking error
    tracking_err = np.sqrt(np.mean((target_dof_pos - dof_pos) ** 2))
    risk_items['tracking'] = min(tracking_err / RISK_TRACKING_ERROR, 1.0)

    # Torque ratio
    if torque_limits is not None and len(tau_est) > 0:
        n = min(len(tau_est), len(torque_limits))
        ratio = np.max(np.abs(tau_est[:n]) / (torque_limits[:n] + 1e-6))
        risk_items['torque'] = min(ratio / RISK_TORQUE_RATIO, 1.0)
    else:
        risk_items['torque'] = 0.0

    # Redis age
    risk_items['redis'] = min(redis_age / REDIS_STALE_THRESHOLD, 1.0) if redis_age else 0.0

    # Loop timing
    risk_items['loop_dt'] = min(loop_dt / (control_dt * RISK_LOOP_DT_RATIO), 1.0)

    # Combined score (weighted)
    weights = {'roll': 2.0, 'pitch': 2.0, 'roll_rate': 1.5, 'pitch_rate': 1.5,
               'tracking': 1.0, 'torque': 1.5, 'redis': 0.5, 'loop_dt': 0.5}
    total_w = sum(weights.values())
    score = sum(weights[k] * risk_items.get(k, 0) for k in weights) / total_w

    return float(score), risk_items


# ═══════════════════════════════════════════════════════════════════════════════
# Main controller
# ═══════════════════════════════════════════════════════════════════════════════

class RealTimePolicyControllerV2:
    def __init__(self, policy_path, config_path, device='cuda', net='eno1',
                 log_path=None, stand_test=False, dry_run=False,
                 replay_mimic=None, recovery_test=False):
        # ── Modes ──
        self.stand_test = stand_test
        self.dry_run = dry_run
        self.replay_mimic_path = replay_mimic
        self.recovery_test = recovery_test

        # ── Redis ──
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            print(f"[WARN] Redis unavailable: {e}")

        # ── Robot env ──
        self.config = Config(config_path)
        if not dry_run:
            self.env = G1RealWorldEnv(net=net, config=self.config)
        else:
            self.env = None

        # ── Policy ──
        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        print(f"Policy loaded from {policy_path}")

        # ── Parameters (match sim exactly) ──
        self.num_actions = 23
        self.default_dof_pos = np.concatenate(
            [self.config.default_angles, self.config.arm_waist_target], axis=0
        ).astype(np.float32)

        self.ang_vel_scale = self.config.ang_vel_scale
        self.dof_vel_scale = self.config.dof_vel_scale
        self.dof_pos_scale = self.config.dof_pos_scale
        self.action_scale = self.config.action_scale
        self.ankle_idx = [4, 5, 10, 11]
        self.control_dt = self.config.control_dt

        # ── Torque limits (for risk detection) ──
        self.torque_limits = np.array([
            88, 139, 88, 139, 50, 50,
            88, 139, 88, 139, 50, 50,
            88, 50, 50,
            25, 25, 25, 25,
            25, 25, 25, 25,
        ], dtype=np.float32)

        # ── Observation history ──
        self.n_mimic_obs = 8 + self.num_actions  # 31
        self.n_proprio = self.n_mimic_obs + 3 + 2 + 3 * self.num_actions  # 105
        self.history_len = 10
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio, dtype=np.float32))

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # ── State ──
        self.last_target_dof_pos = self.default_dof_pos.copy()
        self.last_valid_mimic = None
        self.last_valid_mimic_full = None
        self.frame_id = 0
        self.mode = "SAFE_STAND"      # TRACKING / RECOVERY / SAFE_STAND
        self.mode_entry_time = 0.0
        self.stable_frames = 0
        self.redis_miss_count = 0
        self.ramp = 0.0
        self.start_time = None

        # ── Logging ──
        self.log_path = log_path
        self._csv_writer = None
        self._csv_file = None
        if log_path and not dry_run:
            self._csv_file = open(log_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                'timestamp', 'frame_id', 'loop_dt', 'redis_age', 'mode',
                'roll', 'pitch', 'yaw',
                'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                'dof_pos_0', 'dof_vel_0', 'target_0',
                'raw_action_0', 'last_action_0', 'tau_est_0',
                'joint_tracking_error', 'risk_score', 'ramp',
            ])

        # ── Replay data ──
        self._replay_data = None
        self._replay_idx = 0
        if replay_mimic:
            self._replay_data = np.load(replay_mimic)
            print(f"Loaded replay mimic: {self._replay_data.shape}")

    # ── Robot helpers ────────────────────────────────────────────────────────

    def _reset_robot(self):
        if self.dry_run:
            print("[DRY-RUN] Skipping robot reset.")
            return
        print("Zero torque. Press START to move to default position ...")
        self.env.zero_torque_state()
        print("Press START to go to default pos ...")
        self.env.move_to_default_pos()
        print("Press A to continue ...")
        self.env.default_pos_state()
        print("Ready.")

    def _get_dof_pos(self):
        """Return 23-dof robot state (or zeros for dry-run)."""
        if self.dry_run:
            return (np.zeros(self.num_actions, dtype=np.float32),
                    np.zeros(self.num_actions, dtype=np.float32),
                    np.array([0,0,0,1], dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(self.num_actions, dtype=np.float32))
        dof_pos, dof_vel, quat, ang_vel = self.env.get_robot_state()
        tau_est = getattr(self.env, 'tauj', np.zeros(self.num_actions, dtype=np.float32))
        return dof_pos, dof_vel, quat, ang_vel, tau_est

    def _send_action(self, target_dof_pos, wrist_dof_pos):
        if self.dry_run:
            return
        self.env.send_robot_action(
            target_dof_pos, kp_scale=1.0, kd_scale=1.0,
            left_wrist_roll=float(wrist_dof_pos[0]),
            right_wrist_roll=float(wrist_dof_pos[1]),
        )

    # ── Redis helpers ────────────────────────────────────────────────────────

    def _get_mimic_from_redis(self):
        """Read mimic_obs from Redis. Returns (mimic_obs, redis_age, frame_id).

        On failure, returns (None, inf, -1).
        """
        if self._replay_data is not None:
            # Replay mode
            if self._replay_idx >= len(self._replay_data):
                return None, float('inf'), -1
            mimic = self._replay_data[self._replay_idx].astype(np.float32)
            self._replay_idx += 1
            return mimic, 0.0, self._replay_idx

        if self.redis_client is None:
            return None, float('inf'), -1

        try:
            raw = self.redis_client.get("action_mimic_g1")
            if raw is None:
                return None, float('inf'), -1

            # Try new JSON format {timestamp, frame_id, action_mimic}
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and 'action_mimic' in data:
                    redis_ts = data.get('timestamp', 0)
                    redis_fid = data.get('frame_id', -1)
                    mimic = np.array(data['action_mimic'], dtype=np.float32)
                else:
                    # Legacy format: plain list
                    redis_ts = 0
                    redis_fid = -1
                    mimic = np.array(data, dtype=np.float32)
            except (json.JSONDecodeError, TypeError):
                return None, float('inf'), -1

            age = time.time() - redis_ts if redis_ts > 0 else 0.0
            return mimic, age, redis_fid
        except Exception as e:
            print(f"[WARN] Redis read error: {e}")
            return None, float('inf'), -1

    # ── Safe stand ───────────────────────────────────────────────────────────

    def _enter_safe_stand(self):
        if self.mode != "SAFE_STAND":
            print(f"[MODE] Switching to SAFE_STAND (was {self.mode})")
        self.mode = "SAFE_STAND"
        self.mode_entry_time = time.time()
        self.stable_frames = 0

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, **kwargs):
        if self._csv_writer is None:
            return
        row = [
            f"{kwargs.get('timestamp', time.time()):.6f}",
            str(kwargs.get('frame_id', self.frame_id)),
            f"{kwargs.get('loop_dt', 0):.6f}",
            f"{kwargs.get('redis_age', 0):.6f}",
            str(kwargs.get('mode', self.mode)),
            f"{kwargs.get('roll', 0):.6f}",
            f"{kwargs.get('pitch', 0):.6f}",
            f"{kwargs.get('yaw', 0):.6f}",
            f"{kwargs.get('ang_vel_x', 0):.6f}",
            f"{kwargs.get('ang_vel_y', 0):.6f}",
            f"{kwargs.get('ang_vel_z', 0):.6f}",
            f"{kwargs.get('dof_pos_0', 0):.6f}",
            f"{kwargs.get('dof_vel_0', 0):.6f}",
            f"{kwargs.get('target_0', 0):.6f}",
            f"{kwargs.get('raw_action_0', 0):.6f}",
            f"{kwargs.get('last_action_0', 0):.6f}",
            f"{kwargs.get('tau_est_0', 0):.6f}",
            f"{kwargs.get('joint_tracking_error', 0):.6f}",
            f"{kwargs.get('risk_score', 0):.4f}",
            f"{kwargs.get('ramp', self.ramp):.4f}",
        ]
        self._csv_writer.writerow(row)

    # ═══════════════════════════════════════════════════════════════════════════
    # Main loop
    # ═══════════════════════════════════════════════════════════════════════════

    def run(self):
        if not self.dry_run and not self.stand_test:
            self._reset_robot()

        print(f"[MODE] Starting in SAFE_STAND")
        print(f"  Stand test: {self.stand_test}")
        print(f"  Dry run: {self.dry_run}")
        print(f"  Replay: {self.replay_mimic_path is not None}")
        print(f"  Recovery test: {self.recovery_test}")
        print(f"  Log: {self.log_path or 'disabled'}")

        self.start_time = time.time()
        self._enter_safe_stand()

        # If stand-test: send default pose and exit
        if self.stand_test:
            print("[STAND-TEST] Holding default pose for 10 seconds ...")
            t0 = time.time()
            while time.time() - t0 < 10.0:
                if not self.dry_run:
                    self.env.send_robot_action(
                        self.default_dof_pos, kp_scale=1.0, kd_scale=1.0)
                time.sleep(self.control_dt)
            print("[STAND-TEST] Done.")
            self._cleanup()
            return

        try:
            while True:
                t_start = time.time()

                # ── Check remote ──
                if (not self.dry_run
                        and self.env.remote_controller.button[KeyMap.select] == 1):
                    print("Select pressed. Exiting.")
                    break

                # ── Get robot state ──
                dof_pos, dof_vel, quat, ang_vel, tau_est = self._get_dof_pos()
                rpy = quatToEuler(quat)

                # ── Get mimic_obs from Redis ──
                mimic_raw, redis_age, redis_fid = self._get_mimic_from_redis()

                # ── Handle missing/stale Redis ──
                if mimic_raw is None:
                    self.redis_miss_count += 1
                    if self.redis_miss_count > REDIS_MAX_MISS_FRAMES:
                        print(f"[WARN] Redis missing {self.redis_miss_count} frames → SAFE_STAND")
                        self._enter_safe_stand()
                    # Hold last valid mimic
                    mimic_raw = (self.last_valid_mimic_full
                                 if self.last_valid_mimic_full is not None
                                 else np.zeros(33, dtype=np.float32))
                else:
                    self.redis_miss_count = 0
                    if redis_age > REDIS_STALE_THRESHOLD:
                        print(f"[WARN] Redis stale: {redis_age:.3f}s")
                        if self.last_valid_mimic_full is not None:
                            mimic_raw = self.last_valid_mimic_full
                    else:
                        self.last_valid_mimic_full = mimic_raw.copy()

                action_mimic, wrist_dof_pos = extract_mimic_obs_to_body_and_wrist(mimic_raw)
                self.last_valid_mimic = action_mimic.copy()

                # ── Recovery test: artificially trigger recovery ──
                if self.recovery_test and self.frame_id > 500 and self.frame_id < 700:
                    redis_age = 999.0  # Force stale

                # ── Build observation ──
                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[self.ankle_idx] = 0.0

                obs_proprio = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action,
                ])

                # Publish proprio to Redis
                if self.redis_client:
                    self.redis_client.set("state_body_g1", json.dumps(obs_proprio.tolist()))

                obs_full = np.concatenate([action_mimic, obs_proprio])
                obs_hist = np.array(self.proprio_history_buf).flatten()
                obs_buf = np.concatenate([obs_full, obs_hist])
                self.proprio_history_buf.append(obs_full)

                # ── Policy inference ──
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                raw_action = np.clip(raw_action, -10.0, 10.0)
                self.last_action = raw_action.copy()

                # ── Action ramp ──
                elapsed_total = time.time() - self.start_time
                self.ramp = min(1.0, elapsed_total / RAMP_TIME)

                # ── Compute target with ramp ──
                target_dof_pos = (self.default_dof_pos
                                  + self.ramp * raw_action * self.action_scale)

                # ── Rate limiting ──
                dq = target_dof_pos - self.last_target_dof_pos
                dq = np.clip(dq, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP)
                target_dof_pos = self.last_target_dof_pos + dq

                # ── Risk detection ──
                loop_dt = time.time() - t_start
                risk_score, risk_items = compute_risk_score(
                    rpy[0], rpy[1], ang_vel, dof_pos, target_dof_pos,
                    tau_est, self.torque_limits, redis_age, loop_dt,
                    self.control_dt,
                )

                # ── Control mode state machine ──
                if risk_score >= RISK_HIGH:
                    if self.mode != "RECOVERY":
                        print(f"[MODE] TRACKING → RECOVERY (risk={risk_score:.3f})")
                    self.mode = "RECOVERY"
                    self.mode_entry_time = time.time()
                    self.stable_frames = 0
                elif risk_score <= RISK_LOW:
                    self.stable_frames += 1
                    if (self.stable_frames >= RISK_STABLE_FRAMES
                            and self.mode == "RECOVERY"
                            and time.time() - self.mode_entry_time > RECOVERY_MIN_DURATION):
                        print(f"[MODE] RECOVERY → TRACKING (stable {self.stable_frames} frames)")
                        self.mode = "TRACKING"
                        self.stable_frames = 0
                else:
                    self.stable_frames = 0

                if self.mode == "SAFE_STAND":
                    # Force ramp to 0, blend toward default pose
                    recovery_alpha = min(1.0, (time.time() - self.mode_entry_time) / RECOVERY_RAMP_TIME)
                    target_dof_pos = (self.last_target_dof_pos * (1 - recovery_alpha)
                                      + self.default_dof_pos * recovery_alpha)
                    self.ramp = max(0.0, self.ramp - 0.01)

                elif self.mode == "RECOVERY":
                    # Reduce amplitude, gradual return toward default
                    recovery_alpha = min(1.0, (time.time() - self.mode_entry_time) / RECOVERY_RAMP_TIME)
                    safe_target = (self.last_target_dof_pos * (1 - recovery_alpha)
                                   + self.default_dof_pos * recovery_alpha)
                    # Blend reduced-policy target with safe target
                    blend = min(1.0, recovery_alpha * 2)  # faster blend during recovery
                    target_dof_pos = target_dof_pos * (1 - blend) + safe_target * blend
                    target_dof_pos = (target_dof_pos * RECOVERY_ACTION_SCALE
                                      + self.default_dof_pos * (1 - RECOVERY_ACTION_SCALE))

                # ── Send action ──
                self._send_action(target_dof_pos, wrist_dof_pos)
                self.last_target_dof_pos = target_dof_pos.copy()

                # ── Logging ──
                if self.frame_id % 5 == 0:  # log every 5th frame to keep file size manageable
                    self._log(
                        timestamp=time.time(),
                        loop_dt=loop_dt,
                        redis_age=redis_age,
                        roll=float(rpy[0]),
                        pitch=float(rpy[1]),
                        yaw=float(rpy[2]),
                        ang_vel_x=float(ang_vel[0]),
                        ang_vel_y=float(ang_vel[1]),
                        ang_vel_z=float(ang_vel[2]),
                        dof_pos_0=float(dof_pos[0]),
                        dof_vel_0=float(dof_vel[0]),
                        target_0=float(target_dof_pos[0]),
                        raw_action_0=float(raw_action[0]),
                        last_action_0=float(self.last_action[0]),
                        tau_est_0=float(tau_est[0]) if len(tau_est) > 0 else 0.0,
                        joint_tracking_error=float(
                            np.sqrt(np.mean((target_dof_pos - dof_pos) ** 2))),
                        risk_score=risk_score,
                    )

                # ── Rate limiting ──
                self.frame_id += 1
                elapsed = time.time() - t_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)

        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt.")
        except Exception as e:
            print(f"[ERROR] Main loop: {e}")
            traceback.print_exc()
        finally:
            self._cleanup()

    def _cleanup(self):
        if self._csv_file:
            self._csv_file.close()
            print(f"[LOG] Saved to {self.log_path}")
        if not self.dry_run and self.env is not None:
            print("[INFO] Entering zero torque state ...")
            self.env.zero_torque_state()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Enhanced real-robot policy controller (v2)")
    HERE = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--policy_path", default="../assets/twist_general_motion_tracker.pt")
    parser.add_argument("--config_path", default=os.path.join(HERE, "robot_control/configs/g1.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--net", default="eno1")
    parser.add_argument("--log_dir", default=os.path.join(HERE, "logs"),
                        help="Directory for CSV deployment logs")

    # Test modes
    parser.add_argument("--stand-test", action="store_true",
                        help="Stand only, no GMR, just hold default pose")
    parser.add_argument("--replay-mimic", default=None,
                        help="Path to .npy file with recorded mimic_obs for replay")
    parser.add_argument("--dry-run", action="store_true",
                        help="No motor commands, just logging")
    parser.add_argument("--recovery-test", action="store_true",
                        help="Artificially trigger recovery mid-run")

    args = parser.parse_args()

    # Setup logging
    log_path = None
    if args.log_dir and not args.stand_test:
        os.makedirs(args.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"real_deploy_{ts}.csv")

    controller = RealTimePolicyControllerV2(
        policy_path=args.policy_path,
        config_path=args.config_path,
        device=args.device,
        net=args.net,
        log_path=log_path,
        stand_test=args.stand_test,
        dry_run=args.dry_run,
        replay_mimic=args.replay_mimic,
        recovery_test=args.recovery_test,
    )
    controller.run()


if __name__ == "__main__":
    main()
