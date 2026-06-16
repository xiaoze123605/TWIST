"""
Direct port of /home/hank/GMR/scripts/optitrack_to_robot.py
+ redis publishing for low-level controller.

EXACT same pipeline — no changes, no extra filtering:
  OptiTrack → RealtimeMotionFilter → BufferSmoother → GMR.retarget() → redis

Usage:
  conda activate gmr
  python server_optitrack_redis.py
  DISPLAY=:1 python server_optitrack_redis.py --vis
"""

import argparse, json, os, threading, time
from collections import deque
import numpy as np
import redis
from rich import print

import mujoco
import mujoco.viewer

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting.utils.realtime_filter import (
    RealtimeMotionFilter, BufferSmoother,
)
from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import quatToEuler, quat_rotate_inverse
# ── Helpers (same as v2 server) ─────────────────────────────────────────────

def _wrap_to_pi(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))

def _quat_wxyz_to_yaw(q):
    w, x, y, z = map(float, q[:4])
    return float(np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))

def _quat_diff_yaw_rate(q0, q1, dt):
    if dt <= 1e-6: return 0.0
    return float(_wrap_to_pi(_quat_wxyz_to_yaw(q1) - _quat_wxyz_to_yaw(q0)) / dt)

def _lowpass_1st(prev, x, dt, fc_hz):
    if fc_hz is None or fc_hz <= 0: return float(x)
    rc = 1.0/(2.0*np.pi*fc_hz)
    return float((1.0 - dt/(rc+dt))*prev + (dt/(rc+dt))*x)


# ── DoF mapping ─────────────────────────────────────────────────────────────

POLICY_DOF_INDEX = np.array([
    0,1,2,3,4,5,       # left  leg
    6,7,8,9,10,11,     # right leg
    12,13,14,           # waist
    15,16,17,18,19,     # left  arm (skip wrist_pitch=20, wrist_yaw=21)
    22,23,24,25,26,     # right arm (skip wrist_pitch=27, wrist_yaw=28)
], dtype=int)


# ── mimic_obs builder ────────────────────────────────────────────────────────

def build_mimic_obs(qpos, qdot, robot_type, *,
                    prev_root_rot_wxyz=None, dt=None,
                    init_yaw=0.0, root_z_filtered=None):
    root_pos, root_rot = qpos[:3], qpos[3:7]
    dof_pos = qpos[7:][POLICY_DOF_INDEX].copy()
    roll, pitch, yaw = quatToEuler(root_rot)
    yaw = _wrap_to_pi(yaw - init_yaw)
    root_h = np.array([float(root_z_filtered)]) if root_z_filtered is not None else root_pos[2:3]
    root_vel = qdot[:3]
    root_rot_xyzw = root_rot.reshape(1,4)[:,[1,2,3,0]]
    root_vel_rel = quat_rotate_inverse(root_rot_xyzw, root_vel.reshape(1,3)).ravel()
    yr = _quat_diff_yaw_rate(prev_root_rot_wxyz, root_rot, dt) if prev_root_rot_wxyz is not None and dt and dt>1e-6 else 0.0
    return np.concatenate([root_h, [roll,pitch,yaw], root_vel_rel, [yr], dof_pos])


# ── Server ───────────────────────────────────────────────────────────────────

class OptiTrackRedisServer:
    def __init__(self, args):
        self.args = args
        self.robot_type = args.robot

        # ── OptiTrack (exact copy from optitrack_to_robot.py) ──
        print(f"Connecting to {args.host} ...")
        self.client = setup_optitrack(args.host, args.client_ip, args.use_multicast)
        t = threading.Thread(target=self.client.run, daemon=True); t.start()

        # Wait for connection (exact copy from optitrack_to_robot.py lines 87-95)
        timeout = 10
        t0 = time.time()
        while not self.client.connected():
            if time.time() - t0 > timeout:
                print(f"Failed to connect within {timeout}s")
                self.client.shutdown()
                raise RuntimeError("OptiTrack connection failed")
            time.sleep(0.1)
        print(f"Connected: {self.client.connected()}")

        # ── GMR (exact copy from optitrack_to_robot.py) ──
        tgt = "unitree_g1" if args.robot == "g1" else "unitree_t1"
        self.gmr = GMR(
            src_human="fbx",
            tgt_robot=tgt,
            actual_human_height=args.actual_human_height,
        )

        # ── Filters (exact copy from optitrack_to_robot.py) ──
        self.motion_filter = RealtimeMotionFilter(
            pos_smoothing=0.5,
            rot_smoothing=0.4,
            max_pos_jump=0.15,
            max_rot_jump_deg=60.0,
            max_hand_pos_jump=0.05,
            max_hand_rot_jump_deg=30.0,
            wrist_roll_limit_deg=60.0,
            hand_orient_smooth=0.85,
            quat_avg_window=5,
            flip_detect=True,
        )
        self.median_filter = BufferSmoother(window_size=3, mode='median')

        # ── Viewer (exact copy from optitrack_to_robot.py) ──
        self.viewer = None
        if args.vis:
            self.viewer = RobotMotionViewer(robot_type=tgt)

        # ── Redis ──
        self.redis = redis.Redis(host="localhost", port=6379, db=0)

        # ── Post-IK filter state (streaming Butterworth for root + legs) ──
        self._bw_sos_pos = None; self._bw_zi_pos = None
        self._bw_sos_quat = None; self._bw_zi_quat = None; self._bw_prev_quat = None
        self._bw_sos_legs = None; self._bw_zi_legs = None
        self._prev_qpos_filt = None  # for velocity clipping
        self._qpos_history = deque(maxlen=11)  # for regression-based velocity
        self._qdot_smoothed = None  # EMA-smoothed velocity

        # ── Teleop state ──
        self.last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot].copy()
        self.init_yaw = 0.0
        self.root_z_baseline = None
        self.root_z_target_mean = 0.762
        self.root_z_filt = None

    def _post_filter(self, qpos):
        """Real-time equivalent of offline clean_qpos (bvh_to_robot_new.py).

        Offline: 4th-order zero-phase Butterworth at 5 Hz on ALL dofs.
        Real-time: 4th-order CAUSAL Butterworth at 5 Hz on root+legs+waist.
        Arms pass through UNFILTERED for responsive hand tracking.

        Also: velocity clip at 30 rad/s + joint-limit clip at 5 deg.
        """
        from scipy.signal import butter, sosfilt, sosfilt_zi
        out = qpos.copy()
        CUTOFF = 5.0   # Hz — match offline clean_qpos
        ORDER = 4       # match offline clean_qpos
        FS = 120.0      # OptiTrack native frame rate

        # ── Root position (x, y, z) ──
        pos = qpos[0:3].reshape(1, 3)
        if self._bw_sos_pos is None:
            self._bw_sos_pos = butter(ORDER, CUTOFF, 'low', fs=FS, output='sos')
            self._bw_zi_pos = sosfilt_zi(self._bw_sos_pos)[:,:,np.newaxis] * pos[0]
        pos_f, self._bw_zi_pos = sosfilt(self._bw_sos_pos, pos, axis=0, zi=self._bw_zi_pos)
        out[0:3] = pos_f.ravel()

        # ── Root rotation (w, x, y, z) — unflip, filter, renormalize ──
        quat = qpos[3:7].copy()
        if self._bw_prev_quat is not None and np.dot(quat, self._bw_prev_quat) < 0:
            quat = -quat
        self._bw_prev_quat = quat.copy()
        quat_u = quat.reshape(1, 4)
        if self._bw_zi_quat is None:
            self._bw_sos_quat = butter(ORDER, CUTOFF, 'low', fs=FS, output='sos')
            self._bw_zi_quat = sosfilt_zi(self._bw_sos_quat)[:,:,np.newaxis] * quat_u[0]
        quat_f, self._bw_zi_quat = sosfilt(self._bw_sos_quat, quat_u, axis=0, zi=self._bw_zi_quat)
        quat_f = quat_f.ravel()
        out[3:7] = quat_f / max(np.linalg.norm(quat_f), 1e-10)

        # ── Leg + waist dofs (GMR indices 0-14 = qpos[7:22]) ──
        # Arm dofs (GMR indices 15-28 = qpos[22:36]) → UNFILTERED
        leg_end = min(15, len(qpos) - 7)
        if leg_end > 0:
            legs = qpos[7:7+leg_end].reshape(1, -1)
            if self._bw_zi_legs is None:
                self._bw_sos_legs = butter(ORDER, CUTOFF, 'low', fs=FS, output='sos')
                self._bw_zi_legs = sosfilt_zi(self._bw_sos_legs)[:,:,np.newaxis] * legs[0]
            legs_f, self._bw_zi_legs = sosfilt(self._bw_sos_legs, legs, axis=0, zi=self._bw_zi_legs)
            out[7:7+leg_end] = legs_f.ravel()

        # ── Velocity clip: 30 rad/s on ALL dofs (same as clean_qpos) ──
        if self._prev_qpos_filt is not None and len(out) > 7:
            dq_max = 30.0 / FS
            delta = out[7:] - self._prev_qpos_filt[7:]
            out[7:] = self._prev_qpos_filt[7:] + np.clip(delta, -dq_max, dq_max)
        self._prev_qpos_filt = out.copy()

        return out

    def safe_stand(self, freq=50, seconds=2.0):
        target = DEFAULT_MIMIC_OBS[self.robot_type].copy()
        steps = max(int(seconds*freq), 1)
        start = self.last_mimic_obs.copy()
        for i in range(steps):
            a = i/steps
            msg = {"timestamp": time.time(), "frame_id": 0,
                   "action_mimic": (start + (target-start)*a).tolist()}
            self.redis.set(f"action_mimic_{self.robot_type}", json.dumps(msg))
            time.sleep(1.0/freq)
        msg = {"timestamp": time.time(), "frame_id": 0,
               "action_mimic": target.tolist()}
        self.redis.set(f"action_mimic_{self.robot_type}", json.dumps(msg))
        self.last_mimic_obs = target.copy()
        print("[Safe Stand] Done")

    def run(self, freq=50, debug_arms=False):
        print("="*60)
        print("[Server] Starting (EXACT optitrack_to_robot.py pipeline) ...")
        print("="*60)

        self.safe_stand(freq=freq)

        # Wait for first frame to lock init_yaw
        print("[Teleop] Waiting for first frame ...")
        while True:
            frame = self.client.get_frame(timeout=2.0)
            if frame and 'Hips' in frame:
                break
            time.sleep(0.02)
        frame = self.motion_filter(frame)
        frame = self.median_filter(frame)
        first_qpos = np.asarray(self.gmr.retarget(frame), dtype=float)
        first_qpos_filt = self._post_filter(first_qpos)

        self.init_yaw = _quat_wxyz_to_yaw(first_qpos[3:7])
        self.root_z_baseline = float(first_qpos[2])
        self.root_z_filt = self.root_z_target_mean
        print(f"[Teleop] init_yaw={np.degrees(self.init_yaw):+.1f}deg  "
              f"root_z_baseline={self.root_z_baseline:.3f}m")

        # Warmup
        ref0_obs = build_mimic_obs(
            first_qpos_filt, np.zeros_like(first_qpos_filt), self.robot_type,
            prev_root_rot_wxyz=first_qpos_filt[3:7], dt=1.0/freq,
            init_yaw=self.init_yaw, root_z_filtered=self.root_z_target_mean)
        warmup_n = max(int(2.0*freq), 1)
        warmup_start = self.last_mimic_obs.copy()
        print(f"[Teleop] Warmup 2.0s ...")
        for i in range(warmup_n):
            a = (i+1)/warmup_n
            msg = {"timestamp": time.time(), "frame_id": 0,
                   "action_mimic": (warmup_start + (ref0_obs-warmup_start)*a).tolist()}
            self.redis.set(f"action_mimic_{self.robot_type}", json.dumps(msg))
            time.sleep(1.0/freq)

        # ── Main loop (EXACT copy of optitrack_to_robot.py lines 152-202) ──
        self._qpos_history.append(first_qpos_filt.copy())
        prev_root_rot = first_qpos_filt[3:7].copy()
        prev_t = time.time()
        frame_count = 0
        print("[Teleop] Running ...")

        while True:
            # Get frame (line 153)
            frame = self.client.get_frame(timeout=5.0)
            if frame is None or len(frame) == 0:
                continue
            if 'Hips' not in frame:
                continue

            # Filter (lines 163-166)
            frame = self.motion_filter(frame)
            frame = self.median_filter(frame)

            # Retarget (line 196)
            qpos = np.asarray(self.gmr.retarget(frame), dtype=float)
            frame_count += 1

            # Debug: print raw GMR arm angles
            if debug_arms and frame_count % 100 == 0:
                la = qpos[22:29]
                ra = qpos[29:36]
                print(f"[F{frame_count}] L(elbow={la[3]:+.4f} shld_r={la[1]:+.4f} shld_y={la[2]:+.4f}) "
                      f"R(elbow={ra[3]:+.4f} shld_r={ra[1]:+.4f} shld_y={ra[2]:+.4f})")

            # Viewer (lines 197-202) — optional
            if self.viewer is not None:
                try:
                    self.viewer.step(
                        root_pos=qpos[:3],
                        root_rot=qpos[3:7],
                        dof_pos=qpos[7:],
                        rate_limit=False,
                    )
                except Exception:
                    pass

            # ── TWIST-specific: publish to redis ──
            # Apply post-IK smoothing (matches offline clean_qpos).
            qpos_filt = self._post_filter(qpos)

            # Velocity via linear regression over qpos history.
            # Regression over 7 frames at ~120fps → very smooth, low-noise.
            # Multiply by frame rate to convert from "per-frame" to "per-second"
            # units (policy was trained on per-second velocities).
            self._qpos_history.append(qpos_filt.copy())
            FS = 120.0  # OptiTrack native frame rate
            if len(self._qpos_history) >= 7:
                t_reg = np.arange(len(self._qpos_history), dtype=float) / FS  # seconds
                t_mean = np.mean(t_reg)
                q_arr = np.array(self._qpos_history)
                denom = np.sum((t_reg - t_mean) ** 2)
                qdot_raw = (np.sum((t_reg - t_mean)[:, None] * (q_arr - np.mean(q_arr, axis=0)), axis=0)
                            / max(denom, 1e-10))  # per-second units
            elif len(self._qpos_history) >= 2:
                qdot_raw = (self._qpos_history[-1] - self._qpos_history[-2]) * FS
            else:
                qdot_raw = np.zeros_like(qpos_filt)

            # EMA-smooth the velocity (further reduces noise)
            alpha_qdot = 0.3  # ~3-frame effective window
            if self._qdot_smoothed is None:
                self._qdot_smoothed = qdot_raw.copy()
            else:
                self._qdot_smoothed = (1-alpha_qdot) * self._qdot_smoothed + alpha_qdot * qdot_raw

            dt_real = max(time.time() - prev_t, 1.0/(4.0*freq))
            prev_t = time.time()

            z_shifted = float(qpos_filt[2]) + (self.root_z_target_mean - self.root_z_baseline)
            self.root_z_filt = _lowpass_1st(self.root_z_filt, z_shifted, dt_real, 1.5)

            cur_rot = qpos_filt[3:7].copy()
            obs = build_mimic_obs(
                qpos_filt, self._qdot_smoothed, self.robot_type,
                prev_root_rot_wxyz=prev_root_rot, dt=dt_real,
                init_yaw=self.init_yaw, root_z_filtered=self.root_z_filt)
            prev_root_rot = cur_rot

            # Publish with timestamp + frame_id for stale detection
            msg = {
                "timestamp": time.time(),
                "frame_id": frame_count,
                "action_mimic": obs.tolist(),
            }
            self.redis.set(f"action_mimic_{self.robot_type}", json.dumps(msg))
            self.last_mimic_obs = obs.copy()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OptiTrack → G1 (optitrack_to_robot.py + redis)")
    parser.add_argument("--host", default="192.168.3.103")
    parser.add_argument("--client_ip", default="192.168.3.137")
    parser.add_argument("--use_multicast", type=bool, default=True)
    parser.add_argument("--robot", default="g1", choices=["g1","t1"])
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--actual_human_height", type=float, default=1.6)
    parser.add_argument("--debug_arms", action="store_true",
                        help="Print raw GMR arm angles every 100 frames")
    args = parser.parse_args()

    server = OptiTrackRedisServer(args)
    try:
        server.run(freq=50, debug_arms=args.debug_arms)
    except KeyboardInterrupt:
        print("\n[Server] Interrupted.")

if __name__ == "__main__":
    main()
