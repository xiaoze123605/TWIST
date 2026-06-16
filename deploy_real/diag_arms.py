"""Diagnostic: print raw GMR qpos arm values directly from OptiTrack."""
import sys, threading, time
import numpy as np, mujoco

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting.utils.realtime_filter import RealtimeMotionFilter, BufferSmoother

client = setup_optitrack('192.168.3.103', '192.168.3.137', True)
t = threading.Thread(target=client.run, daemon=True); t.start()
time.sleep(3)
print(f'Connected: {client.connected()}')

gmr = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=1.6, damping=0.5)
mf = RealtimeMotionFilter(pos_smoothing=0.5, rot_smoothing=0.4,
    max_pos_jump=0.15, max_rot_jump_deg=60.0,
    max_hand_pos_jump=0.05, max_hand_rot_jump_deg=30.0,
    wrist_roll_limit_deg=0.0, hand_orient_smooth=0.5,
    quat_avg_window=0, flip_detect=True)
med = BufferSmoother(window_size=3, mode='median')

print('qpos[22:29] = left arm: shld_pitch, shld_roll, shld_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw')
print('qpos[29:36] = right arm: same order')
print()

for i in range(8):
    while True:
        frame = client.get_frame(timeout=3.0)
        if frame and 'Hips' in frame: break
        time.sleep(0.1)
    frame = mf(frame); frame = med(frame)
    qpos = np.asarray(gmr.retarget(frame), dtype=float)
    la, ra = qpos[22:29], qpos[29:36]
    print(f'F{i}: L(elbow={la[3]:+.4f}) | R(elbow={ra[3]:+.4f}, shld_p={ra[0]:+.4f}, shld_r={ra[1]:+.4f}, shld_y={ra[2]:+.4f})')

# Check elbow limits
for side, jname in [('L', 'left_elbow_joint'), ('R', 'right_elbow_joint')]:
    jid = mujoco.mj_name2id(gmr.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0 and gmr.model.jnt_limited[jid]:
        print(f'{side} elbow limit: {gmr.model.jnt_range[jid]}')

client.shutdown()
