"""Diagnostic: force straight elbow and see if IK bends it back."""
import sys, threading, time
import numpy as np, mujoco

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack

client = setup_optitrack('192.168.3.103', '192.168.3.137', True)
t = threading.Thread(target=client.run, daemon=True); t.start()
time.sleep(3)

gmr = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=1.8, damping=0.5)

# Get one frame
while True:
    frame = client.get_frame(timeout=3.0)
    if frame and 'Hips' in frame: break
    time.sleep(0.1)

# Run retarget with DEFAULT config (elbow=0 in qpos0)
q0 = gmr.configuration.data.qpos
print(f'Default qpos0 arm: L_elbow={q0[25]:.2f}, R_elbow={q0[32]:.2f}')
print(f'Default qpos0 arm: L_shld_roll={q0[23]:.2f}, L_shld_yaw={q0[24]:.2f}')
print(f'Default qpos0 arm: R_shld_roll={q0[30]:.2f}, R_shld_yaw={q0[31]:.2f}')

qpos = np.asarray(gmr.retarget(frame), dtype=float)
print(f'After retarget:  L_elbow={qpos[25]:+.4f}, R_elbow={qpos[32]:+.4f}')
print(f'After retarget:  L_shld_roll={qpos[23]:+.4f}, L_shld_yaw={qpos[24]:+.4f}')
print(f'After retarget:  R_shld_roll={qpos[30]:+.4f}, R_shld_yaw={qpos[31]:+.4f}')

# Now try: force elbow=0, shoulder_roll and yaw to 0, and retarget
print('\n--- Forcing arms fully straight (elbow=0, roll=0, yaw=0) ---')
q = gmr.configuration.data.qpos
q[22:29] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # left: fully straight
q[29:36] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # right: fully straight
gmr.configuration.data.qpos = q
mujoco.mj_forward(gmr.model, gmr.configuration.data)

qpos2 = np.asarray(gmr.retarget(frame), dtype=float)
print(f'After forced-straight retarget:')
print(f'  L_elbow={qpos2[25]:+.4f}, R_elbow={qpos2[32]:+.4f}')
print(f'  L_shld_roll={qpos2[23]:+.4f}, L_shld_yaw={qpos2[24]:+.4f}')
print(f'  R_shld_roll={qpos2[30]:+.4f}, R_shld_yaw={qpos2[31]:+.4f}')

# Check IK errors
print(f'\nIK error1: {gmr.error1():.4f}')
print(f'IK error2: {gmr.error2():.4f}')

# Compare: what does the scaled human data look like?
if hasattr(gmr, 'scaled_human_data'):
    sd = gmr.scaled_human_data
    for name in ['RightArm', 'RightForeArm', 'RightHand']:
        if name in sd:
            print(f'Scaled {name}: pos={sd[name][0]}')

client.shutdown()
