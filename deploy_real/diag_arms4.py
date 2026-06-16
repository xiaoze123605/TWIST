"""Diagnostic: test auto_calibrate and different rotation offsets."""
import sys, threading, time
import numpy as np, mujoco

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting.utils import auto_calibrate_human_scale
import copy

client = setup_optitrack('192.168.3.103', '192.168.3.137', True)
t = threading.Thread(target=client.run, daemon=True); t.start()
time.sleep(3)

# Get one frame
while True:
    frame = client.get_frame(timeout=3.0)
    if frame and 'Hips' in frame: break
    time.sleep(0.1)

# Make a copy for testing different GMR configs
raw_frame = {k: [np.array(v[0]), np.array(v[1])] for k, v in frame.items()}

# Test 1: default
gmr1 = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=1.6, damping=0.5)
q1 = gmr1.retarget(copy.deepcopy(raw_frame))
print(f'Test 1 (default, h=1.6): L_elbow={q1[25]:+.4f} R_elbow={q1[32]:+.4f}')

# Test 2: with auto_calibrate
gmr2 = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=1.6, damping=0.5)
auto_calibrate_human_scale(gmr2, copy.deepcopy(raw_frame))
q2 = gmr2.retarget(copy.deepcopy(raw_frame))
print(f'Test 2 (auto_calibrate, h=1.6): L_elbow={q2[25]:+.4f} R_elbow={q2[32]:+.4f}')

# Test 3: no height scaling
gmr3 = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=None, damping=0.5)
q3 = gmr3.retarget(copy.deepcopy(raw_frame))
print(f'Test 3 (no scaling): L_elbow={q3[25]:+.4f} R_elbow={q3[32]:+.4f}')

# Test 4: disable table1 (rotation-only tasks), only table2 (position tasks)
gmr4 = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=1.6, damping=0.5)
gmr4.use_ik_match_table1 = False
q4 = gmr4.retarget(copy.deepcopy(raw_frame))
print(f'Test 4 (table2 only): L_elbow={q4[25]:+.4f} R_elbow={q4[32]:+.4f}')

# Test 5: high damping to stay near starting pose
gmr5 = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=1.6, damping=2.0)
q5 = gmr5.retarget(copy.deepcopy(raw_frame))
print(f'Test 5 (damping=2.0): L_elbow={q5[25]:+.4f} R_elbow={q5[32]:+.4f}')

# Test 6: FORCE straight elbow in config and retarget
q0 = gmr1.configuration.data.qpos
q0[25] = 0.0  # left elbow
q0[32] = 0.0  # right elbow
gmr1.configuration.data.qpos = q0
mujoco.mj_forward(gmr1.model, gmr1.configuration.data)
q6 = gmr1.retarget(copy.deepcopy(raw_frame))
# Do it again - maybe it converges better
q6 = gmr1.retarget(copy.deepcopy(raw_frame))
print(f'Test 6 (force elbow=0 x2): L_elbow={q6[25]:+.4f} R_elbow={q6[32]:+.4f}')

# Print mocap arm rotations for comparison
print(f'\nRaw mocap rotations:')
for name in ['RightArm', 'RightForeArm', 'RightHand']:
    if name in raw_frame:
        print(f'  {name}: {raw_frame[name][1]}')

client.shutdown()
