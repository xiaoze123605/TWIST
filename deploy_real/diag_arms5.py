"""Diagnostic: zero out arm rotation costs to prevent elbow bending."""
import sys, threading, time, copy
import numpy as np

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack

client = setup_optitrack('192.168.3.103', '192.168.3.137', True)
t = threading.Thread(target=client.run, daemon=True); t.start()
time.sleep(3)

# Get 5 frames
frames = []
for _ in range(5):
    while True:
        f = client.get_frame(timeout=3.0)
        if f and 'Hips' in f: break
        time.sleep(0.1)
    frames.append({k: [np.array(v[0]), np.array(v[1])] for k, v in f.items()})

# Test: zero out rotation weights for elbow/wrist tasks
gmr = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=1.6, damping=0.5)

# Find elbow/wrist tasks in table1 and table2, set their orientation_cost to 0
arm_task_frames = ['left_elbow_link', 'left_wrist_yaw_link',
                   'right_elbow_link', 'right_wrist_yaw_link']

for task in gmr.tasks1:
    if hasattr(task, 'frame_name') and task.frame_name in arm_task_frames:
        task.orientation_cost = 0.0  # disable rotation constraint
        print(f'Disabled rotation on table1/{task.frame_name}')

for task in gmr.tasks2:
    if hasattr(task, 'frame_name') and task.frame_name in arm_task_frames:
        task.orientation_cost = 0.0
        print(f'Disabled rotation on table2/{task.frame_name}')

print()
for i, frame in enumerate(frames):
    qpos = np.asarray(gmr.retarget(frame), dtype=float)
    print(f'F{i}: L_elbow={qpos[25]:+.4f} R_elbow={qpos[32]:+.4f} '
          f'L_shld_roll={qpos[23]:+.4f} R_shld_roll={qpos[30]:+.4f}')

client.shutdown()
