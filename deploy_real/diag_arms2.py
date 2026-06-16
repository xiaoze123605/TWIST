"""Diagnostic: check mocap arm lengths vs robot arm lengths."""
import sys, threading, time
import numpy as np, mujoco

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack

client = setup_optitrack('192.168.3.103', '192.168.3.137', True)
t = threading.Thread(target=client.run, daemon=True); t.start()
time.sleep(3)

# Test different height scaling
for actual_h in [1.6, 1.8, None]:
    gmr = GMR(src_human='fbx', tgt_robot='unitree_g1', actual_human_height=actual_h, damping=0.5)

    for attempt in range(3):
        while True:
            frame = client.get_frame(timeout=3.0)
            if frame and 'Hips' in frame: break
            time.sleep(0.1)

        # Measure raw mocap arm length: shoulder to hand distance
        if 'RightArm' in frame and 'RightHand' in frame:
            r_shoulder = np.asarray(frame['RightArm'][0])
            r_hand = np.asarray(frame['RightHand'][0])
            mocap_arm_len = np.linalg.norm(r_hand - r_shoulder)
        else:
            mocap_arm_len = 0

        # Measure robot arm length from FK
        qpos = np.asarray(gmr.retarget(frame), dtype=float)

        # After retarget, scaled human data is available
        if hasattr(gmr, 'scaled_human_data') and 'RightArm' in gmr.scaled_human_data and 'RightHand' in gmr.scaled_human_data:
            scaled_shld = gmr.scaled_human_data['RightArm'][0]
            scaled_hand = gmr.scaled_human_data['RightHand'][0]
            scaled_dist = np.linalg.norm(scaled_hand - scaled_shld)
        else:
            scaled_dist = 0

        # Robot FK: shoulder to hand
        d = gmr.configuration.data
        try:
            r_shld_id = mujoco.mj_name2id(gmr.model, mujoco.mjtObj.mjOBJ_BODY, 'right_shoulder_yaw_link')
            r_hand_id = mujoco.mj_name2id(gmr.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')
            robot_arm_len = np.linalg.norm(d.xpos[r_hand_id] - d.xpos[r_shld_id])
        except:
            robot_arm_len = 0

        ra = qpos[29:36]
        print(f'h={actual_h} F{attempt}: mocap_arm={mocap_arm_len:.3f}m scaled_dist={scaled_dist:.3f}m robot_arm={robot_arm_len:.3f}m elbow={ra[3]:+.3f}')

client.shutdown()
