"""Latest-pose rendering and neutral leg-scale calibration for live GMR."""
import multiprocessing as mp
from queue import Empty, Full
import time
import numpy as np


def _viewer_worker(robot_type, frequency, poses, stop):
    from general_motion_retargeting import RobotMotionViewer
    viewer = None
    try:
        viewer = RobotMotionViewer(robot_type=robot_type, motion_fps=frequency)
        period = 1.0 / frequency
        last = None
        while not stop.is_set():
            start = time.monotonic()
            try:
                last = poses.get(timeout=period)
            except Empty:
                pass
            while True:
                try:
                    last = poses.get_nowait()
                except Empty:
                    break
            if last is not None:
                viewer.step(root_pos=last[:3], root_rot=last[3:7],
                            dof_pos=last[7:], rate_limit=False)
            stop.wait(max(0.0, period - (time.monotonic() - start)))
    finally:
        if viewer is not None:
            viewer.close()


class LatestPoseViewer:
    """Rendering cannot block IK/publication; old poses are dropped."""
    def __init__(self, robot_type, frequency=30.0):
        context = mp.get_context('spawn')
        self.poses = context.Queue(maxsize=1)
        self.stop = context.Event()
        self.process = context.Process(target=_viewer_worker,
            args=(robot_type, frequency, self.poses, self.stop), daemon=True)
        self.process.start()
        self._warned = False

    def step(self, root_pos, root_rot, dof_pos, rate_limit=False):
        if not self.process.is_alive():
            if not self._warned:
                print('[Viewer] renderer stopped; mocap/Redis output continues')
                self._warned = True
            return
        pose = np.concatenate((root_pos, root_rot, dof_pos)).copy()
        try:
            self.poses.put_nowait(pose)
        except Full:
            try:
                self.poses.get_nowait()
            except Empty:
                return
            try:
                self.poses.put_nowait(pose)
            except Full:
                pass

    def close(self):
        self.stop.set()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        self.poses.cancel_join_thread()
        self.poses.close()


def neutral_leg_scales(frames, robot_spans, scale_table):
    """Fit each standing pelvis-to-toe span; reject bent or lifted legs.

    Only the scale changes. Joint angles are never zeroed/subtracted, so
    subsequent knee bends and crouches remain in the input.
    """
    samples = {'Left': [], 'Right': []}
    for frame in frames:
        if not all(name in frame for name in ('Hips', 'LeftUpLeg', 'LeftLeg',
                                              'LeftToeBase', 'RightUpLeg',
                                              'RightLeg', 'RightToeBase')):
            continue
        pelvis = np.asarray(frame['Hips'][0], dtype=float)
        feet = [np.asarray(frame[side+'ToeBase'][0], dtype=float) for side in samples]
        if abs(feet[0][2] - feet[1][2]) > .10:
            continue
        values = {}
        for side in samples:
            hip, knee, toe = [np.asarray(frame[side+name][0], dtype=float)
                              for name in ('UpLeg', 'Leg', 'ToeBase')]
            a, b = hip-knee, toe-knee
            denominator = np.linalg.norm(a)*np.linalg.norm(b)
            if denominator < 1e-8:
                break
            bend = np.pi - np.arccos(np.clip(np.dot(a,b)/denominator, -1, 1))
            span = float((pelvis[2]-toe[2])*scale_table[side+'ToeBase'])
            if not np.isfinite(span) or span < .4 or bend > np.deg2rad(35):
                break
            values[side] = span
        if len(values) == 2:
            for side, span in values.items(): samples[side].append(span)
    if min(map(len, samples.values())) < 8:
        raise ValueError('need at least 8 valid neutral standing frames')
    factors = {side: float(robot_spans[side]/np.median(spans))
               for side, spans in samples.items()}
    if any(not .75 <= factor <= 1.35 for factor in factors.values()):
        raise ValueError(f'implausible standing leg-scale correction: {factors}')
    if abs(factors['Left']-factors['Right']) > .08:
        raise ValueError(f'asymmetric standing calibration: {factors}')
    return factors
