"""Pose landmark extraction using MediaPipe Pose (legacy Solutions API).

Runs fully offline -- the pose model ships inside the mediapipe wheel, no
network calls happen at inference time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# Landmark indices we actually use (MediaPipe BlazePose 33-point topology).
LM = mp_pose.PoseLandmark


@dataclass
class FrameLandmarks:
    frame_idx: int
    t_sec: float
    present: bool
    points: dict = field(default_factory=dict)  # name -> (x, y, visibility), normalized [0,1]

    def get(self, name: str):
        return self.points.get(name)

    def visible(self, name: str, min_vis: float = 0.5) -> bool:
        p = self.points.get(name)
        return bool(p) and p[2] >= min_vis


NAMES_OF_INTEREST = {
    "nose": LM.NOSE,
    "left_ear": LM.LEFT_EAR,
    "right_ear": LM.RIGHT_EAR,
    "left_shoulder": LM.LEFT_SHOULDER,
    "right_shoulder": LM.RIGHT_SHOULDER,
    "left_elbow": LM.LEFT_ELBOW,
    "right_elbow": LM.RIGHT_ELBOW,
    "left_wrist": LM.LEFT_WRIST,
    "right_wrist": LM.RIGHT_WRIST,
    "left_hip": LM.LEFT_HIP,
    "right_hip": LM.RIGHT_HIP,
    "left_knee": LM.LEFT_KNEE,
    "right_knee": LM.RIGHT_KNEE,
    "left_ankle": LM.LEFT_ANKLE,
    "right_ankle": LM.RIGHT_ANKLE,
    "left_heel": LM.LEFT_HEEL,
    "right_heel": LM.RIGHT_HEEL,
    "left_foot_index": LM.LEFT_FOOT_INDEX,
    "right_foot_index": LM.RIGHT_FOOT_INDEX,
}


def extract_landmarks(video_path: str, max_frames: int | None = None, progress_cb=None):
    """Run MediaPipe Pose over every frame of the video.

    Returns (frames: list[FrameLandmarks], fps: float, width: int, height: int, frame_count: int)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    frames: list[FrameLandmarks] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames is not None and idx >= max_frames:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)

            t_sec = idx / fps
            fl = FrameLandmarks(frame_idx=idx, t_sec=t_sec, present=False)
            if result.pose_landmarks:
                fl.present = True
                lm = result.pose_landmarks.landmark
                for name, enum_val in NAMES_OF_INTEREST.items():
                    p = lm[enum_val]
                    fl.points[name] = (p.x, p.y, p.visibility)
            frames.append(fl)
            idx += 1
            if progress_cb and idx % 10 == 0:
                progress_cb(idx, total)

    cap.release()
    return frames, fps, width, height, (total or len(frames))


def midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, min(a[2], b[2]))


def side_preference(fl: FrameLandmarks, min_vis: float = 0.5) -> str:
    """Pick which body side (left/right) is more visible/confident in this frame."""
    left_keys = ["left_shoulder", "left_hip", "left_knee", "left_ankle"]
    right_keys = ["right_shoulder", "right_hip", "right_knee", "right_ankle"]
    left_score = sum(fl.points[k][2] for k in left_keys if k in fl.points)
    right_score = sum(fl.points[k][2] for k in right_keys if k in fl.points)
    return "left" if left_score >= right_score else "right"


def angle_deg(p1, p2) -> float:
    """Angle (degrees) of the line p1->p2 relative to the horizontal, in image coords
    (y grows downward). 0 = horizontal, 90 = vertical."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dy, dx))) if (dx != 0 or dy != 0) else 0.0


def dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
