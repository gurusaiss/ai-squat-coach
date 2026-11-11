"""Unit tests for the deterministic parts of the pipeline (rep detection,
measurement math, verdict logic) using synthetic landmark trajectories.
This does NOT exercise MediaPipe itself (that needs a real video of a real
person); it proves the squat-specific logic we wrote on top of pose output
is correct, independent of tracking quality. Run with: pytest backend/tests
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.pose import FrameLandmarks
from app.pipeline.reps import detect_reps
from app.pipeline.measurements import compute_rep_measurements
from app.pipeline.verdicts import evaluate_rep
from app.pipeline.barbell import BarEstimate


FPS = 30.0


def make_squat_frames(n_reps=3, frames_per_rep=60, back_angle_deg=45.0, depth_ok=True):
    """Synthesize a side-view squat: hip/knee/ankle/shoulder trace a simple
    sinusoidal descent-ascent, with a controllable bottom back angle and
    depth relationship, so we know the "ground truth" verdict in advance."""
    frames = []
    ankle_y = 0.9
    shoulder_x = 0.5
    hip_x = 0.5
    idx = 0
    standing_hip_y = 0.55
    bottom_hip_y = 0.80 if depth_ok else 0.60  # depth_ok -> hip well below knee
    knee_y_const = 0.70  # knee height stays roughly fixed

    for rep in range(n_reps):
        for i in range(frames_per_rep):
            phase = i / frames_per_rep  # 0..1
            # 0->0.5 descent, 0.5->1 ascent
            if phase <= 0.5:
                frac = phase / 0.5
            else:
                frac = 1 - (phase - 0.5) / 0.5
            hip_y = standing_hip_y + frac * (bottom_hip_y - standing_hip_y)

            # back angle varies between standing (~80, near vertical) and bottom (back_angle_deg)
            angle_now = 80 - frac * (80 - back_angle_deg)
            angle_rad = math.radians(angle_now)
            shoulder_y = hip_y - 0.25 * math.sin(angle_rad)
            shoulder_x_now = hip_x - 0.25 * math.cos(angle_rad)

            knee_x = hip_x + 0.02  # slight forward offset
            ankle_x = hip_x

            fl = FrameLandmarks(frame_idx=idx, t_sec=idx / FPS, present=True)
            fl.points = {
                "left_shoulder": (shoulder_x_now, shoulder_y, 0.9),
                "right_shoulder": (shoulder_x_now, shoulder_y, 0.9),
                "left_hip": (hip_x, hip_y, 0.9),
                "right_hip": (hip_x, hip_y, 0.9),
                "left_knee": (knee_x, knee_y_const, 0.9),
                "right_knee": (knee_x, knee_y_const, 0.9),
                "left_ankle": (ankle_x, ankle_y, 0.9),
                "right_ankle": (ankle_x, ankle_y, 0.9),
                "left_foot_index": (ankle_x + 0.05, ankle_y, 0.9),
                "right_foot_index": (ankle_x + 0.05, ankle_y, 0.9),
                "left_heel": (ankle_x - 0.03, ankle_y, 0.9),
                "right_heel": (ankle_x - 0.03, ankle_y, 0.9),
                "left_wrist": (shoulder_x_now, shoulder_y, 0.8),
                "right_wrist": (shoulder_x_now, shoulder_y, 0.8),
                "left_ear": (shoulder_x_now, shoulder_y - 0.1, 0.7),
                "right_ear": (shoulder_x_now, shoulder_y - 0.1, 0.7),
            }
            frames.append(fl)
            idx += 1
    return frames


def test_rep_detection_counts_correctly():
    frames = make_squat_frames(n_reps=4)
    hip_y = [f.points["left_hip"][1] for f in frames]
    present = [True] * len(frames)
    reps = detect_reps(hip_y, present, FPS)
    assert len(reps) == 4, f"expected 4 reps, got {len(reps)}"


def test_depth_meets_when_hip_below_knee():
    frames = make_squat_frames(n_reps=1, depth_ok=True, back_angle_deg=45)
    bar_series = [BarEstimate(f.points["left_wrist"][0], "wrist_midpoint", 0.6) for f in frames]
    hip_y = [f.points["left_hip"][1] for f in frames]
    reps = detect_reps(hip_y, [True] * len(frames), FPS)
    assert len(reps) == 1
    rms, side = compute_rep_measurements(frames, bar_series, reps, FPS)
    assert rms[0].depth_ok is True
    findings = evaluate_rep(rms[0])
    depth_finding = next(f for f in findings if f.criterion_id == "depth")
    assert depth_finding.verdict == "meets"


def test_depth_does_not_meet_when_hip_above_knee():
    frames = make_squat_frames(n_reps=1, depth_ok=False, back_angle_deg=45)
    bar_series = [BarEstimate(f.points["left_wrist"][0], "wrist_midpoint", 0.6) for f in frames]
    hip_y = [f.points["left_hip"][1] for f in frames]
    reps = detect_reps(hip_y, [True] * len(frames), FPS)
    assert len(reps) == 1
    rms, side = compute_rep_measurements(frames, bar_series, reps, FPS)
    assert rms[0].depth_ok is False
    findings = evaluate_rep(rms[0])
    depth_finding = next(f for f in findings if f.criterion_id == "depth")
    assert depth_finding.verdict == "does_not_meet"


def test_back_angle_too_vertical_flagged():
    frames = make_squat_frames(n_reps=1, depth_ok=True, back_angle_deg=75)  # too vertical at bottom
    bar_series = [BarEstimate(f.points["left_wrist"][0], "wrist_midpoint", 0.6) for f in frames]
    hip_y = [f.points["left_hip"][1] for f in frames]
    reps = detect_reps(hip_y, [True] * len(frames), FPS)
    rms, side = compute_rep_measurements(frames, bar_series, reps, FPS)
    findings = evaluate_rep(rms[0])
    back_finding = next(f for f in findings if f.criterion_id == "back_angle_bottom")
    assert back_finding.verdict == "does_not_meet"
    assert back_finding.variant == "too_vertical"


def test_cannot_assess_when_landmarks_missing():
    frames = make_squat_frames(n_reps=1, depth_ok=True)
    for f in frames:
        f.points["left_hip"] = (f.points["left_hip"][0], f.points["left_hip"][1], 0.05)
        f.points["right_hip"] = (f.points["right_hip"][0], f.points["right_hip"][1], 0.05)
    bar_series = [BarEstimate(None, "none", 0.0) for _ in frames]
    hip_y = [f.points["left_hip"][1] for f in frames]
    reps = detect_reps(hip_y, [True] * len(frames), FPS)
    assert len(reps) == 1
    rms, side = compute_rep_measurements(frames, bar_series, reps, FPS)
    findings = evaluate_rep(rms[0])
    depth_finding = next(f for f in findings if f.criterion_id == "depth")
    assert depth_finding.verdict == "cannot_assess"


if __name__ == "__main__":
    test_rep_detection_counts_correctly()
    test_depth_meets_when_hip_below_knee()
    test_depth_does_not_meet_when_hip_above_knee()
    test_back_angle_too_vertical_flagged()
    test_cannot_assess_when_landmarks_missing()
    print("All tests passed.")
