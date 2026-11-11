"""Per-rep geometric measurements, grounded in the criteria defined in
skill/squat_skill.yaml. This module produces raw numbers + confidence/
visibility flags; it does NOT decide meets/does-not-meet/cannot-assess --
that verdict logic lives with the skill (applied by assess.py), keeping the
"document drives the assessment" separation the assignment asks for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import FrameLandmarks, angle_deg, dist
from .reps import Rep
from .skill_loader import criteria_by_id

MIN_VIS = 0.5


def _pick_side(frames: list[FrameLandmarks]) -> str:
    left_score = 0.0
    right_score = 0.0
    keys_l = ["left_shoulder", "left_hip", "left_knee", "left_ankle"]
    keys_r = ["right_shoulder", "right_hip", "right_knee", "right_ankle"]
    for fl in frames:
        for k in keys_l:
            p = fl.points.get(k)
            if p:
                left_score += p[2]
        for k in keys_r:
            p = fl.points.get(k)
            if p:
                right_score += p[2]
    return "left" if left_score >= right_score else "right"


def _facing_sign(frames: list[FrameLandmarks], side: str) -> int:
    """+1 if the toe is at a greater x than the ankle (facing right), else -1."""
    diffs = []
    for fl in frames:
        ankle = fl.points.get(f"{side}_ankle")
        toe = fl.points.get(f"{side}_foot_index")
        if ankle and toe and ankle[2] >= MIN_VIS and toe[2] >= MIN_VIS:
            diffs.append(toe[0] - ankle[0])
    if not diffs:
        return 1
    return 1 if np.median(diffs) >= 0 else -1


@dataclass
class RepMeasurement:
    rep_index: int
    start_t: float
    bottom_t: float
    end_t: float
    bottom_frame: int

    depth_ok: bool | None = None
    depth_margin_norm: float | None = None  # (hip_y - knee_y) / thigh_len ; >=0 meets
    depth_confidence_note: str = ""

    back_angle_bottom_deg: float | None = None
    back_angle_confidence_note: str = ""

    knee_forward_offset_norm: float | None = None  # knee forward of ankle, / shank_len
    knee_confidence_note: str = ""

    bar_max_dev_norm: float | None = None  # max |bar_x - ankle_x| / foot_len across rep
    bar_source_quality: str = ""  # summary of edge_refined vs wrist_midpoint usage
    bar_confidence_note: str = ""

    back_angle_at_30pct_ascent: float | None = None
    back_angle_change_frac: float | None = None  # fraction of (standing-bottom) angle closed by 30% hip-rise
    hip_drive_confidence_note: str = ""
    bar_forward_drift_norm: float | None = None

    head_tilt_proxy_deg: float | None = None
    head_confidence_note: str = "low-confidence proxy only; see skill note"


def _thigh_len(frames, side, idx):
    fl = frames[idx]
    hip, knee = fl.points.get(f"{side}_hip"), fl.points.get(f"{side}_knee")
    if hip and knee:
        return dist(hip, knee)
    return None


def _shank_len(frames, side, idx):
    fl = frames[idx]
    knee, ankle = fl.points.get(f"{side}_knee"), fl.points.get(f"{side}_ankle")
    if knee and ankle:
        return dist(knee, ankle)
    return None


def _foot_len(frames, side, idx):
    fl = frames[idx]
    heel, toe = fl.points.get(f"{side}_heel"), fl.points.get(f"{side}_foot_index")
    if heel and toe:
        d = dist(heel, toe)
        if d > 1e-4:
            return d
    return None


def compute_rep_measurements(
    frames: list[FrameLandmarks],
    bar_x_series: list,  # list[BarEstimate] aligned to frames
    reps: list[Rep],
    fps: float,
) -> tuple[list[RepMeasurement], str]:
    side = _pick_side(frames)
    facing = _facing_sign(frames, side)

    out: list[RepMeasurement] = []
    for rep in reps:
        b = rep.bottom_idx
        fl_b = frames[b]
        hip = fl_b.points.get(f"{side}_hip")
        knee = fl_b.points.get(f"{side}_knee")
        ankle = fl_b.points.get(f"{side}_ankle")
        shoulder = fl_b.points.get(f"{side}_shoulder")
        toe = fl_b.points.get(f"{side}_foot_index")

        rm = RepMeasurement(
            rep_index=rep.index,
            start_t=frames[rep.start_idx].t_sec,
            bottom_t=fl_b.t_sec,
            end_t=frames[rep.end_idx].t_sec,
            bottom_frame=b,
        )

        # ---- depth ----
        thigh_len = _thigh_len(frames, side, rep.start_idx) or _thigh_len(frames, side, b)
        if hip and knee and hip[2] >= MIN_VIS and knee[2] >= MIN_VIS and thigh_len:
            margin = (hip[1] - knee[1]) / thigh_len  # y grows downward
            rm.depth_margin_norm = round(float(margin), 4)
            tolerance = criteria_by_id()["depth"]["measurement"]["jitter_tolerance_normalized"]
            rm.depth_ok = margin >= tolerance
        else:
            rm.depth_confidence_note = "hip/knee landmark confidence too low at bottom frame, or thigh scale unavailable"

        # ---- back angle at bottom ----
        if hip and shoulder and hip[2] >= MIN_VIS and shoulder[2] >= MIN_VIS:
            rm.back_angle_bottom_deg = round(angle_deg(hip, shoulder), 1)
        else:
            rm.back_angle_confidence_note = "hip/shoulder landmark confidence too low at bottom frame"

        # ---- knee forward offset ----
        shank_len = _shank_len(frames, side, b)
        if knee and ankle and knee[2] >= MIN_VIS and ankle[2] >= MIN_VIS and shank_len:
            offset = (knee[0] - ankle[0]) * facing / shank_len
            rm.knee_forward_offset_norm = round(float(offset), 4)
        else:
            rm.knee_confidence_note = "knee/ankle landmark confidence too low at bottom frame"

        # ---- bar path over the whole rep ----
        foot_len = _foot_len(frames, side, rep.start_idx) or shank_len
        devs = []
        sources = []
        for i in range(rep.start_idx, rep.end_idx + 1):
            be = bar_x_series[i]
            fl_i = frames[i]
            ank_i = fl_i.points.get(f"{side}_ankle")
            if be and be.x_norm is not None and ank_i and ank_i[2] >= MIN_VIS and be.confidence >= 0.4 and foot_len:
                devs.append(abs(be.x_norm - ank_i[0]) / foot_len)
                sources.append(be.source)
        if devs:
            rm.bar_max_dev_norm = round(float(max(devs)), 4)
            n_edge = sum(1 for s in sources if s == "edge_refined")
            rm.bar_source_quality = f"{n_edge}/{len(sources)} frames edge-refined, rest wrist-midpoint proxy"
        else:
            rm.bar_confidence_note = "bar/ankle not reliably tracked across this rep"

        # ---- hip drive / ascent consistency ----
        if rm.back_angle_bottom_deg is not None:
            standing_angle = _back_angle_at(frames, side, rep.start_idx)
            hip_y_bottom = hip[1] if hip else None
            hip_y_standing = frames[rep.start_idx].points.get(f"{side}_hip")
            hip_y_standing = hip_y_standing[1] if hip_y_standing else None
            if standing_angle is not None and hip_y_bottom is not None and hip_y_standing is not None:
                checkpoint_frac = criteria_by_id()["hip_drive_ascent"]["measurement"]["calibration_ascent_checkpoint_fraction"]
                target_hip_y = hip_y_bottom - checkpoint_frac * (hip_y_bottom - hip_y_standing)
                checkpoint_idx = _find_checkpoint(frames, side, b, rep.end_idx, target_hip_y)
                if checkpoint_idx is not None:
                    angle_cp = _back_angle_at(frames, side, checkpoint_idx)
                    if angle_cp is not None:
                        rm.back_angle_at_30pct_ascent = round(angle_cp, 1)
                        total_change = standing_angle - rm.back_angle_bottom_deg
                        if abs(total_change) > 3:
                            frac = (angle_cp - rm.back_angle_bottom_deg) / total_change
                            rm.back_angle_change_frac = round(float(frac), 3)
                    bar_bottom = bar_x_series[b]
                    bar_cp = bar_x_series[checkpoint_idx]
                    # Pure bar horizontal displacement between bottom and the
                    # early-ascent checkpoint (the ankle/midfoot reference
                    # cancels out algebraically since it barely moves while
                    # the foot stays planted, so we compute it directly).
                    if bar_bottom and bar_cp and foot_len and bar_bottom.x_norm is not None and bar_cp.x_norm is not None:
                        drift = (bar_cp.x_norm - bar_bottom.x_norm) * facing / foot_len
                        rm.bar_forward_drift_norm = round(float(drift), 4)
                else:
                    rm.hip_drive_confidence_note = "ascent too short to sample a 30%-hip-rise checkpoint"
            else:
                rm.hip_drive_confidence_note = "standing reference angle or hip trajectory unavailable"
        else:
            rm.hip_drive_confidence_note = "back angle at bottom unavailable"

        # ---- head/neck tilt proxy (low confidence by design) ----
        ear = fl_b.points.get(f"{side}_ear")
        if ear and shoulder and hip and ear[2] >= MIN_VIS and shoulder[2] >= MIN_VIS and hip[2] >= MIN_VIS:
            torso_angle = angle_deg(hip, shoulder)
            head_angle = angle_deg(shoulder, ear)
            rm.head_tilt_proxy_deg = round(head_angle - torso_angle, 1)

        out.append(rm)

    return out, side


def _back_angle_at(frames, side, idx):
    fl = frames[idx]
    hip, shoulder = fl.points.get(f"{side}_hip"), fl.points.get(f"{side}_shoulder")
    if hip and shoulder and hip[2] >= MIN_VIS and shoulder[2] >= MIN_VIS:
        return angle_deg(hip, shoulder)
    return None


def _find_checkpoint(frames, side, bottom_idx, end_idx, target_hip_y):
    best_idx, best_diff = None, None
    for i in range(bottom_idx, end_idx + 1):
        hip = frames[i].points.get(f"{side}_hip")
        if not hip or hip[2] < MIN_VIS:
            continue
        diff = abs(hip[1] - target_hip_y)
        if best_diff is None or diff < best_diff:
            best_diff, best_idx = diff, i
    return best_idx
