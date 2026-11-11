"""Deterministic verdict engine.

This is the part of the system that actually "applies the document" to the
measurements -- it reads thresholds straight out of skill/squat_skill.yaml
(see that file's header for the document-sourced vs implementation-
calibration distinction) and returns a plain verdict + the evidence behind
it. The LLM, downstream in assess.py, is given these verdicts to turn into
prose -- it does not get to invent or override them. This keeps "does the
conclusion follow the supplied standard" a property of auditable code, not
of a model's generation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .measurements import RepMeasurement
from .skill_loader import criteria_by_id


@dataclass
class Finding:
    criterion_id: str
    rep_index: int
    verdict: str  # "meets" | "does_not_meet" | "cannot_assess"
    variant: str | None  # e.g. "too_vertical" for back_angle_bottom
    timestamp_sec: float
    frame: int
    measurement: dict
    confidence_note: str
    ref_pages: list
    sections: list
    quote: str


def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def evaluate_rep(rm: RepMeasurement) -> list[Finding]:
    crit = criteria_by_id()
    findings: list[Finding] = []

    # ---- depth ----
    c = crit["depth"]
    if rm.depth_margin_norm is None:
        findings.append(Finding(
            "depth", rm.rep_index, "cannot_assess", None, rm.bottom_t, rm.bottom_frame,
            {"depth_margin_norm": None}, rm.depth_confidence_note,
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))
    else:
        verdict = "meets" if rm.depth_ok else "does_not_meet"
        findings.append(Finding(
            "depth", rm.rep_index, verdict, None, rm.bottom_t, rm.bottom_frame,
            {"depth_margin_norm": rm.depth_margin_norm}, "",
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))

    # ---- back angle at bottom ----
    c = crit["back_angle_bottom"]
    lo, hi = c["measurement"]["calibration_band_degrees"]
    if rm.back_angle_bottom_deg is None:
        findings.append(Finding(
            "back_angle_bottom", rm.rep_index, "cannot_assess", None, rm.bottom_t, rm.bottom_frame,
            {"back_angle_deg": None}, rm.back_angle_confidence_note,
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))
    else:
        v = rm.back_angle_bottom_deg
        if lo <= v <= hi:
            verdict, variant = "meets", None
        elif v > hi:
            verdict, variant = "does_not_meet", "too_vertical"
        else:
            verdict, variant = "does_not_meet", "too_horizontal"
        findings.append(Finding(
            "back_angle_bottom", rm.rep_index, verdict, variant, rm.bottom_t, rm.bottom_frame,
            {"back_angle_deg": v}, "",
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))

    # ---- knee travel ----
    c = crit["knee_travel"]
    lo, hi = c["measurement"]["calibration_band_normalized_offset"]
    if rm.knee_forward_offset_norm is None:
        findings.append(Finding(
            "knee_travel", rm.rep_index, "cannot_assess", None, rm.bottom_t, rm.bottom_frame,
            {"knee_forward_offset_norm": None}, rm.knee_confidence_note,
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))
    else:
        v = rm.knee_forward_offset_norm
        verdict = "meets" if lo <= v <= hi else "does_not_meet"
        findings.append(Finding(
            "knee_travel", rm.rep_index, verdict, None, rm.bottom_t, rm.bottom_frame,
            {"knee_forward_offset_norm": v}, "",
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))

    # ---- bar path ----
    c = crit["bar_path"]
    if rm.bar_max_dev_norm is None:
        findings.append(Finding(
            "bar_path", rm.rep_index, "cannot_assess", None, rm.bottom_t, rm.bottom_frame,
            {"bar_max_dev_norm": None}, rm.bar_confidence_note,
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))
    else:
        band_hi = c["measurement"]["calibration_max_normalized_deviation"]
        verdict = "meets" if rm.bar_max_dev_norm <= band_hi else "does_not_meet"
        findings.append(Finding(
            "bar_path", rm.rep_index, verdict, None, rm.bottom_t, rm.bottom_frame,
            {"bar_max_dev_norm": rm.bar_max_dev_norm, "tracking_quality": rm.bar_source_quality}, "",
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))

    # ---- hip drive / ascent consistency ----
    c = crit["hip_drive_ascent"]
    if rm.back_angle_change_frac is None and rm.bar_forward_drift_norm is None:
        findings.append(Finding(
            "hip_drive_ascent", rm.rep_index, "cannot_assess", None, rm.bottom_t, rm.bottom_frame,
            {}, rm.hip_drive_confidence_note,
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))
    else:
        variant = None
        verdict = "meets"
        chest_up_threshold = c["measurement"]["calibration_back_angle_change_frac_threshold"]
        flop_threshold = c["measurement"]["calibration_bar_forward_drift_norm_threshold"]
        if rm.back_angle_change_frac is not None and rm.back_angle_change_frac > chest_up_threshold:
            verdict, variant = "does_not_meet", "chest_up"
        elif rm.bar_forward_drift_norm is not None and rm.bar_forward_drift_norm > flop_threshold:
            verdict, variant = "does_not_meet", "flop_forward"
        findings.append(Finding(
            "hip_drive_ascent", rm.rep_index, verdict, variant, rm.bottom_t, rm.bottom_frame,
            {
                "back_angle_change_frac_at_30pct_hip_rise": rm.back_angle_change_frac,
                "bar_forward_drift_norm": rm.bar_forward_drift_norm,
            }, "",
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))

    # ---- head / eye gaze proxy (always advisory-grade) ----
    c = crit["head_neck_position"]
    if rm.head_tilt_proxy_deg is None:
        findings.append(Finding(
            "head_neck_position", rm.rep_index, "cannot_assess", None, rm.bottom_t, rm.bottom_frame,
            {}, "ear/shoulder/hip landmarks not confidently tracked (common from a distant side view)",
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))
    else:
        max_tilt = c["measurement"]["calibration_max_tilt_deg"]
        verdict = "meets" if rm.head_tilt_proxy_deg <= max_tilt else "does_not_meet"
        findings.append(Finding(
            "head_neck_position", rm.rep_index, verdict, None, rm.bottom_t, rm.bottom_frame,
            {"head_tilt_proxy_deg": rm.head_tilt_proxy_deg}, "low-confidence proxy, not literal eye-gaze tracking",
            c["source"]["ref_pages"], c["source"]["sections"], c["source"]["quote"],
        ))

    return findings
