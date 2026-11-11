"""End-to-end pipeline orchestration for a single uploaded video."""
from __future__ import annotations

from dataclasses import dataclass

import cv2

from .annotate import render_annotated_video
from .assess import generate_writeups
from .barbell import estimate_bar_x
from .measurements import compute_rep_measurements
from .pose import extract_landmarks
from .reps import detect_reps
from .verdicts import evaluate_rep, fmt_time
from .skill_loader import not_assessable_list

MAX_DURATION_SEC = 90
SUPPORTED_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".avi"}


@dataclass
class PipelineResult:
    rep_count: int
    side_used: str
    fps: float
    width: int
    height: int
    findings: list  # list of dict ready for API/JSON
    overall_summary: str
    confidence_caveats: str
    writeup_method: str
    not_assessable: list
    warnings: list


def probe_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file -- unsupported or corrupt format.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    duration = frame_count / fps if fps else 0
    return fps, frame_count, duration


def run_pipeline(input_path: str, output_video_web: str, progress_cb=None) -> PipelineResult:
    warnings = []

    fps, frame_count, duration = probe_video(input_path)
    if duration and duration > MAX_DURATION_SEC:
        warnings.append(
            f"Video is {duration:.0f}s, longer than the {MAX_DURATION_SEC}s demo limit; "
            f"only the first {MAX_DURATION_SEC}s were processed."
        )
    max_frames = int(MAX_DURATION_SEC * fps) if fps else None

    def _progress(done, total):
        if progress_cb:
            pct = min(95, int(10 + 60 * (done / max(1, total or done))))
            progress_cb(pct, f"Tracking pose ({done}/{total or '?'} frames)")

    if progress_cb:
        progress_cb(5, "Reading video and extracting pose landmarks")
    frames, fps, width, height, total = extract_landmarks(input_path, max_frames=max_frames, progress_cb=_progress)

    present_frac = sum(1 for f in frames if f.present) / max(1, len(frames))
    if present_frac < 0.5:
        warnings.append(
            f"A full-body pose was only detected in {present_frac*100:.0f}% of frames. "
            "Check that the lifter is fully in frame and the camera is a side-on view."
        )

    if progress_cb:
        progress_cb(68, "Estimating barbell position")
    cap = cv2.VideoCapture(input_path)
    bar_series = []
    for i, fl in enumerate(frames):
        ok, frame_bgr = cap.read()
        if not ok:
            bar_series.append(estimate_bar_x(None, fl, width, height))
            continue
        bar_series.append(estimate_bar_x(frame_bgr, fl, width, height))
    cap.release()

    if progress_cb:
        progress_cb(75, "Detecting repetitions")
    hip_y = []
    present_mask = []
    for fl in frames:
        p = fl.points.get("left_hip") or fl.points.get("right_hip")
        hip_y.append(p[1] if p else None)
        present_mask.append(fl.present)
    reps = detect_reps(hip_y, present_mask, fps)

    if not reps:
        raise ValueError(
            "No repetitions could be detected. This usually means the lifter "
            "was not fully visible/trackable, the clip doesn't contain a full "
            "squat rep, or the camera is not a side view."
        )

    if progress_cb:
        progress_cb(82, "Computing measurements against the reference document")
    rep_measurements, side = compute_rep_measurements(frames, bar_series, reps, fps)

    all_findings = []
    for rm in rep_measurements:
        all_findings.extend(evaluate_rep(rm))

    if progress_cb:
        progress_cb(88, "Generating grounded explanations (LLM agent)")
    writeups, method = generate_writeups(all_findings)
    writeup_by_key = {w["key"]: w for w in writeups.get("findings", [])}

    findings_out = []
    for f in all_findings:
        key = f"{f.criterion_id}|{f.rep_index}"
        wu = writeup_by_key.get(key, {})
        findings_out.append({
            "criterion_id": f.criterion_id,
            "rep_index": f.rep_index,
            "verdict": f.verdict,
            "variant": f.variant,
            "timestamp": fmt_time(f.timestamp_sec),
            "timestamp_sec": round(f.timestamp_sec, 2),
            "frame": f.frame,
            "measurement": f.measurement,
            "confidence_note": f.confidence_note,
            "document_ref_pages": f.ref_pages,
            "document_sections": f.sections,
            "document_quote": f.quote,
            "explanation": wu.get("explanation", ""),
            "feedback": wu.get("feedback", ""),
        })

    if progress_cb:
        progress_cb(93, "Rendering annotated video")
    raw_path = output_video_web.replace(".mp4", "_raw.mp4")
    render_annotated_video(input_path, raw_path, output_video_web, frames, bar_series, reps, side, fps, width, height)

    if progress_cb:
        progress_cb(100, "Done")

    return PipelineResult(
        rep_count=len(reps),
        side_used=side,
        fps=fps,
        width=width,
        height=height,
        findings=findings_out,
        overall_summary=writeups.get("overall_summary", ""),
        confidence_caveats=writeups.get("confidence_caveats", ""),
        writeup_method=method,
        not_assessable=not_assessable_list(),
        warnings=warnings,
    )
