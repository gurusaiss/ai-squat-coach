"""Renders the annotated/overlay video: skeleton landmarks, bar-path marker,
rep numbering and bottom-position markers, plus live back-angle/depth read-
outs. Tracked (observed) points are drawn solid; the approximated bar
position is drawn with a dashed/hollow marker and a text tag, so a viewer
can visually distinguish observed evidence from estimates, per the
assignment's instruction to "distinguish observed points from estimates."
"""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np

from .measurements import MIN_VIS
from .pose import FrameLandmarks, angle_deg
from .reps import Rep

COLOR_TRACKED = (60, 220, 60)      # green BGR
COLOR_BAR_EST = (0, 200, 255)      # orange BGR
COLOR_BOTTOM = (60, 60, 255)       # red BGR
COLOR_TEXT_BG = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)


def _pt(fl: FrameLandmarks, side: str, name: str, w: int, h: int):
    p = fl.points.get(f"{side}_{name}")
    if not p or p[2] < MIN_VIS:
        return None
    return (int(p[0] * w), int(p[1] * h))


def render_annotated_video(
    input_path: str,
    output_path_raw: str,
    output_path_web: str,
    frames: list[FrameLandmarks],
    bar_x_series: list,
    reps: list[Rep],
    side: str,
    fps: float,
    width: int,
    height: int,
):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path_raw, fourcc, fps, (width, height))

    bottom_frames = {r.bottom_idx: r.index for r in reps}
    rep_ranges = [(r.start_idx, r.end_idx, r.index) for r in reps]
    rep_starts = {r.start_idx for r in reps}
    current_trail: list[tuple[int, int]] = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= len(frames):
            # Beyond the analyzed window (e.g. video truncated at the demo
            # duration cap) -- stop here rather than appending unannotated
            # tail footage that carries no findings.
            break

        fl = frames[idx]
        overlay = frame.copy()

        shoulder = _pt(fl, side, "shoulder", width, height)
        hip = _pt(fl, side, "hip", width, height)
        knee = _pt(fl, side, "knee", width, height)
        ankle = _pt(fl, side, "ankle", width, height)
        toe = _pt(fl, side, "foot_index", width, height)
        ear = _pt(fl, side, "ear", width, height)

        bones = [(shoulder, hip), (hip, knee), (knee, ankle), (ankle, toe), (ear, shoulder)]
        for a, b in bones:
            if a and b:
                cv2.line(overlay, a, b, COLOR_TRACKED, 3, cv2.LINE_AA)
        for p in (shoulder, hip, knee, ankle, toe, ear):
            if p:
                cv2.circle(overlay, p, 5, COLOR_TRACKED, -1, cv2.LINE_AA)

        # bar path trail: reset at the start of each rep, accumulate the
        # estimated bar position frame-by-frame, and draw it as a
        # translucent polyline -- this is the "track the barbell and show
        # its path" requirement, not just a per-frame dot.
        if idx in rep_starts:
            current_trail = []

        # bar estimate marker (hollow ring to signal "estimated", not observed)
        be = bar_x_series[idx] if idx < len(bar_x_series) else None
        if be and be.x_norm is not None and shoulder:
            bx = int(be.x_norm * width)
            by = shoulder[1]
            current_trail.append((bx, by))

            if len(current_trail) >= 2:
                trail_overlay = overlay.copy()
                pts = np.array(current_trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(trail_overlay, [pts], False, COLOR_BAR_EST, 2, cv2.LINE_AA)
                cv2.addWeighted(trail_overlay, 0.55, overlay, 0.45, 0, overlay)

            cv2.circle(overlay, (bx, by), 14, COLOR_BAR_EST, 3, cv2.LINE_AA)
            cv2.putText(overlay, "bar (est.)", (bx + 18, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BAR_EST, 1, cv2.LINE_AA)
            if ankle:
                # dashed vertical reference line through the midfoot proxy
                for y in range(0, height, 14):
                    cv2.line(overlay, (ankle[0], y), (ankle[0], min(y + 7, height)), (120, 120, 120), 1, cv2.LINE_AA)

        # depth reference line at knee height
        if knee:
            cv2.line(overlay, (0, knee[1]), (width, knee[1]), (255, 180, 0), 1, cv2.LINE_AA)

        # HUD text
        lines = []
        if idx in bottom_frames:
            lines.append(f"REP {bottom_frames[idx]} - BOTTOM")
        else:
            for s, e, rnum in rep_ranges:
                if s <= idx <= e:
                    lines.append(f"Rep {rnum}")
                    break
        if shoulder and hip:
            lines.append(f"back angle: {angle_deg(fl.points[f'{side}_hip'], fl.points[f'{side}_shoulder']):.0f} deg")
        if hip and knee:
            lines.append("depth: at/below knee" if fl.points[f"{side}_hip"][1] >= fl.points[f"{side}_knee"][1] else "depth: above knee")

        y0 = 28
        for i, text in enumerate(lines):
            y = y0 + i * 26
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (10, y - th - 6), (16 + tw, y + 6), COLOR_TEXT_BG, -1)
            cv2.putText(overlay, text, (13, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)

        if idx in bottom_frames:
            cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), COLOR_BOTTOM, 6)

        writer.write(overlay)
        idx += 1

    cap.release()
    writer.release()

    # mux to a browser-friendly H.264 mp4 (mp4v from OpenCV is not reliably
    # playable in <video> tags across browsers)
    cmd = [
        "ffmpeg", "-y", "-i", output_path_raw,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path_web,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    try:
        os.remove(output_path_raw)
    except OSError:
        pass
