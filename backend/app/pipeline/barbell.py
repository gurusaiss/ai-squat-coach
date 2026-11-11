"""Lightweight barbell x-position approximation.

LIMITATION (disclosed in skill/squat_skill.yaml and BUILD_NOTES.md): this is
NOT a trained object detector for barbell plates. Within a 4-6 hour build
timebox we use a transparent, debuggable heuristic instead of pretending to
have verified plate-tracking:

  1. Primary signal: the midpoint of the two wrist landmarks from MediaPipe
     Pose. In a back-squat grip the hands stay on the bar, so wrist x
     tracks bar x closely (not bar y -- the bar is on the back, not the
     hands, so we only trust the HORIZONTAL component, which is what the
     bar-path criterion needs).
  2. Refinement: within a horizontal band around the shoulder line (where
     the bar actually rests in a low-bar squat) we look for the strongest
     long horizontal edge (via Canny + HoughLinesP) near the wrist-implied
     x position, and snap to it when found with high confidence. This
     catches cases where the plate silhouette is visually clearer than the
     hand position.

Every value returned carries a confidence/source tag so the assessment
layer can downgrade or withhold a verdict instead of trusting a shaky
estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BarEstimate:
    x_norm: float | None  # normalized [0,1] horizontal position in frame
    source: str  # "wrist_midpoint" | "edge_refined" | "none"
    confidence: float  # 0..1


def estimate_bar_x(frame_bgr, fl, width: int, height: int) -> BarEstimate:
    lw = fl.get("left_wrist")
    rw = fl.get("right_wrist")
    ls = fl.get("left_shoulder")
    rs = fl.get("right_shoulder")

    wrist_candidates = [p for p in (lw, rw) if p and p[2] >= 0.3]
    if not wrist_candidates:
        return BarEstimate(None, "none", 0.0)

    x_wrist = sum(p[0] for p in wrist_candidates) / len(wrist_candidates)
    conf_wrist = sum(p[2] for p in wrist_candidates) / len(wrist_candidates)

    shoulders = [p for p in (ls, rs) if p and p[2] >= 0.3]
    if not shoulders or frame_bgr is None:
        return BarEstimate(x_wrist, "wrist_midpoint", min(conf_wrist, 0.6))

    y_shoulder = sum(p[1] for p in shoulders) / len(shoulders)

    refined = _refine_with_edges(frame_bgr, x_wrist, y_shoulder, width, height)
    if refined is not None:
        return BarEstimate(refined, "edge_refined", min(conf_wrist + 0.15, 0.85))
    return BarEstimate(x_wrist, "wrist_midpoint", min(conf_wrist, 0.6))


def _refine_with_edges(frame_bgr, x_wrist_norm: float, y_shoulder_norm: float, width: int, height: int):
    try:
        cy = int(y_shoulder_norm * height)
        band_half = max(10, int(0.05 * height))
        y0, y1 = max(0, cy - band_half), min(height, cy + band_half)
        if y1 - y0 < 5:
            return None
        band = frame_bgr[y0:y1, :]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=40,
            minLineLength=int(0.08 * width), maxLineGap=10,
        )
        if lines is None:
            return None

        x_wrist_px = x_wrist_norm * width
        best = None
        best_score = -1.0
        for line in lines:
            x1, ly1, x2, ly2 = line[0]
            if abs(ly1 - ly2) > 6:  # keep near-horizontal lines only (the bar, not limbs)
                continue
            length = abs(x2 - x1)
            mid_x = (x1 + x2) / 2.0
            dist_from_wrist = abs(mid_x - x_wrist_px)
            if dist_from_wrist > 0.25 * width:
                continue
            score = length - dist_from_wrist * 0.5
            if score > best_score:
                best_score = score
                best = mid_x
        if best is None:
            return None
        return best / width
    except Exception:
        return None
