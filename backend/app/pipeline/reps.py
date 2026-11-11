"""Repetition segmentation from a smoothed hip-height trajectory."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Rep:
    index: int
    start_idx: int
    bottom_idx: int
    end_idx: int


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _find_local_maxima(y: np.ndarray, min_prominence: float, min_distance: int):
    """Simple peak finder (no scipy dependency). Returns sorted peak indices."""
    peaks = []
    n = len(y)
    for i in range(1, n - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            # walk to find local window extent for prominence estimate
            left = i
            while left > 0 and y[left - 1] <= y[left]:
                left -= 1
            right = i
            while right < n - 1 and y[right + 1] <= y[right]:
                right += 1
            base = min(y[max(0, left - 1)], y[min(n - 1, right + 1)])
            prominence = y[i] - base
            if prominence >= min_prominence:
                peaks.append(i)
    # enforce min distance between peaks, keeping the higher one
    peaks.sort(key=lambda idx: -y[idx])
    kept = []
    for idx in peaks:
        if all(abs(idx - k) >= min_distance for k in kept):
            kept.append(idx)
    return sorted(kept)


def detect_reps(hip_y: list[float], present_mask: list[bool], fps: float) -> list[Rep]:
    """hip_y: per-frame normalized hip y (image coords, larger = lower/closer to ground).
    Bottom of a rep = local maximum of hip_y (lifter is lowest)."""
    n = len(hip_y)
    if n < int(fps * 0.5):
        return []

    y = np.array([v if v is not None else np.nan for v in hip_y], dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < n * 0.5:
        return []
    # fill small gaps by interpolation so smoothing/peak-finding is stable
    idxs = np.arange(n)
    y_filled = np.interp(idxs, idxs[valid], y[valid])

    window = max(3, int(fps * 0.2))
    smoothed = _moving_average(y_filled, window)

    rng = smoothed.max() - smoothed.min()
    if rng < 1e-4:
        return []
    min_prominence = rng * 0.25
    min_distance = int(fps * 0.8)

    bottoms = _find_local_maxima(smoothed, min_prominence, min_distance)
    if not bottoms:
        return []

    reps = []
    for i, b in enumerate(bottoms):
        # start = previous local minimum (standing) before b, or video start
        start = 0
        for j in range(b, 0, -1):
            if smoothed[j] <= smoothed[j - 1]:
                continue
            start = j
            break
        search_start = 0 if i == 0 else bottoms[i - 1]
        local_min_idx = search_start + int(np.argmin(smoothed[search_start:b + 1]))
        start = local_min_idx

        search_end = n - 1 if i == len(bottoms) - 1 else bottoms[i + 1]
        local_min_idx_end = b + int(np.argmin(smoothed[b:search_end + 1]))
        end = local_min_idx_end

        reps.append(Rep(index=i + 1, start_idx=start, bottom_idx=b, end_idx=end))
    return reps
