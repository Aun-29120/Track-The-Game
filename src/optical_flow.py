"""
Lucas-Kanade optical flow -- the cheap, per-frame mechanism that covers
the ~800 non-keyframe frames without any VLM call. Pure pixel motion,
no semantic understanding of "player"; it just follows a small patch of
pixel intensities frame to frame.

Runs in ORIGINAL pixel space (not ruler space), since it's tracking real
image content. Conversion to/from ruler units happens at the call site.

Known weakness, as flagged in the thinking cap: error accumulates over
many frames, and a point can be lost outright under fast motion, blur, or
occlusion. That's why every keyframe resets these points against a fresh
Kalman-corrected position rather than letting drift compound indefinitely.
"""
from __future__ import annotations
import numpy as np
import cv2

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def propagate_points(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    points_px: np.ndarray,  # shape (N, 2), pixel coords in prev_gray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (new_points_px, status) where status[i] == 1 if point i was
    tracked successfully, 0 if lost (e.g. occluded, moved out of frame).
    Callers should fall back to the Kalman prediction alone for any point
    with status == 0."""
    if len(points_px) == 0:
        return points_px, np.array([], dtype=np.uint8)

    pts = points_px.reshape(-1, 1, 2).astype(np.float32)
    new_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **LK_PARAMS)
    return new_pts.reshape(-1, 2), status.reshape(-1)


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
