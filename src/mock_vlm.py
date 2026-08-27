"""
TEST-ONLY stand-in for the real VLM call. Finds colored blobs via basic
OpenCV thresholding instead of asking a model to look at the image.

This exists purely so we can validate the tracker (Kalman + Hungarian +
optical flow + rendering) end-to-end against the synthetic clip from
make_test_clip.py without spending any OpenRouter budget while the
pipeline itself is still being shaken out. It is NOT part of the actual
submission pipeline and must not be used on real footage -- it only knows
about the exact synthetic colors make_test_clip.py draws.
"""
from __future__ import annotations
import numpy as np
import cv2

from ruler_overlay import RulerGeometry
from vlm_client import KeyframeResult
from schema import FrameDetections, PlayerDetection, BallDetection

# BGR colors, matching make_test_clip.py exactly
TEAM_A_BGR = (190, 210, 0)
TEAM_B_BGR = (60, 60, 230)
BALL_BGR = (30, 200, 250)
COLOR_TOL = 30


def _find_blobs(frame_bgr: np.ndarray, target_bgr: tuple[int, int, int]) -> list[tuple[float, float]]:
    lower = np.array([max(0, c - COLOR_TOL) for c in target_bgr])
    upper = np.array([min(255, c + COLOR_TOL) for c in target_bgr])
    mask = cv2.inRange(frame_bgr, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) < 15:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        centers.append((cx, cy))
    return centers


def mock_detect_frame(frame_bgr: np.ndarray, geo: RulerGeometry) -> FrameDetections:
    players = []
    for i, (px, py) in enumerate(_find_blobs(frame_bgr, TEAM_A_BGR)):
        rx, ry = geo.orig_px_to_ruler(px, py)
        players.append(PlayerDetection(frame_label=f"a{i}", x=rx, y=ry, team="A"))
    for i, (px, py) in enumerate(_find_blobs(frame_bgr, TEAM_B_BGR)):
        rx, ry = geo.orig_px_to_ruler(px, py)
        players.append(PlayerDetection(frame_label=f"b{i}", x=rx, y=ry, team="B"))

    ball_blobs = _find_blobs(frame_bgr, BALL_BGR)
    ball = None
    if ball_blobs:
        rx, ry = geo.orig_px_to_ruler(*ball_blobs[0])
        ball = BallDetection(x=rx, y=ry, visible=True)

    return FrameDetections(players=players, ball=ball)


def detect_all_keyframes_mock(
    original_frames: dict[int, np.ndarray],
    geo: RulerGeometry,
) -> dict[int, KeyframeResult]:
    return {
        idx: KeyframeResult(frame_index=idx, detections=mock_detect_frame(frame, geo))
        for idx, frame in original_frames.items()
    }
