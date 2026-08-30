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
    
    # helper to mock bounding boxes
    def _add_players(centers, label_prefix):
        for i, (px, py) in enumerate(centers):
            # mock bounding box of 40x80 pixels around the center
            ymin = max(0, py - 40)
            xmin = max(0, px - 20)
            ymax = min(frame_bgr.shape[0], py + 40)
            xmax = min(frame_bgr.shape[1], px + 20)
            
            # convert to ruler coordinates
            rx_min, ry_min = geo.orig_px_to_ruler(xmin, ymin)
            rx_max, ry_max = geo.orig_px_to_ruler(xmax, ymax)
            
            box_2d = [ry_min, rx_min, ry_max, rx_max]
            players.append(PlayerDetection(label="player", box_2d=box_2d))

    _add_players(_find_blobs(frame_bgr, TEAM_A_BGR), "a")
    _add_players(_find_blobs(frame_bgr, TEAM_B_BGR), "b")

    # Ball detection
    ball_det = None
    ball_centers = _find_blobs(frame_bgr, BALL_BGR)
    if ball_centers:
        bpx, bpy = ball_centers[0]
        bymin = max(0, bpy - 10)
        bxmin = max(0, bpx - 10)
        bymax = min(frame_bgr.shape[0], bpy + 10)
        bxmax = min(frame_bgr.shape[1], bpx + 10)
        brx_min, bry_min = geo.orig_px_to_ruler(bxmin, bymin)
        brx_max, bry_max = geo.orig_px_to_ruler(bxmax, bymax)
        ball_det = BallDetection(box_2d=[bry_min, brx_min, bry_max, brx_max])

    return FrameDetections(detections=players, ball=ball_det)


def detect_all_keyframes_mock(
    original_frames: dict[int, np.ndarray],
    geo: RulerGeometry,
) -> dict[int, KeyframeResult]:
    return {
        idx: KeyframeResult(frame_index=idx, detections=mock_detect_frame(frame, geo))
        for idx, frame in original_frames.items()
    }
