"""
All drawing happens here, on the ORIGINAL frame.
Markers are ellipses placed at the player's feet, in the style the brief
asked for ("think of the markers under the players' feet in FIFA") --
distinct colors per team, a distinct marker for whoever is judged to have
the ball, and a distinct highlight for the ball itself.

Note: OpenCV draws the shapes natively on the numpy arrays.
"""
from __future__ import annotations
import numpy as np
import cv2
import re

from config import CFG

# Comprehensive lookup table for standard sports colors (RGB)
COLOR_LOOKUP = {
    "red": (230, 40, 40),
    "blue": (40, 80, 230),
    "navy": (10, 20, 80),
    "white": (240, 240, 240),
    "black": (30, 30, 30),
    "yellow": (230, 210, 30),
    "green": (40, 180, 60),
    "teal": (0, 210, 190),
    "orange": (240, 130, 20),
    "purple": (120, 40, 200),
    "gray": (160, 160, 160),
    "unknown": (160, 160, 160),
}
FALLBACK_COLOR = (200, 200, 200)

BALL_COLOR = (250, 200, 30)      # yellow/gold
ON_BALL_COLOR = (255, 255, 255)  # white ring for whoever has the ball
DRAW_UNKNOWN = False  # Set True to render gray markers for referees; False to suppress


def _get_color_from_string(color_str: str) -> tuple[int, int, int]:
    if not color_str:
        return FALLBACK_COLOR
    normalized = color_str.strip().lower()
    for name, rgb in COLOR_LOOKUP.items():
        if name in normalized:
            return rgb
    return FALLBACK_COLOR


def _marker_ellipse(frame_bgr: np.ndarray, cx: float, cy: float, axes: tuple[int, int], color_rgb: tuple[int, int, int], width: int = 4) -> None:
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    cv2.ellipse(
        frame_bgr, 
        (int(cx), int(cy)), 
        axes, 
        0, 0, 360, 
        color_bgr, 
        width
    )


def compute_possession(players: list[dict], ball_xy: tuple[float, float] | None, geo) -> int | None:
    """Nearest player (in native pixels) to the ball, strictly within 50px."""
    if ball_xy is None or not players:
        return None
    
    bx_px, by_px = geo.ruler_to_orig_px(*ball_xy)
    best_id, best_d = None, float("inf")
    
    for p in players:
        team = p.get("team", "Gray").lower()
        if team == "gray" or team == "unknown":
            continue
        rx, ry = p["xy"]
        px, py = geo.ruler_to_orig_px(rx, ry)
        d = ((px - bx_px) ** 2 + (py - by_px) ** 2) ** 0.5
        if d < best_d:
            best_d, best_id = d, p["track_id"]
            
    if best_d < 50.0:
        return best_id
    return None


def render_frame(frame_bgr: np.ndarray, geo, snapshot: dict) -> np.ndarray:
    """snapshot: {'players': [...], 'ball': {...}}"""
    players = snapshot.get("players", [])
    ball = snapshot.get("ball")
    ball_xy = ball["xy"] if ball else None
    on_ball_id = compute_possession(players, ball_xy, geo)
    
    for p in players:
        team = p.get("team", "Gray").lower()
        
        # Suppress unknown/referee markers unless DRAW_UNKNOWN is True
        if (team == "gray" or team == "unknown") and not DRAW_UNKNOWN:
            continue
            
        rx, ry = p["xy"]
        x_foot, y_foot = geo.ruler_to_orig_px(rx, ry)
        
        raw_w = 40
        if p.get("bbox_px") is not None:
            _, _, raw_w, _ = p["bbox_px"]
        
        clamped_w = min(raw_w, 80)
        axes = (int(clamped_w * 0.6), int(clamped_w * 0.25))
        
        color_rgb = _get_color_from_string(team)
        _marker_ellipse(frame_bgr, x_foot, y_foot, axes, color_rgb)
        
        if p.get("track_id") is not None and p["track_id"] == on_ball_id:
            axes_on = (axes[0] + 4, axes[1] + 4)
            _marker_ellipse(frame_bgr, x_foot, y_foot, axes_on, ON_BALL_COLOR, width=3)

    if ball_xy is not None:
        bx_px, by_px = geo.ruler_to_orig_px(*ball_xy)
        r = int(CFG.marker_radius_px * 0.5)
        ball_color_bgr = (BALL_COLOR[2], BALL_COLOR[1], BALL_COLOR[0])
        # Filled ellipse for ball
        cv2.ellipse(frame_bgr, (int(bx_px), int(by_px)), (r, r), 0, 0, 360, ball_color_bgr, -1)
        # Outline for ball
        cv2.ellipse(frame_bgr, (int(bx_px), int(by_px)), (r, r), 0, 0, 360, (0, 0, 0), 1)

    return frame_bgr