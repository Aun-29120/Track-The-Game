"""
All drawing happens here, on the ORIGINAL frame (never the ruler-overlaid
one the VLM saw). Per the rules carried over from Task 1: VLMs look,
Pillow draws.

Markers are ellipses placed at the player's feet, in the style the brief
asked for ("think of the markers under the players' feet in FIFA") --
distinct colors per team, a distinct marker for whoever is judged to have
the ball, and a distinct highlight for the ball itself.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

from config import CFG

TEAM_COLORS = {
    "A": (0, 210, 190),   # teal
    "B": (230, 60, 60),   # red
}
BALL_COLOR = (250, 200, 30)      # yellow/gold
ON_BALL_COLOR = (255, 255, 255)  # white ring for whoever has the ball
BALL_TRIANGLE_COLOR = (220, 30, 30)  # red triangle marker above the ball


def _marker_ellipse(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: int, color, width: int = 4) -> None:
    draw.ellipse([cx - r, cy - r * 0.45, cx + r, cy + r * 0.45], outline=color, width=width)


def _triangle_above(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, color) -> None:
    """Downward-pointing filled triangle whose tip sits just above (cx, cy)."""
    gap = size * 0.8
    tip = (cx, cy - gap)
    left = (cx - size, cy - gap - size * 1.6)
    right = (cx + size, cy - gap - size * 1.6)
    draw.polygon([tip, left, right], fill=color, outline=(0, 0, 0))


def compute_possession(players: list[dict], ball_xy: tuple[float, float] | None) -> int | None:
    """Nearest player (in ruler units) to the ball, within a threshold.
    Deliberately NOT asked of the VLM -- once everything is in the same
    grounded coordinate frame this is just arithmetic, and it's cheaper
    and more consistent than a per-frame model judgment call."""
    if ball_xy is None or not players:
        return None
    bx, by = ball_xy
    best_id, best_d = None, float("inf")
    for p in players:
        px, py = p["xy"]
        d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        if d < best_d:
            best_d, best_id = d, p["track_id"]
    if best_d <= CFG.on_ball_distance_threshold_norm:
        return best_id
    return None


def render_frame(frame_bgr: np.ndarray, geo, snapshot: dict) -> np.ndarray:
    """snapshot: TrackManager.snapshot() output, in ruler-space coords."""
    img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB for Pillow
    draw = ImageDraw.Draw(img)

    players = snapshot["players"]
    ball = snapshot.get("ball")
    ball_xy = ball["xy"] if ball else None
    on_ball_id = compute_possession(players, ball_xy)

    for p in players:
        rx, ry = p["xy"]
        px, py = geo.ruler_to_orig_px(rx, ry)
        color = TEAM_COLORS.get(p["team"], (200, 200, 200))
        _marker_ellipse(draw, px, py, CFG.marker_radius_px, color)
        if p["track_id"] == on_ball_id:
            _marker_ellipse(draw, px, py, CFG.marker_radius_px + 6, ON_BALL_COLOR, width=3)

    if ball_xy is not None:
        bx_px, by_px = geo.ruler_to_orig_px(*ball_xy)
        r = CFG.marker_radius_px * 0.5
        draw.ellipse([bx_px - r, by_px - r, bx_px + r, by_px + r], fill=BALL_COLOR, outline=(0, 0, 0))
        _triangle_above(draw, bx_px, by_px, r * 1.1, BALL_TRIANGLE_COLOR)

    return np.array(img)[:, :, ::-1]  # back to BGR for cv2 video writer