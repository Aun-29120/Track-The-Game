"""
Structured output schema for the per-keyframe VLM call.

Coordinates are in RULER units (0 - 1000 on both axes), matching the
synthetic ruler drawn on the image the VLM actually sees.

The schema enforces:
  - "players": array of individual player bounding boxes [ymin, xmin, ymax, xmax]
  - "ball": optional single bounding box for the ball

Box sanity filters:
  - Player boxes wider than 250 ruler units or with landscape aspect ratio
    (width > height) are rejected as macro-boxes.
  - Coordinates outside [0, 1000] by more than COORD_TOLERANCE are rejected.
"""
from __future__ import annotations
import json
from typing import Optional, Literal
from pydantic import BaseModel, Field

RULER_MAX = 1000.0
COORD_TOLERANCE = 50.0  # ruler units of allowed overshoot

def _sanitize_coord(v: object) -> Optional[float]:
    """Clips mild overshoot into [0, RULER_MAX]; returns None (caller should drop the detection)
    for anything beyond COORD_TOLERANCE."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if -COORD_TOLERANCE <= v <= RULER_MAX + COORD_TOLERANCE:
        return round(min(max(v, 0.0), RULER_MAX), 3)
    return None

class PlayerDetection(BaseModel):
    label: Literal["player"]
    box_2d: list[float] = Field(..., min_length=4, max_length=4)  # [ymin, xmin, ymax, xmax]
    team: str
    jersey_number: Optional[str] = None


class BallDetection(BaseModel):
    box_2d: list[float] = Field(..., min_length=4, max_length=4)  # [ymin, xmin, ymax, xmax]


class FrameDetections(BaseModel):
    detections: list[PlayerDetection] = Field(default_factory=list)
    ball: Optional[BallDetection] = None
    pitch_boundary: Optional[list[tuple[float, float]]] = None


def get_response_json_schema(team_a_color: str, team_b_color: str) -> dict:
    return {
        "name": "frame_detections",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "pitch_boundary": {
                    "type": ["array", "null"],
                    "description": "A list of [y, x] coordinate pairs that trace the topmost visible boundary of the green playing surface (e.g., the touchline or goal-line) ordered from left to right along the X-axis. Include 3 to 5 points. If the boundary is a straight horizontal line, 2 points are sufficient.",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "players": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": ["player"]},
                            "team": {"type": "string", "enum": [f"Team {team_a_color}", f"Team {team_b_color}", "Gray"]},
                            "jersey_number": {"type": ["string", "null"]},
                            "box_2d": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4
                            }
                        },
                        "required": ["label", "box_2d", "team"],
                        "additionalProperties": False,
                    },
                },
                "ball": {
                    "type": ["object", "null"],
                    "properties": {
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        }
                    },
                    "required": ["box_2d"],
                    "additionalProperties": False,
                }
            },
            "required": ["pitch_boundary", "players", "ball"],
            "additionalProperties": False,
        },
    }




def parse_vlm_response(raw_json: str) -> Optional[FrameDetections]:
    parsed, _ = parse_vlm_response_verbose(raw_json)
    return parsed


def parse_vlm_response_verbose(raw_json: str) -> tuple[Optional[FrameDetections], Optional[str]]:
    try:
        raw = json.loads(raw_json)
    except Exception as e:
        return None, f"not valid JSON: {e} | raw[:300]={raw_json[:300]!r}"
    if not isinstance(raw, dict):
        return None, f"top-level JSON was not an object (got {type(raw).__name__})"

    # --- Parse players ---
    # Accept either "players" or legacy "detections" key
    dets_raw = raw.get("players") or raw.get("detections") or []
    if not isinstance(dets_raw, list):
        dets_raw = []

    dropped_coord = 0
    dropped_macro = 0
    kept = []
    for p in dets_raw[:60]:
        if not isinstance(p, dict):
            continue
        
        box = p.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue
            
        ymin = _sanitize_coord(box[0])
        xmin = _sanitize_coord(box[1])
        ymax = _sanitize_coord(box[2])
        xmax = _sanitize_coord(box[3])
        
        if any(c is None for c in (ymin, xmin, ymax, xmax)):
            dropped_coord += 1
            continue
        
        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w <= 0 or box_h <= 0:
            dropped_coord += 1
            continue
            
        team = str(p.get("team", "Gray"))
        jersey = p.get("jersey_number")
        if jersey is not None:
            jersey = str(jersey)
        kept.append({"label": "player", "box_2d": [ymin, xmin, ymax, xmax], "team": team, "jersey_number": jersey})
        
    # --- Parse ball ---
    ball_parsed = None
    ball_raw = raw.get("ball")
    if isinstance(ball_raw, dict):
        ball_box = ball_raw.get("box_2d")
        if isinstance(ball_box, list) and len(ball_box) == 4:
            bymin = _sanitize_coord(ball_box[0])
            bxmin = _sanitize_coord(ball_box[1])
            bymax = _sanitize_coord(ball_box[2])
            bxmax = _sanitize_coord(ball_box[3])
            if all(c is not None for c in (bymin, bxmin, bymax, bxmax)):
                ball_parsed = BallDetection(box_2d=[bymin, bxmin, bymax, bxmax])

    # --- Parse pitch_boundary ---
    pitch_boundary = None
    if "pitch_boundary" in raw and isinstance(raw["pitch_boundary"], list):
        pts = []
        for pt in raw["pitch_boundary"]:
            if isinstance(pt, list) and len(pt) == 2:
                y = _sanitize_coord(pt[0])
                x = _sanitize_coord(pt[1])
                if y is not None and x is not None:
                    pts.append((y, x))
        
        # Defensive parsing
        if len(pts) >= 1:
            # Sort by x ascending
            pts.sort(key=lambda p: p[1])
            pitch_boundary = pts

    try:
        parsed = FrameDetections(
            detections=[PlayerDetection(**p) for p in kept[:30]],
            ball=ball_parsed,
            pitch_boundary=pitch_boundary,
        )
    except Exception as e:
        snippet = raw_json[:300].replace("\n", " ")
        return None, f"{e} | raw[:300]={snippet!r}"

    notes = []
    if dropped_coord:
        notes.append(f"{dropped_coord} player(s) with out-of-range coords")
    if dropped_macro:
        notes.append(f"{dropped_macro} macro-box(es) rejected")
    note = "; ".join(notes) if notes else None
    return parsed, note