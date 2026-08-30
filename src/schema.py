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

# Sanity limits for individual player boxes (in ruler units)
MAX_PLAYER_BOX_WIDTH = 250.0
MAX_PLAYER_BOX_HEIGHT = 500.0

TeamLabel = Literal["A", "B"]


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


class BallDetection(BaseModel):
    box_2d: list[float] = Field(..., min_length=4, max_length=4)  # [ymin, xmin, ymax, xmax]


class FrameDetections(BaseModel):
    detections: list[PlayerDetection] = Field(default_factory=list)
    ball: Optional[BallDetection] = None


RESPONSE_JSON_SCHEMA = {
    "name": "frame_detections",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "players": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["player"]},
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        }
                    },
                    "required": ["label", "box_2d"],
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
        "required": ["players", "ball"],
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
        
        # Box sanity filter: reject macro-boxes
        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w <= 0 or box_h <= 0:
            dropped_coord += 1
            continue
        if box_w > MAX_PLAYER_BOX_WIDTH or box_h > MAX_PLAYER_BOX_HEIGHT:
            dropped_macro += 1
            continue
        # Reject landscape boxes (players are always portrait)
        if box_w > box_h:
            dropped_macro += 1
            continue
            
        kept.append({"label": "player", "box_2d": [ymin, xmin, ymax, xmax]})
        
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

    try:
        parsed = FrameDetections(
            detections=[PlayerDetection(**p) for p in kept[:30]],
            ball=ball_parsed,
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