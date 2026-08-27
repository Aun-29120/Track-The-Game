"""
Structured output schema for the per-keyframe VLM call.

Coordinates are in RULER units (0.0 - 10.0 on both axes), matching the
synthetic ruler drawn on the image the VLM actually sees. This is the
grounding trick from Task 1: the model reads off a scale instead of
guessing whether it should output relative or absolute pixel coordinates.

In practice we see THREE distinct failure modes, and they need three
different responses:
  1. Ruler-reading noise on an otherwise-correct response -- 10.2 instead
     of 9.8. Safe to clip into range.
  2. A one-off garbage value on an otherwise-correct response (a single
     player at y=340 while everyone else is normal). Drop just that
     detection.
  3. The WHOLE response is on a different scale -- Gemini in particular
     has a strong training-time habit of normalizing spatial coordinates
     to a 0-1000 grid, regardless of what the prompt asks for, and it
     surfaces unpredictably call-to-call even with an identical prompt.
     Treating every value in a 0-1000 response as "case 2 garbage" would
     drop nearly every player in that frame for no good reason -- the
     model actually read the frame fine, it just answered in the wrong
     units. So before doing per-value work, we check whether the
     response as a WHOLE looks like it's using a known alternate scale,
     and rescale everything in it first. Only what's left over after
     that gets the noise/garbage treatment.

A single unrecognized team label (referee, goalkeeper in a third kit)
gets that one detection dropped rather than failing the entire players
list. Anything actually structurally broken (not JSON, wrong shape)
still drops the whole keyframe -- the tracker falls back to its own
prediction for that frame.
"""
from __future__ import annotations
import json
from typing import Optional, Literal
from pydantic import BaseModel, Field

RULER_MAX = 10.0
COORD_TOLERANCE = 1.0  # ruler units of allowed overshoot before a value counts as "wrong system", not "noisy"

# Candidate whole-response scale factors to try, in order of how likely
# they are: 1.0 = model used our ruler correctly. 0.01 = model answered
# on a 0-1000 grid (Gemini's common normalized-coordinate convention) and
# needs dividing by 100 to land back on our 0-10 ruler. 0.1 = same idea
# for a 0-100 grid, seen less often but cheap to check for.
CANDIDATE_SCALES = (1.0, 0.01, 0.1)
MIN_VALID_FRACTION = 0.6  # a candidate scale must explain at least this fraction of a frame's values to be trusted

TeamLabel = Literal["A", "B"]


def _infer_scale(values: list[float]) -> float:
    """Picks whichever candidate scale puts the most values inside the
    valid (tolerant) range. Checks identity FIRST and keeps it as long as
    it already explains most values -- an alternate scale like 0.01 will
    trivially "explain" already-correct small numbers too (shrinking a
    valid 3.2 to 0.032 still lands inside a lenient window), so only look
    for a rescale when identity is clearly failing, and only accept one
    that explains a clear majority."""
    if not values:
        return 1.0
    lo, hi = -COORD_TOLERANCE, RULER_MAX + COORD_TOLERANCE

    def valid_fraction(scale: float) -> float:
        return sum(1 for v in values if lo <= v * scale <= hi) / len(values)

    if valid_fraction(1.0) >= MIN_VALID_FRACTION:
        return 1.0

    best_scale, best_frac = 1.0, valid_fraction(1.0)
    for scale in CANDIDATE_SCALES[1:]:
        frac = valid_fraction(scale)
        if frac > best_frac:
            best_scale, best_frac = scale, frac
    return best_scale if best_frac >= MIN_VALID_FRACTION else 1.0


def _sanitize_coord(v: object, scale: float = 1.0) -> Optional[float]:
    """Applies the inferred frame-level scale, then clips mild overshoot
    into [0, RULER_MAX]; returns None (caller should drop the detection)
    for anything beyond COORD_TOLERANCE even after rescaling."""
    try:
        v = float(v) * scale
    except (TypeError, ValueError):
        return None
    if -COORD_TOLERANCE <= v <= RULER_MAX + COORD_TOLERANCE:
        return round(min(max(v, 0.0), RULER_MAX), 3)
    return None


class PlayerDetection(BaseModel):
    # frame-local label only -- NOT a persistent identity.
    # e.g. "1", "2"... assigned fresh by the VLM each call, meaningless across frames.
    frame_label: str
    x: float = Field(..., ge=0.0, le=RULER_MAX)
    y: float = Field(..., ge=0.0, le=RULER_MAX)
    team: TeamLabel


class BallDetection(BaseModel):
    x: float = Field(..., ge=0.0, le=RULER_MAX)
    y: float = Field(..., ge=0.0, le=RULER_MAX)
    visible: bool = True


class FrameDetections(BaseModel):
    players: list[PlayerDetection] = Field(default_factory=list)
    ball: Optional[BallDetection] = None


# JSON schema handed to OpenRouter for structured/constrained generation.
# Kept in sync with FrameDetections by hand (small enough that a codegen
# step would be overkill).
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
                        "frame_label": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "team": {"type": "string", "enum": ["A", "B"]},
                    },
                    "required": ["frame_label", "x", "y", "team"],
                    "additionalProperties": False,
                },
            },
            "ball": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "visible": {"type": "boolean"},
                },
                "required": ["x", "y", "visible"],
                "additionalProperties": False,
            },
        },
        "required": ["players", "ball"],
        "additionalProperties": False,
    },
}


def parse_vlm_response(raw_json: str) -> Optional[FrameDetections]:
    """Returns None (not a raised exception) on malformed output, so the
    caller can fall back gracefully -- a bad keyframe should never take
    down the whole pipeline. Kept for backwards compatibility; prefer
    parse_vlm_response_verbose, which also tells you *why* it failed and
    tolerates a single bad team label instead of discarding everything."""
    parsed, _ = parse_vlm_response_verbose(raw_json)
    return parsed


def _normalize_team(raw: object) -> Optional[str]:
    """Best-effort mapping of near-miss team labels despite the system
    prompt's explicit "A"/"B" instruction (e.g. "Team A", "team_b", " A ").
    Returns None for anything that isn't clearly A or B -- a referee or
    third-kit goalkeeper should be DROPPED, not guessed into a team."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper().replace("TEAM", "").replace("_", "").replace(" ", "")
    return s if s in ("A", "B") else None


def parse_vlm_response_verbose(raw_json: str) -> tuple[Optional[FrameDetections], Optional[str]]:
    """Parses + validates one keyframe's response. Returns
    (parsed_or_None, note_or_None). `note` is a diagnostic string when
    parsing fails outright, or an informational note (e.g. "rescaled by
    0.01 (0-1000 grid); dropped 2 player(s): bad coords") when it
    succeeds after correction -- callers should log it either way."""
    try:
        raw = json.loads(raw_json)
    except Exception as e:
        return None, f"not valid JSON: {e} | raw[:300]={raw_json[:300]!r}"
    if not isinstance(raw, dict):
        return None, f"top-level JSON was not an object (got {type(raw).__name__})"

    players_raw = raw.get("players") if isinstance(raw.get("players"), list) else []
    ball_raw = raw.get("ball") if isinstance(raw.get("ball"), dict) else None

    # Look at every coordinate in the response BEFORE dropping anything --
    # a whole-response scale mismatch (see module docstring) needs to be
    # caught before per-value filtering, or it just looks like every
    # player individually has garbage coordinates.
    all_vals: list[float] = []
    for p in players_raw:
        if isinstance(p, dict):
            for k in ("x", "y"):
                v = p.get(k)
                if isinstance(v, (int, float)):
                    all_vals.append(float(v))
    if ball_raw:
        for k in ("x", "y"):
            v = ball_raw.get(k)
            if isinstance(v, (int, float)):
                all_vals.append(float(v))
    scale = _infer_scale(all_vals)

    dropped_team: list[object] = []
    dropped_coord = 0
    kept = []
    for p in players_raw[:60]:  # hard cap before we even do per-item work
        if not isinstance(p, dict):
            continue
        team = _normalize_team(p.get("team"))
        if team is None:
            dropped_team.append(p.get("team"))
            continue
        x = _sanitize_coord(p.get("x"), scale)
        y = _sanitize_coord(p.get("y"), scale)
        if x is None or y is None:
            dropped_coord += 1
            continue
        kept.append({**p, "team": team, "x": x, "y": y})
    raw["players"] = kept[:30]  # sanity bound, not a hard sport rule -- guards against a bad call hallucinating dozens

    if ball_raw:
        bx = _sanitize_coord(ball_raw.get("x"), scale)
        by = _sanitize_coord(ball_raw.get("y"), scale)
        # a garbage ball reading is worse than no ball this keyframe --
        # drop it entirely rather than pin it to a frame edge.
        raw["ball"] = {**ball_raw, "x": bx, "y": by} if (bx is not None and by is not None) else None
    else:
        raw["ball"] = None

    try:
        parsed = FrameDetections.model_validate(raw)
    except Exception as e:
        snippet = raw_json[:300].replace("\n", " ")
        return None, f"{e} | raw[:300]={snippet!r}"

    notes = []
    if scale != 1.0:
        notes.append(f"rescaled coords by {scale:g} (looked like a 0-{RULER_MAX / scale:g} grid)")
    if dropped_team:
        notes.append(f"{len(dropped_team)} player(s) with unrecognized team label {dropped_team}")
    if dropped_coord:
        notes.append(f"{dropped_coord} player(s) with out-of-range coords")
    note = "; ".join(notes) if notes else None
    return parsed, note