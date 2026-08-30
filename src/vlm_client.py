"""
Async client for calling a VLM (default: Gemini flash) through OpenRouter,
one call per keyframe, run concurrently with a semaphore so we stay inside
the latency budget without blowing past rate limits.

Uses OpenRouter's structured-output support (response_format: json_schema)
so responses conform to schema.RESPONSE_JSON_SCHEMA -- but we still
validate on the way out (see schema.parse_vlm_response), because
constrained generation can still return schema-valid-but-semantically-bad
output (e.g. all zeros), and because not every model on OpenRouter
actually honors strict mode identically.
"""
from __future__ import annotations
import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from PIL import Image

from config import CFG
from schema import FrameDetections, RESPONSE_JSON_SCHEMA, parse_vlm_response_verbose

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert sports vision analyzer. Your task is to extract bounding boxes for all active soccer players and the ball.

The image has a ruler overlay along the top (X-axis) and left (Y-axis) edges, both scaled 0 to 1000. Use these rulers to read precise coordinates.

Strict constraints:
* Pitch Isolation: Only detect players actively positioned on the green pitch. Completely ignore photographers, sideline staff, advertising boards, and people in the stands.
* Ball Independence: Provide a single, independent bounding box for the soccer ball. It must not be grouped with a player. If the ball is not visible, set "ball" to null.
* Player Independence: Do not group multiple players into a single bounding box. Every player must have their own unique bounding box.
* Tight Cropping: Keep the player bounding boxes tightly cropped to their physical body to minimize the amount of background grass and shadow included.
* Ratio Limits: Ensure the width is never greater than the height for any player bounding box.
* Coordinates are integers on the 0-1000 ruler scale: [ymin, xmin, ymax, xmax].

Return ONLY valid JSON matching this schema — no markdown, no commentary:

{
  "players": [
    {"label": "player", "box_2d": [ymin, xmin, ymax, xmax]}
  ],
  "ball": {"box_2d": [ymin, xmin, ymax, xmax]}
}"""


@dataclass
class KeyframeResult:
    frame_index: int
    detections: Optional[FrameDetections]
    raw_error: Optional[str] = None
    cost_usd: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _call_once(client: httpx.AsyncClient, img_b64: str) -> dict:
    resp = await client.post(
        f"{CFG.openrouter_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {CFG.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": CFG.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Detect each individual player and the ball in this frame. One box per person, one box for the ball."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                },
            ],
            "response_format": {"type": "json_schema", "json_schema": RESPONSE_JSON_SCHEMA},
            "temperature": 0.0,
            # Ask OpenRouter to include per-request cost in the response
            # usage object, so we can report real $/video instead of
            # estimating from a hardcoded price table that can go stale.
            "usage": {"include": True},
        },
        timeout=CFG.vlm_timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


async def detect_keyframe(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    frame_index: int,
    ruled_frame: Image.Image,
) -> KeyframeResult:
    img_b64 = _encode_image(ruled_frame)
    last_err = None
    async with sem:
        for attempt in range(CFG.vlm_max_retries + 1):
            try:
                data = await _call_once(client, img_b64)
                raw = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                parsed, note = parse_vlm_response_verbose(raw)
                if parsed is not None:
                    if note:
                        logger.info("Keyframe %d: %s", frame_index, note)
                    return KeyframeResult(
                        frame_index=frame_index,
                        detections=parsed,
                        cost_usd=usage.get("cost"),
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                    )
                last_err = f"schema validation failed on attempt {attempt}: {note}"
            except Exception as e:  # network error, timeout, non-200, etc.
                last_err = str(e)
                await asyncio.sleep(0.5 * (attempt + 1))
    logger.warning("Keyframe %d failed after retries: %s", frame_index, last_err)
    return KeyframeResult(frame_index=frame_index, detections=None, raw_error=last_err)


async def detect_all_keyframes(
    ruled_frames: dict[int, Image.Image],
) -> dict[int, KeyframeResult]:
    """ruled_frames: {frame_index: ruler-overlaid PIL image}
    Fires all calls concurrently, bounded by CFG.max_concurrent_vlm_calls."""
    sem = asyncio.Semaphore(CFG.max_concurrent_vlm_calls)
    async with httpx.AsyncClient() as client:
        tasks = [
            detect_keyframe(client, sem, idx, img)
            for idx, img in ruled_frames.items()
        ]
        results = await asyncio.gather(*tasks)
    return {r.frame_index: r for r in results}


def get_openrouter_credits() -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (total_credits, total_usage, credit_limit) or (None, None, None) on error."""
    if CFG.use_mock_vlm or not CFG.openrouter_api_key:
        return None, None, None
    try:
        import httpx
        with httpx.Client() as client:
            resp = client.get(
                f"{CFG.openrouter_base_url}/auth/key",
                headers={"Authorization": f"Bearer {CFG.openrouter_api_key}"},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            limit = data.get("limit")
            usage = data.get("usage")
            limit_remaining = data.get("limit_remaining")
            return limit, usage, limit_remaining
    except Exception as e:
        logger.warning(f"Could not fetch OpenRouter credits: {e}")
        return None, None, None