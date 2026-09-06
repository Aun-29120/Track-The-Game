"""
Async client for calling a VLM (default: Gemini flash) through OpenRouter.
Implements a two-pass architecture:
1. Bootstrapping: Extracts two primary team colors from Frame 0.
2. Concurrent Perception: Classifies all players on keyframes into those two colors or 'Gray'.
"""
from __future__ import annotations
import asyncio
import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from PIL import Image

from config import CFG
from schema import FrameDetections, parse_vlm_response_verbose

logger = logging.getLogger(__name__)


def get_system_prompt(team_a_color: str, team_b_color: str) -> str:
    return f"""You are an expert sports vision analyzer. Your task is to extract bounding boxes for all active soccer players and the ball, and identify the visible pitch boundary.

The image has a ruler overlay along the top (X-axis) and left (Y-axis) edges, both scaled 0 to 1000. Use these rulers to read precise coordinates.

Pitch Boundary Detection:
* "pitch_boundary": Return a list of [y, x] coordinate pairs (between 3 and 5 points) that trace the true perspective-skewed topmost visible boundary of the green playing surface (e.g., the touchline or goal-line). 
  - IMPORTANT: Your points MUST span close to the entire visible width of the image (from x near 0 to x near 1000). 
  - If the pitch ends and the goal-line is visible, trace down the goal-line to exclude the out-of-bounds area behind the goal. 
  - Order them from left to right along the X-axis.

Strict constraints:
* Pitch Isolation: You must ONLY detect players whose feet are physically standing on the visibly green playing surface (below pitch_boundary). Any person located above this boundary must be excluded.
* Perimeter Exclusion: Anything appearing on, behind, or above the perimeter advertising hoardings, or off the pitch, MUST be strictly ignored. Do not draw boxes on advertising text, photographers, ball boys, sideline staff, or people standing behind the goal nets, no matter how much they resemble a player.
* Ball Independence: Provide a single, independent bounding box for the soccer ball. It must not be grouped with a player. If the ball is not visible, set "ball" to null.
* Player Independence: Do not group multiple players into a single bounding box. Every player must have their own unique bounding box.
* Tight Cropping: Keep the player bounding boxes tightly cropped to their physical body to minimize the amount of background grass and shadow included.
* Coordinates are integers on the 0-1000 ruler scale: [ymin, xmin, ymax, xmax].

Classification & Identity:
* Classify each detected player as belonging to Team {team_a_color}, Team {team_b_color}, or 'Gray'. Output this exactly as the "team" string.
* CRITICAL (Referees): Anyone wearing a kit color that clearly matches NEITHER Team {team_a_color} nor Team {team_b_color} MUST be classified as "Gray". This includes all match officials and referees.
* CRITICAL (Goalkeepers): ANY player acting as a goalkeeper — visible near their own goal, typically wearing gloves or a distinct kit — MUST ALWAYS be classified as "Gray" regardless of kit color. Even if the goalkeeper's kit happens to resemble Team {team_a_color} or Team {team_b_color}, classify them as "Gray". There are exactly two goalkeepers in the match and BOTH must always be "Gray".
* Identify the jersey number of the player if it is clearly visible. If it is illegible, partially obscured, or facing away, set "jersey_number" to null. Do not guess.

Return ONLY valid JSON matching this schema — no markdown, no commentary:

{{
  "pitch_boundary": [[y, x], [y, x]],
  "players": [
    {{"label": "player", "team": "Color", "jersey_number": "10", "box_2d": [ymin, xmin, ymax, xmax]}}
  ],
  "ball": {{"box_2d": [ymin, xmin, ymax, xmax]}}
}}"""


@dataclass
class KeyframeResult:
    frame_index: int
    detections: Optional[FrameDetections]
    raw_error: Optional[str] = None
    cost_usd: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


MAX_IMAGE_SIDE = 1280

def _encode_image(img: Image.Image) -> str:
    w, h = img.size
    max_side = max(w, h)
    if max_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max_side
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def bootstrap_team_colors(frame: Image.Image) -> tuple[str, str]:
    """Pass 1: Blockingly ask the VLM for the two primary team colors."""
    if CFG.use_mock_vlm:
        return "White", "Navy"

    img_b64 = await asyncio.to_thread(_encode_image, frame)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{CFG.openrouter_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {CFG.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": CFG.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this sports game image and identify the two primary jersey colors worn by the opposing teams. Return ONLY a JSON object with 'team_a' and 'team_b' as string keys containing the two colors. Nothing else."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            colors = json.loads(content)
            return str(colors.get("team_a", "White")), str(colors.get("team_b", "Navy"))
        except Exception as e:
            logger.warning(f"Failed to bootstrap colors from JSON '{content}': {e}")
            return "White", "Navy"


async def _call_once(client: httpx.AsyncClient, img_b64: str, team_a_color: str, team_b_color: str) -> dict:
    from schema import get_response_json_schema
    resp = await client.post(
        f"{CFG.openrouter_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {CFG.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": CFG.model,
            "messages": [
                {"role": "system", "content": get_system_prompt(team_a_color, team_b_color)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Detect each individual player and the ball in this frame. One box per person, one box for the ball."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                },
            ],
            "response_format": {"type": "json_schema", "json_schema": get_response_json_schema(team_a_color, team_b_color)},
            "temperature": 0.0,
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
    team_a_color: str,
    team_b_color: str,
) -> KeyframeResult:
    img_b64 = await asyncio.to_thread(_encode_image, ruled_frame)
    last_err = None
    
    for attempt in range(CFG.vlm_max_retries + 1):
        try:
            async with sem:
                logger.info(f"Keyframe {frame_index}: Request SENT")
                data = await _call_once(client, img_b64, team_a_color, team_b_color)
                logger.info(f"Keyframe {frame_index}: Request RECEIVED")
                    
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
        except Exception as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                logger.warning(f"Keyframe {frame_index}: Rate limited (429) on attempt {attempt}")
            last_err = str(e)
            
        if attempt < CFG.vlm_max_retries:
            await asyncio.sleep(0.5 * (attempt + 1))
            
    logger.warning("Keyframe %d failed after retries: %s", frame_index, last_err)
    return KeyframeResult(frame_index=frame_index, detections=None, raw_error=last_err)


async def detect_all_keyframes(
    player_frames: dict[int, Image.Image],
    team_a_color: str,
    team_b_color: str,
) -> dict[int, KeyframeResult]:
    """Fires all calls concurrently, bounded by CFG.max_concurrent_vlm_calls."""
    sem = asyncio.Semaphore(CFG.max_concurrent_vlm_calls)
    limits = httpx.Limits(max_connections=CFG.max_concurrent_vlm_calls, max_keepalive_connections=CFG.max_concurrent_vlm_calls)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            detect_keyframe(client, sem, idx, img, team_a_color, team_b_color)
            for idx, img in player_frames.items()
        ]
        results = await asyncio.gather(*tasks)
    return {r.frame_index: r for r in results}


def get_openrouter_credits() -> tuple[Optional[float], Optional[float], Optional[float]]:
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