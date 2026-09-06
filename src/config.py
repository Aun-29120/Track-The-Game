"""
Central config for the tracker. Every knob mentioned in the brainstorm
(keyframe interval, model choice, ruler scale, tracker thresholds) lives
here so the ablation runs are just "change one value here, rerun."
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- VLM ---
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # model string as used by OpenRouter, e.g. "google/gemini-2.5-flash"
    model: str = os.getenv("VLM_MODEL", "google/gemini-3.5-flash-lite")
    max_concurrent_vlm_calls: int = int(os.getenv("MAX_CONCURRENT_VLM_CALLS", "64"))
    vlm_timeout_s: float = float(os.getenv("VLM_TIMEOUT_S", "20"))
    vlm_max_retries: int = 2

    # --- Sampling ---
    keyframe_interval: int = int(os.getenv("KEYFRAME_INTERVAL", "16"))  # every Nth frame goes to the VLM

    # --- Testing ---
    # When true, pipeline.py uses mock_vlm.py (color-blob CV) instead of a
    # real API call. For shaking out the tracker/pipeline against the
    # synthetic clip before spending real budget on real footage.
    use_mock_vlm: bool = os.getenv("USE_MOCK_VLM", "false").lower() == "true"

    # --- Ruler / grounding overlay ---
    ruler_scale_max: float = 1000.0   # ruler reads 0 -> 1000 along each edge
    ruler_thickness_px: int = 40    # thickness of the ruler band drawn on each edge
    ruler_major_tick_every: float = 100.0
    ruler_minor_tick_every: float = 50.0

    # --- Tracking ---
    # Movement/teleportation gates: reject a VLM measurement outright if it's
    # implausibly far from the last known position.
    max_player_jump_norm: float = 150.0    # ruler units; a player can't plausibly move further than this between keyframes
    max_ball_jump_norm: float = 250.0      # ball moves faster than players, so a looser gate
    max_interpolation_gap_frames: int = 24 # max gap (in frames) to interpolate across. gaps larger than this are left blank.

    # Assignment costs for identity (added to spatial distance)
    jersey_match_reward: float = 1000.0
    jersey_mismatch_penalty: float = 20.0
    team_mismatch_penalty: float = 50.0

    # duplicate detection dedup within a single keyframe's response.
    duplicate_detection_distance_norm: float = 20.0

    # --- Debugging ---
    debug_logging: bool = os.getenv("DEBUG_LOGGING", "false").lower() == "true"

    # --- Rendering ---
    marker_radius_px: int = 14
    on_ball_distance_threshold_norm: float = 60.0  # in ruler units; nearest player within this counts as "on the ball"

    # --- Budget / latency targets (from the brief, for the report) ---
    target_latency_s: float = 15.0
    max_latency_s: float = 25.0
    target_cost_usd: float = 1.0


CFG = Config()