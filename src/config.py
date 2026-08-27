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
    max_concurrent_vlm_calls: int = int(os.getenv("MAX_CONCURRENT_VLM_CALLS", "30"))
    vlm_timeout_s: float = float(os.getenv("VLM_TIMEOUT_S", "8"))
    vlm_max_retries: int = 2

    # --- Sampling ---
    keyframe_interval: int = int(os.getenv("KEYFRAME_INTERVAL", "8"))  # every Nth frame goes to the VLM

    # --- Testing ---
    # When true, pipeline.py uses mock_vlm.py (color-blob CV) instead of a
    # real API call. For shaking out the tracker/pipeline against the
    # synthetic clip before spending real budget on real footage.
    use_mock_vlm: bool = os.getenv("USE_MOCK_VLM", "false").lower() == "true"

    # --- Ruler / grounding overlay ---
    ruler_scale_max: float = 10.0   # ruler reads 0.0 -> 10.0 along each edge
    ruler_thickness_px: int = 40    # thickness of the ruler band drawn on each edge
    ruler_major_tick_every: float = 1.0
    ruler_minor_tick_every: float = 0.5

    # --- Tracking ---
    max_track_age_frames: int = 20       # drop a track if unmatched for this many frames
    max_assignment_cost: float = 0.75    # above this, treat as no-match (spawn new / age out)
    kalman_process_noise: float = 1e-2
    kalman_measurement_noise: float = 1e-1

    # Assignment cost weights (position/appearance/team), from the error-report
    # fix: replaced fake point-box IoU with real point-distance cost.
    position_cost_weight: float = 0.65
    appearance_cost_weight: float = 0.20
    team_cost_weight: float = 0.15

    # Movement/teleportation gates: reject a VLM measurement outright if it's
    # implausibly far from the Kalman prediction, rather than letting one bad
    # detection hijack an established track's identity.
    max_player_jump_norm: float = 1.5    # ruler units; a player can't plausibly move further than this between keyframes
    max_ball_jump_norm: float = 2.5      # ball moves faster than players, so a looser gate

    # New-track spawn sanity gates: an unmatched VLM detection shouldn't
    # automatically become a persistent identity.
    min_new_track_spawn_distance_norm: float = 0.5  # too close to an existing track -> treat as noise/duplicate, not a new player
    min_team_confidence_margin: float = 0.05         # ambiguous team appearance -> don't spawn

    # Duplicate-detection dedup within a single keyframe's response.
    duplicate_detection_distance_norm: float = 0.2

    # --- Appearance sampling (torso, not feet) ---
    torso_offset_frac_height: float = 0.06   # fraction of frame height to sample above the foot point
    saturation_mask_threshold: int = 40      # HSV saturation below this is treated as background (grass/pitch) and masked out
    team_prototype_ema_alpha: float = 0.1    # how fast persistent team-color prototypes adapt to lighting changes

    # --- Debugging ---
    debug_logging: bool = os.getenv("DEBUG_LOGGING", "false").lower() == "true"

    # --- Rendering ---
    marker_radius_px: int = 14
    on_ball_distance_threshold_norm: float = 0.6  # in ruler units; nearest player within this counts as "on the ball"

    # --- Budget / latency targets (from the brief, for the report) ---
    target_latency_s: float = 15.0
    max_latency_s: float = 25.0
    target_cost_usd: float = 1.0


CFG = Config()