"""
Offline Two-Pass Tracking Pipeline.

Pass 1: Bootstraps colors from Frame 0, then fires sparse concurrent VLM keyframe requests.
Pass 2: Interpolates the keyframes into a dense temporal map, then renders all frames.
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from config import CFG
from ruler_overlay import add_ruler, RulerGeometry
from vlm_client import detect_all_keyframes, bootstrap_team_colors
from interpolator import interpolate_pipeline

from renderer import render_frame
from video_io import read_all_frames, write_video_stream

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    n_frames: int
    n_keyframes: int
    n_keyframes_failed: int
    elapsed_s: float
    total_cost_usd: Optional[float]
    cost_is_partial: bool


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_bgr[:, :, ::-1])


def select_keyframe_indices(n_frames: int, interval: int) -> list[int]:
    return list(range(0, n_frames, interval))


def run_pipeline(input_path: str, output_path: str) -> RunStats:
    t0 = time.time()
    frames, fps = read_all_frames(input_path)
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames read from input video")

    player_keyframe_idxs = set(select_keyframe_indices(n_frames, CFG.keyframe_interval))
    logger.info("Selected %d frames as full keyframes", len(player_keyframe_idxs))

    # Bootstrapping & Keyframe Prep (Overlapped)
    logger.info(f"Start pipeline at {time.time() - t0:.2f}s")
    pil_frame0 = _bgr_to_pil(frames[0])
    ruled0, geo0 = add_ruler(pil_frame0)
    
    player_ruled_frames: dict[int, Image.Image] = {}
    
    async def overlap_prep():
        # Start the network call
        bootstrap_task = asyncio.create_task(bootstrap_team_colors(ruled0))
        
        # Do the CPU-bound prep in a thread
        def prep_rulers():
            res = {}
            g = None
            for idx in player_keyframe_idxs:
                pil_frame = _bgr_to_pil(frames[idx])
                ruled, g = add_ruler(pil_frame)
                res[idx] = ruled
            return res, g
            
        res, geo = await asyncio.to_thread(prep_rulers)
        team_a, team_b = await bootstrap_task
        return team_a, team_b, res, geo
        
    team_a, team_b, player_ruled_frames, geo = asyncio.run(overlap_prep())
    logger.info("Bootstrapped colors: Team A='%s', Team B='%s'", team_a, team_b)

    # Asynchronous Perception (Pass 1)
    t_vlm_start = time.time()
    if CFG.use_mock_vlm:
        from mock_vlm import detect_all_keyframes_mock
        original_kf_frames = {idx: frames[idx] for idx in player_keyframe_idxs}
        # mock returns a dict mapping frame_idx -> KeyframeResult, need to inject colors
        results = detect_all_keyframes_mock(original_kf_frames, geo, team_a, team_b)
    else:
        logger.info(f"Start VLM calls at {time.time() - t0:.2f}s"); results = asyncio.run(detect_all_keyframes(player_ruled_frames, team_a, team_b))
    t_vlm = time.time() - t_vlm_start
    
    n_failed = sum(1 for r in results.values() if r.detections is None)
    if n_failed:
        logger.warning("%d/%d keyframes failed VLM detection/validation", n_failed, len(results))

    total_cost = 0.0
    cost_missing = False
    for r in results.values():
        if r.cost_usd is not None:
            total_cost += r.cost_usd
        elif not CFG.use_mock_vlm:
            cost_missing = True

    # Temporal Intelligence & Interpolation (Pass 2 prep)
    logger.info("Interpolating dense coordinate map...")
    kf_detections = {idx: r.detections for idx, r in results.items() if r.detections is not None}
    
    # --- Task 2: Persist raw detections for future debugging ---
    import json
    import os
    raw_out_path = os.path.splitext(output_path)[0] + "_raw.json"
    try:
        with open(raw_out_path, "w") as f:
            # We must serialize the Pydantic models to dictionaries
            dump_data = {
                str(idx): (det.model_dump() if hasattr(det, "model_dump") else det.dict()) 
                for idx, det in kf_detections.items()
            }
            json.dump(dump_data, f, indent=2)
        logger.info(f"Saved raw detections to {raw_out_path}")
    except Exception as e:
        logger.warning(f"Failed to save raw detections: {e}")
        
    logger.info(f"Start interpolation at {time.time() - t0:.2f}s"); dense_map = interpolate_pipeline(kf_detections, n_frames, geo)

    # Rendering (Pass 2)
    logger.info(f"Start render at {time.time() - t0:.2f}s"); t_render_start = time.time()
    write_video_stream(output_path, frames, dense_map, geo, fps)
    logger.info("Render & Video write: %.1fs", time.time() - t_render_start)

    return RunStats(
        n_frames=n_frames,
        n_keyframes=len(player_keyframe_idxs),
        n_keyframes_failed=n_failed,
        elapsed_s=time.time() - t0,
        total_cost_usd=total_cost if not cost_missing else None,
        cost_is_partial=cost_missing,
    )