"""
Orchestrates the full clip -> annotated clip pipeline, tying together
every module built so far:

  video_io      -- read frames in, write annotated frames out
  ruler_overlay -- ground keyframes for the VLM
  vlm_client    -- sparse, concurrent VLM calls (perception)
  tracker       -- Kalman + Hungarian + appearance (identity/continuity)
  optical_flow  -- cheap per-frame propagation between keyframes
  renderer      -- Pillow drawing onto the original frames

Frame-type split, as designed in the thinking cap:
  keyframe      -> VLM detects -> tracker.step_keyframe (predict+assign+correct)
  non-keyframe  -> optical flow propagates each track's point one frame
                   forward; tracker.step_predict_only advances the Kalman
                   state to match (keeps velocity estimates sane even
                   though optical flow, not Kalman, supplies the position).
"""
from __future__ import annotations
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
from PIL import Image

from config import CFG
from ruler_overlay import add_ruler, RulerGeometry
from vlm_client import detect_all_keyframes
from tracker import TrackManager
from team_identity import TeamClassifier

from renderer import render_frame
from video_io import read_all_frames, write_video

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    n_frames: int
    n_keyframes: int
    n_keyframes_failed: int
    elapsed_s: float
    total_cost_usd: Optional[float]
    cost_is_partial: bool  # True if some keyframes didn't report cost (e.g. mock mode, or provider omitted it)


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

    keyframe_idxs = select_keyframe_indices(n_frames, CFG.keyframe_interval)
    logger.info("Selected %d/%d frames as keyframes (interval=%d)",
                len(keyframe_idxs), n_frames, CFG.keyframe_interval)

    # 1. build ruler-overlaid images for every keyframe, and remember the
    #    geometry for each (constant per-clip since resolution doesn't change,
    #    but keeping it per-frame keeps this correct if that ever changes)
    ruled_frames: dict[int, Image.Image] = {}
    geo: RulerGeometry | None = None
    for idx in keyframe_idxs:
        pil_frame = _bgr_to_pil(frames[idx])
        ruled, geo = add_ruler(pil_frame)
        ruled_frames[idx] = ruled

    # 2. fire all VLM keyframe calls concurrently (or use the mock CV
    #    detector against the synthetic clip -- see config.use_mock_vlm)
    t_vlm_start = time.time()
    if CFG.use_mock_vlm:
        from mock_vlm import detect_all_keyframes_mock
        original_kf_frames = {idx: frames[idx] for idx in keyframe_idxs}
        results = detect_all_keyframes_mock(original_kf_frames, geo)
    else:
        results = asyncio.run(detect_all_keyframes(ruled_frames))
    t_vlm = time.time() - t_vlm_start
    n_failed = sum(1 for r in results.values() if r.detections is None)
    if n_failed:
        logger.warning("%d/%d keyframes failed VLM detection/validation", n_failed, len(results))
        for idx, r in results.items():
            if r.detections is None:
                logger.warning("  keyframe %d: %s", idx, r.raw_error)
    logger.info("VLM phase: %.1fs for %d keyframes (%.2fs/keyframe avg)",
                t_vlm, len(keyframe_idxs), t_vlm / max(1, len(keyframe_idxs)))

    # 3. walk the clip frame by frame: keyframes update via VLM+Kalman+Hungarian,
    #    everything else propagates via optical flow (+Kalman predict to keep
    #    velocity estimates coherent).
    manager = TrackManager()
    team_classifier = TeamClassifier()
    out_frames: list[np.ndarray] = []
    # KCF trackers for each player, mapping track_id to cv2.TrackerKCF
    kcf_trackers = {}
    
    t_loop_start = time.time()

    for idx in range(n_frames):
        frame_bgr = frames[idx]


        if idx in keyframe_idxs and results[idx].detections is not None:
            raw_detections = results[idx].detections
            
            # Filter small/out-of-bounds/edge detections
            frame_h, frame_w = frame_bgr.shape[:2]
            EDGE_MARGIN_PX = 15
            valid_players = []
            for p in raw_detections.detections:
                r_ymin, r_xmin, r_ymax, r_xmax = p.box_2d
                px_xmin, px_ymin = geo.ruler_to_orig_px(r_xmin, r_ymin)
                px_xmax, px_ymax = geo.ruler_to_orig_px(r_xmax, r_ymax)
                
                if px_xmax <= px_xmin or px_ymax <= px_ymin:
                    continue
                if (px_xmax - px_xmin) < 15 or (px_ymax - px_ymin) < 15:
                    continue
                
                # Spatial edge filter: drop boxes whose center is within 15px of frame border
                cx = (px_xmin + px_xmax) / 2.0
                cy = (px_ymin + px_ymax) / 2.0
                if cx < EDGE_MARGIN_PX or cx > (frame_w - EDGE_MARGIN_PX):
                    continue
                if cy < EDGE_MARGIN_PX or cy > (frame_h - EDGE_MARGIN_PX):
                    continue
                    
                # Spatial Horizon & Pitch Filter (AND-gate logic)
                foot_y = px_ymax
                foot_x = cx
                
                # 1. Horizon test: drop if above 250px
                if foot_y < 250:
                    continue
                    
                # 2. Pitch test: drop if foot coordinates are not on valid green grass
                crop_y0 = max(0, int(foot_y) - 2)
                crop_y1 = min(frame_h, int(foot_y) + 3)
                crop_x0 = max(0, int(foot_x) - 2)
                crop_x1 = min(frame_w, int(foot_x) + 3)
                
                if crop_y1 > crop_y0 and crop_x1 > crop_x0:
                    foot_patch = frame_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
                    hsv = cv2.cvtColor(foot_patch, cv2.COLOR_BGR2HSV)
                    # H: 35-85, S: >40, V: >40
                    grass_mask = cv2.inRange(hsv, (35, 41, 41), (85, 255, 255))
                    if np.count_nonzero(grass_mask) == 0:
                        continue
                        
                valid_players.append(p)
            
            raw_detections.detections = valid_players
            manager.step_keyframe(idx, frame_bgr, geo, raw_detections, team_classifier)
            
            # reset KCF trackers to the freshly corrected Kalman positions
            kcf_trackers = {}
            for t in manager.player_tracks:
                if t.last_bbox_px is None:
                    continue
                    
                x, y, w, h = t.last_bbox_px
                
                # Clamp bounding box to image bounds
                x1 = max(0, min(int(x), frame_w - 1))
                y1 = max(0, min(int(y), frame_h - 1))
                new_w = max(5, min(int(w), frame_w - x1))
                new_h = max(5, min(int(h), frame_h - y1))
                
                # Initialize KCF tracker
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame_bgr, (x1, y1, new_w, new_h))
                kcf_trackers[t.track_id] = tracker
                
        elif idx in keyframe_idxs and results[idx].detections is None:
            # failed keyframe -- fall back to prediction only, same as a non-keyframe
            manager.step_predict_only(idx)

        else:
            # non-keyframe: KCF updates player bounding boxes; Kalman
            # predict keeps state/uncertainty consistent for the next
            # keyframe's correction step.
            
            manager.step_predict_only(idx)
            
            # Update KCF trackers
            for tid, tracker in list(kcf_trackers.items()):
                prev_rx, prev_ry = None, None
                for t in manager.player_tracks:
                    if t.track_id == tid:
                        prev_rx, prev_ry = t.current_xy()
                        break
                        
                success, bbox = tracker.update(frame_bgr)
                if success:
                    x_min, y_min, w, h = bbox
                    frame_width = frame_bgr.shape[1]
                    frame_height = frame_bgr.shape[0]
                    
                    x_min = max(0, min(int(x_min), frame_width - 1))
                    y_min = max(0, min(int(y_min), frame_height - 1))
                    w = max(5, min(int(w), frame_width - x_min))
                    h = max(5, min(int(h), frame_height - y_min))
                    
                    # Recover feet coordinate from tracked bounding box
                    px = int(x_min + w / 2)
                    py = int(y_min + h)
                    
                    displacement = 0.0
                    if prev_rx is not None:
                        prev_px, prev_py = geo.ruler_to_orig_px(prev_rx, prev_ry)
                        displacement = math.hypot(px - prev_px, py - prev_py)
                    
                    if displacement > 35.0:
                        success = False
                    elif 0 < px < frame_bgr.shape[1] - 1 and 0 < py < frame_bgr.shape[0] - 1:
                        # Find the corresponding track and correct its kalman filter
                        for t in manager.player_tracks:
                            if t.track_id == tid:
                                rx, ry = geo.orig_px_to_ruler(px, py)
                                t.correct(rx, ry)
                                break
                                
                if not success:
                    # KCF lost the track or moved too fast, rely purely on Kalman prediction
                    del kcf_trackers[tid]
            
        # Apply temporal majority-vote smoothing to each track's team label
        team_classifier.smooth_tracks(manager.player_tracks)
        snapshot = manager.snapshot()
        out_frames.append(render_frame(frame_bgr, geo, snapshot))

    t_loop = time.time() - t_loop_start
    logger.info("Tracking+render loop: %.1fs for %d frames (%.3fs/frame avg)",
                t_loop, n_frames, t_loop / max(1, n_frames))

    t_write_start = time.time()
    write_video(output_path, out_frames, fps)
    t_write = time.time() - t_write_start
    logger.info("Video write: %.1fs", t_write)
    elapsed = time.time() - t0

    costs = [r.cost_usd for r in results.values() if r.cost_usd is not None]
    cost_is_partial = len(costs) < len(results)
    total_cost = sum(costs) if costs else None

    return RunStats(
        n_frames=n_frames,
        n_keyframes=len(keyframe_idxs),
        n_keyframes_failed=n_failed,
        elapsed_s=elapsed,
        total_cost_usd=total_cost,
        cost_is_partial=cost_is_partial,
    )