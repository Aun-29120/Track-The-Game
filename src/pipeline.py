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
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from config import CFG
from ruler_overlay import add_ruler, RulerGeometry
from vlm_client import detect_all_keyframes
from tracker import TrackManager
from team_identity import TeamIdentityResolver
from optical_flow import propagate_points, to_gray
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
    team_resolver = TeamIdentityResolver()
    out_frames: list[np.ndarray] = []
    prev_gray = None
    # per-track pixel points carried by optical flow between keyframes
    flow_points: dict[int, tuple[float, float]] = {}

    t_loop_start = time.time()

    for idx in range(n_frames):
        frame_bgr = frames[idx]
        curr_gray = to_gray(frame_bgr)

        if idx in keyframe_idxs and results[idx].detections is not None:
            detections = team_resolver.resolve(results[idx].detections, frame_bgr, geo)
            manager.step_keyframe(idx, frame_bgr, geo, detections)
            # reset flow points to the freshly corrected Kalman positions
            flow_points = {
                t.track_id: geo.ruler_to_orig_px(*t.current_xy())
                for t in manager.player_tracks
            }
            if manager.ball_track is not None:
                flow_points[0] = geo.ruler_to_orig_px(*manager.ball_track.current_xy())

        elif idx in keyframe_idxs and results[idx].detections is None:
            # failed keyframe -- fall back to prediction only, same as a non-keyframe
            manager.step_predict_only(idx)

        else:
            # non-keyframe: optical flow carries points forward; Kalman
            # predict keeps state/uncertainty consistent for the next
            # keyframe's correction step.
            if prev_gray is not None and flow_points:
                ids = list(flow_points.keys())
                pts = np.array([flow_points[i] for i in ids], dtype=np.float32)
                new_pts, status = propagate_points(prev_gray, curr_gray, pts)
                for i, tid in enumerate(ids):
                    if status[i]:
                        flow_points[tid] = (float(new_pts[i][0]), float(new_pts[i][1]))
                    # if lost, we simply stop updating this point from flow;
                    # Kalman's own prediction (below) becomes the fallback.

            manager.step_predict_only(idx)
            # nudge Kalman state toward the optical-flow-tracked position where available,
            # so the two signals don't diverge silently between corrections.
            for t in manager.player_tracks:
                if t.track_id in flow_points and geo is not None:
                    rx, ry = geo.orig_px_to_ruler(*flow_points[t.track_id])
                    t.kf.statePost[0, 0] = rx
                    t.kf.statePost[1, 0] = ry

        snapshot = manager.snapshot()
        out_frames.append(render_frame(frame_bgr, geo, snapshot))
        prev_gray = curr_gray

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