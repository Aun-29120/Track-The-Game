"""
Offline temporal linker and interpolator.
Takes asynchronous VLM keyframe detections and builds a dense 
dictionary mapping every frame to its detections.
"""
from __future__ import annotations
import math
from typing import Optional

from config import CFG
from schema import FrameDetections, PlayerDetection, BallDetection
from scipy.optimize import linear_sum_assignment
import numpy as np


class Track:
    def __init__(self, track_id: int):
        self.track_id = track_id
        # map frame_idx -> dict
        self.observations = {}
        self.team_votes = []
        self.last_jersey_number = None

    def add_obs(self, frame_idx: int, box: list[float], team: str, jersey_number: Optional[str] = None):
        self.observations[frame_idx] = box
        if team.lower() not in ["gray", "unknown"]: 
            self.team_votes.append(team)
        if jersey_number is not None:
            self.last_jersey_number = jersey_number

    def get_final_team(self) -> str:
        if not self.team_votes:
            return "Gray"
        from collections import Counter
        return Counter(self.team_votes).most_common(1)[0][0]


def box_distance(box1: list[float], box2: list[float]) -> float:
    y1, x1, y2, x2 = box1
    y3, x3, y4, x4 = box2
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
    return math.hypot(cx1 - cx2, cy1 - cy2)


def interpolate_coordinate(c1: float, c2: float, f1: int, f2: int, f: int) -> float:
    if f1 == f2:
        return c1
    return c1 + (c2 - c1) * (f - f1) / (f2 - f1)


def interpolate_box(box1: list[float], box2: list[float], f1: int, f2: int, f: int) -> list[float]:
    return [
        interpolate_coordinate(box1[0], box2[0], f1, f2, f),
        interpolate_coordinate(box1[1], box2[1], f1, f2, f),
        interpolate_coordinate(box1[2], box2[2], f1, f2, f),
        interpolate_coordinate(box1[3], box2[3], f1, f2, f),
    ]


def interpolate_pipeline(
    kf_results: dict[int, FrameDetections], 
    total_frames: int, 
    geo,
) -> dict[int, dict]:
    """
    1. Match players across keyframes using distance.
    2. Drop tracks that only exist in 1 keyframe (hallucinations).
    3. Linearly interpolate the remaining tracks.
    4. Interpolate the ball.
    """
    sorted_kfs = sorted(list(kf_results.keys()))
    if not sorted_kfs:
        return {i: {"players": [], "ball": None} for i in range(total_frames)}

    active_tracks = []
    finished_tracks = []
    next_track_id = 0

    # Match players
    for kf in sorted_kfs:
        raw_dets = kf_results[kf].detections
        if not raw_dets:
            continue
            
        pitch_boundary = kf_results[kf].pitch_boundary
        dets = []
        for d in raw_dets:
            y2 = d.box_2d[2]
            x_center = (d.box_2d[1] + d.box_2d[3]) / 2.0
            
            boundary_y = None
            if pitch_boundary is not None and len(pitch_boundary) > 0:
                if len(pitch_boundary) == 1:
                    boundary_y = pitch_boundary[0][0]
                else:
                    if x_center <= pitch_boundary[0][1]:
                        boundary_y = pitch_boundary[0][0]
                    elif x_center >= pitch_boundary[-1][1]:
                        boundary_y = pitch_boundary[-1][0]
                    else:
                        for i in range(len(pitch_boundary) - 1):
                            pt1 = pitch_boundary[i]
                            pt2 = pitch_boundary[i+1]
                            if pt1[1] <= x_center <= pt2[1]:
                                if pt2[1] == pt1[1]:
                                    boundary_y = pt1[0]
                                else:
                                    t = (x_center - pt1[1]) / (pt2[1] - pt1[1])
                                    boundary_y = pt1[0] + t * (pt2[0] - pt1[0])
                                break
            
            if boundary_y is not None and y2 < boundary_y:
                continue
            dets.append(d)
            
        if not dets:
            continue
            
        # --- Task 1: Intra-keyframe Deduplication ---
        deduped = []
        for d in dets:
            is_dup = False
            for i, kept_d in enumerate(deduped):
                if box_distance(d.box_2d, kept_d.box_2d) < CFG.duplicate_detection_distance_norm:
                    is_dup = True
                    # Keep the one with a jersey number if there is a conflict
                    if d.jersey_number is not None and kept_d.jersey_number is None:
                        deduped[i] = d
                    break
            if not is_dup:
                deduped.append(d)
                
        unmatched_dets = list(deduped)
        
        # Global optimal matching using linear_sum_assignment
        if active_tracks and unmatched_dets:
            cost_matrix = np.zeros((len(active_tracks), len(unmatched_dets)))
            for i, t in enumerate(active_tracks):
                last_kf = max(t.observations.keys())
                last_box = t.observations[last_kf]
                for j, d in enumerate(unmatched_dets):
                    dist = box_distance(last_box, d.box_2d)
                    
                    # Cost starts as spatial distance
                    cost = dist
                    
                    # Apply team consistency signal
                    t_team = t.get_final_team()
                    if t_team not in ["Gray", "Unknown"] and d.team not in ["Gray", "Unknown"]:
                        if t_team != d.team:
                            # If distance is very small, it's almost certainly the same player misclassified.
                            # Don't apply the penalty to prevent the track from being stolen by a further-away teammate.
                            if dist > 30.0:
                                cost += CFG.team_mismatch_penalty
                            
                    # Apply identity signals (jersey number)
                    if t.last_jersey_number is not None and d.jersey_number is not None:
                        if t.last_jersey_number == d.jersey_number:
                            cost -= CFG.jersey_match_reward
                        else:
                            cost += CFG.jersey_mismatch_penalty
                            
                    cost_matrix[i, j] = cost
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Apply assignments, gating on max_player_jump_norm AFTER assignment
            assigned_det_indices = set()
            for r, c in zip(row_ind, col_ind):
                t = active_tracks[r]
                d = unmatched_dets[c]
                last_kf = max(t.observations.keys())
                dist = box_distance(t.observations[last_kf], d.box_2d)
                
                gap_frames = kf - last_kf
                multiplier = min(gap_frames / CFG.keyframe_interval, 3.0)
                scaled_max_jump = CFG.max_player_jump_norm * multiplier
                
                print(f'dist={dist}, max={scaled_max_jump}, t_team={t.get_final_team()}')
                if dist < scaled_max_jump:
                    t.add_obs(kf, d.box_2d, d.team, d.jersey_number)
                    assigned_det_indices.add(c)
            
            unmatched_dets = [d for j, d in enumerate(unmatched_dets) if j not in assigned_det_indices]

        # Spawn new tracks for remaining detections
        for d in unmatched_dets:
            new_t = Track(next_track_id)
            next_track_id += 1
            new_t.add_obs(kf, d.box_2d, d.team, d.jersey_number)
            active_tracks.append(new_t)
            
    # Combine all tracks
    all_tracks = active_tracks + finished_tracks
    
    # Drop tracks with only 1 observation (temporal outlier rejection)
    valid_tracks = [t for t in all_tracks if len(t.observations) > 1]
    
    # Drop stationary tracks (e.g. ad boards/photographers)
    filtered_tracks = []
    for t in valid_tracks:
        obs_frames = sorted(list(t.observations.keys()))
        first_box = t.observations[obs_frames[0]]
        last_box = t.observations[obs_frames[-1]]
        if box_distance(first_box, last_box) >= 10.0:
            filtered_tracks.append(t)
    valid_tracks = filtered_tracks
    
    # Interpolate Player Tracks
    dense_map = {i: {"players": [], "ball": None} for i in range(total_frames)}
    
    for t in valid_tracks:
        obs_frames = sorted(list(t.observations.keys()))
        for i in range(len(obs_frames) - 1):
            f1 = obs_frames[i]
            f2 = obs_frames[i+1]
            box1 = t.observations[f1]
            box2 = t.observations[f2]
            
            gap_frames = f2 - f1
            if gap_frames > CFG.max_interpolation_gap_frames:
                # Too large a gap, treat as disconnected and only render f1
                y1, x1, y2, x2 = box1
                cx = (x1 + x2) / 2
                px_xmin, px_ymin = geo.ruler_to_orig_px(x1, y1)
                px_xmax, px_ymax = geo.ruler_to_orig_px(x2, y2)
                bbox_px = (px_xmin, px_ymin, px_xmax - px_xmin, px_ymax - px_ymin)
                dense_map[f1]["players"].append({
                    "track_id": t.track_id,
                    "team": t.get_final_team(),
                    "xy": (cx, y2),
                    "bbox_px": bbox_px,
                })
                continue

            # Linear interpolation for all frames between f1 and f2 (exclusive of f2, inclusive of f1)
            for f in range(f1, f2):
                interp_box = interpolate_box(box1, box2, f1, f2, f)
                # Compute foot coordinate and width for renderer
                y1, x1, y2, x2 = interp_box
                cx = (x1 + x2) / 2
                w = x2 - x1
                
                # We need bounding box in pixels for rendering
                px_xmin, px_ymin = geo.ruler_to_orig_px(x1, y1)
                px_xmax, px_ymax = geo.ruler_to_orig_px(x2, y2)
                bbox_px = (px_xmin, px_ymin, px_xmax - px_xmin, px_ymax - px_ymin)
                
                dense_map[f]["players"].append({
                    "track_id": t.track_id,
                    "team": t.get_final_team(),
                    "xy": (cx, y2), # Ruler coordinates of foot
                    "bbox_px": bbox_px,
                })
        
        # Also add the final observation explicitly for the very last keyframe
        last_f = obs_frames[-1]
        if last_f < total_frames:
            box = t.observations[last_f]
            y1, x1, y2, x2 = box
            cx = (x1 + x2) / 2
            w = x2 - x1
            px_xmin, px_ymin = geo.ruler_to_orig_px(x1, y1)
            px_xmax, px_ymax = geo.ruler_to_orig_px(x2, y2)
            bbox_px = (px_xmin, px_ymin, px_xmax - px_xmin, px_ymax - px_ymin)
            
            dense_map[last_f]["players"].append({
                "track_id": t.track_id,
                "team": t.get_final_team(),
                "xy": (cx, y2),
                "bbox_px": bbox_px,
            })

    # Interpolate Ball
    ball_obs = {}
    for kf in sorted_kfs:
        b = kf_results[kf].ball
        if b is not None:
            ball_obs[kf] = b.box_2d
            
    ball_frames = sorted(list(ball_obs.keys()))
    
    # --- Ball Diagnostics ---
    if CFG.debug_logging:
        n_kfs = len(sorted_kfs)
        n_ball_kfs = len(ball_frames)
        ball_detect_frac = n_ball_kfs / max(1, n_kfs)
        print(f"\n[DIAGNOSTIC] Ball detected in {n_ball_kfs}/{n_kfs} keyframes ({ball_detect_frac:.1%})")
    
        jump_distances = []
        if ball_frames:
            for i in range(len(ball_frames) - 1):
                f1 = ball_frames[i]
                f2 = ball_frames[i+1]
                if f2 - f1 <= CFG.keyframe_interval: # Only look at consecutive keyframes in either scale
                    box1 = ball_obs[f1]
                    box2 = ball_obs[f2]
                    dist = box_distance(box1, box2)
                    jump_distances.append(dist)
        if jump_distances:
            avg_jump = sum(jump_distances) / len(jump_distances)
            max_jump = max(jump_distances)
            over_threshold = sum(1 for d in jump_distances if d > CFG.max_ball_jump_norm)
            print(f"[DIAGNOSTIC] Consecutive KF jump distances: avg={avg_jump:.1f}, max={max_jump:.1f}")
            print(f"[DIAGNOSTIC] Jumps > CFG.max_ball_jump_norm ({CFG.max_ball_jump_norm}): {over_threshold}/{len(jump_distances)}")
        else:
            print("[DIAGNOSTIC] No consecutive ball detections to compute jump distances.")
        print("-" * 50 + "\n")
    # ------------------------

    if ball_frames:
        for i in range(len(ball_frames) - 1):
            f1 = ball_frames[i]
            f2 = ball_frames[i+1]
            box1 = ball_obs[f1]
            box2 = ball_obs[f2]
            
            gap_frames = f2 - f1
            multiplier = min(gap_frames / CFG.keyframe_interval, 3.0)
            scaled_max_ball_jump = CFG.max_ball_jump_norm * multiplier

            # Sanity check: if ball jumped too far OR gap is too large, do not interpolate between them
            dist = box_distance(box1, box2)
            if dist > scaled_max_ball_jump or gap_frames > CFG.max_interpolation_gap_frames:
                # Just hold the ball at f1 for a bit, don't interpolate all the way to f2
                y1, x1, y2, x2 = box1
                dense_map[f1]["ball"] = {"xy": ((x1 + x2) / 2, (y1 + y2) / 2)}
                # The jump is too big, treat them as disconnected segments
                continue
                
            for f in range(f1, f2):
                interp_box = interpolate_box(box1, box2, f1, f2, f)
                y1, x1, y2, x2 = interp_box
                dense_map[f]["ball"] = {"xy": ((x1 + x2) / 2, (y1 + y2) / 2)}
                
        # Last ball frame
        last_bf = ball_frames[-1]
        if last_bf < total_frames:
            box = ball_obs[last_bf]
            y1, x1, y2, x2 = box
            dense_map[last_bf]["ball"] = {"xy": ((x1 + x2) / 2, (y1 + y2) / 2)}

    return dense_map
