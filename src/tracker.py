"""
The identity-persistence layer. This is the actual answer to "how do you
keep an identity attached across frames that were never aware of each
other" -- the VLM never sees more than one frame at a time; everything
here is what stitches those independent detections into a continuous
per-player track.

Two things happen here, kept deliberately separate:
  1. Kalman filter (per track) -- predicts a track's position forward
     every frame, corrects itself against a matched VLM detection at
     keyframes, weighted by its own uncertainty.
  2. Hungarian assignment (per keyframe) -- decides WHICH detection
     corresponds to WHICH existing track, using IoU-of-small-boxes as the
     primary cost and appearance (color histogram) as a tiebreaker.
"""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from config import CFG
from appearance import histogram_similarity


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _new_kalman() -> cv2.KalmanFilter:
    """State: [x, y, vx, vy]. Measurement: [x, y]. Constant-velocity model."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )
    
    process_noise = CFG.kalman_process_noise
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
    
    measurement_noise = CFG.kalman_measurement_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
    
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


@dataclass
class Track:
    track_id: int
    kf: cv2.KalmanFilter
    histogram: Optional[np.ndarray] = None
    age_since_match: int = 0
    hit_streak: int = 1  # number of consecutive matches, for confirming tracks
    alive: bool = True
    history: list[tuple[int, float, float]] = field(default_factory=list)  # (frame_idx, x, y)
    last_bbox_px: Optional[tuple[float, float, float, float]] = None # [x, y, w, h]
    team: str = "A"

    def predict(self) -> tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, x: float, y: float) -> None:
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)
        self.age_since_match = 0
        self.hit_streak += 1

    def current_xy(self) -> tuple[float, float]:
        s = self.kf.statePost
        return float(s[0, 0]), float(s[1, 0])


class TrackManager:
    """Owns all player tracks. One instance per clip. 
    Call `step_keyframe` at VLM keyframes and `step_predict_only`
    on the in-between frames covered by optical flow."""

    def __init__(self) -> None:
        self._next_id = itertools.count(1)
        self.player_tracks: list[Track] = []
        self.ball_position: Optional[tuple[float, float]] = None  # ruler-space (rx, ry)
        self.ball_miss_count: int = 0

    # ---- keyframe update: predict, assign, correct ----

    def step_keyframe(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        geo,  # RulerGeometry
        detections,  # schema.FrameDetections
        team_classifier,  # TeamClassifier instance to compute torso histograms
    ) -> None:
        # 0. Extract ball position (bypasses all player logic)
        ball_updated = False
        if detections.ball is not None:
            bymin, bxmin, bymax, bxmax = detections.ball.box_2d
            px_xmin, px_ymin = geo.ruler_to_orig_px(bxmin, bymin)
            px_xmax, px_ymax = geo.ruler_to_orig_px(bxmax, bymax)
            ball_cx = (px_xmin + px_xmax) / 2.0
            ball_cy = (px_ymin + px_ymax) / 2.0
            
            # Sanity check: reject if it jumps > 150px from previous position
            valid_jump = True
            if self.ball_position is not None:
                old_rx, old_ry = self.ball_position
                old_px, old_py = geo.ruler_to_orig_px(old_rx, old_ry)
                dist = ((old_px - ball_cx)**2 + (old_py - ball_cy)**2)**0.5
                if dist > 150.0:
                    valid_jump = False
            
            if valid_jump:
                self.ball_position = geo.orig_px_to_ruler(ball_cx, ball_cy)
                self.ball_miss_count = 0
                ball_updated = True
                
        if not ball_updated:
            self.ball_miss_count += 1
            if self.ball_miss_count >= 2:
                self.ball_position = None

        # 1. predict every existing track forward
        preds = {t.track_id: t.predict() for t in self.player_tracks}

        # 2. Extract bounding boxes and apply NMS
        raw_dets = []
        bboxes_px = []
        scores = []
        for d in detections.detections:
            r_ymin, r_xmin, r_ymax, r_xmax = d.box_2d
            px_xmin, px_ymin = geo.ruler_to_orig_px(r_xmin, r_ymin)
            px_xmax, px_ymax = geo.ruler_to_orig_px(r_xmax, r_ymax)
            
            w = max(1, px_xmax - px_xmin)
            h = max(1, px_ymax - px_ymin)
            bboxes_px.append([px_xmin, px_ymin, w, h])
            scores.append(1.0)
            raw_dets.append((d.box_2d, [px_xmin, px_ymin, w, h]))

        if bboxes_px:
            indices = cv2.dnn.NMSBoxes(bboxes_px, scores, 0.5, 0.4)
            if len(indices) > 0:
                indices = indices.flatten()
            else:
                indices = []
        else:
            indices = []

        # kept detections after NMS
        dets = [raw_dets[i] for i in indices]

        n_tracks, n_dets = len(self.player_tracks), len(dets)
        
        det_hists = []
        det_feet = []
        for d_ruler, d_px in dets:
            x, y, w, h = d_px
            # Foot anchor for tracking
            foot_px_x = x + w / 2.0
            foot_px_y = y + h
            # Project foot back to ruler space for Kalman filter state
            foot_ruler_x, foot_ruler_y = geo.orig_px_to_ruler(foot_px_x, foot_px_y)
            det_feet.append((foot_ruler_x, foot_ruler_y))
            
            # Compute histogram for appearance matching
            hist = team_classifier.extract_jersey_histogram(frame_bgr, d_px)
            det_hists.append(hist)

        if n_tracks and n_dets:
            cost = np.ones((n_tracks, n_dets), dtype=np.float32) * 1e6
            for i, t in enumerate(self.player_tracks):
                track_ruler_x, track_ruler_y = preds[t.track_id]
                
                # estimate current bbox for IoU based on last known w,h and current predicted foot
                if t.last_bbox_px is not None:
                    _, _, tw, th = t.last_bbox_px
                    tpx_foot_x, tpx_foot_y = geo.ruler_to_orig_px(track_ruler_x, track_ruler_y)
                    track_bbox = (tpx_foot_x - tw/2, tpx_foot_y - th, tw, th)
                else:
                    track_bbox = (0, 0, 0, 0)
                
                for j, (d_ruler, d_px) in enumerate(dets):
                    foot_rx, foot_ry = det_feet[j]
                    dist = _dist(track_ruler_x, track_ruler_y, foot_rx, foot_ry)
                    
                    if dist > CFG.max_player_jump_norm:
                        continue
                    
                    if t.last_bbox_px is not None:
                        iou = _iou(track_bbox, tuple(d_px))
                        dist_cost = 1.0 - iou
                    else:
                        dist_cost = min(1.0, dist / CFG.max_player_jump_norm)
                        
                    hist_sim = histogram_similarity(t.histogram, det_hists[j]) if t.histogram is not None and det_hists[j] is not None else 0.5
                    appearance_cost = 1.0 - hist_sim
                    
                    # No team cost anymore, appearance is primary discriminator for identity
                    cost[i, j] = (
                        CFG.position_cost_weight * dist_cost
                        + CFG.appearance_cost_weight * appearance_cost
                    )
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = np.array([], dtype=int), np.array([], dtype=int)

        matched_track_ids: set[int] = set()
        matched_det_idxs: set[int] = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] > CFG.max_assignment_cost:
                continue  # too costly to trust -- treat as no-match
            t = self.player_tracks[r]
            
            foot_rx, foot_ry = det_feet[c]
            _, d_px = dets[c]
            
            px, py = preds[t.track_id]
            if _dist(px, py, foot_rx, foot_ry) > CFG.max_player_jump_norm:
                continue
            
            t.correct(foot_rx, foot_ry)
            if det_hists[c] is not None:
                t.histogram = det_hists[c]
            t.last_bbox_px = tuple(d_px)
            t.history.append((frame_idx, foot_rx, foot_ry))
            
            # Record histogram in the team classifier for k-means tracking
            team_classifier.record_track_histogram(t.track_id, det_hists[c], frame_idx)
            
            matched_track_ids.add(t.track_id)
            matched_det_idxs.add(c)

        # 3. age out unmatched tracks; spawn tracks for unmatched detections
        for t in self.player_tracks:
            if t.track_id not in matched_track_ids:
                t.age_since_match += 1
                t.hit_streak = 0
                
        # Prune ALL dead tracks (missed_keyframes > 0)
        alive_tracks = []
        for t in self.player_tracks:
            if t.age_since_match == 0:
                alive_tracks.append(t)
        self.player_tracks = alive_tracks
        
        for j, (d_ruler, d_px) in enumerate(dets):
            if j in matched_det_idxs:
                continue
            
            foot_rx, foot_ry = det_feet[j]
            if self._too_close_to_existing(foot_rx, foot_ry):
                continue
                
            new_track = self._spawn_player(frame_idx, foot_rx, foot_ry, det_hists[j], d_px)
            self.player_tracks.append(new_track)
            
            if det_hists[j] is not None:
                team_classifier.record_track_histogram(new_track.track_id, det_hists[j], frame_idx)

    # ---- non-keyframe update: pure prediction (paired with optical flow in pipeline.py) ----

    def step_predict_only(self, frame_idx: int) -> None:
        for t in self.player_tracks:
            x, y = t.predict()
            t.history.append((frame_idx, x, y))

    def _too_close_to_existing(self, foot_rx: float, foot_ry: float) -> bool:
        """True if `foot_rx, foot_ry` lands close enough to an existing (still-alive)
        track that it should be treated as noise/duplicate rather than a
        brand-new player."""
        for t in self.player_tracks:
            tx, ty = t.current_xy()
            if _dist(tx, ty, foot_rx, foot_ry) <= CFG.min_new_track_spawn_distance_norm:
                return True
        return False

    def _spawn_player(self, frame_idx: int, rx: float, ry: float, hist: Optional[np.ndarray], bbox_px: tuple) -> Track:
        kf = _new_kalman()
        kf.statePost = np.array([[rx], [ry], [0], [0]], dtype=np.float32)
        t = Track(track_id=next(self._next_id), kf=kf, histogram=hist, last_bbox_px=tuple(bbox_px))
        t.history.append((frame_idx, rx, ry))
        return t

    def snapshot(self) -> dict:
        """Current per-track ruler-space positions, for rendering."""
        ball_dict = {"xy": self.ball_position} if self.ball_position else None
        return {
            "players": [
                {"track_id": t.track_id, "xy": t.current_xy(), "team": t.team, "bbox_px": t.last_bbox_px}
                for t in self.player_tracks if (t.hit_streak >= 2 or t.age_since_match == 0)
            ],
            "ball": ball_dict,
        }