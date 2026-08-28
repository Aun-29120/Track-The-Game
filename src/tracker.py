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
from appearance import sample_torso_histogram, histogram_similarity


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)

# Detections/tracks are represented as points; for IoU purposes we treat
# each point as the center of a small fixed-size box. This is a stand-in
# for real bounding boxes since our detections are point-based (feet
# position), but it gives IoU the same "closer AND same-size" behavior
# instead of raw distance, and it degrades gracefully to "distance-like"
# when boxes don't overlap at all.
BOX_HALF_SIZE_NORM = 0.35  # in ruler units


def _box_from_point(x: float, y: float, half: float = BOX_HALF_SIZE_NORM) -> tuple[float, float, float, float]:
    return (x - half, y - half, x + half, y + half)


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _new_kalman() -> cv2.KalmanFilter:
    """State: [x, y, vx, vy]. Measurement: [x, y]. Constant-velocity model,
    as discussed in the thinking cap -- known to lag on sharp direction
    changes, flagged there as an accepted limitation."""
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
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * CFG.kalman_process_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * CFG.kalman_measurement_noise
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


@dataclass
class Track:
    track_id: int
    team: str  # "A" / "B" / "ball"
    kf: cv2.KalmanFilter
    histogram: Optional[np.ndarray] = None
    age_since_match: int = 0
    alive: bool = True
    history: list[tuple[int, float, float]] = field(default_factory=list)  # (frame_idx, x, y)

    def predict(self) -> tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, x: float, y: float) -> None:
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)
        self.age_since_match = 0

    def current_xy(self) -> tuple[float, float]:
        s = self.kf.statePost
        return float(s[0, 0]), float(s[1, 0])


class TrackManager:
    """Owns all player tracks (+ a single ball track). One instance per
    clip. Call `step_keyframe` at VLM keyframes and `step_predict_only`
    on the in-between frames covered by optical flow."""

    def __init__(self) -> None:
        self._next_id = itertools.count(1)
        self.player_tracks: list[Track] = []
        self.ball_track: Optional[Track] = None

    # ---- keyframe update: predict, assign, correct ----

    def step_keyframe(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        geo,  # RulerGeometry, for converting ruler coords -> pixel coords for histogram sampling
        detections,  # schema.FrameDetections
    ) -> None:
        # 1. predict every existing track forward
        preds = {t.track_id: t.predict() for t in self.player_tracks}

        # 2. dedup detections within this single keyframe's response --
        #    two same-team detections sitting on top of each other are
        #    almost always one player double-counted by the VLM, not two
        #    players. Left unfiltered, both would either fight over the
        #    same track in the Hungarian assignment or -- worse -- one
        #    matches and the other spawns a duplicate ghost track.
        dets = self._dedup_detections(detections.players)
        n_tracks, n_dets = len(self.player_tracks), len(dets)
        frame_h_px = frame_bgr.shape[0]
        det_hists = []
        for d in dets:
            px, py = geo.ruler_to_orig_px(d.x, d.y)
            det_hists.append(sample_torso_histogram(frame_bgr, px, py, frame_h_px))

        if n_tracks and n_dets:
            cost = np.ones((n_tracks, n_dets), dtype=np.float32)
            for i, t in enumerate(self.player_tracks):
                px, py = preds[t.track_id]
                box_t = _box_from_point(px, py)
                for j, d in enumerate(dets):
                    box_d = _box_from_point(d.x, d.y)
                    iou_cost = 1.0 - _iou(box_t, box_d)
                    if t.team != d.team:
                        # cross-team match should basically never happen;
                        # penalize heavily rather than forbid outright
                        # (VLM team classification can itself be wrong).
                        iou_cost = min(1.0, iou_cost + 0.5)
                    hist_sim = histogram_similarity(t.histogram, det_hists[j]) if t.histogram is not None else 0.5
                    appearance_cost = 1.0 - hist_sim
                    cost[i, j] = (
                        CFG.position_cost_weight * iou_cost
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
            d = dets[c]
            # Teleportation gate: even a cheap-enough IoU/appearance match
            # can be flat-out wrong if the VLM hallucinated a position far
            # from where this track actually was. A real player can't
            # cover more than max_player_jump_norm ruler-units between two
            # keyframes, so anything past that is treated as noise -- the
            # track just ages instead of getting its identity hijacked and
            # teleported across the pitch.
            px, py = preds[t.track_id]
            if _dist(px, py, d.x, d.y) > CFG.max_player_jump_norm:
                continue
            t.correct(d.x, d.y)
            t.histogram = det_hists[c]
            t.history.append((frame_idx, d.x, d.y))
            matched_track_ids.add(t.track_id)
            matched_det_idxs.add(c)

        # 3. age out unmatched tracks; spawn tracks for unmatched detections
        for t in self.player_tracks:
            if t.track_id not in matched_track_ids:
                t.age_since_match += 1
        self.player_tracks = [
            t for t in self.player_tracks if t.age_since_match <= CFG.max_track_age_frames
        ]
        for j, d in enumerate(dets):
            if j in matched_det_idxs:
                continue
            # Spawn sanity gate: an unmatched detection sitting right on
            # top of a track that just failed to match (e.g. rejected by
            # the teleport gate above, or lost this keyframe to appearance
            # noise) is almost always that same player, not a new one.
            # Without this, a single noisy keyframe can spawn a duplicate
            # track next to an existing player, and the two flicker/fight
            # for the marker on every subsequent keyframe.
            if self._too_close_to_existing(d):
                continue
            new_track = self._spawn_player(frame_idx, d, det_hists[j])
            self.player_tracks.append(new_track)

        # 4. ball -- single object, no assignment problem, just correct-or-coast
        if detections.ball is not None and detections.ball.visible:
            if self.ball_track is None:
                self.ball_track = self._spawn_ball(frame_idx, detections.ball)
            else:
                bpx, bpy = self.ball_track.predict()
                # Same teleportation gate as players, just with a looser
                # threshold since the ball legitimately moves faster.
                if _dist(bpx, bpy, detections.ball.x, detections.ball.y) <= CFG.max_ball_jump_norm:
                    self.ball_track.correct(detections.ball.x, detections.ball.y)
                    self.ball_track.history.append((frame_idx, detections.ball.x, detections.ball.y))
                # else: implausible jump -- predict() above already advanced
                # the filter, so it simply coasts on prediction this keyframe.
        elif self.ball_track is not None:
            self.ball_track.predict()  # coast on prediction alone this keyframe

    # ---- non-keyframe update: pure prediction (paired with optical flow in pipeline.py) ----

    def step_predict_only(self, frame_idx: int) -> None:
        for t in self.player_tracks:
            x, y = t.predict()
            t.history.append((frame_idx, x, y))
        if self.ball_track is not None:
            x, y = self.ball_track.predict()
            self.ball_track.history.append((frame_idx, x, y))

    @staticmethod
    def _dedup_detections(dets: list) -> list:
        """Drops near-duplicate detections of the same team within one
        keyframe's response, keeping the first of each cluster. Compares
        same-team pairs only -- two different-team detections standing
        close together is normal (players marking each other), not a
        duplicate."""
        kept: list = []
        for d in dets:
            if any(
                d.team == k.team and _dist(d.x, d.y, k.x, k.y) <= CFG.duplicate_detection_distance_norm
                for k in kept
            ):
                continue
            kept.append(d)
        return kept

    def _too_close_to_existing(self, det) -> bool:
        """True if `det` lands close enough to an existing (still-alive)
        track that it should be treated as noise/duplicate rather than a
        brand-new player -- e.g. a track that narrowly missed matching
        this keyframe (appearance blip, or rejected by the teleport gate)
        shouldn't get a duplicate spawned right next to it."""
        for t in self.player_tracks:
            tx, ty = t.current_xy()
            if _dist(tx, ty, det.x, det.y) <= CFG.min_new_track_spawn_distance_norm:
                return True
        return False

    def _spawn_player(self, frame_idx: int, det, hist: np.ndarray) -> Track:
        kf = _new_kalman()
        kf.statePost = np.array([[det.x], [det.y], [0], [0]], dtype=np.float32)
        t = Track(track_id=next(self._next_id), team=det.team, kf=kf, histogram=hist)
        t.history.append((frame_idx, det.x, det.y))
        return t

    def _spawn_ball(self, frame_idx: int, ball) -> Track:
        kf = _new_kalman()
        kf.statePost = np.array([[ball.x], [ball.y], [0], [0]], dtype=np.float32)
        t = Track(track_id=0, team="ball", kf=kf)
        t.history.append((frame_idx, ball.x, ball.y))
        return t

    def snapshot(self) -> dict:
        """Current per-track ruler-space positions, for rendering / ball-possession logic."""
        return {
            "players": [
                {"track_id": t.track_id, "team": t.team, "xy": t.current_xy()}
                for t in self.player_tracks
            ],
            "ball": {"xy": self.ball_track.current_xy()} if self.ball_track else None,
        }