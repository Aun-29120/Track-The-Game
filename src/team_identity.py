"""
Team classification using 3D CIELAB features.

1. Dynamic Color Profiling
   Accumulates mean L*, A*, B* features from the upper torso of all tracked players
   over the first 45 frames. Applies a strict green pitch mask to exclude turf.
   Runs K-Means (k=2) on these accumulated features to find Team 0 and Team 1.

2. ID-Locked Team Assignment
   Tracks accumulate team votes over consecutive keyframes. After 5 consecutive
   votes for the same team, the assignment is permanently locked to prevent
   sliding/falling players from flipping colors.

3. Referee Outlier Rejection
   If a detection's minimum ΔE (CIELAB Euclidean distance) to both team centroids
   exceeds 35, the track is classified as "unknown" (referee/outlier) and its
   marker is suppressed in the renderer.

4. Temporal Smoothing (Per-Track Majority Vote)
   Before locking, maintains a rolling 15-frame window of team votes for each
   track_id to prevent flickering.
"""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from typing import Optional

import cv2
import numpy as np

from config import CFG
from schema import FrameDetections

VOTE_WINDOW = 15          # frames of history per track for majority vote
ACCUMULATION_FRAMES = 45  # wait this many frames before calibrating k-means
LOCK_STREAK = 5           # consecutive same-team votes required to lock
REFEREE_DELTA_E = 35.0    # ΔE threshold: if min(dist0, dist1) > this, classify as referee


class TeamClassifier:
    """One instance per clip. Accumulates features over 45 frames, 
    calibrates k-means centers, and applies temporal smoothing with
    ID-locked assignments and referee outlier rejection."""

    def __init__(self) -> None:
        self._calibrated: bool = False
        self._kmeans_centers: Optional[np.ndarray] = None
        self._vote_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))
        
        self._accumulated_hists: list[np.ndarray] = []
        
        # We need a fallback track-to-team mapping for frames 0-44
        self._early_centers: Optional[np.ndarray] = None
        
        # ID-locked team assignments: once a track accumulates LOCK_STREAK
        # consecutive identical votes, its team is permanently frozen here.
        self._locked_teams: dict[int, str] = {}
        
        # Consecutive vote streak tracker (track_id -> (team, count))
        self._vote_streak: dict[int, tuple[str, int]] = {}

    # Minimum Y coordinate (in pixels) for a valid on-pitch detection.
    PITCH_Y_MIN_PX = 150.0

    def extract_jersey_histogram(self, frame_bgr: np.ndarray, bbox_px: tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Spatial filter → upper torso crop → grass mask → mean L*A*B*."""
        x, y, w, h = bbox_px
        fh, fw = frame_bgr.shape[:2]
        ymax = y + h

        # 1. Spatial filtering: reject ghost trackers above the pitch boundary
        if ymax < self.PITCH_Y_MIN_PX:
            return None

        # 2. Upper Torso Crop
        crop_w = int(w * 0.5)
        crop_h = int(h * 0.5)
        crop_x = int(x + (w - crop_w) / 2)
        crop_y = int(y + (h - crop_h) / 2)

        # Clamp to frame bounds
        crop_x = max(0, min(crop_x, fw - crop_w))
        crop_y = max(0, min(crop_y, fh - crop_h))
        crop_x2 = min(fw, crop_x + crop_w)
        crop_y2 = min(fh, crop_y + crop_h)

        if crop_x2 <= crop_x or crop_y2 <= crop_y:
            return None

        torso_bgr = frame_bgr[crop_y:crop_y2, crop_x:crop_x2]

        # 3. HSV masking: filter out green grass and dark shadows (Hue: 35-85)
        torso_hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(torso_hsv, (35, 40, 40), (85, 255, 255))
        valid_mask = cv2.bitwise_not(grass_mask)

        # Fall back to unmasked if less than 10 valid pixels remain
        if np.count_nonzero(valid_mask) < 10:
            valid_mask = np.ones(valid_mask.shape, dtype=np.uint8) * 255

        # 4. CIELAB conversion: mean L*, A*, B* for K-Means clustering
        torso_lab = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2LAB)
        mean_lab = list(cv2.mean(torso_lab, mask=valid_mask)[:3])
        
        return np.array(mean_lab, dtype=np.float32)

    def record_track_histogram(self, track_id: int, hist: Optional[np.ndarray], frame_idx: int) -> None:
        if hist is None:
            return
            
        if not self._calibrated:
            self._accumulated_hists.append(hist)
            if frame_idx >= ACCUMULATION_FRAMES:
                self._calibrate()
            elif frame_idx % 5 == 0 and len(self._accumulated_hists) >= 4:
                # Update early centers for display purposes before frame 45
                self._update_early_centers()

    def _sort_centers_by_lightness(self, centers: np.ndarray) -> np.ndarray:
        """Sorts centers so centers[0] is High L* (White/Cyan marker) 
        and centers[1] is Low L* (Navy/Red marker)."""
        # centers has shape (2, 3), where column 0 is L*
        if centers[0, 0] < centers[1, 0]:
            return np.array([centers[1], centers[0]])
        return centers

    def _update_early_centers(self) -> None:
        if len(self._accumulated_hists) < 4:
            return
        data = np.array(self._accumulated_hists, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        self._early_centers = self._sort_centers_by_lightness(centers)

    def _calibrate(self) -> None:
        if len(self._accumulated_hists) < 2:
            return
            
        data = np.array(self._accumulated_hists, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        self._kmeans_centers = self._sort_centers_by_lightness(centers)
        self._calibrated = True

    def smooth_tracks(self, player_tracks) -> None:
        """Assign team via ΔE distance, referee outlier rejection, and ID-locked assignment."""
        centers = self._kmeans_centers if self._calibrated else self._early_centers
        
        for t in player_tracks:
            tid = t.track_id
            
            # If this track is already permanently locked, apply immediately and skip
            if tid in self._locked_teams:
                t.team = self._locked_teams[tid]
                continue
            
            if centers is not None and t.histogram is not None:
                dist0 = np.linalg.norm(t.histogram - centers[0])
                dist1 = np.linalg.norm(t.histogram - centers[1])
                min_dist = min(dist0, dist1)
                
                # Referee outlier rejection: if too far from BOTH centroids
                if min_dist > REFEREE_DELTA_E:
                    team_vote = "unknown"
                else:
                    team_vote = "A" if dist0 < dist1 else "B"
                
                self._vote_history[tid].append(team_vote)
                
                # Track consecutive streak for locking
                prev_team, prev_count = self._vote_streak.get(tid, (None, 0))
                if team_vote == prev_team:
                    new_count = prev_count + 1
                else:
                    new_count = 1
                self._vote_streak[tid] = (team_vote, new_count)
                
                # Lock if streak hits threshold
                if new_count >= LOCK_STREAK:
                    self._locked_teams[tid] = team_vote
                    t.team = team_vote
                    continue
                    
            elif centers is None and len(self._vote_history[tid]) == 0:
                self._vote_history[tid].append("A")  # Arbitrary fallback
                
            if len(self._vote_history[tid]) > 0:
                # If not yet locked, use rolling majority vote to assign the display team
                counts = Counter(self._vote_history[tid])
                t.team = counts.most_common(1)[0][0]