"""
Resolves the VLM's frame-local "A"/"B" team labels into a persistent
global team identity across independently-called keyframes.

Every keyframe is a fresh, isolated VLM call with no memory of the
previous one (see vlm_client.py / schema.py docstrings). Its "A"/"B"
labels are only guaranteed to be *internally* consistent within that one
call -- nothing stops keyframe 2 calling the teal kit "B" when keyframe 1
called it "A". appearance.py's docstring already flags this ("the VLM's
own 'A'/'B' labels are frame-local and can't be trusted to mean the same
team across independent calls"), and config.py already has a knob for the
fix (team_prototype_ema_alpha) -- but nothing ever built or called the
resolver. Left as-is, tracker.py's cost function (which penalizes
cross-team matches) and renderer.py (which colors directly off
Track.team) both silently break whenever a keyframe's labels happen to be
inverted relative to the last one: assignment cost spikes for the
correct match, sometimes enough to force a mismatch or a spurious new
track, and the same physical player's marker changes color. That's the
"same player, different colour" symptom.

Fix: keep two persistent appearance prototypes, one per GLOBAL label
("A", "B"), seeded from the first usable keyframe. On every later
keyframe, compare that keyframe's raw "A"/"B" groups against both
prototypes and pick whichever assignment (identity or swap) maximizes
total similarity, then relabel every detection before it reaches the
tracker. Prototypes drift slowly (EMA) so gradual lighting changes don't
break the match, but no single keyframe can flip which color means "A".
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from config import CFG
from appearance import sample_torso_histogram, histogram_similarity
from schema import FrameDetections


class TeamIdentityResolver:
    """One instance per clip -- state (the two prototypes) must persist
    across the whole video, the same way TrackManager does."""

    def __init__(self) -> None:
        self.prototypes: dict[str, Optional[np.ndarray]] = {"A": None, "B": None}

    def resolve(self, detections: FrameDetections, frame_bgr: np.ndarray, geo) -> FrameDetections:
        """Returns a FrameDetections whose player.team values are globally
        consistent. Cheap no-op (returns the input unchanged) once no swap
        is needed, so callers don't pay a copy on the common case."""
        frame_h_px = frame_bgr.shape[0]
        raw_a = [d for d in detections.players if d.team == "A"]
        raw_b = [d for d in detections.players if d.team == "B"]
        hist_a = self._group_histogram(frame_bgr, raw_a, geo, frame_h_px)
        hist_b = self._group_histogram(frame_bgr, raw_b, geo, frame_h_px)

        if self.prototypes["A"] is None and self.prototypes["B"] is None:
            # First usable keyframe: nothing to compare against yet, so
            # this call's labels *become* the global definition of "A"/"B".
            self.prototypes["A"] = hist_a
            self.prototypes["B"] = hist_b
            return detections

        swap = self._should_swap(hist_a, hist_b)
        mapping = {"A": "B", "B": "A"} if swap else {"A": "A", "B": "B"}
        self._update_prototypes(mapping, hist_a, hist_b)

        if not swap:
            return detections
        relabeled = [d.model_copy(update={"team": mapping[d.team]}) for d in detections.players]
        return detections.model_copy(update={"players": relabeled})

    def _should_swap(self, hist_a: Optional[np.ndarray], hist_b: Optional[np.ndarray]) -> bool:
        straight = (histogram_similarity(hist_a, self.prototypes["A"])
                    + histogram_similarity(hist_b, self.prototypes["B"]))
        swapped = (histogram_similarity(hist_a, self.prototypes["B"])
                   + histogram_similarity(hist_b, self.prototypes["A"]))
        return swapped > straight

    def _update_prototypes(self, mapping: dict[str, str], hist_a, hist_b) -> None:
        alpha = CFG.team_prototype_ema_alpha
        for raw_label, hist in (("A", hist_a), ("B", hist_b)):
            if hist is None:
                continue  # this keyframe had no players of this raw label -- don't drift the prototype on nothing
            target = mapping[raw_label]
            proto = self.prototypes[target]
            self.prototypes[target] = hist if proto is None else (1 - alpha) * proto + alpha * hist

    @staticmethod
    def _group_histogram(frame_bgr, dets, geo, frame_h_px: int) -> Optional[np.ndarray]:
        hists = []
        for d in dets:
            px, py = geo.ruler_to_orig_px(d.x, d.y)
            h = sample_torso_histogram(frame_bgr, px, py, frame_h_px)
            if h is not None and h.sum() > 0:
                hists.append(h)
        if not hists:
            return None
        return np.mean(hists, axis=0).astype(np.float32)