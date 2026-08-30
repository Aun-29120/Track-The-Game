"""
Cheap appearance signal used only as a tiebreaker in track assignment,
for the case motion/IoU alone can't resolve: two players crossing paths.
Also used to build persistent per-team appearance prototypes (see
team_identity.py) since the VLM's own "A"/"B" labels are frame-local and
can't be trusted to mean the same team across independent calls.

Deliberately NOT a learned re-id embedding (that's the DeepSORT approach)
-- a color histogram of the kit is enough when we're only disambiguating
two team colors, not many visually distinct individuals.

FIX (error report #2): sampling used to be centered directly on the foot
point, which mostly captures grass/pitch/shadow, not the kit. The patch is
now offset upward toward the torso, and low-saturation pixels (grass,
lines, shadow) are masked out of the histogram so they don't dilute the
kit color signal.
"""
from __future__ import annotations
import numpy as np
import cv2

from config import CFG

PATCH_HALF_SIZE = 18  # pixels, half-width of the sampled patch around a point


def sample_histogram(frame_bgr: np.ndarray, x_px: float, y_px: float, mask_low_saturation: bool = True) -> np.ndarray:
    """HSV hue histogram of a small patch centered on (x_px, y_px) in the
    ORIGINAL (un-ruled) frame. Low-saturation pixels (grass, lines,
    shadow) are masked out by default so background doesn't dilute the
    kit-color signal. Returns a normalized 1D histogram."""
    h, w = frame_bgr.shape[:2]
    x0 = max(0, int(x_px) - PATCH_HALF_SIZE)
    x1 = min(w, int(x_px) + PATCH_HALF_SIZE)
    y0 = max(0, int(y_px) - PATCH_HALF_SIZE)
    y1 = min(h, int(y_px) + PATCH_HALF_SIZE)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(32, dtype=np.float32)
    patch = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    mask = None
    if mask_low_saturation:
        sat = hsv[:, :, 1]
        mask = (sat > CFG.saturation_mask_threshold).astype(np.uint8) * 255
        if mask.sum() == 0:
            # entire patch was low-saturation (e.g. patch landed fully on
            # pitch) -- fall back to unmasked rather than returning an
            # empty/zero histogram that would look "similar to everything"
            mask = None

    hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def sample_torso_histogram(frame_bgr: np.ndarray, foot_x_px: float, foot_y_px: float, frame_h_px: int) -> np.ndarray:
    """Samples the kit/torso rather than the feet -- offsets upward from
    the VLM's reported foot position by a fraction of frame height before
    sampling, so the patch lands on the shirt instead of the pitch."""
    offset_px = frame_h_px * CFG.torso_offset_frac_height
    torso_y = max(0.0, foot_y_px - offset_px)
    return sample_histogram(frame_bgr, foot_x_px, torso_y)


def histogram_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Computes visual similarity between two CIELAB feature vectors.
    Returns 1.0 for identical features, down to 0.0 for completely different.
    """
    if f1 is None or f2 is None or f1.size != 3 or f2.size != 3:
        return 0.5

    # Euclidean distance in CIELAB space.
    # Max possible distance is ~255 (if mapped to 8-bit). We'll tune the scale.
    dist = np.linalg.norm(f1 - f2)
    
    # Scale distance to similarity (tune threshold 100.0 empirically)
    similarity = 1.0 - (dist / 100.0)
    return float(max(0.0, min(1.0, similarity)))