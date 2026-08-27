"""
Synthetic ruler overlay -- the grounding trick carried over from Task 1.

VLMs don't get explicit image dimensions; they patch the image into
tokens and rely on training exposure for an approximate sense of
position. Drawing a literal ruler (0.0 -> 10.0) along the top and left
edges gives the model a scale to read off instead of guessing whether to
output relative or absolute coordinates.

The ruler is drawn on a COPY of the frame that only the VLM sees. All
Pillow drawing of the final annotations happens on the original,
ruler-free frame later in renderer.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

from config import CFG


@dataclass
class RulerGeometry:
    """Maps between ruler units (0-10) and pixel coords of the ORIGINAL
    (un-ruled) frame, given how much band we added on each edge."""
    orig_w: int
    orig_h: int
    band_px: int
    scale_max: float = CFG.ruler_scale_max

    def ruler_to_orig_px(self, rx: float, ry: float) -> tuple[float, float]:
        """Ruler units -> pixel coordinates in the ORIGINAL frame."""
        px = (rx / self.scale_max) * self.orig_w
        py = (ry / self.scale_max) * self.orig_h
        return px, py

    def orig_px_to_ruler(self, px: float, py: float) -> tuple[float, float]:
        rx = (px / self.orig_w) * self.scale_max
        ry = (py / self.orig_h) * self.scale_max
        return rx, ry


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_ruler(frame: Image.Image) -> tuple[Image.Image, RulerGeometry]:
    """
    Returns a new, larger image with a ruled band added along the top and
    left edges, plus the geometry needed to map returned ruler coordinates
    back onto the ORIGINAL frame's pixel space.

    The original frame content is placed unscaled into the bottom-right
    region of the new canvas, so the ruler ticks correspond 1:1 with
    positions in the original image.
    """
    band = CFG.ruler_thickness_px
    w, h = frame.size
    geo = RulerGeometry(orig_w=w, orig_h=h, band_px=band)

    canvas = Image.new("RGB", (w + band, h + band), "white")
    canvas.paste(frame, (band, band))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(max(10, band // 3))

    scale_max = CFG.ruler_scale_max
    major = CFG.ruler_major_tick_every
    minor = CFG.ruler_minor_tick_every

    # top ruler (x axis) -- spans the width of the ORIGINAL frame, offset by `band`
    n_minor = int(scale_max / minor)
    for i in range(n_minor + 1):
        val = i * minor
        x = band + (val / scale_max) * w
        is_major = abs(val % major) < 1e-6
        tick_h = band * (0.6 if is_major else 0.3)
        draw.line([(x, band - tick_h), (x, band)], fill="black", width=2 if is_major else 1)
        if is_major:
            draw.text((x + 2, 2), f"{val:g}", fill="black", font=font)

    # left ruler (y axis)
    for i in range(n_minor + 1):
        val = i * minor
        y = band + (val / scale_max) * h
        is_major = abs(val % major) < 1e-6
        tick_w = band * (0.6 if is_major else 0.3)
        draw.line([(band - tick_w, y), (band, y)], fill="black", width=2 if is_major else 1)
        if is_major:
            draw.text((2, y + 2), f"{val:g}", fill="black", font=font)

    # corner label so the model knows what it's looking at
    draw.rectangle([0, 0, band, band], fill="white")
    draw.text((3, band // 2 - 5), "0,0", fill="black", font=_get_font(max(9, band // 4)))

    return canvas, geo
