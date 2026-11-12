"""Annotated output image."""

from __future__ import annotations

import cv2
import numpy as np

from .measure import DEFAULT_DIAMETER, DIAMETER_NAMES, MEASURED, Organoid, diameter_line

# BGR colours
MEASURED_COLOR = (0, 255, 255)  # yellow
EDGE_COLOR = (0, 165, 255)  # orange: measured, but cut off by the image edge
EXCLUDED_COLOR = (80, 80, 255)  # light red
TEXT_BG = (0, 0, 0)

LABEL_MODES = ("both", "id", "diameter", "none")


def to_bgr8(img: np.ndarray) -> np.ndarray:
    """Convert any grayscale/colour, 8/16-bit/float image to 8-bit BGR for drawing."""
    out = img
    if out.dtype != np.uint8:
        out = out.astype(np.float64)
        lo, hi = np.percentile(out, (0.5, 99.5))
        out = np.clip((out - lo) / max(hi - lo, 1e-9), 0, 1)
        out = (out * 255).astype(np.uint8)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    return out.copy()


def annotate(
    img: np.ndarray,
    organoids: list[Organoid],
    units_per_px: float,
    label: str = "both",
    show_excluded: bool = True,
    diameter_method: str = DEFAULT_DIAMETER,
    unit: str = "um",
    legend: bool = True,
) -> np.ndarray:
    """Draw outlines and labels on a copy of ``img``.

    Measured organoids: yellow outline and label (``label``: "both" = id and diameter, "id",
    "diameter" or "none"), plus a line whose length is the reported diameter (see
    ``measure.diameter_line``). Measured organoids touching the image edge: orange, and a "*"
    after the diameter. Excluded objects (fragments, too small, edge if excluded): red outline.
    A legend in the top-left corner names the diameter definition and calibration.
    """
    if label not in LABEL_MODES:
        raise ValueError(f"label must be one of {LABEL_MODES}")
    out = to_bgr8(img)
    h, w = out.shape[:2]
    # Scale line and text size with image size so labels stay legible at any resolution.
    thickness = max(1, round(w / 1000))
    # Text scales with image size, but shrinks for small organoids so labels don't cover them.
    sizes = [o.equivalent_diameter_px for o in organoids if o.status == MEASURED]
    typical = float(np.median(sizes)) if sizes else w
    font_scale = max(0.25, min(w / 4000, typical / 320))

    for o in organoids:
        pts = o.contour.astype(np.int32).reshape(-1, 1, 2)
        if o.status != MEASURED:
            if show_excluded:
                cv2.polylines(out, [pts], True, EXCLUDED_COLOR, thickness, cv2.LINE_AA)
            continue
        color = EDGE_COLOR if o.touches_border else MEASURED_COLOR
        cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
        p1, p2 = (tuple(int(round(v)) for v in p) for p in diameter_line(o, diameter_method))
        cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

        diameter = f"{o.diameter_px(diameter_method) * units_per_px:.0f}" + ("*" if o.touches_border else "")
        lines = {"both": [f"#{o.id}", diameter], "id": [f"#{o.id}"], "diameter": [diameter]}.get(label, [])
        if lines:
            _put_text_block(out, lines, o.centroid, font_scale, max(1, thickness))

    if legend:
        unit_txt = "um" if unit in ("um", "µm") else unit
        _put_legend(out, [
            f"Diameter ({unit_txt}): {DIAMETER_NAMES.get(diameter_method, diameter_method)}",
            f"Line = diameter. Scale {units_per_px:.3f} {unit_txt}/px",
            "* / orange = touches image edge (underestimated)",
        ], max(0.4, w / 4500), 1)
    return out


def _put_legend(img, lines, font_scale, thickness, font=cv2.FONT_HERSHEY_SIMPLEX):
    sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]
    line_h = max(s[0][1] + s[1] for s in sizes) + 4
    block_w = max(s[0][0] for s in sizes)
    cv2.rectangle(img, (0, 0), (block_w + 12, line_h * len(lines) + 8), TEXT_BG, cv2.FILLED)
    for i, (text, ((_, th), _)) in enumerate(zip(lines, sizes)):
        cv2.putText(img, text, (6, 4 + i * line_h + th + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _put_text_block(img, lines, center_xy, font_scale, thickness, font=cv2.FONT_HERSHEY_SIMPLEX):
    sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]
    line_h = max(s[0][1] + s[1] for s in sizes) + 2
    block_w = max(s[0][0] for s in sizes)
    block_h = line_h * len(lines)
    x0 = int(center_xy[0] - block_w / 2)
    y0 = int(center_xy[1] - block_h / 2)
    cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + block_w + 2, y0 + block_h + 1), TEXT_BG, cv2.FILLED)
    for i, (text, ((tw, th), base)) in enumerate(zip(lines, sizes)):
        tx = int(center_xy[0] - tw / 2)
        ty = y0 + i * line_h + th + 1
        cv2.putText(img, text, (tx, ty), font, font_scale, MEASURED_COLOR, thickness, cv2.LINE_AA)
