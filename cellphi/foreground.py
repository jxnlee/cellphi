"""Classic (non-ML) image analysis that supports Cellpose for brightfield organoid images:
estimating the typical organoid size, and recovering solid dark organoids Cellpose missed.

Assumes organoids are darker than the background (brightfield). The image alone can't tell
reliably which side is background: in densely packed wells the image border is mostly dark
organoids, and touching organoids form one connected dark network (both rules failed on more
than a third of the reference images). So dark objects are assumed, unless Cellpose's outlines
show otherwise (then the fallback does nothing). Use --no-auto-size for non-brightfield images.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

# Rescued blobs must be at least this solid (area / convex hull area). One organoid is ~0.95+;
# clusters of touching organoids and shadows are lower.
RESCUE_MIN_SOLIDITY = 0.9
# ...and at least this fraction of the typical organoid diameter.
RESCUE_MIN_RELATIVE_SIZE = 0.5


def to_gray8(img: np.ndarray) -> np.ndarray:
    """Any image (gray/BGR/BGRA, 8/16-bit/float) -> 8-bit grayscale."""
    g = img
    if g.ndim == 3:
        g = cv2.cvtColor(g[..., :3].astype(np.float32), cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = g.astype(np.float64)
        lo, hi = np.percentile(g, (0.5, 99.5))
        g = (np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    return g


def dark_foreground(gray: np.ndarray, masks: np.ndarray | None = None, fill_holes: bool = True) -> np.ndarray | None:
    """Pixels darker than the background (Otsu threshold).

    ``fill_holes`` fills enclosed light areas (e.g. a see-through organoid's centre); note that
    it also fills background gaps fully enclosed by a ring of touching organoids.

    Returns None if ``masks`` (Cellpose's outlines) show objects that are *not* darker than the
    unlabelled background. Without masks, dark objects are assumed (brightfield).
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if masks is not None and (masks > 0).any() and (masks == 0).any():
        if np.median(g[masks > 0]) >= np.median(g[masks == 0]):
            return None
    fg = ndimage.binary_opening(g < t, iterations=2)
    return ndimage.binary_fill_holes(fg) if fill_holes else fg


def estimate_diameter_px(gray: np.ndarray) -> float | None:
    """Typical organoid diameter in pixels, from the dark foreground's distance transform.

    Each organoid's centre is a local maximum of the distance to the background, even when
    organoids touch. The area-weighted median of those radii is the size of the organoids that
    cover most of the foreground. Runs on a 512 px copy (~10 ms). Returns None if there is no
    usable dark foreground. On the reference data it was within 0.9-1.3x of the size Cellpose finds.
    """
    h, w = gray.shape
    s = 512 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    # No hole filling: in densely packed wells it would merge whole clusters into one blob.
    fg = dark_foreground(small, fill_holes=False)
    if fg is None or not fg.any():
        return None
    fg = ndimage.binary_opening(fg, iterations=1)
    dt = ndimage.distance_transform_edt(fg)
    ys, xs = np.nonzero((dt == ndimage.maximum_filter(dt, size=7)) & (dt >= 2))
    if ys.size == 0:
        return None
    # One peak per organoid: a large organoid has a ridge of several local maxima, which would
    # otherwise each count as an organoid. Keep a peak only if no larger kept peak covers it.
    order = np.argsort(-dt[ys, xs])
    kept_xy, kept_r = [], []
    for i in order[:5000]:
        p, r = np.array([xs[i], ys[i]], float), dt[ys[i], xs[i]]
        if kept_xy and np.any(np.hypot(*(np.array(kept_xy) - p).T) < np.array(kept_r)):
            continue
        kept_xy.append(p)
        kept_r.append(r)
    radii = np.sort(np.array(kept_r))
    cum = np.cumsum(radii**2)
    r = radii[np.searchsorted(cum, cum[-1] / 2)]
    return float(2 * r / s)


def add_missed_dark_blobs(masks: np.ndarray, gray: np.ndarray, typical_diameter_px: float | None) -> tuple[np.ndarray, int]:
    """Add solid dark blobs that Cellpose left unlabelled, as new labels.

    Cellpose's current model misses very large, featureless dark organoids (e.g. one big organoid
    alone in a well, or one far larger than its neighbours). Those are easy for thresholding: a
    dark, solid, roughly convex region. Only regions that are (a) not already labelled, (b) solid
    (single organoid, not a cluster or shadow) and (c) not tiny are added.
    Returns (new_masks, number_added).
    """
    fg = dark_foreground(gray, masks)
    if fg is None:
        return masks, 0
    fg = fg & ~ndimage.binary_dilation(masks > 0, iterations=3)
    fg = ndimage.binary_opening(fg, iterations=3)
    labels, _ = ndimage.label(fg)
    min_d = RESCUE_MIN_RELATIVE_SIZE * (typical_diameter_px or 0)
    min_area = max(np.pi * (min_d / 2) ** 2, 50)
    out = masks.astype(np.int32, copy=True)
    next_label = int(out.max()) + 1
    added = 0
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        comp = labels[sl] == i
        if comp.sum() < min_area:
            continue
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=cv2.contourArea)
        solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        if solidity < RESCUE_MIN_SOLIDITY:
            continue
        region = out[sl]
        region[comp & (region == 0)] = next_label
        next_label += 1
        added += 1
    return out, added
