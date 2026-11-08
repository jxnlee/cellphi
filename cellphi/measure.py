"""Measurements on Cellpose label masks. Pure numpy/OpenCV, no model needed (unit-testable)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

MEASURED = "measured"
EXCLUDED_BORDER = "touches_border"
EXCLUDED_SMALL = "too_small"
EXCLUDED_FRAGMENT = "overlap_fragment"

# Objects within this many pixels of the image edge count as cut off.
BORDER_MARGIN_PX = 2
# Overlap-fragment detection (see measure_masks). Real organoids measured ~1.05-1.2
# elongation; overlap lenses ~1.6-2.1 and are mostly surrounded by other organoids.
FRAGMENT_MIN_ELONGATION = 1.5
FRAGMENT_MIN_CONTACT = 0.5
# Also a fragment: an object this enclosed by others and smaller than this x the typical size.
ENCLOSED_MIN_CONTACT = 0.9
ENCLOSED_MAX_RELATIVE_SIZE = 0.5
# A fragment is given back to an organoid only if this share of it lies in the organoid's convex hull.
FRAGMENT_MIN_OWNERSHIP = 0.2

# Offsets from a pixel centre to its four corners.
_CORNERS = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


@dataclass
class Organoid:
    label: int  # label value in the Cellpose mask
    area_px: int
    centroid: tuple[float, float]  # (x, y), pixels
    feret_px: float  # maximum Feret diameter, pixels
    feret_endpoints: tuple[tuple[float, float], tuple[float, float]]  # (x, y) pairs
    contour: np.ndarray = field(repr=False)  # (N, 2) int (x, y), outer boundary for drawing
    touches_border: bool = False
    status: str = MEASURED
    id: int | None = None  # 1..n for measured organoids, matches the image annotation and CSV

    @property
    def equivalent_diameter_px(self) -> float:
        """Diameter of a circle with the same area."""
        return float(np.sqrt(4.0 * self.area_px / np.pi))


def max_feret(points_xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Maximum Feret (caliper) diameter of a set of pixel coordinates.

    ``points_xy`` are integer pixel coordinates (pixel centres). Each pixel covers a unit
    square, so the distance is measured between the outer pixel corners rather than the
    pixel centres. This removes a ~1 px underestimate that otherwise biases every
    measurement. The maximum pairwise distance is always between convex-hull vertices,
    so only those are compared.

    Returns (diameter_px, endpoint1, endpoint2) with endpoints as (x, y) floats.
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("No points given.")
    hull = cv2.convexHull(pts).reshape(-1, 2)
    corners = (hull[:, None, :] + _CORNERS[None, :, :]).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(corners).reshape(-1, 2).astype(np.float64)
    d = cdist(hull, hull)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return float(d[i, j]), hull[i], hull[j]


def _measure_region(binary: np.ndarray, y0: int, x0: int, shape: tuple[int, int], border_margin: int) -> dict:
    """Measure one binary region located at (y0, x0) in an image of ``shape``."""
    sub = binary.astype(np.uint8)
    offset = np.array([x0, y0])
    ys, xs = np.nonzero(sub)
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    feret, p1, p2 = max_feret(np.concatenate([c.reshape(-1, 2) for c in contours]) + offset)
    h, w = shape
    return dict(
        area_px=int(len(xs)),
        centroid=(float(xs.mean() + x0), float(ys.mean() + y0)),
        feret_px=feret,
        feret_endpoints=(tuple(p1), tuple(p2)),
        contour=max(contours, key=cv2.contourArea).reshape(-1, 2) + offset,
        # Cellpose masks often stop a pixel or two short of the edge, hence the margin.
        touches_border=bool(
            y0 + ys.min() <= border_margin
            or x0 + xs.min() <= border_margin
            or y0 + ys.max() >= h - 1 - border_margin
            or x0 + xs.max() >= w - 1 - border_margin
        ),
    )


def measure_masks(
    masks: np.ndarray,
    exclude_border: bool = True,
    border_margin: int = BORDER_MARGIN_PX,
    min_diameter_px: float = 0.0,
    exclude_fragments: bool = True,
) -> list[Organoid]:
    """Measure every labelled object in a Cellpose mask (0 = background, 1..N = objects).

    Objects are flagged (kept in the result, but not given an id) when they are:
    - ``overlap_fragment``: the lens-shaped region where two organoids overlap, which
      Cellpose labels as a separate object. Recognised as elongated (Feret diameter
      >= FRAGMENT_MIN_ELONGATION x the equal-area circle diameter) and mostly enclosed by
      other organoids (>= FRAGMENT_MIN_CONTACT of its outline touches them); or small
      (< ENCLOSED_MAX_RELATIVE_SIZE x typical) and almost fully enclosed by organoids. Its pixels
      are added back to the organoids whose outline it lies within.
    - ``touches_border``: within ``border_margin`` px of the image edge, so cut off.
    - ``too_small``: Feret diameter below ``min_diameter_px``.
    Measured organoids get ids 1..n in reading order.
    """
    if masks.ndim != 2:
        raise ValueError(f"Expected a 2D label mask, got shape {masks.shape}.")
    h, w = masks.shape
    pad = 2  # neighbourhood (px) used to find touching objects

    # Pass 1: measure each label on its own and find which labels it touches.
    info: dict[int, dict] = {}
    for label, sl in enumerate(ndimage.find_objects(masks), start=1):
        if sl is None:
            continue
        y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        y1, x1 = min(sl[0].stop + pad, h), min(sl[1].stop + pad, w)
        window = masks[y0:y1, x0:x1]
        me = window == label
        ring = window[ndimage.binary_dilation(me, iterations=pad) & ~me]
        r = _measure_region(me, y0, x0, (h, w), border_margin)
        r["contact"] = float((ring > 0).mean()) if ring.size else 0.0
        r["neighbours"] = set(np.unique(ring[ring > 0]).tolist())
        r["box"] = (y0, y1, x0, x1)
        r["eq_diam"] = np.sqrt(4.0 * r["area_px"] / np.pi)
        r["fragment"] = (
            exclude_fragments
            and r["feret_px"] / r["eq_diam"] >= FRAGMENT_MIN_ELONGATION
            and r["contact"] >= FRAGMENT_MIN_CONTACT
        )
        info[label] = r

    # Small pieces almost completely enclosed by organoids (e.g. the gap where several meet).
    others = [r["eq_diam"] for r in info.values() if not r["fragment"]]
    if exclude_fragments and others:
        typical = float(np.median(others))
        for r in info.values():
            if r["contact"] >= ENCLOSED_MIN_CONTACT and r["eq_diam"] < ENCLOSED_MAX_RELATIVE_SIZE * typical:
                r["fragment"] = True

    # Pass 2: give each fragment back to the organoids it overlaps. A lens cut out of an
    # organoid leaves a bite whose chord the organoid's convex hull spans, so a good part
    # of the lens lies inside that hull. An organoid that merely touches the lens's tip
    # covers almost none of it and does not get it.
    for label, r in info.items():
        frags = [n for n in r["neighbours"] if info.get(n, {}).get("fragment")]
        if r["fragment"] or not frags:
            continue
        boxes = [r["box"]] + [info[f]["box"] for f in frags]
        y0, x0 = min(b[0] for b in boxes), min(b[2] for b in boxes)
        y1, x1 = max(b[1] for b in boxes), max(b[3] for b in boxes)
        window = masks[y0:y1, x0:x1]
        hull = np.zeros(window.shape, np.uint8)
        pts = cv2.convexHull(r["contour"].astype(np.int32)) - np.array([x0, y0], np.int32)
        cv2.fillConvexPoly(hull, pts, 1)
        hull = hull.astype(bool)
        owned = [f for f in frags if (hull & (window == f)).sum() >= FRAGMENT_MIN_OWNERSHIP * info[f]["area_px"]]
        if owned:
            union = np.isin(window, [label, *owned])
            r.update(_measure_region(union, y0, x0, (h, w), border_margin))

    organoids: list[Organoid] = []
    for label, r in info.items():
        if r["fragment"]:
            status = EXCLUDED_FRAGMENT
        elif exclude_border and r["touches_border"]:
            status = EXCLUDED_BORDER
        elif r["feret_px"] < min_diameter_px:
            status = EXCLUDED_SMALL
        else:
            status = MEASURED
        organoids.append(
            Organoid(
                label=label,
                area_px=r["area_px"],
                centroid=r["centroid"],
                feret_px=r["feret_px"],
                feret_endpoints=r["feret_endpoints"],
                contour=r["contour"],
                touches_border=r["touches_border"],
                status=status,
            )
        )

    number_reading_order([o for o in organoids if o.status == MEASURED])
    return organoids


def number_reading_order(organoids: list[Organoid]) -> None:
    """Assign ids 1..n in reading order: rows (one typical diameter tall) top to bottom,
    then left to right within a row, so numbers are easy to find on the image."""
    if not organoids:
        return
    row_height = max(float(np.median([o.feret_px for o in organoids])), 1.0)
    ordered = sorted(organoids, key=lambda o: (int(o.centroid[1] // row_height), o.centroid[0]))
    for i, o in enumerate(ordered, start=1):
        o.id = i
