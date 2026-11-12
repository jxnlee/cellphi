"""CSV output."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .measure import DEFAULT_DIAMETER, EXCLUDED_BORDER, EXCLUDED_FRAGMENT, EXCLUDED_SMALL, MEASURED, Organoid


def organoid_rows(
    image_name: str, organoids: list[Organoid], units_per_px: float, unit: str, diameter_method: str = DEFAULT_DIAMETER
) -> list[dict]:
    """One row per measured organoid, ordered by id."""
    rows = []
    for o in sorted((o for o in organoids if o.status == MEASURED), key=lambda o: o.id):
        rows.append(
            {
                "image": image_name,
                "id": o.id,
                f"diameter_{unit}": round(o.diameter_px(diameter_method) * units_per_px, 1),
                "touches_edge": "yes" if o.touches_border else "no",
                f"equivalent_diameter_{unit}": round(o.equivalent_diameter_px * units_per_px, 1),
                f"max_feret_diameter_{unit}": round(o.feret_px * units_per_px, 1),
                f"min_feret_diameter_{unit}": round(o.min_feret_px * units_per_px, 1),
                f"area_{unit}2": round(o.area_px * units_per_px**2, 0),
                "centroid_x_px": round(o.centroid[0], 1),
                "centroid_y_px": round(o.centroid[1], 1),
                f"{unit}_per_px": round(units_per_px, 6),
            }
        )
    return rows


def _stats(values: list[float], prefix: str, unit: str) -> dict:
    fmt = lambda v: round(v, 1) if v is not None else ""  # noqa: E731
    return {
        f"{prefix}mean_diameter_{unit}": fmt(statistics.fmean(values) if values else None),
        f"{prefix}median_diameter_{unit}": fmt(statistics.median(values) if values else None),
        f"{prefix}sd_diameter_{unit}": fmt(statistics.stdev(values) if len(values) > 1 else None),
        f"{prefix}min_diameter_{unit}": fmt(min(values) if values else None),
        f"{prefix}max_diameter_{unit}": fmt(max(values) if values else None),
    }


def summary_row(
    image_name: str, organoids: list[Organoid], units_per_px: float, unit: str, diameter_method: str = DEFAULT_DIAMETER
) -> dict:
    """Per-image counts and diameter statistics.

    Statistics are given for all measured organoids, and separately for complete ones only
    (not touching the image edge; an edge-cut organoid's diameter is underestimated).
    """
    measured = [o for o in organoids if o.status == MEASURED]
    d_all = [o.diameter_px(diameter_method) * units_per_px for o in measured]
    d_complete = [o.diameter_px(diameter_method) * units_per_px for o in measured if not o.touches_border]
    return {
        "image": image_name,
        "n_measured": len(measured),
        "n_touching_edge": sum(o.touches_border for o in measured),
        "n_excluded_edge": sum(o.status == EXCLUDED_BORDER for o in organoids),
        "n_excluded_small": sum(o.status == EXCLUDED_SMALL for o in organoids),
        "n_excluded_overlap_fragments": sum(o.status == EXCLUDED_FRAGMENT for o in organoids),
        **_stats(d_all, "", unit),
        **_stats(d_complete, "complete_", unit),
        "diameter_method": diameter_method,
        f"{unit}_per_px": round(units_per_px, 6),
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        fieldnames += [k for k in r if k not in fieldnames]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
