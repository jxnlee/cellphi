"""CSV output."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .measure import EXCLUDED_BORDER, EXCLUDED_FRAGMENT, EXCLUDED_SMALL, MEASURED, Organoid


def organoid_rows(image_name: str, organoids: list[Organoid], units_per_px: float, unit: str) -> list[dict]:
    """One row per measured organoid, ordered by id."""
    rows = []
    for o in sorted((o for o in organoids if o.status == MEASURED), key=lambda o: o.id):
        rows.append(
            {
                "image": image_name,
                "id": o.id,
                f"diameter_{unit}": round(o.feret_px * units_per_px, 2),
                f"equivalent_diameter_{unit}": round(o.equivalent_diameter_px * units_per_px, 2),
                f"area_{unit}2": round(o.area_px * units_per_px**2, 1),
                "centroid_x_px": round(o.centroid[0], 1),
                "centroid_y_px": round(o.centroid[1], 1),
                f"{unit}_per_px": round(units_per_px, 6),
            }
        )
    return rows


def summary_row(image_name: str, organoids: list[Organoid], units_per_px: float, unit: str) -> dict:
    d = [o.feret_px * units_per_px for o in organoids if o.status == MEASURED]
    fmt = lambda v: round(v, 2) if v is not None else ""  # noqa: E731
    return {
        "image": image_name,
        "n_measured": len(d),
        "n_excluded_border": sum(o.status == EXCLUDED_BORDER for o in organoids),
        "n_excluded_small": sum(o.status == EXCLUDED_SMALL for o in organoids),
        "n_excluded_overlap_fragments": sum(o.status == EXCLUDED_FRAGMENT for o in organoids),
        f"mean_diameter_{unit}": fmt(statistics.fmean(d) if d else None),
        f"median_diameter_{unit}": fmt(statistics.median(d) if d else None),
        f"sd_diameter_{unit}": fmt(statistics.stdev(d) if len(d) > 1 else None),
        f"min_diameter_{unit}": fmt(min(d) if d else None),
        f"max_diameter_{unit}": fmt(max(d) if d else None),
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
