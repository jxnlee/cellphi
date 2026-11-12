"""Checks on the real images that don't need Cellpose (fast; these run in CI).

Skipped if reference/ or input/ isn't present.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cellphi.foreground import add_missed_dark_blobs, estimate_diameter_px, to_gray8
from cellphi.measure import measure_masks
from cellphi.pipeline import DEFAULT_CALIBRATION, read_image

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests" / "data" / "reference_lines.json").read_text())
SINGLE_ORGANOID_WELLS = ROOT / "input" / "Midbrain Organoids" / "Plate 4-3 FTY_CM"


def reference_image(well: str):
    path = ROOT / "reference" / well / REFERENCE["wells"][well]["image"]
    if not path.exists():
        pytest.skip(f"{path} not present")
    return read_image(path)


def test_reference_data_is_consistent():
    # The manual lines were drawn with the same calibration Cellphi uses by default.
    assert REFERENCE["um_per_px_2048"] == pytest.approx(DEFAULT_CALIBRATION.units_per_px(2048, 1536))
    for well, info in REFERENCE["wells"].items():
        assert len(info["lines"]) >= 0.6 * info["n_manual"], well
        for line in info["lines"]:
            assert 0 <= line["x"] < 2048 and 0 <= line["y"] < 1536


@pytest.mark.parametrize("well", sorted(REFERENCE["wells"]))
def test_size_estimate_matches_manual_measurements(well):
    img = reference_image(well)
    upp = DEFAULT_CALIBRATION.units_per_px(img.shape[1], img.shape[0])
    estimate = estimate_diameter_px(to_gray8(img)) * upp
    manual = np.median([line["length_um"] for line in REFERENCE["wells"][well]["lines"]])
    # Observed: 1.02-1.33x (the estimate leans towards the larger organoids, which cover more area).
    assert 0.8 * manual <= estimate <= 1.6 * manual


@pytest.mark.parametrize("well", ["A1", "A2", "A3", "B1", "B2"])
def test_fallback_finds_single_large_organoid(well):
    files = sorted((SINGLE_ORGANOID_WELLS / well).glob("*.jpg"))
    if not files:
        pytest.skip("Plate 4-3 images not present")
    gray = to_gray8(read_image(files[0]))
    estimate = estimate_diameter_px(gray)
    masks, added = add_missed_dark_blobs(np.zeros(gray.shape, np.int32), gray, estimate)
    # One organoid, nothing else (A3 also has a dark shadow band at the edge, which must be left out).
    assert added == 1
    (o,) = measure_masks(masks)
    upp = DEFAULT_CALIBRATION.units_per_px(gray.shape[1], gray.shape[0])
    # Cellphi measured 1293-1835 um for these organoids (equal-area diameter).
    assert 1100 <= o.equivalent_diameter_px * upp <= 2000
