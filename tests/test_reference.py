"""Accuracy against the lab's manual ImageJ measurements (reference/Plate 1 CM_CM).

Runs Cellpose, so it is slow (~1 min on a GPU, several minutes on CPU) and not part of the
default test run:  pytest -m slow

tests/data/reference_lines.json holds where each hand-drawn diameter line is and its length
(made by tools/extract_reference_lines.py from the ImageJ Results.csv + screenshot).
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "reference"
DATA = json.loads((ROOT / "tests" / "data" / "reference_lines.json").read_text())

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def comparisons():
    pytest.importorskip("cellpose")
    if not REF_DIR.exists():
        pytest.skip("reference/ folder not present")
    from cellphi.pipeline import process_image
    from cellphi.segment import Segmenter

    segmenter = Segmenter()
    rows = []  # (manual_um, cellphi_um, touches_edge) per matched organoid
    n_lines = n_split_or_merged = n_missed = 0
    for well, info in DATA["wells"].items():
        res = process_image(REF_DIR / well / info["image"], segmenter)
        by_label = {o.label: o for o in res.measured}
        hits: dict[int, list] = {}
        for line in info["lines"]:
            n_lines += 1
            x, y = int(round(line["x"])), int(round(line["y"]))
            win = res.masks[max(y - 6, 0): y + 7, max(x - 6, 0): x + 7]
            labels = win[win > 0]
            if labels.size == 0 or int(np.bincount(labels).argmax()) not in by_label:
                n_missed += 1
                continue
            hits.setdefault(int(np.bincount(labels).argmax()), []).append(line)
        for label, lines in hits.items():
            if len(lines) > 1:  # one outline covering several hand-measured organoids
                n_split_or_merged += len(lines)
                continue
            o = by_label[label]
            rows.append((lines[0]["length_um"], o.diameter_px(res.diameter_method) * res.units_per_px, o.touches_border))
    return dict(rows=rows, n_lines=n_lines, n_missed=n_missed, n_merged=n_split_or_merged)


def test_finds_hand_measured_organoids(comparisons):
    found = len(comparisons["rows"]) / comparisons["n_lines"]
    assert found >= 0.95, comparisons


def test_diameters_match_manual_measurements(comparisons):
    manual, auto, _ = map(np.array, zip(*comparisons["rows"]))
    ratio = auto / manual
    # Reference run (2026-11): median ratio 1.00, mean abs. error 6.8%, 79% within 10%.
    assert 0.97 <= np.median(ratio) <= 1.03
    assert np.mean(np.abs(ratio - 1)) < 0.09
    assert np.mean(np.abs(ratio - 1) < 0.10) >= 0.70
