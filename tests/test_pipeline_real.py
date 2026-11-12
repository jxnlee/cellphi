"""Full pipeline (with Cellpose) on a few very different real wells, via the command line.

Slow and needs the Cellpose model, so not part of the default run or CI:  pytest -m slow
Expected values come from a full run whose annotated images were checked by eye (2026-11).
If a change moves these numbers, look at the annotated images before updating them.
"""

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"

pytestmark = pytest.mark.slow

# image (relative to input/) -> (organoids measured, median diameter in um)
EXPECTED = {
    "Midbrain Organoids/Plate 1 CM_CM/A1/day 23 2-28-23 KS217537.jpg": (97, 382.3),   # dense, touching
    "Midbrain Organoids/Plate 3 CM_CM/A1/day 23 2-28-23 KS217590.jpg": (38, 594.7),   # large organoids
    "Midbrain Organoids/Plate 4-3 FTY_CM/A1/day 23 2-28-23 KS217645.jpg": (1, 1293.2),  # single organoid (fallback)
    "Cortical Organoids/Plate 4 FTY_CM/A1/day 23 2-28-23 KS217704.jpg": (159, 262.5),  # semi-transparent
    "Cortical Organoids/Plate 4 FTY_CM/B1/day 23 2-28-23 KS217711.jpg": (131, 286.7),  # one giant + many small
}


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    pytest.importorskip("cellpose")
    files = [INPUT / name for name in EXPECTED]
    if not all(f.exists() for f in files):
        pytest.skip("input images not present")
    from cellphi.cli import main

    out = tmp_path_factory.mktemp("results")
    assert main([*map(str, files), "-o", str(out)]) == 0
    with (out / "summary.csv").open() as f:
        rows = {r["image"]: r for r in csv.DictReader(f)}
    # Single files are listed by file name
    return {name: rows[Path(name).name] for name in EXPECTED}, out


@pytest.mark.parametrize("name", list(EXPECTED))
def test_counts_and_sizes_match_reviewed_run(summary, name):
    rows, _ = summary
    n_expected, median_expected = EXPECTED[name]
    row = rows[name]
    assert int(row["n_measured"]) == pytest.approx(n_expected, rel=0.08, abs=1)
    assert float(row["median_diameter_um"]) == pytest.approx(median_expected, rel=0.05)


def test_outputs_written(summary):
    _, out = summary
    assert len(list(out.glob("*_annotated.jpg"))) == len(EXPECTED)
    csvs = list(out.glob("*_organoids.csv"))
    assert len(csvs) == len(EXPECTED) and all(f.stat().st_size > 0 for f in csvs)
