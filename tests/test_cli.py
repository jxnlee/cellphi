"""End-to-end CLI tests without Cellpose (fast; these run in CI).

Cellpose is replaced by a stand-in "segmenter" that thresholds dark regions, so everything
else (folder search, size estimate, measurement, drawing, CSV output) runs for real.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy import ndimage

import cellphi.segment
from cellphi.cli import main


class _Device:
    type = "cpu"


class FakeSegmenter:
    """Labels connected dark regions. Records the size hints it was given."""

    hints: list = []

    def __init__(self, gpu=True, model="cpsam"):
        self.device, self.device_name = _Device(), "CPU (fake segmenter for tests)"

    def segment(self, img, settings=None):
        FakeSegmenter.hints.append(getattr(settings, "diameter_px", None))
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        labels, _ = ndimage.label(ndimage.binary_opening(gray < t, iterations=2))
        return labels.astype(np.int32)


@pytest.fixture(autouse=True)
def fake_cellpose(monkeypatch):
    FakeSegmenter.hints = []
    monkeypatch.setattr(cellphi.segment, "Segmenter", FakeSegmenter)


def write_image(path: Path, disks, shape=(768, 1024)):
    """Light background with dark disks (cx, cy, r)."""
    img = np.full(shape, 215, np.uint8)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for cx, cy, r in disks:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r**2] = 20
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_cli_end_to_end(tmp_path):
    data = tmp_path / "data"
    write_image(data / "Plate 1" / "A1" / "img1.jpg", [(200, 200, 60), (500, 400, 80), (1010, 300, 50)])
    write_image(data / "Plate 1" / "A2" / "img2.jpg", [(300, 300, 100)])
    out = tmp_path / "results"

    assert main([str(data), "-o", str(out), "--units-per-px", "2.0"]) == 0

    assert (out / "Plate 1" / "A1" / "img1_annotated.jpg").exists()
    assert (out / "Plate 1" / "A2" / "img2_annotated.jpg").exists()
    assert not (out / "organoids.csv").exists()  # one CSV per image, not a combined one
    a1 = read_csv(out / "Plate 1" / "A1" / "img1_organoids.csv")
    assert {r["image"] for r in a1} == {"Plate 1/A1/img1.jpg"}
    assert len(a1) == 3
    assert len(read_csv(out / "Plate 1" / "A2" / "img2_organoids.csv")) == 1
    edge = [r for r in a1 if r["touches_edge"] == "yes"]
    assert len(edge) == 1  # the disk cut off by the right edge is measured and flagged
    complete = sorted(float(r["diameter_um"]) for r in a1 if r["touches_edge"] == "no")
    assert complete == pytest.approx([240, 320], rel=0.03)  # 2 * r * 2.0 um/px

    summary = {r["image"]: r for r in read_csv(out / "summary.csv")}
    assert summary["Plate 1/A1/img1.jpg"]["n_touching_edge"] == "1"
    assert float(summary["Plate 1/A2/img2.jpg"]["median_diameter_um"]) == pytest.approx(400, rel=0.03)

    # The organoid size was estimated from each image and passed to the segmenter.
    assert all(h is not None and h > 50 for h in FakeSegmenter.hints)


def test_cli_rerun_skips_its_own_outputs(tmp_path):
    data = tmp_path / "data"
    write_image(data / "img.jpg", [(300, 300, 80)])
    assert main([str(data), "-o", str(data / "results")]) == 0
    assert main([str(data), "-o", str(data / "results")]) == 0
    assert len(read_csv(data / "results" / "summary.csv")) == 1


def test_cli_options(tmp_path):
    write_image(tmp_path / "img.jpg", [(300, 300, 80), (1015, 300, 50), (700, 400, 10)])
    out = tmp_path / "out"
    rc = main([str(tmp_path / "img.jpg"), "-o", str(out), "--units-per-px", "1",
               "--exclude-edge", "--min-diameter", "50", "--diameter", "max_feret", "--no-auto-size"])
    assert rc == 0
    (row,) = read_csv(out / "img_organoids.csv")
    assert float(row["diameter_um"]) == float(row["max_feret_diameter_um"])
    s = read_csv(out / "summary.csv")[0]
    assert s["n_excluded_edge"] == "1" and s["n_excluded_small"] == "1"
    assert FakeSegmenter.hints == [None]  # --no-auto-size: no size hint


def test_image_without_organoids_gets_csv_with_header(tmp_path):
    from cellphi.results import ORGANOID_COLUMNS

    write_image(tmp_path / "empty.jpg", [])
    assert main([str(tmp_path / "empty.jpg"), "-o", str(tmp_path / "out")]) == 0
    with (tmp_path / "out" / "empty_organoids.csv").open() as f:
        assert f.read().strip().split(",") == ORGANOID_COLUMNS("um")
    assert read_csv(tmp_path / "out" / "summary.csv")[0]["n_measured"] == "0"


def test_cli_default_input_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main([]) == 1  # no ./input folder
    assert "input" in capsys.readouterr().err
    write_image(tmp_path / "input" / "x.png", [(300, 300, 80)])
    assert main([]) == 0
    assert (tmp_path / "results" / "x_annotated.jpg").exists()
