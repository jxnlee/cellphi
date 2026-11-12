import numpy as np
import pytest

from cellphi.foreground import add_missed_dark_blobs, dark_foreground, estimate_diameter_px


def brightfield(shape=(768, 1024), disks=(), bg=220, fg=15, seed=0):
    """Synthetic brightfield image: dark disks (cx, cy, r) on a light, slightly noisy background."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, bg, np.float64) + rng.normal(0, 4, shape)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for cx, cy, r in disks:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r**2] = fg
    return np.clip(img, 0, 255).astype(np.uint8)


def grid_disks(r, step, shape=(768, 1024)):
    return [(x, y, r) for y in range(step, shape[0] - step // 2, step) for x in range(step, shape[1] - step // 2, step)]


@pytest.mark.parametrize("r", [20, 45, 90])
def test_estimate_diameter_of_separate_disks(r):
    img = brightfield(disks=grid_disks(r, int(2.6 * r)))
    assert estimate_diameter_px(img) == pytest.approx(2 * r, rel=0.15)


def test_estimate_diameter_of_touching_disks():
    r = 40
    img = brightfield(disks=grid_disks(r, 2 * r))  # neighbours touch
    assert estimate_diameter_px(img) == pytest.approx(2 * r, rel=0.2)


def test_estimate_single_large_organoid():
    img = brightfield(disks=[(500, 380, 250)])
    assert estimate_diameter_px(img) == pytest.approx(500, rel=0.1)


def test_estimate_in_densely_packed_well():
    # Organoids fill most of the frame and the image border is mostly dark.
    r = 60
    img = brightfield(disks=grid_disks(r, int(2.05 * r)) + [(0, 0, 200), (1024, 768, 200)])
    assert estimate_diameter_px(img) == pytest.approx(2 * r, rel=0.2)


def test_rescue_adds_missed_solid_dark_organoid():
    img = brightfield(disks=[(300, 300, 150), (800, 300, 40)])
    masks = np.zeros(img.shape, np.int32)
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    masks[(xx - 800) ** 2 + (yy - 300) ** 2 <= 40**2] = 1  # Cellpose found the small one only
    out, added = add_missed_dark_blobs(masks, img, typical_diameter_px=80)
    assert added == 1
    new = out == 2
    assert new.sum() == pytest.approx(np.pi * 150**2, rel=0.05)
    assert (out == 1).sum() == (masks == 1).sum()  # existing outline untouched


def test_rescue_skips_clusters_and_small_specks():
    # Two touching disks (not solid as one region) and a speck far below the typical size.
    img = brightfield(disks=[(300, 300, 80), (460, 300, 80), (800, 600, 8)])
    out, added = add_missed_dark_blobs(np.zeros(img.shape, np.int32), img, typical_diameter_px=160)
    assert added == 0


def test_dark_foreground_polarity_from_masks():
    img = brightfield(disks=[(300, 300, 60)])
    masks = np.zeros(img.shape, np.int32)
    masks[250:350, 250:350] = 1
    assert dark_foreground(img, masks) is not None
    assert dark_foreground(255 - img, masks) is None
