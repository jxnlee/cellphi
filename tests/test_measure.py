import numpy as np
import pytest

from cellphi.calibration import Calibration
from cellphi.pipeline import DEFAULT_CALIBRATION
from cellphi.measure import (
    EXCLUDED_BORDER,
    EXCLUDED_FRAGMENT,
    EXCLUDED_SMALL,
    MEASURED,
    max_feret,
    measure_masks,
)
from cellphi.results import organoid_rows, summary_row


def disk_mask(shape, center, radius, label=1, masks=None):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    m = np.zeros(shape, np.int32) if masks is None else masks
    m[(xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2] = label
    return m


def test_feret_rectangle_is_exact_diagonal():
    # A 30 x 40 pixel block spans exactly 30 x 40 units -> diagonal 50.
    m = np.zeros((100, 100), np.int32)
    m[10:40, 20:60] = 1
    (o,) = measure_masks(m)
    assert o.feret_px == pytest.approx(50.0)


def test_feret_single_pixel():
    d, _, _ = max_feret(np.array([[5, 5]]))
    assert d == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("radius", [10, 37.5, 80])
def test_feret_disk_matches_true_diameter(radius):
    m = disk_mask((300, 300), (150, 150), radius)
    (o,) = measure_masks(m)
    # Pixelated disk: within ~1 px of the true diameter (2r).
    assert o.feret_px == pytest.approx(2 * radius, abs=1.5)
    assert o.equivalent_diameter_px == pytest.approx(2 * radius, rel=0.03)


def test_edge_organoids_are_measured_and_flagged_by_default():
    m = disk_mask((200, 200), (100, 100), 30, label=1)
    disk_mask((200, 200), (5, 100), 20, label=2, masks=m)  # cut off by the left edge
    orgs = {o.label: o for o in measure_masks(m)}
    assert orgs[1].status == MEASURED and not orgs[1].touches_border
    assert orgs[2].status == MEASURED and orgs[2].touches_border and orgs[2].id is not None


def test_edge_exclusion_and_min_size():
    m = disk_mask((200, 200), (100, 100), 30, label=1)
    disk_mask((200, 200), (5, 100), 20, label=2, masks=m)  # cut off by the left edge
    disk_mask((200, 200), (170, 170), 4, label=3, masks=m)  # small fragment
    orgs = {o.label: o for o in measure_masks(m, exclude_border=True, min_diameter_px=20)}
    assert orgs[1].status == MEASURED and orgs[1].id == 1
    assert orgs[2].status == EXCLUDED_BORDER and orgs[2].id is None
    assert orgs[3].status == EXCLUDED_SMALL


def test_overlap_lens_is_excluded_and_given_back():
    # Two overlapping disks; the overlap (lens) gets its own label, as Cellpose does.
    shape = (300, 400)
    a = disk_mask(shape, (150, 150), 60) > 0
    b = disk_mask(shape, (250, 150), 60) > 0
    m = np.zeros(shape, np.int32)
    m[a] = 1
    m[b] = 2
    m[a & b] = 3
    orgs = {o.label: o for o in measure_masks(m)}
    assert orgs[3].status == EXCLUDED_FRAGMENT and orgs[3].id is None
    for label in (1, 2):
        assert orgs[label].status == MEASURED
        # With the lens given back, each disk measures its full size again.
        assert orgs[label].area_px == pytest.approx(np.pi * 60**2, rel=0.02)
        assert orgs[label].feret_px == pytest.approx(120, abs=1.5)

    kept = {o.label: o.status for o in measure_masks(m, exclude_fragments=False)}
    assert kept[3] == MEASURED


def test_isolated_elongated_object_is_not_a_fragment():
    m = np.zeros((200, 200), np.int32)
    m[90:110, 40:160] = 1  # 20 x 120 bar, nothing around it
    (o,) = measure_masks(m)
    assert o.status == MEASURED


def test_border_margin_catches_masks_stopping_short_of_edge():
    # Cellpose masks often end 1 px before the edge.
    m = disk_mask((200, 200), (100, 190), 30)
    m[-1, :] = 0
    m[:, -1] = 0
    (o,) = measure_masks(m)
    assert o.touches_border
    (o,) = measure_masks(m, exclude_border=True)
    assert o.status == EXCLUDED_BORDER


def test_ids_in_reading_order():
    m = np.zeros((300, 300), np.int32)
    disk_mask(m.shape, (250, 50), 20, label=1, masks=m)  # top right
    disk_mask(m.shape, (50, 55), 20, label=2, masks=m)  # top left (slightly lower)
    disk_mask(m.shape, (150, 200), 20, label=3, masks=m)  # bottom
    ids = {o.label: o.id for o in measure_masks(m)}
    assert ids == {2: 1, 1: 2, 3: 3}


def test_default_calibration_matches_manual_measurements():
    # 1800 um = 370 px on a 926 px wide export -> 2.20 um/px on the 2048x1536 camera images.
    assert DEFAULT_CALIBRATION.units_per_px(2048, 1536) == pytest.approx(2.1996, abs=1e-4)
    assert DEFAULT_CALIBRATION.units_per_px(1024, 768) == pytest.approx(2 * 2.1996, abs=1e-3)


def test_diameter_definitions_on_rectangle():
    m = np.zeros((100, 100), np.int32)
    m[10:40, 20:60] = 1  # 30 x 40 px
    (o,) = measure_masks(m)
    assert o.diameter_px("max_feret") == pytest.approx(50.0)
    assert o.diameter_px("min_feret") == pytest.approx(30.0, abs=0.1)
    assert o.diameter_px("equivalent") == pytest.approx(np.sqrt(4 * 1200 / np.pi))
    assert 30 < o.diameter_px("mean_feret") < 50
    with pytest.raises(ValueError):
        o.diameter_px("bogus")


def test_diameter_definitions_agree_on_a_disk():
    (o,) = measure_masks(disk_mask((300, 300), (150, 150), 80))
    for method in ("equivalent", "max_feret", "mean_feret", "min_feret"):
        assert o.diameter_px(method) == pytest.approx(160, rel=0.02), method


def test_calibration_scales_with_resolution():
    cal = Calibration(1800, 370, reference_width=926, reference_height=695)
    assert cal.units_per_px(926, 695) == pytest.approx(1800 / 370)
    assert cal.units_per_px(2048, 1536) == pytest.approx(1800 / 370 * 926 / 2048)
    assert Calibration.from_units_per_px(2.2).units_per_px(5000) == pytest.approx(2.2)


def test_calibration_warns_on_aspect_mismatch():
    cal = Calibration(1800, 370, reference_width=926, reference_height=695)
    with pytest.warns(UserWarning):
        cal.units_per_px(2048, 2048)


def test_results_rows():
    m = disk_mask((200, 200), (100, 100), 30)
    orgs = measure_masks(m)
    rows = organoid_rows("a.jpg", orgs, 2.0, "um")
    assert rows[0]["id"] == 1 and rows[0]["touches_edge"] == "no"
    assert rows[0]["diameter_um"] == pytest.approx(orgs[0].equivalent_diameter_px * 2.0, abs=0.06)
    assert rows[0]["max_feret_diameter_um"] == pytest.approx(orgs[0].feret_px * 2.0, abs=0.06)
    rows = organoid_rows("a.jpg", orgs, 2.0, "um", diameter_method="max_feret")
    assert rows[0]["diameter_um"] == rows[0]["max_feret_diameter_um"]
    s = summary_row("a.jpg", orgs, 2.0, "um")
    assert s["n_measured"] == 1 and s["sd_diameter_um"] == "" and s["diameter_method"] == "equivalent"


def test_summary_separates_complete_and_edge_organoids():
    m = disk_mask((200, 200), (100, 100), 30, label=1)
    disk_mask((200, 200), (5, 100), 20, label=2, masks=m)
    s = summary_row("a.jpg", measure_masks(m), 1.0, "um")
    assert s["n_measured"] == 2 and s["n_touching_edge"] == 1
    assert s["complete_median_diameter_um"] == pytest.approx(60, abs=1.5)


@pytest.mark.parametrize("method", ["equivalent", "max_feret", "mean_feret", "min_feret"])
def test_diameter_line_length_equals_reported_diameter(method):
    from cellphi.measure import diameter_line

    m = np.zeros((300, 300), np.int32)
    yy, xx = np.mgrid[:300, :300]
    m[((xx - 150) / 90) ** 2 + ((yy - 140) / 60) ** 2 <= 1] = 1  # ellipse 180 x 120 px
    (o,) = measure_masks(m)
    p1, p2 = diameter_line(o, method)
    assert np.hypot(*(p2 - p1)) == pytest.approx(o.diameter_px(method), rel=1e-6)
    # the line crosses the organoid: its midpoint lies inside the outline
    mid = np.round((p1 + p2) / 2).astype(int)
    assert m[mid[1], mid[0]] == 1


def test_min_feret_line_spans_the_narrow_axis():
    from cellphi.measure import diameter_line

    m = np.zeros((200, 200), np.int32)
    m[50:80, 20:180] = 1  # 30 px tall, 160 px wide bar
    (o,) = measure_masks(m)
    p1, p2 = diameter_line(o, "min_feret")
    assert abs(p2[0] - p1[0]) < 1 and abs(p2[1] - p1[1]) == pytest.approx(30, abs=0.5)  # vertical
    assert min(p1[1], p2[1]) == pytest.approx(49.5, abs=0.5)  # from the top edge...
    assert max(p1[1], p2[1]) == pytest.approx(79.5, abs=0.5)  # ...to the bottom edge


def test_annotate_draws_legend_and_line():
    from cellphi.draw import annotate

    img = np.full((300, 400, 3), 200, np.uint8)
    m = np.zeros((300, 400), np.int32)
    m[100:200, 150:250] = 1
    out = annotate(img, measure_masks(m), 2.2)
    assert (out[5:30, 5:200] < 50).mean() > 0.5  # dark legend box top-left
    plain = annotate(img, measure_masks(m), 2.2, legend=False, label="none")
    inside = plain[105:195, 155:245]  # square interior, away from the outline
    assert (inside != 200).any(axis=-1).sum() > 80  # a diameter line crosses it
