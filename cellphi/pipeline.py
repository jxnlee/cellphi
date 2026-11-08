"""Read an image, segment, measure."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .measure import Organoid, measure_masks

# Lab default: a scale bar of 1800 um measured as 370 px on a 926x695 px export of the
# microscope image (the raw camera images are 2048x1536). Assumes the same field of view.
DEFAULT_CALIBRATION = Calibration(
    bar_length=1800.0, bar_pixels=370.0, reference_width=926, reference_height=695, unit="um"
)

# Use the size-hinted Cellpose pass only if it measures at least this many times more organoids.
AUTO_SIZE_MIN_GAIN = 1.25

IMAGE_EXTENSIONS ={".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageResult:
    name: str
    image: np.ndarray = field(repr=False)  # as read (BGR / grayscale)
    masks: np.ndarray = field(repr=False)
    organoids: list[Organoid] = field(repr=False)
    units_per_px: float = 1.0
    unit: str = "um"
    size_hint: float | None = None  # organoid size (units) given to Cellpose, if one was used

    @property
    def measured(self) -> list[Organoid]:
        return sorted((o for o in self.organoids if o.id is not None), key=lambda o: o.id)

    def diameters(self) -> list[float]:
        return [o.feret_px * self.units_per_px for o in self.measured]


def read_image(path: str | Path) -> np.ndarray:
    """Read an image at full resolution (BGR or grayscale, 8 or 16 bit)."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None and path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        img = tifffile.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _for_cellpose(img: np.ndarray) -> np.ndarray:
    """BGR(A) -> RGB; grayscale unchanged."""
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def process_image(
    image: str | Path | np.ndarray,
    segmenter,
    calibration: Calibration = DEFAULT_CALIBRATION,
    settings=None,
    expected_diameter: float | None = None,
    min_diameter: float = 0.0,
    exclude_border: bool = True,
    exclude_fragments: bool = True,
    auto_size: bool = True,
    name: str | None = None,
) -> ImageResult:
    """Segment and measure one image at its native resolution.

    expected_diameter / min_diameter are in calibration units (e.g. um).
    If expected_diameter is not given and auto_size is on, a second Cellpose pass with an
    estimated organoid size is tried and used when it finds clearly more organoids.
    """
    from .segment import SegmentationSettings

    if isinstance(image, (str, Path)):
        name = name or Path(image).name
        img = read_image(image)
    else:
        img = image
        name = name or "image"

    h, w = img.shape[:2]
    upp = calibration.units_per_px(w, h)

    settings = settings or SegmentationSettings()
    x = _for_cellpose(img)

    def measure(m):
        return measure_masks(
            m,
            exclude_border=exclude_border,
            min_diameter_px=(min_diameter or 0.0) / upp,
            exclude_fragments=exclude_fragments,
        )

    def n_measured(orgs):
        return sum(o.id is not None for o in orgs)

    size_hint_px = None
    if expected_diameter:
        size_hint_px = expected_diameter / upp
        masks = segmenter.segment(x, dataclasses.replace(settings, diameter_px=size_hint_px))
        organoids = measure(masks)
    else:
        masks = segmenter.segment(x, settings)
        organoids = measure(masks)
        if auto_size:
            # Cellpose works best when it knows the object size. Estimate it from the first
            # pass and retry; keep the retry only if it finds clearly more organoids (the first
            # pass gives slightly tighter outlines when it works).
            first = measure_masks(masks, exclude_border=False, exclude_fragments=False)
            if first:
                est = float(np.median([o.equivalent_diameter_px for o in first]))
                masks2 = segmenter.segment(x, dataclasses.replace(settings, diameter_px=est))
                organoids2 = measure(masks2)
                if n_measured(organoids2) >= AUTO_SIZE_MIN_GAIN * max(n_measured(organoids), 1):
                    masks, organoids, size_hint_px = masks2, organoids2, est

    return ImageResult(
        name=name,
        image=img,
        masks=masks,
        organoids=organoids,
        units_per_px=upp,
        unit=calibration.unit,
        size_hint=size_hint_px * upp if size_hint_px else None,
    )


def find_images(inputs: list[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of image paths."""
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTENSIONS)
        elif p.exists():
            paths.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    return paths
