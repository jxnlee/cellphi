"""Read an image, segment, measure."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .foreground import add_missed_dark_blobs, estimate_diameter_px, to_gray8
from .measure import DEFAULT_DIAMETER, Organoid, measure_masks

# Lab default: 1800 um = 370 px, measured on a 926x695 px export of the 2048x1536 camera
# images, i.e. 2.20 um/px at full resolution. Confirmed against the lab's manual ImageJ
# measurements (reference/Plate 1 CM_CM): six hand-drawn lines gave 2.19-2.24 um/px.
DEFAULT_CALIBRATION = Calibration(
    bar_length=1800.0, bar_pixels=370.0, reference_width=926, reference_height=695, unit="um"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageResult:
    name: str
    image: np.ndarray = field(repr=False)  # as read (BGR / grayscale)
    masks: np.ndarray = field(repr=False)
    organoids: list[Organoid] = field(repr=False)
    units_per_px: float = 1.0
    unit: str = "um"
    diameter_method: str = DEFAULT_DIAMETER
    size_hint: float | None = None  # organoid size (units) given to Cellpose
    n_rescued: int = 0  # dark organoids added by the fallback (Cellpose missed them)

    @property
    def measured(self) -> list[Organoid]:
        return sorted((o for o in self.organoids if o.id is not None), key=lambda o: o.id)

    def diameters(self) -> list[float]:
        return [o.diameter_px(self.diameter_method) * self.units_per_px for o in self.measured]


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
    exclude_border: bool = False,
    exclude_fragments: bool = True,
    auto_size: bool = True,
    rescue_dark: bool = True,
    diameter_method: str = DEFAULT_DIAMETER,
    name: str | None = None,
) -> ImageResult:
    """Segment and measure one image at its native resolution.

    1. Organoid size: ``expected_diameter`` (calibration units) if given; otherwise estimated
       from the image (``auto_size``). Cellpose finds far more organoids when told their size
       (reference data: 99% vs 69% of hand-measured organoids).
    2. One Cellpose pass with that size.
    3. ``rescue_dark``: add solid dark organoids Cellpose missed (very large / isolated ones).
    4. Measure. Organoids touching the image edge are measured and flagged, or excluded with
       ``exclude_border``. ``min_diameter`` is in calibration units.
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
    gray = to_gray8(img)

    settings = settings or SegmentationSettings()
    if expected_diameter:
        hint_px = expected_diameter / upp
    elif auto_size:
        hint_px = estimate_diameter_px(gray)
    else:
        hint_px = None
    if hint_px:
        settings = dataclasses.replace(settings, diameter_px=hint_px)

    masks = segmenter.segment(_for_cellpose(img), settings)
    n_rescued = 0
    if rescue_dark:
        masks, n_rescued = add_missed_dark_blobs(masks, gray, hint_px)

    organoids = measure_masks(
        masks,
        exclude_border=exclude_border,
        min_diameter_px=(min_diameter or 0.0) / upp,
        exclude_fragments=exclude_fragments,
        diameter_method=diameter_method,
    )
    return ImageResult(
        name=name,
        image=img,
        masks=masks,
        organoids=organoids,
        units_per_px=upp,
        unit=calibration.unit,
        diameter_method=diameter_method,
        size_hint=hint_px * upp if hint_px else None,
        n_rescued=n_rescued,
    )


@dataclass(frozen=True)
class ImageInput:
    path: Path
    name: str  # path relative to the input folder (e.g. "day3/A1.jpg"); used for output names and CSVs


def _is_cellphi_output(path: Path) -> bool:
    return path.stem.endswith(("_annotated", "_masks"))


def find_images(
    inputs: list[str | Path], recursive: bool = True, exclude_dirs: list[str | Path] = ()
) -> list[ImageInput]:
    """Expand files and folders into a sorted list of images.

    Folders are searched recursively (unless ``recursive=False``). Hidden folders (e.g. .git,
    .venv), folders in ``exclude_dirs`` (e.g. the output folder) and Cellphi's own output files
    are skipped. Names are kept unique so outputs from different subfolders don't overwrite
    each other.
    """
    excluded = [Path(d).resolve() for d in exclude_dirs]
    found: list[ImageInput] = []
    seen_paths: set[Path] = set()
    used_names: set[str] = set()

    def add(path: Path, name: str, root_name: str) -> None:
        resolved = path.resolve()
        if resolved in seen_paths:
            return
        seen_paths.add(resolved)
        if name in used_names:  # same relative name from two different input folders
            name = f"{root_name}/{name}"
        base, n = name, 2
        while name in used_names:
            name = f"{Path(base).with_suffix('')}_{n}{Path(base).suffix}"
            n += 1
        used_names.add(name)
        found.append(ImageInput(path, name))

    # With several input folders, prefix names with the folder's own name so that, e.g.,
    # "Cortical/Plate 1/A1" and "Midbrain/Plate 1/A1" stay apart in the outputs.
    prefix_dirs = sum(Path(i).is_dir() for i in inputs) > 1

    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files: list[Path] = []
            for dirpath, dirnames, filenames in os.walk(p):
                d = Path(dirpath)
                # Prune hidden and excluded folders in place so os.walk doesn't descend into them.
                dirnames[:] = sorted(
                    n for n in dirnames
                    if not n.startswith(".") and not any((d / n).resolve() == e for e in excluded)
                )
                files += [d / f for f in filenames if not f.startswith(".")]
                if not recursive:
                    break
            for f in sorted(files):
                if f.suffix.lower() in IMAGE_EXTENSIONS and not _is_cellphi_output(f):
                    rel = f.relative_to(p).as_posix()
                    add(f, f"{p.resolve().name}/{rel}" if prefix_dirs else rel, p.resolve().name)
        elif p.is_file():
            add(p, p.name, p.resolve().parent.name)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    return found
