"""Cellpose segmentation. Cellpose/torch are imported lazily so the measurement code
can be used and tested without them."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SegmentationSettings:
    """Cellpose settings. The defaults work for most brightfield organoid images.

    diameter_px:        typical organoid diameter in pixels. None = let Cellpose use the image
                        as is. Setting it makes Cellpose rescale internally so organoids are
                        the size it works best on. Masks are still returned at full resolution.
    flow_threshold:     higher = accept more (irregular) shapes; lower = stricter. 0.4 default.
    cellprob_threshold: lower (e.g. -1) = larger masks / find fainter organoids; higher = smaller/fewer.
    min_size_px:        discard masks smaller than this many pixels (area).
    """

    diameter_px: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size_px: int = 15
    max_size_fraction: float = 0.4
    batch_size: int = 8


def pick_device(gpu: bool = True):
    """Return (torch.device, description). Tries CUDA (NVIDIA), then MPS (Apple Silicon), then CPU."""
    import torch

    if gpu:
        if torch.cuda.is_available():
            return torch.device("cuda"), f"GPU ({torch.cuda.get_device_name(0)})"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps"), "Apple GPU (MPS)"
    return torch.device("cpu"), "CPU"


class Segmenter:
    """Wraps a Cellpose model. Load it once and reuse it across images."""

    def __init__(self, gpu: bool = True, model: str = "cpsam"):
        from cellpose import models

        self.device, self.device_name = pick_device(gpu)
        use_gpu = self.device.type != "cpu"
        self.model = models.CellposeModel(gpu=use_gpu, device=self.device, pretrained_model=model)

    def segment(self, img: np.ndarray, settings: SegmentationSettings | None = None) -> np.ndarray:
        """Return a 2D integer label mask (0 = background, 1..N = organoids) at the input's resolution."""
        s = settings or SegmentationSettings()
        masks, _flows, _styles = self.model.eval(
            img,
            diameter=s.diameter_px,
            flow_threshold=s.flow_threshold,
            cellprob_threshold=s.cellprob_threshold,
            min_size=s.min_size_px,
            max_size_fraction=s.max_size_fraction,
            batch_size=s.batch_size,
        )
        return np.asarray(masks)
