"""Pixel -> physical unit conversion."""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    """Converts pixel lengths to physical units (e.g. micrometers).

    The calibration comes from a scale bar: ``bar_length`` units span ``bar_pixels`` pixels
    on a reference image that is ``reference_width`` pixels wide.

    If the image being measured has the same field of view but a different resolution
    (e.g. the reference was a downsized export of the same camera frame), the scale is
    adjusted in proportion to the width, so images are measured at full native resolution
    instead of being resized.

    If ``reference_width`` is None, the scale bar is assumed to have been measured on
    an image of the same resolution as the one being measured.
    """

    bar_length: float
    bar_pixels: float
    reference_width: int | None = None
    reference_height: int | None = None
    unit: str = "um"

    def __post_init__(self) -> None:
        if self.bar_length <= 0 or self.bar_pixels <= 0:
            raise ValueError("Scale bar length and pixel count must be positive.")
        if self.reference_width is not None and self.reference_width <= 0:
            raise ValueError("Reference width must be positive.")

    @classmethod
    def from_units_per_px(cls, units_per_px: float, unit: str = "um") -> "Calibration":
        """A calibration given directly as units per pixel at the measured image's resolution."""
        return cls(bar_length=units_per_px, bar_pixels=1.0, unit=unit)

    def units_per_px(self, image_width: int, image_height: int | None = None) -> float:
        """Physical size of one pixel for an image of the given dimensions."""
        factor = self.bar_length / self.bar_pixels
        if self.reference_width is None:
            return factor
        if image_height is not None and self.reference_height is not None:
            ref_aspect = self.reference_width / self.reference_height
            img_aspect = image_width / image_height
            if abs(ref_aspect - img_aspect) / ref_aspect > 0.01:
                warnings.warn(
                    f"Image aspect ratio ({image_width}x{image_height}) differs from the "
                    f"calibration reference ({self.reference_width}x{self.reference_height}). "
                    "The image may be cropped relative to the reference, so the scale may be wrong.",
                    stacklevel=2,
                )
        return factor * self.reference_width / image_width

    def describe(self, image_width: int, image_height: int | None = None) -> str:
        upp = self.units_per_px(image_width, image_height)
        if self.reference_width is None:
            src = f"{self.bar_length:g} {self.unit} = {self.bar_pixels:g} px"
        else:
            src = (
                f"{self.bar_length:g} {self.unit} = {self.bar_pixels:g} px "
                f"on a {self.reference_width} px wide reference"
            )
        return f"{upp:.4f} {self.unit}/px ({src})"
