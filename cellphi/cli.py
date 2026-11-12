"""Command-line interface: ``cellphi IMAGE_OR_FOLDER ... [options]``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .calibration import Calibration
from .draw import LABEL_MODES, annotate
from .measure import DEFAULT_DIAMETER, DIAMETER_METHODS
from .pipeline import DEFAULT_CALIBRATION, find_images, process_image
from .results import organoid_rows, summary_row, write_csv


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 926x695") from None


def build_parser() -> argparse.ArgumentParser:
    d = DEFAULT_CALIBRATION
    p = argparse.ArgumentParser(
        prog="cellphi",
        description="Segment organoids with Cellpose and measure their diameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="*", default=["input"],
                   help="image files and/or folders (folders are searched recursively, including subfolders); "
                        "default: the 'input' folder")
    p.add_argument("--no-recursive", action="store_true", help="only use images directly inside the given folders")
    p.add_argument("-o", "--output", default="results", help="output folder")
    p.add_argument("--version", action="version", version=f"cellphi {__version__}")

    cal = p.add_argument_group(
        "calibration",
        "How pixels convert to real units. Either give --units-per-px directly, or a scale bar "
        "(--scale-bar LENGTH PIXELS) plus the size of the image it was measured on (--reference-size). "
        f"Default: {d.bar_length:g} {d.unit} = {d.bar_pixels:g} px on a "
        f"{d.reference_width}x{d.reference_height} image.",
    )
    cal.add_argument("--units-per-px", type=float, help="real units per pixel at the input images' resolution")
    cal.add_argument("--scale-bar", nargs=2, type=float, metavar=("LENGTH", "PIXELS"),
                     help="scale bar length in real units and in pixels")
    cal.add_argument("--reference-size", type=_size, metavar="WxH",
                     help="size of the image the scale bar was measured on (omit if same as input images)")
    cal.add_argument("--unit", default=d.unit, help="unit name used in outputs")

    m = p.add_argument_group("measurement")
    m.add_argument("--diameter", choices=DIAMETER_METHODS, default=DEFAULT_DIAMETER,
                   help="how diameter is defined: 'equivalent' = diameter of a circle with the same area "
                        "(matches manual ImageJ measurements best); 'max_feret' / 'min_feret' = longest / "
                        "shortest caliper width; 'mean_feret' = average caliper width")
    m.add_argument("--exclude-edge", action="store_true",
                   help="exclude organoids touching the image edge (by default they are measured and "
                        "flagged, since their diameter is underestimated)")
    m.add_argument("--keep-fragments", action="store_true",
                   help="do not exclude the lens-shaped regions where two organoids overlap")
    m.add_argument("--min-diameter", type=float, default=0.0,
                   help="exclude objects smaller than this diameter (real units), e.g. debris or fragments")

    s = p.add_argument_group("segmentation (Cellpose)")
    s.add_argument("--cpu", action="store_true", help="do not use a GPU even if available")
    s.add_argument("--model", default="cpsam", help="Cellpose model name or path to a custom/fine-tuned model")
    s.add_argument("--expected-diameter", type=float,
                   help="typical organoid diameter (real units). Normally estimated automatically; "
                        "set it if organoids are missed or split")
    s.add_argument("--no-auto-size", action="store_true",
                   help="don't estimate the organoid size from the image; run Cellpose without a size hint")
    s.add_argument("--no-dark-fallback", action="store_true",
                   help="don't add solid dark organoids that Cellpose missed (e.g. a single very large one)")
    s.add_argument("--flow-threshold", type=float, default=0.4,
                   help="raise (e.g. 0.6) to accept more irregular shapes, lower to be stricter")
    s.add_argument("--cellprob-threshold", type=float, default=0.0,
                   help="lower (e.g. -1) to find fainter organoids / larger outlines, raise for fewer/smaller")

    o = p.add_argument_group("output")
    o.add_argument("--label", choices=LABEL_MODES, default="both", help="text drawn on each organoid")
    o.add_argument("--hide-excluded", action="store_true", help="do not draw outlines of excluded objects")
    o.add_argument("--save-masks", action="store_true", help="also save the label masks (16-bit PNG)")
    return p


def calibration_from_args(args) -> Calibration:
    if args.units_per_px is not None:
        if args.scale_bar or args.reference_size:
            raise SystemExit("Use either --units-per-px or --scale-bar/--reference-size, not both.")
        return Calibration.from_units_per_px(args.units_per_px, unit=args.unit)
    if args.scale_bar:
        ref_w, ref_h = args.reference_size if args.reference_size else (None, None)
        return Calibration(args.scale_bar[0], args.scale_bar[1], ref_w, ref_h, unit=args.unit)
    if args.reference_size:
        raise SystemExit("--reference-size needs --scale-bar.")
    d = DEFAULT_CALIBRATION
    return Calibration(d.bar_length, d.bar_pixels, d.reference_width, d.reference_height, unit=args.unit)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    warnings.simplefilter("default", UserWarning)
    warnings.filterwarnings("ignore", module=r"cellpose\..*")
    warnings.filterwarnings("ignore", message=r".*[Ss]parse.*")

    calibration = calibration_from_args(args)
    out = Path(args.output)
    try:
        images = find_images(args.inputs, recursive=not args.no_recursive, exclude_dirs=[out])
    except FileNotFoundError as e:
        hint = " (give image files or folders, e.g. 'cellphi my_images/')" if args.inputs == ["input"] else ""
        print(f"Error: {e}{hint}", file=sys.stderr)
        return 1
    if not images:
        print("Error: no images found.", file=sys.stderr)
        return 1
    print(f"Found {len(images)} image(s).")

    from .segment import SegmentationSettings, Segmenter

    print(f"Loading Cellpose model '{args.model}'...")
    segmenter = Segmenter(gpu=not args.cpu, model=args.model)
    print(f"Running on {segmenter.device_name}.")
    if segmenter.device.type == "cpu":
        print("  (CPU is slow: expect a minute or more per image.)")

    out.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []

    for i, item in enumerate(images, start=1):
        path = item.path
        t0 = time.time()
        print(f"[{i}/{len(images)}] {item.name}")
        settings = SegmentationSettings(flow_threshold=args.flow_threshold, cellprob_threshold=args.cellprob_threshold)
        try:
            res = process_image(
                path, segmenter, calibration, settings,
                name=item.name,
                expected_diameter=args.expected_diameter,
                min_diameter=args.min_diameter,
                exclude_border=args.exclude_edge,
                exclude_fragments=not args.keep_fragments,
                auto_size=not args.no_auto_size,
                rescue_dark=not args.no_dark_fallback,
                diameter_method=args.diameter,
            )
        except Exception as e:  # keep going with the other images
            print(f"  Error processing {path}: {e}", file=sys.stderr)
            continue

        h, w = res.image.shape[:2]
        print(f"  {w}x{h} px, calibration {calibration.describe(w, h)}")
        if res.size_hint and not args.expected_diameter:
            print(f"  estimated typical organoid size: {res.size_hint:.0f} {res.unit}")
        if res.n_rescued:
            print(f"  added {res.n_rescued} dark organoid(s) that Cellpose missed")
        # Mirror the input's subfolders in the output folder.
        stem = out / Path(item.name).with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{stem}_annotated.jpg",
                    annotate(res.image, res.organoids, res.units_per_px, label=args.label,
                             show_excluded=not args.hide_excluded, diameter_method=args.diameter, unit=res.unit),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if args.save_masks:
            cv2.imwrite(f"{stem}_masks.png", res.masks.astype(np.uint16))

        all_rows += organoid_rows(res.name, res.organoids, res.units_per_px, res.unit, args.diameter)
        summary = summary_row(res.name, res.organoids, res.units_per_px, res.unit, args.diameter)
        summaries.append(summary)
        excluded = [f"{summary[k]} {what}" for k, what in [
            ("n_excluded_edge", "at edge"), ("n_excluded_overlap_fragments", "overlap fragments"),
            ("n_excluded_small", "too small")] if summary[k]]
        print(f"  {summary['n_measured']} organoids measured ({summary['n_touching_edge']} touching the edge)"
              + (f", excluded: {', '.join(excluded)}" if excluded else "")
              + f"; median diameter {summary[f'median_diameter_{res.unit}']} {res.unit}"
              f"  [{time.time() - t0:.1f}s]")

    if not summaries:
        return 1
    write_csv(out / "organoids.csv", all_rows)
    write_csv(out / "summary.csv", summaries)
    print(f"Results written to {out}/ (organoids.csv, summary.csv, *_annotated.jpg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
