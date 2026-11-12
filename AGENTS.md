# AGENTS.md — Cellphi

Guidance for AI coding agents (Claude Code, Codex, etc.) and human contributors working in this repo.
`CLAUDE.md` just imports this file; edit this one.

## What this project is

Cellphi measures **organoids** in brightfield microscope images:
1. estimates the typical organoid size from the image (classic image analysis),
2. segments organoids with [Cellpose](https://github.com/MouseLand/cellpose) (v4, `cpsam`) using that size,
3. adds solid dark organoids Cellpose missed, cleans up overlap fragments, flags edge-cut organoids,
4. measures each organoid's diameter (default: equal-area diameter) in µm,
5. writes an annotated image + CSVs.

The end user is a biology researcher, **not an ML expert**. Defaults must work without tuning,
and any model/parameter choice should be explained in plain language.

## Goals (priority order)

1. **Diameter per organoid**: annotated image + CSV mapping organoid ID → diameter (µm). Done.
2. Simple **CLI** (`cellphi`). Done.
3. Later: more metrics (circularity), plots.
4. Later: an organoid-specific model (fine-tuned Cellpose), maybe a GUI.

Non-goals: classifying live vs dead organoids.

## The data this is tuned for

`input/` (not tracked): `Cortical Organoids/` and `Midbrain Organoids/`, organised as
`Plate …/<well>/day 23 … KS2176xx.jpg`, one image per well. All images are 2048×1536 brightfield JPEGs.
- Mostly solid black organoids on a light background, with sizes from ~250 µm to ~4 mm (median ~400–650 µm).
  Some wells are dense, with many touching organoids. Cortical Plate 4 organoids are semi-transparent.
- Unusual cases: Midbrain `Plate 4-3` = one very large organoid per well (one image overexposed, with a
  dark shadow band at the edge). Midbrain `Plate 5` has blue fibres. `Plate 3-2 B3` is blurry.

`reference/Plate 1 CM_CM/<well>/` (Cortical): the lab's manual ImageJ measurements for six wells,
with `Results.csv` (one row per hand-drawn diameter line: `Length` in µm, `Angle`) and a `Clipboard.jpg`
screenshot of the lines. Each well folder also has a second, unmeasured image. The measured one is the one in `input/`.
`tools/extract_reference_lines.py` turns these into `tests/data/reference_lines.json` (line positions
recovered from the screenshot, 285 of 350 lines matched one-to-one).

The user may give new data. Keep defaults working for "similar" data, and keep the pipeline functional
on different data (other resolutions, contrast, sizes).

## Target environments

Develop on **WSL2 (Linux, RTX 5070 GPU → needs the CUDA ≥ 12.8 torch build)**. Must also run on **macOS**
(CPU/MPS), **Google Colab**, and shared Linux machines. So:
- GPU optional with automatic fallback (`segment.pick_device`: CUDA → MPS → CPU);
- no GUI-only dependencies (PyQt) in the install; use `opencv-python-headless`;
- `requirements.txt` installs everything (incl. `-e .`) and uses `--extra-index-url .../cu128` for torch.
  That works with pip on all platforms. On macOS `uv` may fail on it because uv uses the first index that has the package.

## Layout

| Path | What it is |
|---|---|
| `cellphi/calibration.py` | `Calibration`: scale bar (length, px, reference image size) → units/px for any image width. |
| `cellphi/foreground.py` | Classic image analysis: `estimate_diameter_px` (size hint), `add_missed_dark_blobs` (fallback), `dark_foreground`, `to_gray8`. |
| `cellphi/segment.py` | `Segmenter` (wraps `CellposeModel`, lazy imports torch/cellpose), `SegmentationSettings`. |
| `cellphi/measure.py` | `measure_masks()`: per-label measurement (Feret max/min/mean, area), fragment/edge/size handling, IDs, `DIAMETER_METHODS`. Pure numpy/OpenCV. |
| `cellphi/pipeline.py` | `process_image()`: read → size estimate → Cellpose → fallback → measure. `DEFAULT_CALIBRATION`. `find_images()`: recursive folder search. |
| `cellphi/draw.py` | `annotate()`: outlines (yellow / orange = edge / red = excluded), labels, the diameter line (`measure.diameter_line`), legend. |
| `cellphi/results.py` | CSV rows (`organoid_rows`, `summary_row`, `write_csv`, `ORGANOID_COLUMNS`). Output: one `<image>_organoids.csv` per image next to its `_annotated.jpg` (user request, replaced the combined organoids.csv), plus one `summary.csv` for all images. |
| `cellphi/cli.py` | `cellphi` command (argparse). |
| `tests/` | Fast (CI): synthetic unit tests, `test_cli.py` (CLI with a fake segmenter), `test_real_images.py` (no model). Slow: `test_reference.py`, `test_pipeline_real.py` (Cellpose). |
| `tools/extract_reference_lines.py` | Rebuilds `tests/data/reference_lines.json` from `reference/`. |
| `tools/check_before_push.sh`, `.githooks/pre-push` | Local pre-push check (fast + slow tests). |
| `.github/workflows/tests.yml`, `requirements-ci.txt` | GitHub Actions CI (fast tests, no model). |
| `notebooks/example.ipynb` | Example for Jupyter/Colab. |

## Dev workflow

```bash
source .venv/bin/activate         # venv with Python 3.12; pip install -r requirements.txt
pytest -q                         # fast tests (~5 s), no Cellpose: what CI runs
pytest -m slow                    # Cellpose on real images: accuracy vs manual + reviewed run (~80 s GPU)
tools/check_before_push.sh        # both of the above; also the optional pre-push hook (.githooks/)
cellphi -o /tmp/results --save-masks   # full run on ./input, ~6 s/image on the RTX 5070
```

CI (`.github/workflows/tests.yml`): fast tests on ubuntu (3.10, 3.12), macOS, Windows, installed from
`requirements-ci.txt` + `pip install -e . --no-deps` (no Cellpose/torch). So **nothing imported at module
level may need cellpose/torch**: keep those imports inside functions (`segment.py`), as now. The fast CLI
test (`tests/test_cli.py`) replaces `cellphi.segment.Segmenter` with a threshold-based fake; keep the CLI
importing `Segmenter` from `cellphi.segment` inside `main()` so the monkeypatch works. If you add a core
dependency, add it to `requirements-ci.txt` too.

Tests: `test_measure.py`, `test_foreground.py`, `test_find_images.py`, `test_cli.py` (synthetic);
`test_real_images.py` (real images, no model); `test_reference.py`, `test_pipeline_real.py` (slow, Cellpose).

To iterate on measurement logic without re-running Cellpose, save masks once (`--save-masks`) and call
`measure_masks` on them. **Always look at annotated images** after changing segmentation or measurement.
Summary stats alone have hidden problems more than once.

Useful unsupervised check for this data: Otsu-threshold the dark pixels and compute **coverage**
(share of dark pixels inside an outline) and **spill** (share of outline pixels that are light).
Current pipeline: mean coverage 0.99 over all 84 images. The only low one is Plate 4-3 A3 (0.69),
where the uncovered dark pixels are an edge shadow and correctly have no outline. Spill is ~0.06,
higher (~0.2) for the semi-transparent Cortical Plate 4 organoids, whose light rims are correctly included.

## Decisions and the evidence behind them

- **Calibration 2.20 µm/px** at 2048×1536: 1800 µm = 370 px on a 926×695 export. The user first said
  370 px applies to the 2048 image (4.865 µm/px), but the manual ImageJ lines in `reference/` all imply
  2.19–2.24 µm/px, and the user then confirmed 2.20. The test `test_default_calibration_matches_manual_measurements` pins this.
- **Size hint from the image, one Cellpose pass.** Without a `diameter` hint, `cpsam` covered only ~33% of the
  dark organoid pixels on this data (69% of hand-measured organoids found). The old approach (estimate the
  size from a first Cellpose pass, then retry) failed when the first pass found junk (e.g. a 12 px estimate
  for a single 600 px organoid). `estimate_diameter_px` uses the distance transform of the Otsu dark
  foreground: local maxima ≈ organoid radii, one peak per organoid (non-maximum suppression), area-weighted median.
  It is within 0.93–1.31× of Cellpose's own median size, takes ~15 ms, and is twice as fast overall as two passes.
  - No hole filling in the estimate (it merged packed clusters into one blob: 8× overestimates).
  - No polarity check from the image alone. Border-based and connectivity-based "which side is background"
    rules were wrong on 51 and 38 of the 84 images (dense wells). Dark objects are assumed. The fallback checks
    polarity against Cellpose's outlines.
- **Dark-blob fallback** (`add_missed_dark_blobs`): Cellpose finds nothing for single very large, featureless
  dark organoids at any hint or threshold tried. Unlabelled dark regions (3 px gap from existing outlines)
  that are solid (≥ 0.9) and ≥ 0.5× the typical diameter are added. On the 84 images it adds exactly 14,
  all checked by eye: 5 single-organoid wells, 1 giant organoid, and 8 edge-cut organoids.
- **Default diameter = equivalent (equal-area) diameter.** Against 284 hand-measured organoids: median ratio
  1.004, mean abs. error 6.9%, 79% within 10%. Max Feret reads +10%, so the lab measures a "typical"
  diameter, not the longest line. Max/min Feret are still in the CSV; `--diameter` switches.
- **The drawn line is the reported diameter** (user request). `diameter_line`: exact longest line for max_feret;
  otherwise the chord through the centroid whose length is closest to the diameter, scaled to exactly that length.
  An earlier version placed it by caliper extent instead, and the lines left the outline on irregular organoids.
- **Edge organoids are measured and flagged** (user's choice), with a 2 px margin (`BORDER_MARGIN_PX`),
  because Cellpose masks often stop 1 px short. `summary.csv` has `complete_` stats without them.
- **Overlap fragments** (semi-transparent organoids overlapping): a lens-shaped label (Feret/eq-diam ≥ 1.5,
  ≥ 50% of its outline touching others), or a small piece ≥ 90% enclosed and < 0.5× typical size, is excluded
  and its pixels are given to the organoids whose convex hull holds ≥ 20% of it. Rarely triggers on the current
  (opaque) data, but is kept for semi-transparent organoids.
- **Robustness:** half-resolution copies give the same µm results within 2% (and are ~5× faster), and
  darker/low-contrast copies give identical results.

## Known limitations

- Organoids mostly hidden behind others are measured from their visible part; not flagged.
- Assumes brightfield (dark organoids, light background). For other modalities use `--no-auto-size`
  (and `--no-dark-fallback`), or give `--expected-diameter`.
- Thresholds were tuned on this data. Re-check on new image types by looking at annotated output.
- Fine-tuning Cellpose on a few hand-corrected images would likely beat the fallback rules.

## Conventions

- OpenCV BGR when drawing (`(0, 255, 255)` = yellow). Cellpose gets RGB (`pipeline._for_cellpose`).
- `cv2.findContours` gives `(x, y)`; numpy indexing is `(row, col)` = `(y, x)`.
- Keep measurement/foreground logic free of Cellpose/IO so it stays unit-testable on synthetic images.
- Lengths in µm, with the calibration recorded in the CSV.
- Tunable thresholds are module-level constants with a comment on where the value came from.
