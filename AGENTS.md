# AGENTS.md — Cellphi

Guidance for AI coding agents (Claude Code, Codex, etc.) and human contributors working in this repo.
`CLAUDE.md` just imports this file; edit this one.

## What this project is

Cellphi measures **organoids** in brightfield microscope images:
1. segments each organoid with [Cellpose](https://github.com/MouseLand/cellpose) (v4, `cpsam` model),
2. cleans up the segmentation (overlap fragments, edge-cut organoids),
3. measures each organoid's **maximum Feret diameter**, converts it to µm,
4. writes an annotated image + CSVs.

The end user is a biology researcher, **not an ML expert**. Defaults must work without tuning,
and any model/parameter choice should be explained in plain language.

## Goals (priority order)

1. **Diameter per organoid**: annotated image + CSV mapping organoid ID → diameter (µm). *Top priority.* Done.
2. Simple **CLI** (`cellphi`). Done.
3. Later: more metrics (min Feret, circularity), plots.
4. Later: an organoid-specific model (fine-tuned Cellpose), maybe a GUI.

Non-goals: classifying live vs dead organoids.

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
| `cellphi/segment.py` | `Segmenter` (wraps `CellposeModel`, lazy imports torch/cellpose), `SegmentationSettings`. |
| `cellphi/measure.py` | `measure_masks()`: per-label measurement, fragment/border/size exclusion, IDs. Pure numpy/OpenCV. |
| `cellphi/pipeline.py` | `process_image()`: read → segment (with auto size retry) → measure. `DEFAULT_CALIBRATION`. |
| `cellphi/draw.py` | `annotate()`: outlines, Feret lines, `#id`/diameter labels. |
| `cellphi/results.py` | CSV rows (`organoid_rows`, `summary_row`, `write_csv`). |
| `cellphi/cli.py` | `cellphi` command (argparse). |
| `tests/test_measure.py` | Unit tests on synthetic masks (no model needed). |
| `notebooks/example.ipynb` | Example for Jupyter/Colab. |
| `input/` | Sample images, all 2048×1536 JPEG. |

The old `cellphi.py`/`util.py`/`test,py`/`test.ipynb` and generated outputs were removed (still in git commit `0cb48ec`).

## Dev workflow

```bash
source .venv/bin/activate         # venv with Python 3.12; pip install -r requirements.txt
pytest -q                         # fast, no Cellpose needed
cellphi input/ -o /tmp/results    # full run; ~10 s/image on the RTX 5070
```

To iterate on measurement logic without re-running Cellpose, save masks once
(`cellphi ... --save-masks`, or `np.save` from `Segmenter.segment`) and call `measure_masks` on them.
**Always look at the annotated images** after changing measurement or segmentation logic. Summary
stats alone hid problems (e.g. a 690 µm "organoid" that was two organoids plus a lens).

## Measurement decisions (and the evidence behind them)

- **Native resolution + scaled calibration.** The old workflow resized images to 926×695 before applying
  1800 µm/370 px. We now keep 2048×1536 and use `(1800/370)·(926/2048) ≈ 2.20 µm/px`: same units, finer outlines.
  **Open question for the user:** was the 926×695 scale-bar image the same field of view as the camera
  images (just downsized)? If not, the default calibration is wrong. (`util.py` used to claim 2048×1536
  for the same 370 px, which contradicted the notebook.)
- **Feret on pixel corners.** `max_feret` measures between outer pixel corners of the convex hull, not
  pixel centres (removes a ~1 px underestimate). The tests check that a 30×40 px block measures exactly 50.
- **Border exclusion uses a 2 px margin** (`BORDER_MARGIN_PX`). Cellpose masks often stop 1 px short of the edge.
  Complete organoids near the edge were ≥ 6 px away in the samples.
- **Overlap fragments.** Cellpose can't output overlapping masks, so where two organoids overlap the lens
  becomes its own label. Detected as `feret/equivalent_diameter ≥ 1.5` AND `≥ 50 %` of its outline touching
  other labels. In the samples, real organoids were 1.05–1.2, lenses 1.6–2.1 with 0.8–1.0 contact.
  Also a fragment: `≥ 90 %` enclosed and `< 0.5×` the median size (gaps where 3–4 organoids meet).
  A fragment's pixels are merged back into an organoid only if ≥ 20 % of the fragment lies inside that
  organoid's convex hull. Without this check, organoids that merely touched a lens tip absorbed it and measured ~600 µm.
- **Auto size retry.** `cpsam` without a `diameter` hint failed badly on `KS213728.jpg` (solid black organoids:
  4 measured). With `diameter` ≈ median from a first pass it found 85. The hint makes outlines ~1–2 px looser,
  so the hinted pass is used only if it measures ≥ 1.25× more organoids (`AUTO_SIZE_MIN_GAIN`).
  Cellpose 4 has no `invert` argument.

Sample results (default settings): 30 → 118 measured, median 360 µm; 36 → 116, 362 µm; 52 → 711, 140 µm;
KS213728 → 85, 413 µm; test → 677, 147 µm.

## Known limitations

- Organoids mostly hidden behind others are measured from their visible part (underestimated); not flagged.
- Thresholds were tuned on 5 sample images. Re-check on new image types by looking at annotated output.
- Fine-tuning Cellpose on a few hand-corrected images would likely beat the post-processing rules.

## Conventions

- OpenCV BGR when drawing (`(0, 255, 255)` = yellow). Cellpose gets RGB (`pipeline._for_cellpose`).
- `cv2.findContours` gives `(x, y)`; numpy indexing is `(row, col)` = `(y, x)`.
- Keep measurement logic free of Cellpose/IO so it stays unit-testable on synthetic masks.
- Lengths in µm, with the calibration recorded in the CSV.
- Tunable thresholds are module-level constants in `measure.py`/`pipeline.py` with a comment on where the value came from.
