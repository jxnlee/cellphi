# Cellphi

Cellphi measures organoids in brightfield microscope images. It finds every organoid with
[Cellpose](https://www.cellpose.org/) and measures each one's **diameter** in micrometers.

For each image you get:
- an **annotated image**: every organoid is outlined and labelled with its number and diameter;
- a **CSV file** with one row per organoid (number, diameter, other size measures, position, whether it touches the image edge);
- a **summary CSV** with the count, mean, median and spread of diameters per image.

Cellphi's diameters agree with manual ImageJ measurements (see [Accuracy](#accuracy)).

## Installation

You need Python 3.10 or newer.

```bash
git clone <this repo> cellphi && cd cellphi
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # installs everything, including GPU-enabled PyTorch
```

- **GPU:** On Linux/Windows with an NVIDIA GPU, the GPU is used automatically. Without one it runs on CPU, just slower.
- **Mac:** the Python that ships with macOS (3.9) is too old. Install Python 3.12 from
  [python.org](https://www.python.org/downloads/) or with `brew install python@3.12`, then create the
  environment with `python3.12 -m venv .venv`. Use plain `pip` as shown above, not `uv pip`:
  `uv` can fail on a Mac because of the extra PyTorch download source.
  - *Apple Silicon (M1–M4):* the Mac's GPU is used automatically.
  - *Intel Macs:* runs on CPU only and needs Python 3.12 or older, because PyTorch has no newer builds for Intel Macs.
- **Google Colab:** Choose *Runtime → Change runtime type → GPU*, then run `!pip install -r requirements.txt` in a cell.
- **Slow connection:** PyTorch is a large download (several GB). If it times out, run the same command again;
  already downloaded files are reused.
- The first run downloads the Cellpose model (about 1 GB).

## Usage

```bash
cellphi input/ -o results/                              # every image in a folder, including all subfolders
cellphi "input/Midbrain Organoids/Plate 1 CM_CM/A1"     # one well
cellphi input/ --no-recursive                           # only images directly in input/, not subfolders
cellphi --help                                          # all options
```

Folders are searched recursively for `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff` and `.bmp` files.
The output folder mirrors the input's subfolders
(e.g. `input/Cortical Organoids/Plate 1 CM_CM/A1/x.jpg` → `results/Cortical Organoids/Plate 1 CM_CM/A1/x_annotated.jpg`),
and the CSVs list each image by that path. Hidden folders, the output folder and Cellphi's own output
images are skipped, so it is safe to re-run on the same folder.

About 6 seconds per image on an NVIDIA laptop GPU (RTX 5070).

### Output

| File | Contents |
|---|---|
| `<image>_annotated.jpg` | **Yellow** outline: measured organoid, labelled `#number` and diameter in µm, with a **line showing that diameter** (its length is the reported value). **Orange** outline and `*` after the diameter: measured, but cut off by the image edge. **Red** outline: excluded (see below). A legend in the top-left corner names the diameter definition and the scale. |
| `organoids.csv` | One row per measured organoid: `image, id, diameter_um, touches_edge, equivalent_diameter_um, max_feret_diameter_um, min_feret_diameter_um, area_um2, centroid_x_px, centroid_y_px, um_per_px`. `id` matches the `#number` on the image. |
| `summary.csv` | One row per image: counts, and mean/median/sd/min/max diameter, both for all measured organoids and for `complete_` ones only (not touching the edge). |

### What "diameter" means

Organoids aren't perfect circles, so "diameter" needs a definition. The default, `equivalent`, is the
diameter of a circle with the same area as the organoid. It matched the lab's manual ImageJ measurements
best: on average less than 1% different, with no tendency to read high or low.
On the annotated image, the drawn line is that diameter: for `max_feret` the longest line across; for the
others a line through the organoid's centre, in the direction where the organoid is that wide, so it runs edge to
edge like a hand-drawn diameter.
Other definitions are in the CSV, and `--diameter` picks which one is used as `diameter_um` and drawn:

| `--diameter` | Meaning | Compared with manual measurements |
|---|---|---|
| `equivalent` (default) | diameter of the equal-area circle | +0.4% (median), 6.9% average difference |
| `mean_feret` | caliper width averaged over all directions | +2.8%, 7.0% |
| `max_feret` | longest caliper width (longest line across) | +9.8%, 11.9% |
| `min_feret` | shortest caliper width | −5.5%, 10.0% |

### Organoids at the image edge

An organoid cut off by the image edge is only partly visible, so its diameter is too small. These are
measured but flagged (`touches_edge = yes`, orange outline, `*`). Use `summary.csv`'s `complete_…` columns,
or filter `touches_edge = no` in `organoids.csv`, for statistics without them. `--exclude-edge` leaves them out entirely.

### What gets excluded (red outlines)

- **Overlap fragments**: where two semi-transparent organoids overlap, Cellpose can label the overlap as a
  separate object. Cellphi recognises these and gives the overlap back to the organoids it belongs to.
- **Too small** (optional): `--min-diameter 200` drops anything under 200 µm, such as debris.
- **Edge organoids**, only with `--exclude-edge`.

Use `--keep-fragments` to keep overlap fragments.

### How it finds organoids

1. **Estimate the typical organoid size** from the image (dark organoids on a light background). Cellpose
   finds far more organoids when told their size: on the manually measured wells, 99% of organoids with
   the size estimate vs. 69% without.
2. **Cellpose** outlines the organoids.
3. **Fallback for dark organoids Cellpose missed.** Cellpose's current model can miss very large,
   uniformly black organoids (e.g. a single big organoid in a well, or one far bigger than its neighbours).
   Any solid, round dark region left without an outline is added as an organoid. Clusters and shadows are not.
   The output says when this happens.
4. **Measure** every outline.

Options if something goes wrong:

| Option | Use it when |
|---|---|
| `--expected-diameter 500` | The size estimate is off (organoids missed or split). Give the typical diameter in µm. |
| `--no-auto-size` | Images aren't dark organoids on a light background (e.g. fluorescence). |
| `--no-dark-fallback` | Dark non-organoid objects are being added. |
| `--flow-threshold`, `--cellprob-threshold` | Advanced Cellpose settings (see `--help`). |
| `--model PATH` | Use a fine-tuned Cellpose model. |
| `--cpu` | Force CPU. |
| `--save-masks` | Also save the raw segmentation (16-bit PNG, one value per organoid). |
| `--label id` / `diameter` / `none` | Less text on the annotated image (default shows both). |

## Calibration: getting µm right

The built-in default is the lab's calibration: **1800 µm = 370 px on a 926×695 export** of the camera image,
which is **2.20 µm per pixel** on the full 2048×1536 images. This matches the lab's manual ImageJ
measurements. Every run prints the calibration used, and the CSVs record it.

Images with a different resolution but the same field of view are scaled automatically (a 1024×768
export gets 4.40 µm/px). Cellphi warns if an image's shape doesn't match the reference, which suggests cropping.

For a different microscope, objective or camera setting, calibrate again:

1. Take an image of a scale bar or stage micrometer with the **same settings** as your organoid images.
2. Measure the scale bar's length in pixels (e.g. with the line tool in ImageJ/Fiji).
3. Pass the length, the pixels, and the size of that image:
   `--scale-bar <µm> <px> --reference-size <W>x<H>`, or the result directly: `--units-per-px <µm per px>`.

## Accuracy

`reference/` holds manual ImageJ measurements for six wells (Cortical Plate 1 CM_CM, 285 organoids).
Cellphi finds 284 of the 285 as separate organoids. Its default diameters have a median ratio of 1.00
to the manual values, differ by 6.9% on average, and 79% are within 10%. Much of the remaining difference is
the natural variation of drawing a single line by hand across a not-quite-round organoid.

Re-run this comparison with `pytest -m slow` (about 1 minute on a GPU).

## Using it from Python / Jupyter

See [notebooks/example.ipynb](notebooks/example.ipynb). In short:

```python
from cellphi import process_image, annotate, organoid_rows, write_csv
from cellphi.segment import Segmenter

segmenter = Segmenter()                       # load the model once
res = process_image("input/Midbrain Organoids/Plate 1 CM_CM/A1/day 23 2-28-23 KS217537.jpg", segmenter)
print(res.diameters())                        # µm, in id order
annotated = annotate(res.image, res.organoids, res.units_per_px)
write_csv("organoids.csv", organoid_rows(res.name, res.organoids, res.units_per_px, res.unit))
```

## Tips for accurate results

- Check the annotated images, especially crowded areas and anything orange (edge) or unusual.
- An organoid mostly hidden behind others is measured from the part that's visible, so it reads small.
- Keep focus, exposure and magnification the same across an experiment, and calibrate once per objective.

## Roadmap

- [x] Command-line tool with annotated image + CSV output
- [x] Batch processing of folders (recursive)
- [x] Tuned and checked against manual measurements
- [ ] More metrics (circularity) and plots
- [ ] Organoid-specific Cellpose model (fine-tuned on a few hand-corrected images)
- [ ] GUI
