# Cellphi

Cellphi measures organoids in microscope images. It finds every organoid with
[Cellpose](https://www.cellpose.org/) and measures each one's **diameter** in micrometers.

For each image you get:
- an **annotated image**: every measured organoid is outlined, with its diameter line, number and diameter in µm;
- a **CSV file** with one row per organoid (number, diameter, area, position);
- a **summary CSV** with the count, mean, median and spread of diameters per image.

## Installation

You need Python 3.10 or newer.

```bash
git clone <this repo> cellphi && cd cellphi
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # installs everything, including GPU-enabled PyTorch
```

- **GPU:** On Linux/Windows with an NVIDIA GPU, the GPU is used automatically. Without one it runs on CPU, just slower.
  Apple Silicon Macs use the Mac's GPU automatically.
- **Mac:** use plain `pip` as shown above. `uv pip install -r requirements.txt` can fail on a Mac
  because of the extra PyTorch download source.
- **Google Colab:** Choose *Runtime → Change runtime type → GPU*, then run `!pip install -r requirements.txt` in a cell.
- **Slow connection:** PyTorch is a large download (several GB). If it times out, run the same command again;
  already downloaded files are reused.
- The first run downloads the Cellpose model (about 1 GB).

## Usage

```bash
cellphi input/ -o results/              # every image in a folder
cellphi input/36.jpg input/52.jpg       # specific images
cellphi --help                          # all options
```

Output in `results/`:

| File | Contents |
|---|---|
| `<image>_annotated.jpg` | Yellow: measured organoids with `#number` and diameter in µm. Red outline: excluded (see below). |
| `organoids.csv` | One row per measured organoid: `image, id, diameter_um, equivalent_diameter_um, area_um2, centroid_x_px, centroid_y_px, um_per_px`. `id` matches the `#number` on the image. |
| `summary.csv` | One row per image: counts (measured and excluded), mean/median/sd/min/max diameter. |

**Diameter** is the *maximum Feret diameter*: the longest straight line across the organoid's outline.
**Equivalent diameter** is the diameter of a circle with the same area. For round organoids the two are close.

### What gets excluded (red outlines)

- **Organoids cut off by the image edge**: their diameter would be too small.
- **Overlap fragments**: where two organoids overlap, Cellpose labels the lens-shaped overlap as a separate
  object. Cellphi recognises these and gives the overlap back to the organoids it belongs to, so their
  outline and area are complete. The same applies to small gaps fully enclosed by organoids.
- **Too small** (optional): use `--min-diameter 100` to drop anything under 100 µm, such as debris.

Use `--keep-border` or `--keep-fragments` to turn off the first two.

### Common options

| Option | Use it when |
|---|---|
| `--units-per-px 2.20` | You know the µm per pixel for your images directly. |
| `--scale-bar 1800 370 --reference-size 926x695` | Your calibration is a scale bar: 1800 µm = 370 px, measured on a 926×695 image. |
| `--expected-diameter 350` | Organoids are missed or split. Tell Cellpose the rough size in µm. |
| `--min-diameter 100` | You want to ignore debris below a size. |
| `--label id` / `--label diameter` / `--label none` | You want less text on the image (default shows both). |
| `--cpu` | You want to force CPU. |
| `--save-masks` | You also want the raw segmentation (16-bit PNG, one value per organoid). |

### Organoid size and Cellpose

Cellpose works best when it knows roughly how big the objects are. Cellphi first segments normally,
then estimates the typical organoid size and tries again with that hint. It keeps the second result only
if it finds clearly more organoids. In the sample images this mattered only for the very dark organoids in
`KS213728.jpg` (found 85 organoids instead of 4). The output says when the hint was used.
`--no-auto-size` skips the second pass, which is faster on CPU.

## Calibration: getting µm right

Measurements are only as accurate as the calibration. To calibrate:

1. Take an image with a scale bar or a stage micrometer, using the **same microscope, objective and camera
   settings** as your organoid images.
2. Measure how many pixels long the scale bar is (e.g. with the line tool in ImageJ/Fiji).
3. Note the **size in pixels of that image**. Pass both: `--scale-bar <µm> <px> --reference-size <W>x<H>`.

Cellphi measures every image at its full resolution. If your images have a different resolution than the
reference but show the same field of view, the scale is adjusted automatically. Cellphi warns you if the
image's shape (aspect ratio) doesn't match the reference, which suggests cropping.

The built-in default is the lab's current calibration: 1800 µm = 370 px on a 926×695 image, which works out
to **2.20 µm per pixel** for the 2048×1536 camera images. Every run prints the calibration used, and the
CSVs record it.

## Using it from Python / Jupyter

See [notebooks/example.ipynb](notebooks/example.ipynb). In short:

```python
from cellphi import process_image, annotate, organoid_rows, write_csv
from cellphi.segment import Segmenter

segmenter = Segmenter()                       # load the model once
res = process_image("input/36.jpg", segmenter)
print(res.diameters())                        # µm, in id order
annotated = annotate(res.image, res.organoids, res.units_per_px)
write_csv("36.csv", organoid_rows(res.name, res.organoids, res.units_per_px, res.unit))
```

## Tips for accurate results

- Check the annotated image, especially in crowded areas. An organoid that is mostly hidden behind
  others is measured from the part that's visible, so its diameter will be too small.
- Keep focus and magnification the same across an experiment, and calibrate once per objective.

## Roadmap

- [x] Command-line tool with annotated image + CSV output
- [x] Exclude organoids at the image edge and overlap fragments
- [x] Measure at native resolution
- [x] Batch processing of folders
- [ ] More metrics (minimum diameter, circularity) and plots
- [ ] Organoid-specific Cellpose model (fine-tuned on a few hand-corrected images)
- [ ] GUI
