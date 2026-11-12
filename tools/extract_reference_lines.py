"""Turn the lab's manual ImageJ measurements in reference/ into test data.

Each reference well folder holds the measured image, an ImageJ ``Results.csv`` (one row per
hand-drawn diameter line: Length in um, Angle) and a ``Clipboard.jpg`` screenshot showing the
lines in yellow. The CSV has no line positions, so they are recovered from the screenshot:
yellow line pieces are detected, pieces split by the number label are joined, and the rebuilt
lines are matched one-to-one to CSV rows by angle and length.

Output: tests/data/reference_lines.json, used by tests/test_reference.py.

    python tools/extract_reference_lines.py [reference_dir] [output_json]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Calibration of the manual measurements: 1800 um = 370 px on a 926 px wide image.
UM_PER_PX_2048 = 1800 / 370 * 926 / 2048


def find_measured_image(well: Path, panel: np.ndarray) -> Path:
    """The well folder has two camera images; pick the one shown in the screenshot."""
    best = None
    for f in sorted(well.glob("*.jpg")):
        if "Clipboard" in f.name:
            continue
        g = cv2.resize(cv2.imread(str(f), 0), (panel.shape[1], panel.shape[0])).astype(float)
        r = np.corrcoef(g.ravel(), panel.ravel())[0, 1]
        if best is None or r > best[0]:
            best = (r, f)
    if best is None or best[0] < 0.8:
        raise RuntimeError(f"No image in {well} matches the screenshot")
    return best[1]


def rebuild_lines(panel_bgr: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """Detect yellow line pieces and join collinear ones. Returns (length, angle_deg, midpoint)."""
    b, g, r = (panel_bgr[..., i].astype(int) for i in range(3))
    yellow = ((np.minimum(r, g) - b) > 50).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(yellow, 1, np.pi / 360, threshold=10, minLineLength=8, maxLineGap=4)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float)
    th = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]) % np.pi
    d = np.c_[np.cos(th), np.sin(th)]
    n = np.c_[-np.sin(th), np.cos(th)]
    parent = list(range(len(segs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(segs)):
        ei = sorted([d[i] @ segs[i, :2], d[i] @ segs[i, 2:]])
        for j in range(i + 1, len(segs)):
            if min(abs(th[i] - th[j]), np.pi - abs(th[i] - th[j])) > np.radians(3):
                continue
            if abs(n[i] @ segs[j, :2] - n[i] @ segs[i, :2]) > 3:
                continue
            ej = sorted([d[i] @ segs[j, :2], d[i] @ segs[j, 2:]])
            if max(ej[0] - ei[1], ei[0] - ej[1]) < 30:  # gap left by the number label
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(segs)):
        groups.setdefault(find(i), []).append(i)
    lines = []
    for idx in groups.values():
        pts = np.concatenate([segs[i].reshape(2, 2) for i in idx])
        t = pts @ d[idx[0]]
        a, z = pts[t.argmin()], pts[t.argmax()]
        lines.append((float(np.hypot(*(z - a))), float(np.degrees(np.arctan2(-(z[1] - a[1]), z[0] - a[0])) % 180), (a + z) / 2))
    return lines


def extract_well(well: Path) -> dict:
    clip_path = next(well.glob("*Clipboard.jpg"))
    res_path = next(well.glob("*Results.csv"))
    clip = cv2.imread(str(clip_path))
    off = clip.shape[1] // 2
    panel = clip[:, off:]
    image = find_measured_image(well, cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY).astype(float))
    scale = 2048 / panel.shape[1]  # screenshot px -> image px
    um_per_screen_px = UM_PER_PX_2048 * scale
    rows = list(csv.DictReader(open(res_path)))
    lines = rebuild_lines(panel)
    rl = np.array([float(r["Length"]) / um_per_screen_px for r in rows])
    ra = np.array([float(r["Angle"]) % 180 for r in rows])
    cost = np.full((len(rows), len(lines)), 1e6)
    for j, (ln, ang, _) in enumerate(lines):
        da = np.minimum(abs(ra - ang), 180 - abs(ra - ang))
        dl = abs(ln / rl - 1)
        ok = (da < 4) & (dl < 0.12)
        cost[ok, j] = (da / 4 + dl / 0.12)[ok]
    ri, ci = linear_sum_assignment(cost)
    matched = [
        {"n": int(i) + 1, "length_um": float(rows[i]["Length"]),
         "x": round(float(lines[j][2][0] * scale), 1), "y": round(float(lines[j][2][1] * scale), 1)}
        for i, j in zip(ri, ci) if cost[i, j] < 1e6
    ]
    return {"image": image.name, "n_manual": len(rows), "lines": sorted(matched, key=lambda m: m["n"])}


def main() -> None:
    ref = Path(sys.argv[1] if len(sys.argv) > 1 else "reference")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "tests/data/reference_lines.json")
    data = {}
    for clip in sorted(ref.rglob("*Clipboard.jpg")):
        well = clip.parent
        key = well.relative_to(ref).as_posix()
        data[key] = extract_well(well)
        print(f"{key}: {len(data[key]['lines'])}/{data[key]['n_manual']} manual lines located ({data[key]['image']})")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"um_per_px_2048": UM_PER_PX_2048, "wells": data}, open(out, "w"), indent=1)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
