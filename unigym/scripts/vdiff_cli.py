"""`unigym vdiff A.jpg B.jpg` -- visual diff for camera forensics (toolkit D).

Pure PIL/numpy, zero sim imports (design doc §5.2).  Produces: side-by-side
composite, abs-diff heatmap, per-image luminance histograms, the metrics
line (mean|d|, mean_a/mean_b, histL1 over 8 gray bins) and a patch table
(default 2x2 quadrants + ``--patches``).  If either input has a photo
sidecar (``<name>.jpg.json``, contract: camera_contract §sidecar) its
camera/tick/world facts are echoed into the report.

Offline-testable core: :func:`image_stats` / :func:`patch_stats` /
:func:`verdict` carry no I/O.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from unigym._optional import require

Image = require("PIL.Image", "record", feature="image visual diff")
ImageDraw = require("PIL.ImageDraw", "record", feature="image visual diff")

PATCH_NONE = "none"
SIDE_W = "side.png"
DIFF_W = "diff.png"
HIST_W = "hist.png"


# --------------------------------------------------------------------------- #
# pure math (unit-locked)
# --------------------------------------------------------------------------- #
def load_rgb(path: str | Path) -> np.ndarray:
    """Image -> (H, W, 3) uint8."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def image_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """mean|d| / means / histL1 (gray 8 bins, diff_dumps.py image_stats port)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape} -- vdiff "
                         f"compares same-resolution frames")
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    mad = float(np.abs(af - bf).mean())
    ga = af.mean(axis=2)
    gb = bf.mean(axis=2)
    ha, _ = np.histogram(ga, bins=8, range=(0, 255))
    hb, _ = np.histogram(gb, bins=8, range=(0, 255))
    ha = ha / max(1, ha.sum())
    hb = hb / max(1, hb.sum())
    return {"mean_abs_diff": mad,
            "mean_a": float(af.mean()), "mean_b": float(bf.mean()),
            "hist_l1": float(np.abs(ha - hb).sum())}


def patch_boxes(h: int, w: int, extra: list[tuple] | None = None) \
        -> list[dict]:
    """Default 2x2 quadrants + user boxes -> [{name, x0, y0, x1, y1}]."""
    boxes = [
        {"name": "q_tl", "x0": 0, "y0": 0, "x1": w // 2, "y1": h // 2},
        {"name": "q_tr", "x0": w // 2, "y0": 0, "x1": w, "y1": h // 2},
        {"name": "q_bl", "x0": 0, "y0": h // 2, "x1": w // 2, "y1": h},
        {"name": "q_br", "x0": w // 2, "y0": h // 2, "x1": w, "y1": h},
    ]
    for i, (x0, y0, x1, y1) in enumerate(extra or []):
        boxes.append({"name": f"p{i}", "x0": int(x0), "y0": int(y0),
                      "x1": int(x1), "y1": int(y1)})
    return boxes


def patch_stats(a: np.ndarray, b: np.ndarray, boxes: list[dict]) -> list[dict]:
    """Per-box mean_a / mean_b / mean|d| (+ the worst box first in caller)."""
    out = []
    for box in boxes:
        sl = (slice(box["y0"], box["y1"]), slice(box["x0"], box["x1"]))
        pa = a[sl].astype(np.float64)
        pb = b[sl].astype(np.float64)
        out.append({**box, "mean_a": float(pa.mean()),
                    "mean_b": float(pb.mean()),
                    "mean_abs_diff": float(np.abs(pa - pb).mean())})
    return out


def verdict(mad: float) -> str:
    """One-word conclusion band (documented thresholds, contract §vdiff):
    identical (==0) / near (<1 gray level) / differ."""
    if mad == 0.0:
        return "identical"
    return "near" if mad < 1.0 else "differ"


def parse_patches(spec: str | None) -> list[tuple]:
    """``x0,y0,x1,y1;x0,y0,x1,y1`` -> tuple list (fail-fast)."""
    if not spec:
        return []
    out = []
    for chunk in spec.split(";"):
        parts = [p for p in chunk.split(",") if p != ""]
        if len(parts) != 4:
            raise ValueError(
                f"--patches expects x0,y0,x1,y1 boxes separated by ';', "
                f"got {chunk!r}")
        vals = [int(float(p)) for p in parts]
        out.append(tuple(vals))
    return out


# --------------------------------------------------------------------------- #
# rendering (PIL)
# --------------------------------------------------------------------------- #
_LUT = None


def _heat_lut() -> np.ndarray:
    """256x3 black->blue->magenta->yellow->white style ramp (numpy-only)."""
    global _LUT
    if _LUT is None:
        stops = np.array([
            [0, 0, 0], [32, 0, 90], [130, 0, 130],
            [255, 120, 0], [255, 255, 180], [255, 255, 255]],
            dtype=np.float64)
        x = np.linspace(0.0, 1.0, 256)
        lut = np.stack([np.interp(x, np.linspace(0, 1, len(stops)),
                                  stops[:, c]) for c in range(3)], axis=1)
        _LUT = lut.astype(np.uint8)
    return _LUT


def render_heatmap(d: np.ndarray) -> Image.Image:
    """abs-diff (H,W) -> LUT-colored heatmap image."""
    g = d.mean(axis=2) if d.ndim == 3 else d.astype(np.float64)
    peak = float(g.max())
    idx = np.zeros(g.shape, dtype=np.uint8) if peak <= 0 else \
        (g / peak * 255).astype(np.uint8)
    return Image.fromarray(_heat_lut()[idx])


def render_hists(a: np.ndarray, b: np.ndarray) -> Image.Image:
    """Two normalized gray histograms side by side (256x120 each)."""
    panels = []
    for img in (a, b):
        g = img.astype(np.float64).mean(axis=2)
        hist, _ = np.histogram(g, bins=32, range=(0, 255))
        hist = hist / max(1, hist.max())
        panel = Image.new("RGB", (256, 120), "white")
        dr = ImageDraw.Draw(panel)
        for i, v in enumerate(hist):
            x = i * 8
            dr.rectangle([x, 119 - int(v * 115), x + 7, 119],
                         fill=(60, 90, 160))
        panels.append(panel)
    out = Image.new("RGB", (256 * 2 + 8, 120), "white")
    out.paste(panels[0], (0, 0))
    out.paste(panels[1], (256 + 8, 0))
    return out


# --------------------------------------------------------------------------- #
# sidecar echo (camera_contract §sidecar)
# --------------------------------------------------------------------------- #
def sidecar_facts(path: Path) -> str | None:
    """One-line sidecar summary for the report (None = no sidecar)."""
    sc = path.with_name(path.name + ".json")
    if not sc.is_file():
        return None
    try:
        doc = json.loads(sc.read_text())
    except (OSError, ValueError):
        return f"{sc.name}: unreadable"
    world = doc.get("world") or {}
    pos = world.get("pos")
    bits = [f"camera={doc.get('camera')}", f"tick={doc.get('tick')}"]
    if pos:
        bits.append(f"pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]")
    bits.append(f"source={doc.get('source')}")
    return f"{sc.name}: " + "  ".join(bits)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_vdiff(a: argparse.Namespace, extra: list[str]) -> int:
    pa, pb = Path(a.a), Path(a.b)
    for p in (pa, pb):
        if not p.is_file():
            print(f"[unigym vdiff] not a file: {p}", file=sys.stderr)
            return 1
    try:
        img_a, img_b = load_rgb(pa), load_rgb(pb)
        stats = image_stats(img_a, img_b)
        boxes = patch_boxes(img_a.shape[0], img_a.shape[1],
                            parse_patches(a.patches))
        patches = patch_stats(img_a, img_b, boxes)
    except ValueError as e:
        print(f"[unigym vdiff] {e}", file=sys.stderr)
        return 1

    out_dir = Path(a.out) if a.out else \
        pa.parent / f"vdiff_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_a.shape[:2]
    side = Image.new("RGB", (w * 2 + 8, h), "black")
    side.paste(Image.fromarray(img_a), (0, 0))
    side.paste(Image.fromarray(img_b), (w + 8, 0))
    side_path = out_dir / SIDE_W
    diff_path = out_dir / DIFF_W
    hist_path = out_dir / HIST_W
    side.save(side_path)
    render_heatmap(np.abs(img_a.astype(np.float64)
                          - img_b.astype(np.float64))).save(diff_path)
    render_hists(img_a, img_b).save(hist_path)

    worst = max(patches, key=lambda p: p["mean_abs_diff"])
    doc = {"ok": True, "a": str(pa), "b": str(pb), **stats,
           "verdict": verdict(stats["mean_abs_diff"]),
           "worst_patch": {k: worst[k] for k in
                           ("name", "x0", "y0", "x1", "y1",
                            "mean_a", "mean_b", "mean_abs_diff")},
           "patches": patches, "written": [str(side_path), str(diff_path),
                                           str(hist_path)],
           "sidecar_a": sidecar_facts(pa), "sidecar_b": sidecar_facts(pb)}
    if a.json:
        print(json.dumps(doc))
        return 0
    print(f"mean|d|={stats['mean_abs_diff']:.2f}  "
          f"mean_a={stats['mean_a']:.1f} mean_b={stats['mean_b']:.1f}  "
          f"histL1(8bins)={stats['hist_l1']:.3f}  "
          f"verdict={doc['verdict']}")
    print(f"top patch: [{worst['x0']}:{worst['y0']},"
          f"{worst['x1']}:{worst['y1']}] mean_a={worst['mean_a']:.1f} "
          f"mean_b={worst['mean_b']:.1f} ({worst['name']})")
    for label, p in (("a", pa), ("b", pb)):
        facts = doc[f"sidecar_{label}"]
        if facts:
            print(f"sidecar {label.upper()}: {facts}")
    print("written: " + " ".join(doc["written"]))
    return 0


def add_vdiff_subparser(sub) -> None:
    """Register the ``vdiff`` command (lazy CLI dispatch)."""
    vd = sub.add_parser("vdiff", help="compare two images: side-by-side + "
                                      "diff heatmap + histogram stats + "
                                      "one-line verdict")
    vd.add_argument("a", help="first image (jpg/png)")
    vd.add_argument("b", help="second image (same resolution expected)")
    vd.add_argument("--out", default=None,
                    help="artifact dir (default: <dir-of-A>/vdiff_<ts>)")
    vd.add_argument("--patches", default=None, metavar="X0,Y0,X1,Y1...",
                    help="extra patch boxes (semicolon-separated), on top "
                         "of the default 2x2 quadrants")
    vd.add_argument("--json", action="store_true")
    vd.set_defaults(fn=cmd_vdiff)
