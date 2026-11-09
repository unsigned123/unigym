"""Minimal PIL line charts for the analyze report (zero new deps).

Same spirit as the webui sparkline (webui/assets/charts.js): a plain
polyline on a gridded canvas -- no matplotlib (absent from .venv_sim).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from unigym._optional import require

Image = require("PIL.Image", "record", feature="analyze report charts")
ImageDraw = require("PIL.ImageDraw", "record", feature="analyze report charts")

W, H = 720, 320
M = {"l": 56, "r": 12, "t": 26, "b": 30}
COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def _downsample(ys: np.ndarray, n: int = 1200) -> np.ndarray:
    if ys.size <= n:
        return ys
    idx = np.linspace(0, ys.size - 1, n).astype(np.int64)
    return ys[idx]


def line_chart(
    path: str | Path,
    title: str,
    series: list[tuple],
    x_label: str = "tick",
    hlines: list[tuple] | None = None,
) -> Path:
    """series = [(label, ys), ...]; one PNG polyline chart (y from data).

    ``hlines`` (the --metric face): [(value, label), ...] predicate
    threshold lines drawn in red over the data range.
    """
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    x0, y0 = M["l"], H - M["b"]
    x1, y1 = W - M["r"], M["t"]
    allv = np.concatenate(
        [np.asarray(ys, dtype=np.float64).ravel() for _, ys in series if np.size(ys)]
        + [np.asarray([v]) for v, _ in (hlines or [])]
    )
    if allv.size == 0 or not np.isfinite(allv).any():
        d.text((x0, (y0 + y1) // 2), "no data", fill="#888")
        img.save(path)
        return Path(path)
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    if hi - lo < 1e-12:
        hi = lo + 1e-12

    def px(i, n):  # tick index -> canvas x
        return x0 + (x1 - x0) * (i / max(1, n - 1))

    def py(v):  # value -> canvas y
        return y1 + (y0 - y1) * (v - lo) / (hi - lo)

    for g in range(5):  # horizontal grid + y ticks
        v = lo + (hi - lo) * g / 4
        gy = py(v)
        d.line([x0, gy, x1, gy], fill="#e5e7eb")
        d.text((4, gy - 6), f"{v:.3g}", fill="#666")
    for v, label in hlines or []:  # predicate threshold lines
        gy = py(v)
        d.line([x0, gy, x1, gy], fill="#dc2626")
        d.text((x0 + 4, gy - 11), f"th {label}", fill="#dc2626")
    x_hi = 0
    for s_i, (label, ys) in enumerate(series):
        ys = _downsample(np.asarray(ys, dtype=np.float64).ravel())
        x_hi = max(0, ys.size - 1)
        color = COLORS[s_i % len(COLORS)]
        pts = [(px(i, ys.size), py(v)) for i, v in enumerate(ys) if np.isfinite(v)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=1)
        d.rectangle([x1 - 130, y1 + 12 * s_i, x1 - 118, y1 + 12 * s_i + 8], fill=color)
        d.text((x1 - 112, y1 + 12 * s_i - 2), label[:18], fill="#333")
    d.text((x0, 6), title[:80], fill="#111")
    d.text((x0, y0 + 6), f"{x_label} 0..{x_hi}", fill="#666")
    img.save(path)
    return Path(path)
