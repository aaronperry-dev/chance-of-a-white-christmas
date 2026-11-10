"""Turn the classified surface into web-ready assets and summary statistics."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from PIL import Image

from .geo import LccGrid, scale_factor, xy_to_lonlat

NODATA = 255


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def render_overlay(cls: np.ndarray, coverage: np.ndarray, colors: list[str], path: Path) -> None:
    """RGBA PNG: class colour, alpha = land coverage (anti-aliased coast)."""
    lut = np.array([_rgb(c) for c in colors], dtype=np.uint8)
    rgba = np.zeros(cls.shape + (4,), dtype=np.uint8)
    ok = cls >= 0
    rgba[ok, :3] = lut[cls[ok]]
    rgba[ok, 3] = np.round(np.clip(coverage[ok], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def write_value_grid(values: np.ndarray, coverage: np.ndarray, path: Path) -> None:
    """uint8 percent grid (255 = outside land), row-major, gzip-compressed.

    The browser reads it with DecompressionStream for exact click look-ups at
    full grid resolution.
    """
    v = np.full(values.shape, NODATA, dtype=np.uint8)
    ok = ~np.isnan(values) & (coverage >= 0.5)
    v[ok] = np.clip(np.round(values[ok]), 0, 100).astype(np.uint8)
    path.write_bytes(gzip.compress(v.tobytes(), compresslevel=9, mtime=0))


def area_stats(values: np.ndarray, cls: np.ndarray, coverage: np.ndarray,
               grid: LccGrid, n_classes: int) -> dict:
    """True-area-weighted shares of the land surface.

    LCC is conformal, not equal-area: a pixel of ``res`` metres covers
    res**2 / k**2 on the ground, k = scale factor at that latitude.
    """
    rows, cols = np.nonzero((cls >= 0) & (coverage > 0))
    w = np.empty(len(rows), np.float64)
    for s in range(0, len(rows), 2_000_000):
        sl = slice(s, s + 2_000_000)
        x, y = grid.pixel_xy(rows[sl], cols[sl])
        _, lat = xy_to_lonlat(x, y)
        w[sl] = coverage[rows[sl], cols[sl]] / scale_factor(lat) ** 2
    v, c = values[rows, cols], cls[rows, cols]
    total = w.sum()
    by_class = np.bincount(c, weights=w, minlength=n_classes) / total
    return {
        "share_by_class": [round(float(x), 5) for x in by_class],
        "share_ge_50": round(float(w[v >= 50].sum() / total), 5),
        "share_lt_10": round(float(w[v < 10].sum() / total), 5),
        "land_area_km2": round(float(total * (grid.res / 1000.0) ** 2)),
    }
