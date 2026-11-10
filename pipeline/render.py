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


def render_overlay(cls: np.ndarray, coverage: np.ndarray, colors: list[str], path: Path,
                   values: np.ndarray | None = None, transparent_below: float | None = None) -> None:
    """RGBA PNG: class colour, alpha = land coverage (anti-aliased coast).

    Cells whose value is below ``transparent_below`` get alpha 0 so the base map shows through.
    """
    lut = np.array([_rgb(c) for c in colors], dtype=np.uint8)
    rgba = np.zeros(cls.shape + (4,), dtype=np.uint8)
    ok = cls >= 0
    if values is not None and transparent_below is not None:
        with np.errstate(invalid="ignore"):
            ok &= ~(values < transparent_below)
    rgba[ok, :3] = lut[cls[ok]]
    rgba[ok, 3] = np.round(np.clip(coverage[ok], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def render_relief(shade: np.ndarray, coverage: np.ndarray, path: Path, quality: int = 80) -> None:
    """Hillshade as a black/white overlay with alpha = |shade - flat| * land coverage.

    Slopes facing away from the light become translucent black, slopes facing it translucent
    white and flat ground is fully transparent, so it composites correctly over every class
    colour (a multiply or soft-light blend would hide relief on the white top class). The page
    scales the whole layer with CSS opacity (relief-strength slider).
    """
    d = (shade.astype(np.float32) - 128.0) / 128.0                 # -1 (shadow) .. +1 (lit)
    alpha = np.clip(np.abs(d), 0, 1) * np.clip(coverage, 0, 1)
    rgb = np.where(d < 0, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, rgb, rgb, np.round(alpha * 255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(path, "WEBP", quality=quality, alpha_quality=85, method=4)


def write_value_grid(values: np.ndarray, coverage: np.ndarray, path: Path) -> None:
    """uint8 percent grid (255 = outside land), row-major, gzip-compressed.

    The browser reads it with DecompressionStream for exact click look-ups at
    full grid resolution.
    """
    v = np.full(values.shape, NODATA, dtype=np.uint8)
    ok = ~np.isnan(values) & (coverage >= 0.5)
    v[ok] = np.clip(np.round(values[ok]), 0, 100).astype(np.uint8)
    path.write_bytes(gzip.compress(v.tobytes(), compresslevel=9, mtime=0))


def area_stats(values: np.ndarray, cls: np.ndarray, coverage: np.ndarray, grid: LccGrid, n_classes: int) -> dict:
    """True-area-weighted shares. LCC is conformal, not equal-area: a pixel covers res^2 / k^2 on the ground."""
    stride = max(1, min(8, min(grid.ny, grid.nx) // 25))
    rows, cols = np.arange(0, grid.ny, stride), np.arange(0, grid.nx, stride)
    cov = coverage[np.ix_(rows, cols)]
    v, c = values[np.ix_(rows, cols)], cls[np.ix_(rows, cols)]
    X, Y = np.meshgrid(grid.x_centers()[cols], grid.y_centers()[rows])
    _, lat = xy_to_lonlat(X, Y)
    w = (cov / scale_factor(lat) ** 2).astype(np.float64)
    ok = (c >= 0) & (w > 0)
    tot = float(w[ok].sum())
    by = [float(w[ok & (c == i)].sum() / tot) for i in range(n_classes)]
    pix_km2 = (grid.res / 1000.0) ** 2 * stride * stride
    vv = np.where(np.isnan(v), -1.0, np.round(v))
    return {
        "basis": "land area",
        "share_by_class": by,
        "share_ge_50": float(w[ok & (vv >= 50)].sum() / tot),
        "share_le_10": float(w[ok & (vv <= 10)].sum() / tot),
        "land_area_km2": round(tot * pix_km2),
    }
