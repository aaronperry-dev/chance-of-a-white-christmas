"""Turn the classified surface into web-ready assets and summary statistics."""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from PIL import Image

from .grid import GridSpec

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


def encode_value_grid(values: np.ndarray, coverage: np.ndarray) -> dict:
    """uint8 percent grid (255 = outside land) as base64, for click look-ups."""
    v = np.full(values.shape, NODATA, dtype=np.uint8)
    ok = ~np.isnan(values) & (coverage >= 0.5)
    v[ok] = np.clip(np.round(values[ok]), 0, 100).astype(np.uint8)
    return {"ny": int(v.shape[0]), "nx": int(v.shape[1]),
            "b64": base64.b64encode(v.tobytes()).decode("ascii")}


def area_stats(values: np.ndarray, cls: np.ndarray, coverage: np.ndarray,
               spec: GridSpec, n_classes: int) -> dict:
    """Area-weighted shares of the land surface.

    In a Mercator-uniform grid a pixel's ground area is proportional to
    cos^2(latitude): its width shrinks with cos(lat) and, because rows are
    spaced uniformly in Mercator y, its height shrinks by cos(lat) again.
    """
    lat = np.radians(spec.lat_centers())
    w = (np.cos(lat) ** 2)[:, None] * coverage
    ok = cls >= 0
    total = float(w[ok].sum())
    by_class = np.bincount(cls[ok], weights=w[ok], minlength=n_classes) / total
    ge = lambda t: float(w[ok & (values >= t)].sum() / total)  # noqa: E731
    return {
        "share_by_class": [round(float(x), 5) for x in by_class],
        "share_ge_50": round(ge(50), 5),
        "share_ge_10": round(ge(10), 5),
        "share_lt_10": round(1 - ge(10), 5),
    }
