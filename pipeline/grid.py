"""Land mask on the LCC grid and classification into NOAA's classes."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import config as C
from .geo import LccGrid, lonlat_to_xy


def project_polygons(polygons):
    """lon/lat polygons (list of rings) -> LCC metres. Each polygon: exterior ring first, then holes."""
    out = []
    for poly in polygons:
        rings = []
        for ring in poly:
            x, y = lonlat_to_xy(ring[:, 0], ring[:, 1])
            rings.append(np.column_stack([x, y]))
        out.append(rings)
    return out


def polygon_extent(projected):
    pts = np.vstack([r for poly in projected for r in poly[:1]])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def rasterize_mask(grid: LccGrid, projected, supersample: int = 2) -> np.ndarray:
    """Land coverage 0..1 per pixel (anti-aliased by supersampling)."""
    ss = supersample
    img = Image.new("L", (grid.nx * ss, grid.ny * ss), 0)
    d = ImageDraw.Draw(img)
    for poly in projected:
        for k, ring in enumerate(poly):
            px = (ring[:, 0] - grid.x0) / grid.res * ss
            py = (grid.y1 - ring[:, 1]) / grid.res * ss
            d.polygon(list(zip(px.tolist(), py.tolist())), fill=255 if k == 0 else 0)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return a.reshape(grid.ny, ss, grid.nx, ss).mean(axis=(1, 3))


def classify(values: np.ndarray, upper_edges=C.CLASS_UPPER_EDGES) -> np.ndarray:
    """Class index 0..len(edges); a class includes its upper edge (10 -> class 0, 10.5 -> class 1). NaN -> -1."""
    v = np.asarray(values, float)
    out = np.searchsorted(np.asarray(upper_edges, float), v, side="left").astype(np.int8)
    out[np.isnan(v)] = -1
    return out
