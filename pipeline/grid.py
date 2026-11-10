"""Land mask and classification on the LCC grid."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .boundary import Polygon
from .geo import LccGrid, lonlat_to_xy


def project_polygons(polygons: list[Polygon]) -> list[Polygon]:
    out = []
    for rings in polygons:
        proj = []
        for ring in rings:
            x, y = lonlat_to_xy(ring[:, 0], ring[:, 1])
            proj.append(np.column_stack([x, y]))
        out.append(proj)
    return out


def polygon_extent(projected: list[Polygon]) -> tuple[float, float, float, float]:
    xs = np.concatenate([r[:, 0] for p in projected for r in p[:1]])
    ys = np.concatenate([r[:, 1] for p in projected for r in p[:1]])
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def rasterize_mask(grid: LccGrid, projected: list[Polygon], supersample: int = 2) -> np.ndarray:
    """Fractional land coverage per pixel (0..1); gives anti-aliased coastlines."""
    ss = supersample
    img = Image.new("L", (grid.nx * ss, grid.ny * ss), 0)
    draw = ImageDraw.Draw(img)
    for rings in projected:
        for i, ring in enumerate(rings):
            px, py = grid.to_pixel(ring[:, 0], ring[:, 1])
            pts = list(zip((px * ss).tolist(), (py * ss).tolist()))
            if len(pts) >= 3:
                draw.polygon(pts, fill=255 if i == 0 else 0)
    return np.asarray(img.reduce(ss), dtype=np.float32) / 255.0


def classify(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Class index per pixel (0..len(edges)); NaN pixels get -1.

    Lower edge inclusive: 10.0 is class 1 for edges [10, 20, ...].
    """
    cls = np.digitize(np.nan_to_num(values, nan=0.0), edges).astype(np.int8)
    cls[np.isnan(values)] = -1
    return cls
