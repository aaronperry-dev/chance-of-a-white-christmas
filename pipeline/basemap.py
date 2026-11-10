"""Base map under the probability surface: a hypsometric (land) and bathymetric (ocean, lakes) colour map.

Drawn at BASE_RES_M over NOAA's whole view extent, once per colour theme. Where the chance of a white
Christmas is below TRANSPARENT_BELOW_PCT the surface is transparent and this is what shows. A second
image shades terrain outside the US (inside the US the 1 km relief layer does that).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import config as C
from . import dem as demmod
from .geo import LccGrid
from .grid import project_polygons, rasterize_mask
from .render import _rgb, render_relief

log = logging.getLogger(__name__)


def base_grid() -> LccGrid:
    x0, y0, x1, y1 = C.NOAA_EXTENT
    p = C.BASE_PAD_M
    return LccGrid.from_extent(x0 - p, y0 - p, x1 + p, y1 + p, C.BASE_RES_M)


def ramp(values: np.ndarray, stops) -> np.ndarray:
    """(n, 3) uint8 colour per value, linear between stops, clamped at both ends."""
    xs = np.array([s[0] for s in stops], float)
    cols = np.array([_rgb(s[1]) for s in stops], float)
    v = np.asarray(values, float)
    out = np.stack([np.interp(v, xs, cols[:, k]) for k in range(3)], axis=-1)
    return np.round(out).astype(np.uint8)


def rasterize_labels(grid: LccGrid, projected) -> np.ndarray:
    """int32 label image: polygon i (1-based) painted with i, holes with 0."""
    img = Image.new("I", (grid.nx, grid.ny), 0)
    d = ImageDraw.Draw(img)
    for i, poly in enumerate(projected, start=1):
        for k, ring in enumerate(poly):
            px = (ring[:, 0] - grid.x0) / grid.res
            py = (grid.y1 - ring[:, 1]) / grid.res
            d.polygon(list(zip(px.tolist(), py.tolist())), fill=i if k == 0 else 0)
    return np.asarray(img, dtype=np.int32)


def water_depth(dem: np.ndarray, land_cov: np.ndarray, lake_labels: np.ndarray) -> np.ndarray:
    """Depth (m, >= 0) for every cell, used where the cell is water.

    Ocean: -elevation. Lakes: the lake's surface (95th percentile of the DEM inside it) minus the
    elevation, so a lake whose DEM is flat reads as uniformly shallow rather than as 'land'.
    """
    depth = np.maximum(-dem, 0.0).astype(np.float32)
    n = int(lake_labels.max())
    if n:
        idx = np.arange(1, n + 1)
        surf = ndimage.labeled_comprehension(dem, lake_labels, idx, lambda v: np.percentile(v, 95), float, 0.0)
        surf_img = np.concatenate([[0.0], surf]).astype(np.float32)[lake_labels]
        lake = lake_labels > 0
        depth[lake] = np.maximum(surf_img[lake] - dem[lake], 0.0)
    return depth


def colorize(dem: np.ndarray, land_cov: np.ndarray, lake: np.ndarray, depth: np.ndarray, theme: str) -> np.ndarray:
    """(ny, nx, 3) uint8 base map for one theme."""
    h, w = dem.shape
    land = ramp(np.maximum(dem, 0.0).ravel(), C.LAND_RAMP[theme]).reshape(h, w, 3).astype(np.float32)
    wat = ramp(depth.ravel(), C.WATER_RAMP[theme]).reshape(h, w, 3).astype(np.float32)
    a = (np.clip(land_cov, 0, 1) * (~lake))[..., None]          # lakes are water even inside a country
    return np.round(a * land + (1 - a) * wat).astype(np.uint8)


def build_base(land_polys, lake_polys, us_projected, out_dir: Path, cache: Path, *, dem: np.ndarray | None = None,
               grid: LccGrid | None = None, zoom: int = C.BASE_DEM_ZOOM) -> dict:
    grid = grid or base_grid()
    if dem is None:
        dem = demmod.dem_on_grid(grid, cache, zoom)
    dem = dem.astype(np.float32)
    land_cov = rasterize_mask(grid, project_polygons(land_polys))
    us_cov = rasterize_mask(grid, us_projected)
    labels = rasterize_labels(grid, project_polygons(lake_polys)) if lake_polys else np.zeros(dem.shape, np.int32)
    lake = labels > 0
    depth = water_depth(dem, land_cov, labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        rgb = colorize(dem, land_cov, lake, depth, theme)
        Image.fromarray(rgb, "RGB").save(out_dir / f"base_{theme}.webp", "WEBP", quality=88, method=4)
    # terrain shading outside the US only (the US has the 1 km relief layer); never over water
    ext_cov = np.clip(land_cov * (~lake) * (1.0 - np.clip(us_cov, 0, 1)), 0, 1)
    render_relief(demmod.hillshade(np.maximum(dem, 0.0), grid.res), ext_cov, out_dir / "relief_ext.webp")
    log.info("base map: %d x %d at %.0f m, %d lakes", grid.nx, grid.ny, grid.res, int(labels.max()))
    return {
        "grid": grid.as_meta(), "light": "base_light.webp", "dark": "base_dark.webp", "relief": "relief_ext.webp",
        "ramps": {t: {"land": [list(s) for s in C.LAND_RAMP[t]], "water": [list(s) for s in C.WATER_RAMP[t]]}
                  for t in ("light", "dark")},
    }
