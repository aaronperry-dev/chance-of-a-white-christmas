"""SYNTHETIC stations for layout/testing only. Never real data.

Used by ``python -m pipeline.build --demo`` so the page can be previewed without
network access. The resulting meta.json carries ``synthetic: true`` and the
page shows a banner saying so. Station values are invented: a latitude trend plus broad invented mountain ranges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .boundary import Polygon
from .geo import LccGrid, lonlat_to_xy
from .grid import rasterize_mask


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _elevation(lon, lat) -> np.ndarray:
    base = 100 + 1300 * _sig((lon + 101.5) / 2.2)
    rockies = 1800 * np.exp(-(((lon + 107.5 + 0.4 * (lat - 40)) / 3.2) ** 2))
    sierra = 1900 * np.exp(-(((lon + 119.6 + 0.6 * (lat - 37)) / 0.9) ** 2)) * _sig(lat - 35.5) * _sig(41 - lat)
    cascades = 1300 * np.exp(-(((lon + 121.6) / 0.7) ** 2)) * _sig(lat - 42)
    appal = 700 * np.exp(-(((lon + 81.5 + 0.9 * (lat - 37)) / 1.1) ** 2)) * _sig(lat - 34.5) * _sig(43 - lat)
    big = base + rockies + sierra + cascades + appal
    return np.maximum(big, 0).astype(np.float32)


def make_demo_stations(polygons: list[Polygon], n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    from .grid import polygon_extent, project_polygons
    proj = project_polygons(polygons)
    xmin, ymin, xmax, ymax = polygon_extent(proj)
    grid = LccGrid.from_extent(xmin, ymin, xmax, ymax, 5000.0)
    land = rasterize_mask(grid, proj, supersample=1) > 0.5
    lon = rng.uniform(-125, -66.5, n * 4)
    lat = rng.uniform(24.5, 49.5, n * 4)
    x, y = lonlat_to_xy(lon, lat)
    px, py = grid.to_pixel(x, y)
    xi, yi = np.clip(px.astype(int), 0, grid.nx - 1), np.clip(py.astype(int), 0, grid.ny - 1)
    keep = land[yi, xi]
    lon, lat, xi, yi = lon[keep][:n], lat[keep][:n], xi[keep][:n], yi[keep][:n]

    elev = np.maximum(_elevation(lon, lat) + rng.normal(0, 40, len(lat)), 0)
    logit = 0.45 * (lat - 43.0) + 0.0022 * (elev - 300.0) + rng.normal(0, 0.35, len(lat))
    prob = np.clip(np.round(100 / (1 + np.exp(-logit))), 0, 100)
    return pd.DataFrame({
        "id": [f"DEMO{i:05d}" for i in range(len(lat))],
        "name": [f"Demo station {i:04d}" for i in range(len(lat))],
        "state": "", "lat": np.round(lat, 4), "lon": np.round(lon, 4),
        "elev_m": np.round(elev, 1), "prob": prob, "years": 30,
    })
