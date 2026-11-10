"""SYNTHETIC terrain and stations for layout/testing only. Never real data.

Used by ``python -m pipeline.build --demo`` so the page can be previewed without
network access. The resulting meta.json carries ``synthetic: true`` and the
page shows a banner saying so. The "terrain" here is invented ridges + noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .boundary import Polygon
from .geo import LccGrid, lonlat_to_xy, xy_to_lonlat
from .grid import project_polygons, rasterize_mask


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fractal(grid: LccGrid, seed: int) -> np.ndarray:
    """Unit-variance multi-octave smooth noise on the grid (cell sizes 60/25/10/4 km)."""
    from scipy.ndimage import gaussian_filter, zoom
    rng = np.random.default_rng(seed)
    out = np.zeros((grid.ny, grid.nx), np.float32)
    for cell_km, amp in [(60, 1.0), (25, 0.6), (10, 0.35), (4, 0.2)]:
        h = max(8, int(grid.ny * grid.res / 1000 / cell_km))
        w = max(8, int(grid.nx * grid.res / 1000 / cell_km))
        small = gaussian_filter(rng.normal(size=(h, w)), 0.8)
        up = zoom(small, (grid.ny / h, grid.nx / w), order=3)[:grid.ny, :grid.nx]
        out += amp * (up / up.std()).astype(np.float32)
    return out / out.std()


def _elevation(lon, lat, f1, f2) -> np.ndarray:
    base = 100 + 1300 * _sig((lon + 101.5) / 2.2)
    rockies = 1800 * np.exp(-(((lon + 107.5 + 0.4 * (lat - 40)) / 3.2) ** 2))
    sierra = 1900 * np.exp(-(((lon + 119.6 + 0.6 * (lat - 37)) / 0.9) ** 2)) * _sig(lat - 35.5) * _sig(41 - lat)
    cascades = 1300 * np.exp(-(((lon + 121.6) / 0.7) ** 2)) * _sig(lat - 42)
    appal = 700 * np.exp(-(((lon + 81.5 + 0.9 * (lat - 37)) / 1.1) ** 2)) * _sig(lat - 34.5) * _sig(43 - lat)
    big = base + rockies + sierra + cascades + appal
    mtn = np.clip((rockies + sierra + cascades + appal) / 700.0, 0, 1)
    return np.maximum(big + 35 * f1 + mtn * 520 * (1 - np.abs(f2)) * (0.6 + 0.4 * f1), 0).astype(np.float32)


def synthetic_dem(grid: LccGrid, seed: int = 3, chunk_rows: int = 256) -> np.ndarray:
    f1, f2 = _fractal(grid, seed), _fractal(grid, seed + 1)
    dem = np.empty((grid.ny, grid.nx), np.float32)
    xc = grid.x_centers()
    for r0 in range(0, grid.ny, chunk_rows):
        rows = np.arange(r0, min(r0 + chunk_rows, grid.ny))
        X, Y = np.meshgrid(xc, grid.y1 - (rows + 0.5) * grid.res)
        lo, la = xy_to_lonlat(X, Y)
        dem[rows] = _elevation(lo, la, f1[rows], f2[rows])
    return dem


def make_demo_stations(polygons: list[Polygon], grid: LccGrid, dem: np.ndarray,
                       n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    land = rasterize_mask(grid, project_polygons(polygons), supersample=1) > 0.5
    lon = rng.uniform(-125, -66.5, n * 4)
    lat = rng.uniform(24.5, 49.5, n * 4)
    x, y = lonlat_to_xy(lon, lat)
    px, py = grid.to_pixel(x, y)
    xi, yi = np.clip(px.astype(int), 0, grid.nx - 1), np.clip(py.astype(int), 0, grid.ny - 1)
    keep = land[yi, xi]
    lon, lat, xi, yi = lon[keep][:n], lat[keep][:n], xi[keep][:n], yi[keep][:n]

    elev = np.maximum(dem[yi, xi] + rng.normal(0, 40, len(lat)), 0)
    logit = 0.45 * (lat - 43.0) + 0.0022 * (elev - 300.0) + rng.normal(0, 0.35, len(lat))
    prob = np.clip(np.round(100 / (1 + np.exp(-logit))), 0, 100)
    return pd.DataFrame({
        "id": [f"DEMO{i:05d}" for i in range(len(lat))],
        "name": [f"Demo station {i:04d}" for i in range(len(lat))],
        "state": "", "lat": np.round(lat, 4), "lon": np.round(lon, 4),
        "elev_m": np.round(elev, 1), "prob": prob, "years": 30,
    })


def synthetic_base_dem(grid: LccGrid, land_polys) -> np.ndarray:
    """SYNTHETIC elevation for the base map: invented terrain on land, sloping sea floor offshore."""
    from scipy.ndimage import gaussian_filter
    from .boundary import Polygon  # noqa: F401  (type only)
    land = rasterize_mask(grid, project_polygons(land_polys), supersample=1)
    dem = synthetic_dem(grid, seed=11)
    sea = -50.0 - 4500.0 * np.clip(1.0 - gaussian_filter(land, 12.0) * 2.2, 0, 1)
    return np.where(land > 0.5, np.maximum(dem, 5.0), sea).astype(np.float32)
