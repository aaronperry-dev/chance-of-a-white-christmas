"""SYNTHETIC stations for layout/testing only. Never real NOAA data.

Used by ``python -m pipeline.build --demo`` so the page can be previewed
without network access. The resulting meta.json carries ``synthetic: true`` and
the page shows a banner saying so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .boundary import Polygon
from .config import CONUS_BBOX
from .grid import GridSpec, rasterize_mask


def make_demo_stations(polygons: list[Polygon], n: int = 2500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    west, south, east, north = CONUS_BBOX
    spec = GridSpec.from_bbox(CONUS_BBOX, 0.1)
    land = rasterize_mask(spec, polygons, supersample=2) > 0.5

    lon = rng.uniform(west, east, n * 3)
    lat = rng.uniform(south, north, n * 3)
    x, y = spec.to_pixel(lon, lat)
    xi, yi = np.clip(x.astype(int), 0, spec.nx - 1), np.clip(y.astype(int), 0, spec.ny - 1)
    keep = land[yi, xi]
    lon, lat = lon[keep][:n], lat[keep][:n]

    base = 1 / (1 + np.exp(-(lat - 41.5) / 2.6))
    rockies = 0.55 * np.exp(-(((lon + 106.5) / 3.5) ** 2)) * (lat > 35)
    sierra = 0.45 * np.exp(-(((lon + 120.3) / 1.1) ** 2)) * (lat > 36.5)
    appal = 0.30 * np.exp(-(((lon + 80.5) / 1.4) ** 2)) * (lat > 36)
    p = base + (rockies + sierra + appal) * (1 - base) + rng.normal(0, 0.04, len(lat))
    prob = np.clip(np.round(p * 100), 0, 100)

    return pd.DataFrame({
        "id": [f"DEMO{i:05d}" for i in range(len(lat))],
        "name": [f"Demo station {i:04d}" for i in range(len(lat))],
        "state": "",
        "lat": np.round(lat, 4), "lon": np.round(lon, 4),
        "elev": np.nan, "prob": prob,
    })
