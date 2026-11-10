"""SYNTHETIC stations for layout/testing only. Never real data.

``python -m pipeline.build --demo`` uses this so the page can be previewed without
network access; meta.json carries ``synthetic: true`` and the page shows a banner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_stations(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-124.5, -67.5, n)
    lat = rng.uniform(25.0, 49.0, n)
    elev = np.maximum(200 + 1500 * np.exp(-(((lon + 108) / 6) ** 2)) + rng.normal(0, 120, n), 0)
    logit = 0.45 * (lat - 43.0) + 0.0022 * (elev - 300.0) + rng.normal(0, 0.35, n)
    prob = np.clip(np.round(100 / (1 + np.exp(-logit))), 0, 100)
    return pd.DataFrame({
        "id": [f"DEMO{i:05d}" for i in range(n)], "name": [f"Demo station {i:04d}" for i in range(n)],
        "state": "", "lat": np.round(lat, 4), "lon": np.round(lon, 4),
        "elev_m": np.round(elev, 1), "prob": prob, "years": 30,
    })
