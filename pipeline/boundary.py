"""Load CONUS state polygons (Natural Earth) for masking the surface to land."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CONUS_BBOX, NON_CONUS

# A polygon is a list of rings; ring 0 is the exterior, the rest are holes.
Polygon = list[np.ndarray]  # each ring: (n, 2) array of lon, lat


def _polys(geom: dict) -> list[list[list[list[float]]]]:
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


def load_conus_polygons(path: Path) -> list[Polygon]:
    west, south, east, north = CONUS_BBOX
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Polygon] = []
    for feat in data["features"]:
        props = feat.get("properties") or {}
        if props.get("admin") not in (None, "United States of America"):
            continue
        if props.get("name") in NON_CONUS:
            continue
        for poly in _polys(feat["geometry"]):
            rings = [np.asarray(r, dtype=float)[:, :2] for r in poly]
            cx, cy = rings[0][:, 0].mean(), rings[0][:, 1].mean()
            # Drop anything whose centroid is well outside CONUS (islands etc.)
            if not (west - 1 <= cx <= east + 1 and south - 1 <= cy <= north + 1):
                continue
            out.append(rings)
    if not out:
        raise ValueError(f"No CONUS polygons found in {path}")
    return out
