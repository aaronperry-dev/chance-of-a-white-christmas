"""Web-Mercator-aligned grid, land mask, and inverse-distance interpolation.

Leaflet's ImageOverlay stretches an image linearly between two lat/lon corners
in *projected* space, so the raster must have rows that are uniform in
Mercator y (not in latitude). ``GridSpec`` builds exactly that, with square
pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from .boundary import Polygon
from .config import IDW_EPS_KM

R_EARTH_KM = 6371.0088


def merc_y(lat_deg):
    phi = np.radians(lat_deg)
    return np.log(np.tan(np.pi / 4 + phi / 2))


def inv_merc_y(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


@dataclass(frozen=True)
class GridSpec:
    west: float
    south: float
    east: float
    north: float
    step: float   # degrees of longitude per pixel
    nx: int
    ny: int
    y_top: float  # Mercator y (radians) of the north edge
    dy: float     # Mercator y (radians) per pixel row (== radians(step))

    @classmethod
    def from_bbox(cls, bbox: tuple[float, float, float, float], step: float) -> "GridSpec":
        west, south, east, north = bbox
        nx = math.ceil((east - west) / step)
        dx = math.radians(step)
        y_top = float(merc_y(north))
        y_bot = float(merc_y(south))
        ny = math.ceil((y_top - y_bot) / dx)
        return cls(
            west=west, south=float(inv_merc_y(y_top - ny * dx)),
            east=west + nx * step, north=north,
            step=step, nx=nx, ny=ny, y_top=y_top, dy=dx,
        )

    def lon_centers(self) -> np.ndarray:
        return self.west + (np.arange(self.nx) + 0.5) * self.step

    def lat_centers(self) -> np.ndarray:
        return inv_merc_y(self.y_top - (np.arange(self.ny) + 0.5) * self.dy)

    def to_pixel(self, lon, lat):
        x = (np.asarray(lon) - self.west) / self.step
        y = (self.y_top - merc_y(np.asarray(lat))) / self.dy
        return x, y

    def as_meta(self) -> dict:
        return {
            "west": self.west, "south": self.south, "east": self.east, "north": self.north,
            "step": self.step, "nx": self.nx, "ny": self.ny,
            "y_top": self.y_top, "dy": self.dy,
        }


def rasterize_mask(spec: GridSpec, polygons: list[Polygon], supersample: int = 3) -> np.ndarray:
    """Fractional land coverage per pixel (0..1); gives anti-aliased coastlines."""
    ss = supersample
    img = Image.new("L", (spec.nx * ss, spec.ny * ss), 0)
    draw = ImageDraw.Draw(img)
    for rings in polygons:
        for i, ring in enumerate(rings):
            x, y = spec.to_pixel(ring[:, 0], ring[:, 1])
            pts = list(zip((x * ss).tolist(), (y * ss).tolist()))
            if len(pts) >= 3:
                draw.polygon(pts, fill=255 if i == 0 else 0)
    return np.asarray(img.reduce(ss), dtype=np.float32) / 255.0


def _xyz(lon_deg, lat_deg) -> np.ndarray:
    lam, phi = np.radians(lon_deg), np.radians(lat_deg)
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])


def interpolate_idw(
    spec: GridSpec,
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    where: np.ndarray,
    k: int = 16,
    power: float = 2.0,
    chunk: int = 100_000,
) -> np.ndarray:
    """IDW from the k nearest stations (great-circle distance) onto ``where`` pixels.

    Returns a float32 (ny, nx) array, NaN outside ``where``. Output is always
    within [min(values), max(values)] - no overshoot.
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    values = np.asarray(values, float)
    if len(values) == 0:
        raise ValueError("no stations to interpolate")
    k = min(k, len(values))
    tree = cKDTree(_xyz(lon, lat))

    rows, cols = np.nonzero(where)
    lons = spec.lon_centers()[cols]
    lats = spec.lat_centers()[rows]
    out = np.full((spec.ny, spec.nx), np.nan, dtype=np.float32)

    for s in range(0, len(rows), chunk):
        sl = slice(s, s + chunk)
        chord, idx = tree.query(_xyz(lons[sl], lats[sl]), k=k, workers=-1)
        if k == 1:
            chord, idx = chord[:, None], idx[:, None]
        d_km = 2 * R_EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))
        w = 1.0 / (d_km + IDW_EPS_KM) ** power
        out[rows[sl], cols[sl]] = (w * values[idx]).sum(axis=1) / w.sum(axis=1)
    return out


def classify(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Class index per pixel (0..len(edges)); NaN pixels get -1.

    Lower edge inclusive: 10.0 is class 1 for edges [10, 20, ...].
    """
    cls = np.digitize(np.nan_to_num(values, nan=0.0), edges).astype(np.int16)
    cls[np.isnan(values)] = -1
    return cls
