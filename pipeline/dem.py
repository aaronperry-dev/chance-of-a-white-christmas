"""Elevation from AWS Open Data Terrain Tiles (Terrarium encoding) resampled onto the LCC grid.

Terrarium PNG: elevation_m = R * 256 + G + B / 256 - 32768. Tiles are Web Mercator
256 px; zoom 7 is ~0.8-1.1 km/px over CONUS. Tiles are cached on disk under
``data/raw/terrarium`` and downloaded in parallel with retries.
"""
from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import map_coordinates

from . import config as C
from .fetch import USER_AGENT, FetchError
from .geo import LccGrid, xy_to_lonlat

log = logging.getLogger(__name__)
TILE = 256


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def encode_terrarium(elev: np.ndarray) -> np.ndarray:
    v = np.asarray(elev, dtype=np.float64) + 32768.0
    r = np.floor(v / 256.0)
    g = np.floor(v - r * 256.0)
    b = np.floor((v - np.floor(v)) * 256.0)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def tile_coords(lon, lat, z: int):
    """Fractional Web-Mercator tile coordinates (x east, y south) of lon/lat at zoom z."""
    n = 2.0 ** z
    lat_r = np.radians(np.clip(np.asarray(lat, float), -85.0511287798, 85.0511287798))
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n
    return x, y


def _get_tile(z: int, x: int, y: int, cache: Path, retries: int = 4) -> np.ndarray:
    path = Path(cache) / str(z) / str(x) / f"{y}.png"
    if path.exists() and path.stat().st_size > 0:
        data = path.read_bytes()
    else:
        url = C.TERRAIN_URL.format(z=z, x=x, y=y)
        last = None
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, timeout=(10, 60), headers={"User-Agent": USER_AGENT})
                if r.status_code == 404:
                    return np.zeros((TILE, TILE), np.float32)       # no tile = open water / no data
                r.raise_for_status()
                data = r.content
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(path)
                break
            except requests.RequestException as exc:
                last = exc
                time.sleep(1.5 * attempt)
        else:
            raise FetchError(f"terrain tile {z}/{x}/{y} failed: {last}")
    return decode_terrarium(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))


def build_mosaic(z: int, x_rng, y_rng, cache: Path, workers: int = 8) -> np.ndarray:
    """Stitch tiles x_rng=(min,max), y_rng=(min,max) (inclusive) into one float32 array."""
    xs, ys = range(x_rng[0], x_rng[1] + 1), range(y_rng[0], y_rng[1] + 1)
    jobs = [(x, y) for y in ys for x in xs]
    log.info("terrain: %d tiles at zoom %d", len(jobs), z)
    with ThreadPoolExecutor(workers) as ex:
        tiles = list(ex.map(lambda j: _get_tile(z, j[0], j[1], cache), jobs))
    out = np.empty((len(ys) * TILE, len(xs) * TILE), np.float32)
    for (x, y), t in zip(jobs, tiles):
        out[(y - y_rng[0]) * TILE:(y - y_rng[0] + 1) * TILE, (x - x_rng[0]) * TILE:(x - x_rng[0] + 1) * TILE] = t
    return out


def sample_mosaic(mosaic: np.ndarray, x_min: int, y_min: int, fx, fy) -> np.ndarray:
    """Bilinear sample at fractional tile coordinates (pixel centres sit at +0.5)."""
    cols = (np.asarray(fx) - x_min) * TILE - 0.5
    rows = (np.asarray(fy) - y_min) * TILE - 0.5
    return map_coordinates(mosaic, [rows, cols], order=1, mode="nearest").astype(np.float32)


def dem_on_grid(grid: LccGrid, cache: Path, z: int = C.DEM_ZOOM, chunk_rows: int = 256) -> np.ndarray:
    """Elevation (m) at every pixel centre of ``grid``, shape (ny, nx)."""
    xc, yc = grid.x_centers(), grid.y_centers()
    # tile range: sample the grid's rows/cols sparsely but always include the far edges
    ri = np.unique(np.r_[np.arange(0, grid.ny, max(1, grid.ny // 64)), grid.ny - 1])
    ci = np.unique(np.r_[np.arange(0, grid.nx, max(1, grid.nx // 64)), grid.nx - 1])
    X, Y = np.meshgrid(xc[ci], yc[ri])
    lon, lat = xy_to_lonlat(X, Y)
    tx, ty = tile_coords(lon, lat, z)
    x_min, x_max = int(np.floor(tx.min())) - 1, int(np.floor(tx.max())) + 1
    y_min, y_max = int(np.floor(ty.min())) - 1, int(np.floor(ty.max())) + 1
    n = 2 ** z
    x_min, x_max, y_min, y_max = max(x_min, 0), min(x_max, n - 1), max(y_min, 0), min(y_max, n - 1)
    mosaic = build_mosaic(z, (x_min, x_max), (y_min, y_max), Path(cache))
    dem = np.empty((grid.ny, grid.nx), np.float32)
    for r0 in range(0, grid.ny, chunk_rows):
        r1 = min(r0 + chunk_rows, grid.ny)
        X, Y = np.meshgrid(xc, yc[r0:r1])
        lon, lat = xy_to_lonlat(X, Y)
        fx, fy = tile_coords(lon, lat, z)
        dem[r0:r1] = sample_mosaic(mosaic, x_min, y_min, fx, fy)
    return dem


def hillshade(dem: np.ndarray, res: float, azimuth: float = C.HILLSHADE_AZIMUTH, altitude: float = C.HILLSHADE_ALTITUDE,
              z_factor: float = C.HILLSHADE_Z, gain: float = C.HILLSHADE_GAIN) -> np.ndarray:
    """uint8 shade, 128 = flat. Light from ``azimuth`` (clockwise from north) at ``altitude`` degrees."""
    d = np.nan_to_num(np.asarray(dem, np.float32), nan=0.0)
    dz_row, dz_col = np.gradient(d, res)
    zx, zy = z_factor * dz_col, -z_factor * dz_row           # east and north slopes (rows run south)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = (-zx * np.sin(az) * np.cos(alt) - zy * np.cos(az) * np.cos(alt) + np.sin(alt)) / np.sqrt(1.0 + zx * zx + zy * zy)
    rel = shade / np.sin(alt)                                # flat ground = 1
    return np.clip(128.0 * (1.0 + gain * (rel - 1.0)), 0, 255).astype(np.uint8)
