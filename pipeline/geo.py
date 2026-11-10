"""ESRI:102009 (North America Lambert Conformal Conic) helpers and the pixel grid.

The grid is a plain north-up raster in LCC metres, so a pixel is a square on the map
exactly as the browser draws it (same projection as NOAA's own maps).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from . import config as C

_FWD = Transformer.from_crs("EPSG:4326", C.LCC_PROJ, always_xy=True)
_INV = Transformer.from_crs(C.LCC_PROJ, "EPSG:4326", always_xy=True)


def lonlat_to_xy(lon, lat):
    return _FWD.transform(lon, lat)


def xy_to_lonlat(x, y):
    return _INV.transform(x, y)


_P1, _P2 = (math.radians(v) for v in (20.0, 60.0))
_N = math.log(math.cos(_P1) / math.cos(_P2)) / math.log(math.tan(math.pi / 4 + _P2 / 2) / math.tan(math.pi / 4 + _P1 / 2))


def scale_factor(lat_deg):
    """LCC point scale factor k at a latitude (1.0 on the standard parallels, <1 between them).

    Spherical formula; within ~0.5 % of the ellipsoidal value, which is plenty for area weights.
    """
    phi = np.radians(np.asarray(lat_deg, float))
    t = lambda p: np.tan(np.pi / 4 + p / 2) ** _N
    return (math.cos(_P1) * t(_P1)) / (np.cos(phi) * t(phi))


@dataclass(frozen=True)
class LccGrid:
    x0: float   # west edge (m)
    y1: float   # north edge (m)
    res: float  # pixel size (m)
    nx: int
    ny: int

    @classmethod
    def from_extent(cls, xmin, ymin, xmax, ymax, res):
        x0 = math.floor(xmin / res) * res
        y1 = math.ceil(ymax / res) * res
        return cls(x0, y1, res, int(math.ceil((xmax - x0) / res)), int(math.ceil((y1 - ymin) / res)))

    @property
    def x1(self) -> float:
        return self.x0 + self.nx * self.res

    @property
    def y0(self) -> float:
        return self.y1 - self.ny * self.res

    def x_centers(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.res

    def y_centers(self) -> np.ndarray:           # row order: north -> south
        return self.y1 - (np.arange(self.ny) + 0.5) * self.res

    def to_pixel(self, x, y):
        """Fractional (col, row) of map coordinates; pixel (i, j) covers [i, i+1) x [j, j+1)."""
        return (np.asarray(x) - self.x0) / self.res, (self.y1 - np.asarray(y)) / self.res

    def pixel_xy(self, rows, cols):
        return self.x0 + (np.asarray(cols) + 0.5) * self.res, self.y1 - (np.asarray(rows) + 0.5) * self.res

    def as_meta(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "res": self.res, "nx": self.nx, "ny": self.ny}
