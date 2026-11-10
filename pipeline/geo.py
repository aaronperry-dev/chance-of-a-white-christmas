"""Projection (ESRI:102009 Lambert Conformal Conic) and the LCC pixel grid."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from . import config as C

_fwd = Transformer.from_crs("EPSG:4326", C.LCC_PROJ, always_xy=True)
_inv = Transformer.from_crs(C.LCC_PROJ, "EPSG:4326", always_xy=True)


def lonlat_to_xy(lon, lat):
    x, y = _fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def xy_to_lonlat(x, y):
    lon, lat = _inv.transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon), np.asarray(lat)


def scale_factor(lat_deg):
    """Local linear scale factor k of the LCC projection at a latitude.

    LCC is conformal, so k depends on latitude only; true ground area of a
    projected pixel is res**2 / k**2.
    """
    p1, p2 = (math.radians(v) for v in C.LCC_STD_PARALLELS)
    t = lambda phi: np.tan(np.pi / 4 + phi / 2)  # noqa: E731
    n = math.log(math.cos(p1) / math.cos(p2)) / math.log(t(p2) / t(p1))
    phi = np.radians(np.asarray(lat_deg, float))
    return (math.cos(p1) / np.cos(phi)) * (t(p1) / t(phi)) ** n


@dataclass(frozen=True)
class LccGrid:
    """North-up raster in LCC metres. Row 0 is the northern edge."""
    x0: float   # west edge
    y1: float   # north edge
    res: float  # metres per pixel
    nx: int
    ny: int

    @classmethod
    def from_extent(cls, xmin, ymin, xmax, ymax, res: float) -> "LccGrid":
        x0 = math.floor(xmin / res) * res
        y1 = math.ceil(ymax / res) * res
        nx = math.ceil((xmax - x0) / res)
        ny = math.ceil((y1 - ymin) / res)
        return cls(x0=x0, y1=y1, res=res, nx=nx, ny=ny)

    @property
    def x1(self) -> float:
        return self.x0 + self.nx * self.res

    @property
    def y0(self) -> float:
        return self.y1 - self.ny * self.res

    def x_centers(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.res

    def y_centers(self) -> np.ndarray:
        return self.y1 - (np.arange(self.ny) + 0.5) * self.res

    def to_pixel(self, x, y):
        return (np.asarray(x) - self.x0) / self.res, (self.y1 - np.asarray(y)) / self.res

    def pixel_xy(self, rows, cols):
        return self.x0 + (np.asarray(cols) + 0.5) * self.res, self.y1 - (np.asarray(rows) + 0.5) * self.res

    def as_meta(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1,
                "res": self.res, "nx": self.nx, "ny": self.ny}
