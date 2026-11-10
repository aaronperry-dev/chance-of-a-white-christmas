"""Station -> surface interpolation: plain inverse-distance weighting (IDW).

The percentage at each grid cell is a distance-weighted mean of the K nearest
stations' values (power 2, with a small softening distance so a cell sitting on a
station gets essentially that station's value). No terrain or other ancillary
data is used: the surface comes only from NOAA's station values.

All distances are in LCC metres (conformal, ~1 % scale error across CONUS).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import config as C


class Estimator:
    def __init__(self, sx, sy, p_pct):
        self.sx, self.sy = np.asarray(sx, float), np.asarray(sy, float)
        self.p = np.asarray(p_pct, float)
        self.tree = cKDTree(np.column_stack([self.sx, self.sy]))

    def predict(self, tx, ty, exclude_self: bool = False) -> np.ndarray:
        """Estimate at target points (metres). ``exclude_self`` drops the nearest
        station (the point itself) - used for leave-one-out validation."""
        k = min(C.IDW_K, len(self.p) - int(exclude_self))
        dist, idx = self.tree.query(np.column_stack([tx, ty]), k=k + int(exclude_self), workers=-1)
        if k + int(exclude_self) == 1:
            dist, idx = dist[:, None], idx[:, None]
        if exclude_self:
            dist, idx = dist[:, 1:], idx[:, 1:]
        w = 1.0 / (dist + C.IDW_EPS_M) ** C.IDW_POWER
        return (w * self.p[idx]).sum(1) / w.sum(1)


def interpolate_grid(est: Estimator, grid, where: np.ndarray, chunk: int = 150_000) -> np.ndarray:
    """Evaluate the estimator on every pixel where ``where`` is True. NaN elsewhere."""
    rows, cols = np.nonzero(where)
    out = np.full((grid.ny, grid.nx), np.nan, dtype=np.float32)
    for s in range(0, len(rows), chunk):
        r, c = rows[s:s + chunk], cols[s:s + chunk]
        x, y = grid.pixel_xy(r, c)
        out[r, c] = est.predict(x, y)
    return out


def loo_cv(sx, sy, p_pct, max_n: int = 20000, seed: int = 0) -> dict:
    """Leave-one-out cross-validation at the stations themselves (percentage points)."""
    est = Estimator(sx, sy, p_pct)
    n = len(est.p)
    sel = np.arange(n)
    if n > max_n:
        sel = np.random.default_rng(seed).choice(n, max_n, replace=False)
    err = est.predict(est.sx[sel], est.sy[sel], exclude_self=True) - est.p[sel]
    return {"n": int(len(sel)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err))}
