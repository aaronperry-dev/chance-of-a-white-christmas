"""Station -> surface estimators.

``idw``      plain inverse-distance weighting of the percentages (baseline).
``terrain``  local regression of logit(probability) on elevation (tricube
             weights over the K nearest stations) + a correction from the regression
             residuals. The regression step lets the surface follow the terrain between
             stations (more snow on the ridges, less in the valleys); the residual step
             pulls it toward the station values. By default (RESID_SMOOTH_M > 0) the
             correction is a Gaussian-weighted mean, which is smooth, so an unusual
             station makes a gentle bump instead of the spike-and-ring ("bull's-eye")
             a 1/distance kernel draws; stations are then reproduced to about 2
             percentage points rather than a fraction of one. RESID_SMOOTH_M = 0 restores
             the exact-honouring IDW correction. The elevation slope is constrained to be
             >= 0 (higher = colder = more snow cover) and is shrunk toward 0 where
             nearby stations hardly differ in elevation.

All distances are in LCC metres (conformal, ~1 % scale error across CONUS).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import config as C


def logit(p_pct: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_pct, float) / 100.0, C.LOGIT_EPS, 1 - C.LOGIT_EPS)
    return np.log(p / (1 - p))


def expit_pct(y: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-y))


class Estimator:
    def __init__(self, sx, sy, sz, p_pct, method: str = "terrain"):
        if method not in {"idw", "terrain"}:
            raise ValueError(method)
        self.method = method
        self.sx, self.sy = np.asarray(sx, float), np.asarray(sy, float)
        self.sz = np.asarray(sz, float)
        self.p = np.asarray(p_pct, float)
        self.ylogit = logit(self.p)
        self.tree = cKDTree(np.column_stack([self.sx, self.sy]))
        if method == "terrain" and np.isnan(self.sz).any():
            raise ValueError("terrain method needs an elevation for every station")

    def predict(self, tx, ty, tz=None, exclude_self: bool = False) -> np.ndarray:
        """Estimate at target points (metres). ``exclude_self`` drops the nearest
        station (the point itself) - used for leave-one-out validation."""
        k = C.IDW_K if self.method == "idw" else C.TERRAIN_K
        k = min(k, len(self.p) - int(exclude_self))
        dist, idx = self.tree.query(np.column_stack([tx, ty]), k=k + int(exclude_self), workers=-1)
        if k + int(exclude_self) == 1:
            dist, idx = dist[:, None], idx[:, None]
        if exclude_self:
            dist, idx = dist[:, 1:], idx[:, 1:]
        if self.method == "idw":
            w = 1.0 / (dist + C.IDW_EPS_M) ** C.IDW_POWER
            return (w * self.p[idx]).sum(1) / w.sum(1)

        yv, zz = self.ylogit[idx], self.sz[idx]
        h = dist[:, -1:] * 1.001 + 1.0
        wt = np.clip(1.0 - (dist / h) ** 3, 0, None) ** 3
        sw = wt.sum(1)
        zbar = (wt * zz).sum(1) / sw
        ybar = (wt * yv).sum(1) / sw
        dz = zz - zbar[:, None]
        sxx = (wt * dz * dz).sum(1) + sw * C.RIDGE_M ** 2
        sxy = (wt * dz * (yv - ybar[:, None])).sum(1)
        b = np.clip(sxy / sxx, 0.0, C.SLOPE_MAX)
        tz = zbar if tz is None else np.where(np.isnan(tz), zbar, tz)
        tz = np.clip(tz, zz.min(1) - C.EXTRAPOLATE_MARGIN_M, zz.max(1) + C.EXTRAPOLATE_MARGIN_M)
        trend = ybar + b * (tz - zbar)
        resid = yv - (ybar[:, None] + b[:, None] * dz)
        if C.RESID_SMOOTH_M > 0:
            # Smooth, gently shrinking correction: Gaussian-weighted mean of the neighbours' residuals.
            # It has no spike at the stations (the 1/d^p kernel's cusp is what draws bull's-eye rings) and
            # fades to zero far from any station, leaving the terrain trend alone there.
            wi = np.exp(-0.5 * (dist / C.RESID_SMOOTH_M) ** 2)
            return expit_pct(trend + (wi * resid).sum(1) / (wi.sum(1) + C.RESID_SHRINK))
        wi = 1.0 / (dist + C.RESID_EPS_M) ** C.RESID_POWER
        return expit_pct(trend + (wi * resid).sum(1) / wi.sum(1))


def interpolate_grid(est: Estimator, grid, where: np.ndarray, z_grid=None, chunk: int = 150_000) -> np.ndarray:
    """Evaluate the estimator on every pixel where ``where`` is True. NaN elsewhere."""
    rows, cols = np.nonzero(where)
    out = np.full((grid.ny, grid.nx), np.nan, dtype=np.float32)
    for s in range(0, len(rows), chunk):
        r, c = rows[s:s + chunk], cols[s:s + chunk]
        x, y = grid.pixel_xy(r, c)
        z = None if z_grid is None else z_grid[r, c].astype(float)
        out[r, c] = est.predict(x, y, z)
    return out


def loo_cv(sx, sy, sz, p_pct, method: str, max_n: int = 20000, seed: int = 0) -> dict:
    """Leave-one-out cross-validation at the stations themselves (percentage points)."""
    est = Estimator(sx, sy, sz, p_pct, method)
    n = len(est.p)
    sel = np.arange(n)
    if n > max_n:
        sel = np.random.default_rng(seed).choice(n, max_n, replace=False)
    pred = est.predict(est.sx[sel], est.sy[sel], est.sz[sel], exclude_self=True)
    err = pred - est.p[sel]
    return {"method": method, "n": int(len(sel)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err))}
