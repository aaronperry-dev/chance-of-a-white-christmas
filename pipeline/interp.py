"""Station -> surface estimators.

``idw``      plain inverse-distance weighting of the percentages (baseline / no-DEM fallback).
``terrain``  geographically weighted quasi-binomial regression on elevation:

                 logit P(x) = a(x) + b(x) * (z - z0(x))

             a and b are fitted from the K nearest stations (tricube weights, iteratively reweighted
             least squares on the station *frequencies*, so 0 % and 100 % readings are handled without
             the logit(0) blow-up of a plain logit transform). b is constrained to [0, SLOPE_MAX]
             (higher = colder = more snow cover) and ridge-shrunk toward 0 where the neighbours hardly
             differ in height.

             Station values are 30-year frequencies, so they are noisy; the fit SMOOTHS them rather than
             honouring them. (An earlier version added the regression residuals back with an inverse-
             distance or Gaussian kernel; that drew a round patch around every unusual station, which is
             what showed up as small circles near the 1 % edge.)

             On the grid the regression is solved on a coarse node lattice (NODE_SPACING_M), a, b and the
             reference elevation z0 are bicubic-upsampled, and the elevation term is applied from the
             full-resolution DEM: smooth regional field, terrain detail at 1 km.

All distances are in LCC metres (conformal, ~1 % scale error across CONUS).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import config as C


def expit(y):
    return 1.0 / (1.0 + np.exp(-y))


def logit_frac(f, years: float | None = None):
    """Empirical logit of a frequency f in [0, 1]: (k + c) / (n + 2c) with a small pseudo-count."""
    n = C.STATION_YEARS_EFF if years is None else years
    c = C.FREQ_PSEUDO
    y = (np.asarray(f, float) * n + c) / (n + 2 * c)
    return np.log(y / (1 - y))


def expit_pct(y: np.ndarray) -> np.ndarray:
    return 100.0 * expit(y)


def _irls(wt, zz, fr, tz0, ridge, slope_max):
    """Weighted quasi-binomial fit of fr on (z - tz0) for each row. Returns a, b."""
    n, c = C.STATION_YEARS_EFF, C.FREQ_PSEUDO
    yadj = (fr * n + c) / (n + 2 * c)
    dz = zz - tz0[:, None]
    m = (wt * yadj).sum(1) / wt.sum(1)
    a = np.log(m / (1 - m))
    b = np.zeros(len(a))
    for _ in range(C.IRLS_ITERS):
        eta = a[:, None] + b[:, None] * dz
        mu = np.clip(expit(eta), 1e-4, 1 - 1e-4)
        v = mu * (1 - mu)
        zr = eta + (yadj - mu) / v                    # working response
        w = wt * v
        sw = w.sum(1)
        zb = (w * dz).sum(1) / sw
        yb = (w * zr).sum(1) / sw
        d2 = dz - zb[:, None]
        b = np.clip((w * d2 * (zr - yb[:, None])).sum(1) / ((w * d2 * d2).sum(1) + sw * ridge ** 2), 0.0, slope_max)
        a = np.clip(yb - b * zb, -C.LOGIT_LIMIT, C.LOGIT_LIMIT)
    return a, b


class Estimator:
    def __init__(self, sx, sy, sz, p_pct, method: str = "terrain", k: int | None = None):
        if method not in {"idw", "terrain"}:
            raise ValueError(method)
        self.method = method
        self.sx, self.sy = np.asarray(sx, float), np.asarray(sy, float)
        self.sz = np.asarray(sz, float)
        self.p = np.asarray(p_pct, float)
        self.frac = np.clip(self.p / 100.0, 0.0, 1.0)
        self.k = int(k) if k else (C.IDW_K if method == "idw" else C.TERRAIN_K)
        self.tree = cKDTree(np.column_stack([self.sx, self.sy]))
        if method == "terrain" and np.isnan(self.sz).any():
            raise ValueError("terrain method needs an elevation for every station")

    # ------------------------------------------------------------------ neighbour search
    def _neighbours(self, tx, ty, exclude_self: bool):
        k = min(self.k, len(self.p) - int(exclude_self))
        dist, idx = self.tree.query(np.column_stack([tx, ty]), k=k + int(exclude_self), workers=-1)
        if k + int(exclude_self) == 1:
            dist, idx = dist[:, None], idx[:, None]
        if exclude_self:
            dist, idx = dist[:, 1:], idx[:, 1:]
        return dist, idx

    # ------------------------------------------------------------------ terrain regression
    def fit(self, tx, ty, tz0, exclude_self: bool = False, chunk: int = 100_000):
        """Local fit at points; returns a (logit at elevation tz0), b (logit/m), neighbour elevation min and max."""
        tx, ty, tz0 = (np.asarray(v, float) for v in (tx, ty, tz0))
        a, b = np.empty(len(tx)), np.empty(len(tx))
        zlo, zhi = np.empty(len(tx)), np.empty(len(tx))
        for s in range(0, len(tx), chunk):
            sl = slice(s, s + chunk)
            dist, idx = self._neighbours(tx[sl], ty[sl], exclude_self)
            h = dist[:, -1:] * 1.001 + 1.0
            wt = np.clip(1.0 - (dist / h) ** 3, 0, None) ** 3
            zz = self.sz[idx]
            a[sl], b[sl] = _irls(wt, zz, self.frac[idx], tz0[sl], C.RIDGE_M, C.SLOPE_MAX)
            zlo[sl], zhi[sl] = zz.min(1), zz.max(1)
        return a, b, zlo, zhi

    def predict(self, tx, ty, tz=None, exclude_self: bool = False) -> np.ndarray:
        """Estimate (percent) at target points (metres). ``exclude_self`` drops the nearest
        station (the point itself) - used for leave-one-out validation."""
        if self.method == "idw":
            dist, idx = self._neighbours(tx, ty, exclude_self)
            w = 1.0 / (dist + C.IDW_EPS_M) ** C.IDW_POWER
            return (w * self.p[idx]).sum(1) / w.sum(1)
        tx, ty = np.asarray(tx, float), np.asarray(ty, float)
        if tz is None:       # no elevation: evaluate at the neighbourhood's own mean height
            dist, idx = self._neighbours(tx, ty, exclude_self)
            tz = self.sz[idx].mean(1)
        tz = np.asarray(tz, float)
        a, b, zlo, zhi = self.fit(tx, ty, tz, exclude_self)
        tzc = np.clip(tz, zlo - C.EXTRAPOLATE_MARGIN_M, zhi + C.EXTRAPOLATE_MARGIN_M)
        return expit_pct(a + b * (tzc - tz))

    # ------------------------------------------------------------------ grid evaluation
    def predict_grid(self, grid, where: np.ndarray, dem: np.ndarray, node_m: float | None = None) -> np.ndarray:
        """Surface on the whole grid: regression on a coarse lattice, bicubic upsampling, DEM term at full res."""
        step = max(1, int(round((node_m or C.NODE_SPACING_M) / grid.res)))
        ny, nx = grid.ny, grid.nx
        nyc, nxc = -(-ny // step), -(-nx // step)
        pad = ((0, nyc * step - ny), (0, nxc * step - nx))
        dem_p = np.pad(np.asarray(dem, np.float32), pad, mode="edge")
        z0 = dem_p.reshape(nyc, step, nxc, step).mean(axis=(1, 3))
        ctr = (np.arange(nyc) + 0.5) * step - 0.5, (np.arange(nxc) + 0.5) * step - 0.5   # pixel-index centres
        R, Cc = np.meshgrid(ctr[0], ctr[1], indexing="ij")
        nxm, nym = grid.pixel_xy(R.ravel(), Cc.ravel())
        a, b, zlo, zhi = self.fit(nxm, nym, z0.ravel())

        def up(F, order):
            f = ndimage.zoom(F.reshape(nyc, nxc).astype(np.float32), step, order=order, mode="nearest", grid_mode=True)
            return f[:ny, :nx]

        out = np.full((ny, nx), np.nan, np.float32)
        A, B, Z0, LO, HI = up(a, 3), np.clip(up(b, 3), 0, C.SLOPE_MAX), up(z0.ravel(), 3), up(zlo, 1), up(zhi, 1)
        dz = np.clip(np.asarray(dem, np.float32), LO - C.EXTRAPOLATE_MARGIN_M, HI + C.EXTRAPOLATE_MARGIN_M) - Z0
        eta = np.clip(A + B * dz, -C.LOGIT_LIMIT, C.LOGIT_LIMIT)
        out[where] = expit_pct(eta[where]).astype(np.float32)
        return out


def interpolate_grid(est: Estimator, grid, where: np.ndarray, z_grid=None, chunk: int = 150_000) -> np.ndarray:
    """Evaluate the estimator on every pixel where ``where`` is True. NaN elsewhere."""
    if est.method == "terrain" and z_grid is not None:
        return est.predict_grid(grid, where, z_grid)
    rows, cols = np.nonzero(where)
    out = np.full((grid.ny, grid.nx), np.nan, dtype=np.float32)
    for s in range(0, len(rows), chunk):
        r, c = rows[s:s + chunk], cols[s:s + chunk]
        x, y = grid.pixel_xy(r, c)
        z = None if z_grid is None else z_grid[r, c].astype(float)
        out[r, c] = est.predict(x, y, z)
    return out


def loo_cv(sx, sy, sz, p_pct, method: str, max_n: int = 20000, seed: int = 0, k: int | None = None) -> dict:
    """Leave-one-out cross-validation at the stations themselves (percentage points)."""
    est = Estimator(sx, sy, sz, p_pct, method, k=k)
    n = len(est.p)
    sel = np.arange(n)
    if n > max_n:
        sel = np.random.default_rng(seed).choice(n, max_n, replace=False)
    pred = est.predict(est.sx[sel], est.sy[sel], est.sz[sel], exclude_self=True)
    err = pred - est.p[sel]
    return {"method": method, "k": est.k, "n": int(len(sel)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err))}


def choose_k(sx, sy, sz, p_pct, candidates=None) -> tuple[int, dict]:
    """K with the lowest leave-one-out RMSE (capped to the station count)."""
    cands = sorted({min(int(k), len(p_pct) - 1) for k in (candidates or C.TERRAIN_K_CANDIDATES) if len(p_pct) > 2})
    res = {k: loo_cv(sx, sy, sz, p_pct, "terrain", k=k) for k in cands}
    best = min(res, key=lambda k: res[k]["rmse"])
    return best, res[best]
