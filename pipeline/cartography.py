"""Cartographic finishing of the classified surface: smooth upsampling and removal of tiny patches."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

TRANSPARENT = 99        # internal code for "no colour" cells while sieving


def fill_nearest(values: np.ndarray) -> np.ndarray:
    """Replace NaNs with the nearest valid value (so a spline can be fitted across the coast)."""
    bad = np.isnan(values)
    if not bad.any():
        return values
    if bad.all():
        return np.zeros_like(values)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return values[tuple(idx)]


def upsample(values: np.ndarray, factor: int, lo: float = 0.0, hi: float = 100.0) -> np.ndarray:
    """Bicubic-spline upsample of a (smooth) float surface; NaN-safe, clipped to [lo, hi]."""
    if factor == 1:
        return values
    f = ndimage.zoom(fill_nearest(np.asarray(values, np.float32)), factor, order=3, mode="nearest", grid_mode=True)
    return np.clip(f, lo, hi, out=f)


def upsample_nearest(a: np.ndarray, factor: int) -> np.ndarray:
    return a if factor == 1 else np.repeat(np.repeat(a, factor, 0), factor, 1)


def relief_mask(dem: np.ndarray, relief_m: float, size: int = 5) -> np.ndarray:
    """True where the DEM varies by more than ``relief_m`` over a ``size``-pixel window (real mountains)."""
    d = np.asarray(dem, np.float32)
    return (ndimage.maximum_filter(d, size=size) - ndimage.minimum_filter(d, size=size)) > relief_m


def sieve(code: np.ndarray, min_px: int, protect: np.ndarray | None = None, valid: np.ndarray | None = None) -> np.ndarray:
    """Merge connected patches of one code smaller than ``min_px`` into the surrounding majority code.

    ``code`` is an int array (class index, TRANSPARENT, or -1 for "outside"). Patches that are mostly on
    ``protect`` cells (terrain with real relief, where a small summit patch is genuine) are kept.
    Returns a new array.
    """
    out = code.copy()
    st = ndimage.generate_binary_structure(2, 1)          # 4-connected
    for k in np.unique(out[out >= 0] if valid is None else out[valid & (out >= 0)]):
        lab, n = ndimage.label(out == k, structure=st)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = np.nonzero((sizes[1:] < min_px))[0] + 1
        if len(small) == 0:
            continue
        if protect is not None:
            frac = ndimage.sum(protect, lab, small) / sizes[small]
            small = small[frac < 0.5]
        objs = ndimage.find_objects(lab)
        for i in small:
            sl = objs[i - 1]
            y0, y1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, out.shape[0])
            x0, x1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, out.shape[1])
            reg = lab[y0:y1, x0:x1] == i
            ring = ndimage.binary_dilation(reg, structure=st, iterations=1) & ~reg
            nb = out[y0:y1, x0:x1][ring]
            nb = nb[nb >= 0]
            if len(nb):
                new = np.bincount(nb).argmax()
                out[y0:y1, x0:x1][reg] = new
    return out
