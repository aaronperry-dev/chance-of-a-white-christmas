"""Vector context for the web map: state lines, Canada/Mexico land, city + state labels.

Coordinates stay in lon/lat (the browser projects them with the same ESRI:102009 definition),
but they are SIMPLIFIED IN THE MAP PROJECTION, in metres. Simplifying in degrees is wrong here:
a border that follows a parallel (the 49th, for one) is a straight line in lon/lat, so degree-based
simplification collapses it to a single chord, yet in Lambert Conformal Conic a parallel is an arc,
and a 27-degree chord lies up to ~80 km off the real border.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from .geo import lonlat_to_xy

REGION = (-142.0, 12.0, -46.0, 62.0)  # lon/lat box the map can show


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Douglas-Peucker simplification (iterative, numpy)."""
    return pts[rdp_mask(pts, eps)]


def rdp_mask(pts: np.ndarray, eps: float) -> np.ndarray:
    """Boolean mask of the vertices Douglas-Peucker keeps (distances in the units of ``pts``)."""
    n = len(pts)
    if n < 5:
        return np.ones(n, bool)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        seg = pts[i + 1:j]
        ab = b - a
        L = np.hypot(*ab)
        if L == 0:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return keep


def simplify_ll(ring_ll: np.ndarray, eps_m: float) -> np.ndarray:
    """Simplify a lon/lat ring so no dropped vertex is more than ``eps_m`` metres (in LCC) from the result."""
    ll = np.asarray(ring_ll, float)[:, :2]
    x, y = lonlat_to_xy(ll[:, 0], ll[:, 1])
    return ll[rdp_mask(np.column_stack([x, y]), eps_m)]


def _ring(r, eps_m: float, nd: int = 3) -> list:
    pts = simplify_ll(r, eps_m)
    if len(pts) < 4:
        return []
    return np.round(pts, nd).tolist()


def _geom_polys(g: dict) -> list:
    return [g["coordinates"]] if g["type"] == "Polygon" else (g["coordinates"] if g["type"] == "MultiPolygon" else [])


def _feature(polys: list, props: dict) -> dict | None:
    out = []
    for poly in polys:
        rings = [_ring(r, EPS[0]) for r in poly]
        if rings and rings[0]:
            out.append([r for r in rings if r])
    if not out:
        return None
    geom = ({"type": "Polygon", "coordinates": out[0]} if len(out) == 1
            else {"type": "MultiPolygon", "coordinates": out})
    return {"type": "Feature", "properties": props, "geometry": geom}


EPS = [500.0]            # metres in the map projection (set per layer below)


def _centroid(ring: np.ndarray) -> tuple[float, float]:
    x, y = ring[:, 0], ring[:, 1]
    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = a.sum() / 2
    if abs(A) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x[:-1] + x[1:]) * a).sum() / (6 * A)), float(((y[:-1] + y[1:]) * a).sum() / (6 * A))


def build_vectors(admin1: Path, countries: Path, places: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    west, south, east, north = C.CONUS_BBOX

    # --- US states (CONUS) + label points
    states, labels = [], []
    for f in json.loads(Path(admin1).read_text(encoding="utf-8"))["features"]:
        p = f.get("properties") or {}
        if p.get("admin") != "United States of America" or p.get("name") in C.NON_CONUS:
            continue
        EPS[0] = 500.0          # ~1/3 pixel at the finest zoom
        feat = _feature(_geom_polys(f["geometry"]), {"name": p.get("name"), "postal": p.get("postal")})
        if feat:
            states.append(feat)
            polys = _geom_polys(f["geometry"])
            big = max(polys, key=lambda pl: np.ptp(np.asarray(pl[0])[:, 0]) * np.ptp(np.asarray(pl[0])[:, 1]))
            cx, cy = _centroid(np.asarray(big[0], float))
            labels.append({"t": p.get("postal"), "la": round(cy, 2), "lo": round(cx, 2)})
    (out_dir / "states.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": states}, separators=(",", ":")))

    # --- Canada / Mexico land for context (coarser)
    ctx = []
    for f in json.loads(Path(countries).read_text(encoding="utf-8"))["features"]:
        p = f.get("properties") or {}
        if p.get("ADM0_A3") not in ("CAN", "MEX"):
            continue
        EPS[0] = 3000.0         # context only; ~1 pixel at the opening zoom
        polys = []
        for poly in _geom_polys(f["geometry"]):
            r = np.asarray(poly[0], float)
            if r[:, 0].max() < REGION[0] or r[:, 0].min() > REGION[2] or r[:, 1].max() < REGION[1] or r[:, 1].min() > REGION[3]:
                continue
            polys.append(poly[:1])           # exterior only: lakes are not needed for context
        feat = _feature(polys, {"name": p.get("ADMIN")})
        if feat:
            ctx.append(feat)
    (out_dir / "context.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": ctx}, separators=(",", ":")))

    # --- cities
    cities = []
    for f in json.loads(Path(places).read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        lo, la = p["longitude"], p["latitude"]
        if not (REGION[0] <= lo <= REGION[2] and REGION[1] <= la <= REGION[3]):
            continue
        usa = p.get("adm0_a3") == "USA"
        if usa and not (west - 0.5 <= lo <= east + 0.5 and south - 0.5 <= la <= north + 0.5):
            continue
        if not usa and p.get("scalerank", 9) > 1:
            continue
        cities.append({"n": p["nameascii"], "la": round(la, 3), "lo": round(lo, 3),
                       "r": int(p.get("scalerank", 9)), "p": int(p.get("pop_max") or 0), "us": int(usa)})
    (out_dir / "places.json").write_text(
        json.dumps({"cities": cities, "states": labels}, separators=(",", ":")))
    return {"states": len(states), "context": len(ctx), "cities": len(cities)}
