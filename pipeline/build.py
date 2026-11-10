"""Pipeline entry point:  python -m pipeline.build [--demo] [--stations FILE] [--res METRES]

  NOAA station table   -+
  AWS terrain tiles    -+->  1 km terrain-aware surface in NOAA's projection and classes
  Natural Earth land   -+     -> site/data/{overlay.png, relief.webp, values.bin.gz,
                                            stations.json, stations.csv, meta.json}
  Natural Earth        ->  site/data/{states.geojson, context.geojson, places.json}  (optional overlays)

NOAA's published tiles bake station dots and labels into the image, and NOAA publishes no raw
grid, so this builds an equivalent surface from the same station values. Terrain (AWS Terrarium
elevation) shapes the surface between stations and supplies the hillshade. NOAA's own tiles stay
available on the page as an optional reference layer.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import dem as demmod
from .boundary import load_conus_polygons, load_lake_polygons, load_land_polygons
from .fetch import FetchError, ensure_file
from . import basemap
from .geo import LccGrid, lonlat_to_xy
from . import cartography as cart
from . import tiles as tilemod
from .grid import classify, polygon_extent, project_polygons, rasterize_mask
from .interp import Estimator, choose_k, interpolate_grid, loo_cv
from .render import area_stats, render_overlay, render_relief, write_value_grid

log = logging.getLogger("pipeline")


def station_stats(prob: np.ndarray) -> dict:
    """Shares are of STATIONS (not land area): the map surface is NOAA's, not ours."""
    cls = np.array([C.class_index(p) for p in prob])
    n = len(prob)
    return {
        "basis": "stations",
        "n": int(n),
        "share_by_class": [float((cls == i).mean()) for i in range(len(C.NOAA_COLORS))],
        "share_ge_50": float((prob >= 50).mean()),
        "share_le_10": float((prob <= 10).mean()),
    }


def make_grid(polygons, res: float = C.GRID_RES_M):
    projected = project_polygons(polygons)
    xmin, ymin, xmax, ymax = polygon_extent(projected)
    pad = C.GRID_PAD_M
    return LccGrid.from_extent(xmin - pad, ymin - pad, xmax + pad, ymax + pad, res), projected


def _station_pixels(st: pd.DataFrame, grid: LccGrid):
    x, y = lonlat_to_xy(st.lon.values, st.lat.values)
    px, py = grid.to_pixel(x, y)
    return (np.asarray(x), np.asarray(y),
            np.clip(py.astype(int), 0, grid.ny - 1), np.clip(px.astype(int), 0, grid.nx - 1))


def _write_overlay(values: np.ndarray, dem: np.ndarray | None, grid: LccGrid, projected, path: Path) -> None:
    """Overlay PNG at OVERLAY_SCALE x the grid: classes are cut from a bicubic upsample of the smooth surface (so
    class edges are smooth curves, not pixel stairs), then tiny flat-terrain patches are merged into their surroundings."""
    f = C.OVERLAY_SCALE
    v = cart.upsample(values, f)
    cov = rasterize_mask(grid, projected, out_factor=f)
    code = classify(v).astype(np.int16)
    code[np.round(v) <= C.TRANSPARENT_AT_OR_BELOW_PCT] = cart.TRANSPARENT
    code[cov <= 0.02] = -1
    protect = cart.upsample_nearest(cart.relief_mask(dem, C.SIEVE_RELIEF_M), f) if dem is not None else None
    min_px = max(1, int(round(C.MIN_REGION_KM2 / (grid.res / 1000.0 / f) ** 2)))
    code = cart.sieve(code, min_px, protect)
    clear = code == cart.TRANSPARENT
    render_overlay(np.where(clear, 0, code).astype(np.int8), cov, C.NOAA_COLORS, path, transparent=clear)


def _write_pyramids(out_dir: Path, grid: LccGrid, has_relief: bool, base_meta: dict | None) -> dict:
    """Tile pyramids for every map image (see tiles.py); returns the page's description of them."""
    ext = (grid.x0, grid.y0, grid.x1, grid.y1)
    out = {"overlay": tilemod.write_pyramid(out_dir / "overlay.png", ext, out_dir, "overlay", fmt="webp", lossless=True)}
    if has_relief:
        out["relief"] = tilemod.write_pyramid(out_dir / "relief.webp", ext, out_dir, "relief", quality=80)
    if base_meta:
        g = base_meta["grid"]
        bext = (g["x0"], g["y0"], g["x1"], g["y1"])
        for key in ("light", "dark"):
            out["base_" + key] = tilemod.write_pyramid(out_dir / base_meta[key], bext, out_dir, "base_" + key, quality=86)
        if base_meta.get("relief"):
            out["relief_ext"] = tilemod.write_pyramid(out_dir / base_meta["relief"], bext, out_dir, "relief_ext", quality=80)
    else:
        for n in ("base_light", "base_dark", "relief_ext"):
            shutil.rmtree(out_dir / "tiles" / n, ignore_errors=True)
    if not has_relief:
        shutil.rmtree(out_dir / "tiles" / "relief", ignore_errors=True)
    return out


def build(stations: pd.DataFrame, polygons, out_dir: Path, *, res: float = C.GRID_RES_M, dem: np.ndarray | None = None,
          grid: LccGrid | None = None, dem_label: str = "", use_terrain: bool = True, dem_zoom: int = C.DEM_ZOOM, base_inputs: dict | None = None,
          synthetic: bool = False, source: dict | None = None, keep_sources: bool = True) -> dict:
    """Build every data asset. ``dem`` (ny, nx) is fetched from AWS when omitted and ``use_terrain`` is set."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = C.CONUS_BBOX
    st = stations[stations.lon.between(west, east) & stations.lat.between(south, north)].reset_index(drop=True)
    if st.empty:
        raise ValueError("no stations inside the CONUS bounding box")
    if "elev_m" not in st:
        st["elev_m"] = np.nan
    log.info("%d stations inside CONUS bbox", len(st))

    projected = project_polygons(polygons)
    if grid is None:
        grid, projected = make_grid(polygons, res)
    log.info("grid %d x %d at %.0f m", grid.nx, grid.ny, grid.res)
    coverage = rasterize_mask(grid, projected)
    land = coverage > 0.02

    if dem is None and use_terrain:
        dem = demmod.dem_on_grid(grid, C.RAW_DIR / C.TERRAIN_CACHE, dem_zoom)
        dem_label = dem_label or "AWS Open Data Terrain Tiles (Terrarium), zoom %d" % dem_zoom
    if dem is not None:
        dem = np.maximum(dem, 0.0).astype(np.float32)       # sea level floor: no coastal cliffs into the sea

    sx, sy, sr, sc = _station_pixels(st, grid)
    prob = st.prob.values.astype(float)
    cv: dict = {}
    method = "idw"
    if dem is not None:
        # station elevation in the SAME (1 km DEM) space as the grid cells it is compared with
        sz = dem[sr, sc].astype(float)
        cv["idw"] = loo_cv(sx, sy, sz, prob, "idw")
        k_best, cv["terrain"] = choose_k(sx, sy, sz, prob)
        log.info("leave-one-out RMSE (pct points): IDW %.2f | terrain-aware %.2f with K=%d  (n=%d)",
                 cv["idw"]["rmse"], cv["terrain"]["rmse"], k_best, cv["terrain"]["n"])
        if cv["terrain"]["rmse"] <= cv["idw"]["rmse"]:
            method = "terrain"
        else:
            log.warning("terrain-aware model did not beat IDW in cross-validation; using IDW (relief is still drawn)")
        est = Estimator(sx, sy, sz, prob, method, k=k_best if method == "terrain" else None)
        values = interpolate_grid(est, grid, land, dem)
    else:
        est = Estimator(sx, sy, np.zeros_like(sx), prob, "idw")
        values = interpolate_grid(est, grid, land, None)
        log.info("no terrain: plain IDW surface")
    cv["chosen"] = method

    cls = classify(values)
    _write_overlay(values, dem, grid, projected, out_dir / "overlay.png")
    write_value_grid(values, coverage, out_dir / "values.bin.gz")
    has_relief = dem is not None
    if has_relief:
        render_relief(demmod.hillshade(dem, grid.res), coverage, out_dir / "relief.webp")
    else:
        (out_dir / "relief.webp").unlink(missing_ok=True)

    base_meta = None
    if base_inputs:
        base_meta = basemap.build_base(base_inputs["land"], base_inputs.get("lakes") or [], projected, out_dir,
                                       C.RAW_DIR / C.TERRAIN_CACHE, dem=base_inputs.get("dem"),
                                       grid=base_inputs.get("grid"), zoom=base_inputs.get("zoom", C.BASE_DEM_ZOOM))
    else:
        for f in ("base_light.webp", "base_dark.webp", "relief_ext.webp"):
            (out_dir / f).unlink(missing_ok=True)

    pyramids = _write_pyramids(out_dir, grid, has_relief, base_meta)
    if not keep_sources:                      # the whole-image files are only inputs to the tiles; do not publish them
        for f in ("overlay.png", "relief.webp", "base_light.webp", "base_dark.webp", "relief_ext.webp"):
            (out_dir / f).unlink(missing_ok=True)

    recs = [{"id": r.id, "n": r.name, "s": r.state, "la": round(r.lat, 4), "lo": round(r.lon, 4),
             "e": None if pd.isna(r.elev_m) else round(float(r.elev_m) * C.M_TO_FT), "p": int(round(r.prob)),
             "y": None if pd.isna(getattr(r, "years", np.nan)) else int(r.years)}
            for r in st.itertuples()]
    (out_dir / "stations.json").write_text(json.dumps(recs, separators=(",", ":")))
    with open(out_dir / "stations.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "state", "lat", "lon", "elevation_ft", "prob_snow_depth_ge_1in_pct", "years"])
        for r in st.itertuples():
            w.writerow([r.id, r.name, r.state, r.lat, r.lon,
                        "" if pd.isna(r.elev_m) else round(float(r.elev_m) * C.M_TO_FT),
                        r.prob, "" if pd.isna(getattr(r, "years", np.nan)) else int(r.years)])

    labels = C.class_labels()
    meta = {
        "title": "Probability of a White Christmas",
        "definition": "Chance of at least 1 inch of snow on the ground on December 25",
        "period": "1991\u20132020 U.S. Climate Normals",
        "synthetic": synthetic,
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": source or {},
        "crs": {"name": C.LCC_NAME, "proj4": C.LCC_PROJ, "extent": list(C.NOAA_EXTENT)},
        "grid": grid.as_meta(),
        "conus_extent": [round(v, 1) for v in polygon_extent(projected)],
        "pyramids": pyramids,
        "likelihood": [{"hi": hi, "label": lab} for hi, lab in C.LIKELIHOOD],
        "surface": {"overlay": "overlay.png", "overlay_scale": C.OVERLAY_SCALE, "relief": "relief.webp" if has_relief else None,
                    "values": "values.bin.gz", "nodata": 255, "method": method,
                    "transparent_at_or_below": C.TRANSPARENT_AT_OR_BELOW_PCT,
                    "dem": dem_label if has_relief else "", "cv": cv},
        "base": base_meta,
        "tiles": {
            "url": C.TILES_URL, "service": C.TILES_SERVICE_PAGE, "size": C.TILE_SIZE,
            "origin": list(C.TILE_ORIGIN), "resolutions": C.TILE_RESOLUTIONS,
            "full_extent": list(C.TILE_FULL_EXTENT), "attribution": C.TILE_ATTRIBUTION,
        },
        "classes": [{"label": lab, "color": C.NOAA_COLORS[i],
                     "lo": 0 if i == 0 else C.CLASS_UPPER_EDGES[i - 1],
                     "hi": 100 if i == len(C.CLASS_UPPER_EDGES) else C.CLASS_UPPER_EDGES[i]}
                    for i, lab in enumerate(labels)],
        "stats": area_stats(values, cls, coverage, grid, len(C.NOAA_COLORS)),
        "station_stats": station_stats(np.round(prob)),     # same rounding as stations.json
        "n_stations": int(len(st)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _vectors(out: Path) -> None:
    """Optional overlays (state lines, place labels). Failure here must not fail the build."""
    try:
        from .webvec import build_vectors
        paths = [ensure_file(u, C.RAW_DIR / f, min_bytes=10_000) for u, f in (
            (C.BOUNDARY_URL, C.BOUNDARY_FILE), (C.COUNTRIES_URL, C.COUNTRIES_FILE), (C.PLACES_URL, C.PLACES_FILE))]
        log.info("vector overlays: %s", build_vectors(*paths, out))
    except (FetchError, OSError, ValueError) as exc:
        log.warning("optional overlays skipped (%s); the map and stations are unaffected", exc)
        for f in ("states.geojson", "context.geojson", "places.json"):
            (out / f).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", choices=["auto", "arcgis", "xlsx"], default="auto",
                    help="auto: NOAA feature layer, falling back to the NCEI spreadsheet")
    ap.add_argument("--stations", type=Path, help="local station table (.xlsx/.csv); implies --source xlsx")
    ap.add_argument("--prob-col", help="exact header of the snow-depth probability column (xlsx)")
    ap.add_argument("--refresh", action="store_true", help="re-download the station layer even if cached")
    ap.add_argument("--out", type=Path, default=C.OUT_DIR)
    ap.add_argument("--no-overlays", action="store_true", help="skip the Natural Earth state-line / label overlays")
    ap.add_argument("--res", type=float, default=C.GRID_RES_M, help="grid cell size in metres (default %(default)s)")
    ap.add_argument("--dem-zoom", type=int, default=C.DEM_ZOOM, help="AWS terrain tile zoom (7 ~ 1 km, 8 ~ 0.5 km)")
    ap.add_argument("--no-base", action="store_true", help="skip the land / bathymetry base map")
    ap.add_argument("--no-terrain", action="store_true", help="skip the AWS terrain download: plain IDW surface, no relief")
    ap.add_argument("--keep-sources", action="store_true", help="also keep the whole-image overlay/relief/base files next to the tiles")
    ap.add_argument("--demo", action="store_true",
                    help="SYNTHETIC stations and terrain (layout testing only; page shows a banner)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    boundary = ensure_file(C.BOUNDARY_URL, C.RAW_DIR / C.BOUNDARY_FILE, min_bytes=10_000)
    polygons = load_conus_polygons(boundary)
    grid, dem, dem_label = None, None, ""

    if args.demo:
        from .demo import make_demo_stations, synthetic_dem
        grid, _ = make_grid(polygons, args.res)
        dem, dem_label = synthetic_dem(grid), "SYNTHETIC terrain"
        stations = make_demo_stations(polygons, grid, dem)
        source = {"name": "SYNTHETIC DEMO", "url": "", "file": ""}
    else:
        stations, source, arcgis_error = None, {}, None
        if args.source in ("auto", "arcgis") and not args.stations:
            try:
                from . import stations_arcgis
                stations = stations_arcgis.load(C.RAW_DIR / C.STATIONS_CACHE, args.refresh)
                source = {"name": "NOAA NCEI via NOAA ArcGIS feature layer", "url": C.FEATURE_URL, "file": C.STATIONS_CACHE}
            except (FetchError, ValueError) as exc:
                if args.source == "arcgis":
                    raise
                arcgis_error = exc
                log.warning("feature layer unavailable (%s); falling back to the NCEI spreadsheet", exc)
        if stations is None:
            from .stations import check_is_workbook, load_stations
            try:
                path = args.stations or ensure_file(C.STATIONS_URL, C.RAW_DIR / C.STATIONS_FILE,
                                                    validate=check_is_workbook)
                stations = load_stations(path, args.prob_col)
            except (FetchError, ValueError) as exc:
                if arcgis_error is None:
                    raise
                raise SystemExit(f"No station data could be loaded.\n  NOAA feature layer: {arcgis_error}\n"
                                 f"  NCEI spreadsheet:   {exc}") from exc
            source = {"name": "NOAA NCEI station spreadsheet", "url": C.STATIONS_URL, "file": Path(path).name}

    base_inputs = None
    if not args.no_base and not args.no_terrain:
        try:
            land = load_land_polygons(ensure_file(C.COUNTRIES_URL, C.RAW_DIR / C.COUNTRIES_FILE, min_bytes=10_000))
            lakes = load_lake_polygons(ensure_file(C.LAKES_URL, C.RAW_DIR / C.LAKES_FILE, min_bytes=1_000))
            base_inputs = {"land": land, "lakes": lakes, "zoom": min(C.BASE_DEM_ZOOM, args.dem_zoom)}
            if args.demo:
                from .demo import synthetic_base_dem
                bg = basemap.base_grid()
                base_inputs.update(grid=bg, dem=synthetic_base_dem(bg, land))
        except (FetchError, OSError, ValueError) as exc:
            log.warning("base map skipped (%s); the page falls back to a plain background", exc)

    meta = build(stations, polygons, args.out, res=args.res, dem=dem, grid=grid, dem_label=dem_label,
                 use_terrain=not args.no_terrain, dem_zoom=args.dem_zoom, base_inputs=base_inputs, synthetic=args.demo, source=source,
                 keep_sources=args.keep_sources)
    if not args.no_overlays:
        _vectors(args.out)

    s = meta["stats"]
    log.info("done: %d stations | %s surface | %.1f%% of land >=50%% | %.1f%% <=10%% | -> %s",
             meta["n_stations"], meta["surface"]["method"], 100 * s["share_ge_50"], 100 * s["share_le_10"], args.out)
    if meta["n_stations"] < 500 and not args.demo:
        log.warning("only %d stations - double-check the input", meta["n_stations"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
