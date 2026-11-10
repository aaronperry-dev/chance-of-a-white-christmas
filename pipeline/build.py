"""Pipeline entry point:  python -m pipeline.build [--demo] [--stations FILE]

  NOAA station table -> LCC (ESRI:102009) grid at ~1 km -> inverse-distance
  surface -> classify with ColorBrewer Blues -> light/dark PNG overlays +
  gz lookup grid + vector context.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .boundary import Polygon, load_conus_polygons
from .fetch import FetchError, ensure_file
from .geo import LccGrid, lonlat_to_xy
from .grid import classify, polygon_extent, project_polygons, rasterize_mask
from .interp import Estimator, interpolate_grid, loo_cv
from .render import area_stats, render_overlay, write_value_grid

log = logging.getLogger("pipeline")


def build(stations: pd.DataFrame, polygons: list[Polygon], out_dir: Path, *,
          res: float = C.GRID_RES_M, synthetic: bool = False, source: dict | None = None) -> dict:
    """Numeric part of the pipeline. Writes all site/data assets except vector context."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = C.CONUS_BBOX
    st = stations[stations.lon.between(west, east) & stations.lat.between(south, north)].reset_index(drop=True)
    if st.empty:
        raise ValueError("no stations inside the CONUS bounding box")
    if "elev_m" not in st:
        st["elev_m"] = np.nan
    log.info("%d stations inside CONUS bbox", len(st))

    proj = project_polygons(polygons)
    xmin, ymin, xmax, ymax = polygon_extent(proj)
    grid = LccGrid.from_extent(xmin - C.GRID_PAD_M, ymin - C.GRID_PAD_M,
                               xmax + C.GRID_PAD_M, ymax + C.GRID_PAD_M, res)
    log.info("LCC grid %d x %d px @ %g m", grid.nx, grid.ny, res)
    coverage = rasterize_mask(grid, proj)

    sx, sy = lonlat_to_xy(st.lon.values, st.lat.values)
    cv = loo_cv(sx, sy, st.prob.values)
    log.info("LOO-CV IDW RMSE %.2f  MAE %.2f  bias %+.2f  (n=%d, percentage points)",
             cv["rmse"], cv["mae"], cv["bias"], cv["n"])

    values = interpolate_grid(Estimator(sx, sy, st.prob.values), grid, coverage > 0)
    cls = classify(values, C.CLASS_EDGES)

    for theme, colors in C.THEME_COLORS.items():
        render_overlay(cls, coverage, colors, out_dir / f"overlay_{theme}.png")
    write_value_grid(values, coverage, out_dir / "values.bin.gz")

    recs = [{"id": r.id, "n": r.name, "s": r.state, "la": round(r.lat, 4), "lo": round(r.lon, 4),
             "e": None if pd.isna(r.elev_m) else round(float(r.elev_m)), "p": int(round(r.prob)),
             "y": None if pd.isna(getattr(r, "years", np.nan)) else int(r.years)}
            for r in st.itertuples()]
    (out_dir / "stations.json").write_text(json.dumps(recs, separators=(",", ":")))
    with open(out_dir / "stations.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "state", "lat", "lon", "elevation_m", "prob_snow_depth_ge_1in_pct", "years"])
        for r in st.itertuples():
            w.writerow([r.id, r.name, r.state, r.lat, r.lon, "" if pd.isna(r.elev_m) else round(float(r.elev_m), 1),
                        r.prob, "" if pd.isna(getattr(r, "years", np.nan)) else int(r.years)])

    labels = C.class_labels()
    meta = {
        "title": "Probability of a White Christmas",
        "definition": "Chance of at least 1 inch of snow on the ground on December 25",
        "period": "1991–2020 U.S. Climate Normals",
        "synthetic": synthetic,
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": source or {},
        "palette": C.PALETTE_NAME,
        "crs": {"name": C.LCC_NAME, "proj4": C.LCC_PROJ, "resolutions": C.CRS_RESOLUTIONS,
                "extent": list(C.NOAA_EXTENT)},
        "grid": grid.as_meta(),
        "overlay": {t: f"overlay_{t}.png" for t in C.THEME_COLORS},
        "values": "values.bin.gz",
        "classes": [{"label": lab, "colors": {t: cols[i] for t, cols in C.THEME_COLORS.items()},
                     "lo": 0 if i == 0 else C.CLASS_EDGES[i - 1],
                     "hi": 100 if i == len(C.CLASS_EDGES) else C.CLASS_EDGES[i]}
                    for i, lab in enumerate(labels)],
        "stats": area_stats(values, cls, coverage, grid, len(labels)),
        "n_stations": int(len(st)),
        "estimation": {"method": "idw", "cv": cv},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _vectors(out: Path) -> None:
    from .webvec import build_vectors
    paths = [ensure_file(u, C.RAW_DIR / f, min_bytes=10_000) for u, f in (
        (C.BOUNDARY_URL, C.BOUNDARY_FILE), (C.COUNTRIES_URL, C.COUNTRIES_FILE), (C.PLACES_URL, C.PLACES_FILE))]
    log.info("vector context: %s", build_vectors(*paths, out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", choices=["auto", "arcgis", "xlsx"], default="auto",
                    help="auto: NOAA feature layer, falling back to the NCEI spreadsheet")
    ap.add_argument("--stations", type=Path, help="local station table (.xlsx/.csv); implies --source xlsx")
    ap.add_argument("--prob-col", help="exact header of the snow-depth probability column (xlsx)")
    ap.add_argument("--refresh", action="store_true", help="re-download the station layer even if cached")
    ap.add_argument("--boundary", type=Path, help="local state-boundary GeoJSON; skips download")
    ap.add_argument("--out", type=Path, default=C.OUT_DIR)
    ap.add_argument("--res", type=float, default=C.GRID_RES_M, help="pixel size in metres (default 1000)")
    ap.add_argument("--demo", action="store_true",
                    help="SYNTHETIC stations (layout testing only; page shows a banner)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    boundary = args.boundary or ensure_file(C.BOUNDARY_URL, C.RAW_DIR / C.BOUNDARY_FILE, min_bytes=10_000)
    polygons = load_conus_polygons(boundary)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        from .demo import make_demo_stations
        stations = make_demo_stations(polygons)
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

    meta = build(stations, polygons, args.out, res=args.res, synthetic=args.demo, source=source)
    _vectors(args.out)

    s = meta["stats"]
    log.info("done: %d stations | %.1f%% of CONUS area >=50%% | %.1f%% <10%% | -> %s",
             meta["n_stations"], 100 * s["share_ge_50"], 100 * s["share_lt_10"], args.out)
    if meta["n_stations"] < 500 and not args.demo:
        log.warning("only %d stations - double-check the input", meta["n_stations"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
