"""Pipeline entry point:  python -m pipeline.build [--demo] [--stations FILE]

  NOAA station table  ->  site/data/{stations.json, stations.csv, meta.json}
  Natural Earth       ->  site/data/{states.geojson, context.geojson, places.json}  (optional overlays)

The map itself is NOAA's own cached tile service (see config.TILES_URL); this
pipeline supplies the station values behind click look-ups, the legend, headline
figures and the optional state-line / label overlays.
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
from .fetch import FetchError, ensure_file

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


def build(stations: pd.DataFrame, out_dir: Path, *, synthetic: bool = False, source: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = C.CONUS_BBOX
    st = stations[stations.lon.between(west, east) & stations.lat.between(south, north)].reset_index(drop=True)
    if st.empty:
        raise ValueError("no stations inside the CONUS bounding box")
    if "elev_m" not in st:
        st["elev_m"] = np.nan
    log.info("%d stations inside CONUS bbox", len(st))

    recs = [{"id": r.id, "n": r.name, "s": r.state, "la": round(r.lat, 4), "lo": round(r.lon, 4),
             "e": None if pd.isna(r.elev_m) else round(float(r.elev_m)), "p": int(round(r.prob)),
             "y": None if pd.isna(getattr(r, "years", np.nan)) else int(r.years)}
            for r in st.itertuples()]
    (out_dir / "stations.json").write_text(json.dumps(recs, separators=(",", ":")))
    with open(out_dir / "stations.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "state", "lat", "lon", "elevation_m", "prob_snow_depth_ge_1in_pct", "years"])
        for r in st.itertuples():
            w.writerow([r.id, r.name, r.state, r.lat, r.lon,
                        "" if pd.isna(r.elev_m) else round(float(r.elev_m), 1),
                        r.prob, "" if pd.isna(getattr(r, "years", np.nan)) else int(r.years)])

    labels = C.class_labels()
    meta = {
        "title": "Probability of a White Christmas",
        "definition": "Chance of at least 1 inch of snow on the ground on December 25",
        "period": "1991–2020 U.S. Climate Normals",
        "synthetic": synthetic,
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": source or {},
        "crs": {"name": C.LCC_NAME, "proj4": C.LCC_PROJ, "extent": list(C.NOAA_EXTENT)},
        "tiles": {
            "url": C.TILES_URL, "service": C.TILES_SERVICE_PAGE, "size": C.TILE_SIZE,
            "origin": list(C.TILE_ORIGIN), "resolutions": C.TILE_RESOLUTIONS,
            "full_extent": list(C.TILE_FULL_EXTENT), "attribution": C.TILE_ATTRIBUTION,
        },
        "classes": [{"label": lab, "color": C.NOAA_COLORS[i],
                     "lo": 0 if i == 0 else C.CLASS_UPPER_EDGES[i - 1],
                     "hi": 100 if i == len(C.CLASS_UPPER_EDGES) else C.CLASS_UPPER_EDGES[i]}
                    for i, lab in enumerate(labels)],
        "stats": station_stats(np.round(st.prob.values.astype(float))),   # same rounding as stations.json
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
    ap.add_argument("--demo", action="store_true",
                    help="SYNTHETIC stations (layout testing only; page shows a banner)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        from .demo import make_demo_stations
        stations = make_demo_stations()
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

    meta = build(stations, args.out, synthetic=args.demo, source=source)
    if not args.no_overlays:
        _vectors(args.out)

    s = meta["stats"]
    log.info("done: %d stations | %.1f%% of stations >=50%% | %.1f%% <=10%% | -> %s",
             meta["n_stations"], 100 * s["share_ge_50"], 100 * s["share_le_10"], args.out)
    if meta["n_stations"] < 500 and not args.demo:
        log.warning("only %d stations - double-check the input", meta["n_stations"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
