"""Pipeline entry point:  python -m pipeline.build [--demo] [--stations FILE]

  NCEI station table -> parse -> IDW surface on a Web-Mercator grid -> mask to
  CONUS -> classify with the ColorBrewer ramp -> PNG overlay + JSON assets.
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
from .fetch import ensure_file
from .grid import GridSpec, classify, interpolate_idw, rasterize_mask
from .render import area_stats, encode_value_grid, render_overlay
from .stations import load_stations

log = logging.getLogger("pipeline")


def build(stations: pd.DataFrame, polygons: list[Polygon], out_dir: Path, *,
          step: float = C.GRID_STEP_DEG, synthetic: bool = False,
          source_file: str = "") -> dict:
    """Run the numeric part of the pipeline and write all site/data assets."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = C.CONUS_BBOX
    st = stations[stations.lon.between(west, east) & stations.lat.between(south, north)]
    if st.empty:
        raise ValueError("no stations inside the CONUS bounding box")
    log.info("%d stations inside CONUS bbox", len(st))

    spec = GridSpec.from_bbox(C.CONUS_BBOX, step)
    log.info("grid %d x %d px (%.3f deg)", spec.nx, spec.ny, step)
    coverage = rasterize_mask(spec, polygons)
    values = interpolate_idw(spec, st.lon.values, st.lat.values, st.prob.values,
                             coverage > 0, k=C.IDW_K, power=C.IDW_POWER)
    cls = classify(values, C.CLASS_EDGES)

    render_overlay(cls, coverage, C.CLASS_COLORS, out_dir / "overlay.png")
    (out_dir / "grid.json").write_text(
        json.dumps(encode_value_grid(values, coverage), separators=(",", ":")))

    recs = [{"id": r.id, "n": r.name, "s": r.state, "la": round(r.lat, 4), "lo": round(r.lon, 4),
             "e": None if pd.isna(r.elev) else round(float(r.elev)), "p": int(round(r.prob))}
            for r in st.itertuples()]
    (out_dir / "stations.json").write_text(json.dumps(recs, separators=(",", ":")))
    with open(out_dir / "stations.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "state", "lat", "lon", "elevation", "prob_snow_depth_ge_1in_pct"])
        for r in st.itertuples():
            w.writerow([r.id, r.name, r.state, r.lat, r.lon,
                        "" if pd.isna(r.elev) else r.elev, r.prob])

    meta = {
        "title": "Probability of a White Christmas",
        "definition": "Chance of at least 1 inch of snow on the ground on December 25",
        "period": "1991–2020 U.S. Climate Normals",
        "synthetic": synthetic,
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": {"name": "NOAA NCEI", "url": C.STATIONS_URL, "file": source_file},
        "palette": C.PALETTE_NAME,
        "grid": spec.as_meta(),
        "bounds": [[spec.south, spec.west], [spec.north, spec.east]],
        "classes": [{"label": lab, "color": col,
                     "lo": 0 if i == 0 else C.CLASS_EDGES[i - 1],
                     "hi": 100 if i == len(C.CLASS_EDGES) else C.CLASS_EDGES[i]}
                    for i, (lab, col) in enumerate(zip(C.class_labels(), C.CLASS_COLORS))],
        "stats": area_stats(values, cls, coverage, spec, len(C.CLASS_COLORS)),
        "n_stations": int(len(st)),
        "interpolation": f"inverse-distance weighting, k={C.IDW_K}, power={C.IDW_POWER:g}",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stations", type=Path, help="local station table (.xlsx/.csv); skips download")
    ap.add_argument("--prob-col", help="exact header of the snow-depth probability column")
    ap.add_argument("--boundary", type=Path, help="local state-boundary GeoJSON; skips download")
    ap.add_argument("--out", type=Path, default=C.OUT_DIR)
    ap.add_argument("--step", type=float, default=C.GRID_STEP_DEG, help="pixel size in degrees")
    ap.add_argument("--demo", action="store_true",
                    help="use SYNTHETIC stations (layout testing only; page shows a banner)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    boundary = args.boundary or ensure_file(C.BOUNDARY_URL, C.RAW_DIR / C.BOUNDARY_FILE)
    polygons = load_conus_polygons(boundary)

    if args.demo:
        from .demo import make_demo_stations
        stations, source_file = make_demo_stations(polygons), "SYNTHETIC DEMO"
    else:
        path = args.stations or ensure_file(C.STATIONS_URL, C.RAW_DIR / C.STATIONS_FILE)
        stations, source_file = load_stations(path, args.prob_col), Path(path).name

    meta = build(stations, polygons, args.out, step=args.step,
                 synthetic=args.demo, source_file=source_file)
    s = meta["stats"]
    log.info("done: %d stations | %.1f%% of CONUS area >=50%% | %.1f%% <10%% | -> %s",
             meta["n_stations"], 100 * s["share_ge_50"], 100 * s["share_lt_10"], args.out)
    if meta["n_stations"] < 500 and not args.demo:
        log.warning("only %d stations - double-check the input table", meta["n_stations"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
