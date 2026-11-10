"""Pipeline tests. The Excel fixtures here are SYNTHETIC and only mimic a plausible
layout (title rows, long headers, distractor columns); they are not NOAA data.
"""
import base64
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pipeline import config as C
from pipeline.build import build
from pipeline.grid import GridSpec, classify, interpolate_idw, inv_merc_y, merc_y, rasterize_mask
from pipeline.render import area_stats
from pipeline.stations import StationFormatError, load_stations

RECT = [[np.array([[-100, 35], [-90, 35], [-90, 45], [-100, 45], [-100, 35]], dtype=float)]]


def write_xlsx(path, columns: dict, title_rows=2):
    df = pd.DataFrame(columns)
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame([["Christmas Day snow statistics (synthetic test fixture)"], ["notes..."]][:title_rows]) \
            .to_excel(xw, header=False, index=False, startrow=0)
        df.to_excel(xw, index=False, startrow=title_rows + 1)


def base_cols(**extra):
    cols = {
        "Station ID": ["A1", "B2", "C3", "D4"],
        "Station Name": ["Alpha", "Bravo", "Charlie", "Delta"],
        "State": ["MN", "CO", "TX", "ME"],
        "Latitude": [45.0, 39.7, 31.0, 46.8],
        "Longitude": [-93.2, -105.0, -97.0, -68.0],
        "Elevation (ft)": [830, 5280, 500, 600],
    }
    cols.update(extra)
    return cols


# ---------------------------------------------------------------- stations
def test_parser_picks_depth_ge1_column_among_distractors(tmp_path):
    p = tmp_path / "s.xlsx"
    write_xlsx(p, base_cols(**{
        "Probability of Snowfall >= 0.1 inch (%)": [30, 20, 1, 40],
        "Probability of Snow Depth >= 5 inch (%)": [20, 10, 0, 60],
        "Probability of Snow Depth >= 1 inch (%)": [55, 53, 1, 87],
    }))
    df = load_stations(p)
    assert list(df.prob) == [55, 53, 1, 87]
    assert {"id", "name", "state", "lat", "lon", "elev", "prob"} <= set(df.columns)
    assert df.name.tolist()[0] == "Alpha"


def test_parser_converts_fractions_and_percent_strings(tmp_path):
    p = tmp_path / "s.xlsx"
    write_xlsx(p, base_cols(**{"Prob. of snow depth >= 1 in": [0.55, 0.53, 0.01, 0.87]}))
    assert load_stations(p).prob.round(1).tolist() == [55.0, 53.0, 1.0, 87.0]
    p2 = tmp_path / "s2.xlsx"
    write_xlsx(p2, base_cols(**{"Prob. of snow depth >= 1 in": ["55%", "53%", "1%", "87%"]}))
    assert load_stations(p2).prob.tolist() == [55, 53, 1, 87]


def test_parser_drops_rows_without_probability(tmp_path):
    p = tmp_path / "s.xlsx"
    write_xlsx(p, base_cols(**{"Prob. snow depth >= 1 in (%)": [55, None, 1, "N/A"]}))
    assert len(load_stations(p)) == 2


def test_parser_refuses_to_guess_between_equal_columns(tmp_path):
    p = tmp_path / "s.xlsx"
    write_xlsx(p, base_cols(**{
        "Prob. snow depth >= 1 in (A)": [55, 53, 1, 87],
        "Prob. snow depth >= 1 in (B)": [50, 50, 2, 80],
    }))
    with pytest.raises(StationFormatError, match="--prob-col"):
        load_stations(p)
    assert load_stations(p, prob_col="Prob. snow depth >= 1 in (B)").prob.tolist() == [50, 50, 2, 80]


def test_parser_errors_clearly_when_no_depth_column(tmp_path):
    p = tmp_path / "s.xlsx"
    write_xlsx(p, base_cols(**{"Snowfall prob (%)": [1, 2, 3, 4]}))
    with pytest.raises(StationFormatError, match="Columns found"):
        load_stations(p)


# ---------------------------------------------------------------- grid
def test_grid_is_mercator_uniform_with_square_pixels():
    spec = GridSpec.from_bbox(C.CONUS_BBOX, 0.1)
    x, y = spec.to_pixel(spec.west, spec.north)
    assert (float(x), float(y)) == pytest.approx((0, 0), abs=1e-9)
    x, y = spec.to_pixel(spec.east, spec.south)
    assert (float(x), float(y)) == pytest.approx((spec.nx, spec.ny), abs=1e-6)
    lat = spec.lat_centers()
    assert np.all(np.diff(lat) < 0)
    # rows are uniform in Mercator y, not in latitude
    assert np.allclose(np.diff(merc_y(lat)), -spec.dy)
    assert inv_merc_y(merc_y(40.0)) == pytest.approx(40.0)


def test_mask_coverage_inside_outside_and_antialiased_edge():
    spec = GridSpec.from_bbox((-105, 30, -85, 50), 0.1)
    cov = rasterize_mask(spec, RECT)
    x, y = spec.to_pixel(-95, 40)
    assert cov[int(y), int(x)] == pytest.approx(1.0)
    x, y = spec.to_pixel(-102.5, 40)  # outside the rectangle
    assert cov[int(y), int(x)] == 0.0
    assert ((cov > 0) & (cov < 1)).any()


def test_idw_constant_exact_and_bounded():
    spec = GridSpec.from_bbox((-105, 30, -85, 50), 0.2)
    where = np.ones((spec.ny, spec.nx), bool)
    lon = np.array([-100.0, -95, -90, -97, -92])
    lat = np.array([34.0, 40, 44, 46, 36])
    const = interpolate_idw(spec, lon, lat, np.full(5, 42.0), where)
    assert np.allclose(const, 42.0, atol=1e-4)

    vals = np.array([0.0, 100, 50, 20, 80])
    out = interpolate_idw(spec, lon, lat, vals, where)
    assert out.min() >= 0 and out.max() <= 100          # no overshoot
    x, y = spec.to_pixel(-95, 40)
    assert out[int(y), int(x)] == pytest.approx(100, abs=1.0)  # ~ station value on top of it


def test_classify_edges_are_lower_inclusive():
    v = np.array([0, 9.99, 10, 19.99, 20, 79.9, 80, 100, np.nan])
    assert classify(v, C.CLASS_EDGES).tolist() == [0, 0, 1, 1, 2, 7, 8, 8, -1]


def test_palette_matches_classes():
    assert len(C.CLASS_COLORS) == len(C.CLASS_EDGES) + 1 == len(C.class_labels())
    assert C.CLASS_COLORS[0] == "#08306b" and C.CLASS_COLORS[-1] == "#f7fbff"  # Blues-9 flipped


def test_area_stats_shares_sum_to_one():
    spec = GridSpec.from_bbox((-105, 30, -85, 50), 0.2)
    cov = np.ones((spec.ny, spec.nx), np.float32)
    values = np.tile(np.linspace(0, 100, spec.nx, dtype=np.float32), (spec.ny, 1))
    s = area_stats(values, classify(values, C.CLASS_EDGES), cov, spec, len(C.CLASS_COLORS))
    assert sum(s["share_by_class"]) == pytest.approx(1.0, abs=1e-3)
    assert s["share_ge_50"] == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------- end to end
def test_build_writes_consistent_assets(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    stations = pd.DataFrame({
        "id": [f"S{i}" for i in range(n)], "name": [f"N{i}" for i in range(n)], "state": "XX",
        "lat": rng.uniform(35.5, 44.5, n), "lon": rng.uniform(-99.5, -90.5, n),
        "elev": np.nan, "prob": rng.uniform(0, 100, n).round(),
    })
    meta = build(stations, RECT, tmp_path, step=0.1, synthetic=True)

    for f in ["overlay.png", "grid.json", "stations.json", "stations.csv", "meta.json"]:
        assert (tmp_path / f).stat().st_size > 0
    g = meta["grid"]
    assert Image.open(tmp_path / "overlay.png").size == (g["nx"], g["ny"])
    grid = json.loads((tmp_path / "grid.json").read_text())
    assert len(base64.b64decode(grid["b64"])) == g["nx"] * g["ny"]
    assert len(meta["classes"]) == len(C.CLASS_COLORS)
    assert sum(meta["stats"]["share_by_class"]) == pytest.approx(1.0, abs=1e-3)
    assert meta["synthetic"] is True and meta["n_stations"] == n
    assert len(json.loads((tmp_path / "stations.json").read_text())) == n
