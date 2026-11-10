"""Pipeline tests.

Fixtures are SYNTHETIC. The ArcGIS records mirror the live layer's shape (three real
records captured from it), the Excel files only mimic a plausible layout, and the
terrain tiles are generated - nothing here is NOAA/AWS data, and no test touches the network.
"""
import base64
import gzip
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pipeline import config as C
from pipeline import stations_arcgis as SA
from pipeline.build import build
from pipeline.geo import LccGrid, lonlat_to_xy, scale_factor, xy_to_lonlat
from pipeline.grid import classify, project_polygons, rasterize_mask
from pipeline.render import area_stats
from pipeline.stations import StationFormatError, load_stations
from pipeline.interp import Estimator, interpolate_grid, loo_cv
from pipeline.webvec import rdp

# a lon/lat rectangle comfortably inside CONUS
RECT = [[np.array([[-104, 35], [-90, 35], [-90, 45], [-104, 45], [-104, 35]], dtype=float)]]


# ============================================================ config
def test_palette_and_themes():
    assert len(C.BLUES_9) == len(C.CLASS_EDGES) + 1 == len(C.class_labels()) == 9
    assert C.THEME_COLORS["light"][0] == "#f7fbff" and C.THEME_COLORS["light"][-1] == "#08306b"
    assert C.THEME_COLORS["dark"] == C.THEME_COLORS["light"][::-1]          # anchor flipped
    r = C.CRS_RESOLUTIONS
    assert all(r[i] == pytest.approx(2 * r[i + 1]) for i in range(len(r) - 1))


# ============================================================ projection (ESRI:102009)
def test_lcc_matches_noaa_projection_definition():
    assert "+lat_0=40" in C.LCC_PROJ and "+lon_0=-96" in C.LCC_PROJ
    assert "+lat_1=20" in C.LCC_PROJ and "+lat_2=60" in C.LCC_PROJ
    x, y = lonlat_to_xy(-96.0, 40.0)                      # origin of the projection
    assert (float(x), float(y)) == pytest.approx((0, 0), abs=1e-6)
    x, y = lonlat_to_xy(-122.3, 47.6)                     # Seattle (also checked in the browser)
    assert (float(x), float(y)) == pytest.approx((-1841985, 1074564), abs=3)
    lon, lat = xy_to_lonlat(x, y)
    assert (float(lon), float(lat)) == pytest.approx((-122.3, 47.6), abs=1e-6)


def test_scale_factor_is_one_on_standard_parallels():
    assert scale_factor(20.0) == pytest.approx(1.0, abs=1e-9)
    assert scale_factor(60.0) == pytest.approx(1.0, abs=1e-9)
    assert 0.9 < float(scale_factor(40.0)) < 1.0          # between the parallels: map is "too small"
    assert float(scale_factor(30.0)) < 1.0 and float(scale_factor(75.0)) > 1.0


def test_grid_geometry():
    g = LccGrid.from_extent(-2_000_050, -1_000_020, 1_000_010, 1_000_030, 1000)
    assert g.x0 % 1000 == 0 and g.y1 % 1000 == 0
    assert g.x1 >= 1_000_010 and g.y0 <= -1_000_020
    px, py = g.to_pixel(g.x0, g.y1)
    assert (float(px), float(py)) == (0.0, 0.0)
    x, y = g.pixel_xy(0, 0)
    assert (float(x), float(y)) == (g.x0 + 500, g.y1 - 500)
    assert np.all(np.diff(g.y_centers()) < 0) and np.all(np.diff(g.x_centers()) > 0)


def test_mask_coverage_inside_outside_and_antialiased_edge():
    proj = project_polygons(RECT)
    xs = np.concatenate([r[:, 0] for p in proj for r in p]); ys = np.concatenate([r[:, 1] for p in proj for r in p])
    g = LccGrid.from_extent(xs.min() - 50_000, ys.min() - 50_000, xs.max() + 50_000, ys.max() + 50_000, 5000)
    cov = rasterize_mask(g, proj)
    cx, cy = lonlat_to_xy(-97, 40)
    px, py = g.to_pixel(cx, cy)
    assert cov[int(py), int(px)] == pytest.approx(1.0)
    assert cov[0, 0] == 0.0
    assert ((cov > 0) & (cov < 1)).any()


def test_classify_edges_are_lower_inclusive():
    v = np.array([0, 9.99, 10, 19.99, 20, 79.9, 80, 100, np.nan])
    assert classify(v, C.CLASS_EDGES).tolist() == [0, 0, 1, 1, 2, 7, 8, 8, -1]


# ============================================================ estimator
def _world(n=300, seed=2):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 900_000, n), rng.uniform(0, 900_000, n)
    p = np.clip(50 + 40 * np.sin(x / 200_000) * np.cos(y / 250_000) + rng.normal(0, 3, n), 1, 99)
    return x, y, np.round(p)


def test_estimator_constant_field_and_bounds():
    x, y, _ = _world(120)
    out = Estimator(x, y, np.full(len(x), 42.0)).predict(np.array([450_000.0, 100_000]), np.array([450_000.0, 800_000]))
    assert np.allclose(out, 42.0, atol=1e-6)
    p = np.random.default_rng(0).uniform(0, 100, len(x))
    out = Estimator(x, y, p).predict(*np.random.default_rng(1).uniform(0, 9e5, (2, 300)))
    assert out.min() >= 0 and out.max() <= 100                       # weighted mean never leaves the data range


def test_estimator_is_close_to_station_value_at_a_station():
    x, y, p = _world(300)
    pred = Estimator(x, y, p).predict(x, y)
    # a station's own weight dominates (softening distance 5 km vs ~50 km to the next station)
    assert np.median(np.abs(pred - p)) < 1.0


def test_loo_cv_reports_sane_errors():
    x, y, p = _world(400)
    cv = loo_cv(x, y, p)
    assert cv["n"] == 400 and 0 < cv["mae"] <= cv["rmse"] < 25


def test_interpolate_grid_only_fills_requested_pixels():
    x, y, p = _world(200)
    g = LccGrid.from_extent(0, 0, 900_000, 900_000, 30_000)
    where = np.zeros((g.ny, g.nx), bool); where[5:10, 5:10] = True
    out = interpolate_grid(Estimator(x, y, p), g, where)
    assert np.isnan(out[~where]).all() and np.isfinite(out[where]).all()


# ============================================================ NOAA feature layer
REAL_RECORDS = [  # three real records returned by the live layer
    {"FID": 1, "STATION": "USC00144665", "LATITUDE": 38.49, "LONGITUDE": -101.3669, "ELEVATION": 1008.6,
     "NAME": "LEOTI, KS US", "month": 12, "day": 25, "Percent_da": 16.5, "Years": 24},
    {"FID": 2, "STATION": "USC00286055", "LATITUDE": 40.4728, "LONGITUDE": -74.4225, "ELEVATION": 33.8,
     "NAME": "NEW BRUNSWICK 3 SE, NJ US", "month": 12, "day": 25, "Percent_da": 16.5, "Years": 30},
    {"FID": 3, "STATION": "USC00331890", "LATITUDE": 40.2403, "LONGITUDE": -81.8711, "ELEVATION": 231.6,
     "NAME": "COSHOCTON WPC PLT, OH US", "month": 12, "day": 25, "Percent_da": 16.5, "Years": 22},
]


def test_pretty_name():
    assert SA.pretty_name("LEOTI, KS US") == ("Leoti", "KS")
    assert SA.pretty_name("NEW BRUNSWICK 3 SE, NJ US") == ("New Brunswick 3 SE", "NJ")
    assert SA.pretty_name("WINSTON-SALEM REYNOLDS AP, NC US") == ("Winston-Salem Reynolds AP", "NC")
    assert SA.pretty_name("WEIRD NAME") == ("Weird Name", "")


def test_to_frame_parses_real_records_and_cleans():
    junk = [
        {"FID": 4, "STATION": "X1", "LATITUDE": 40, "LONGITUDE": -100, "ELEVATION": -999.9, "NAME": "A, NE US",
         "month": 12, "day": 25, "Percent_da": 50.0, "Years": 30},        # sentinel elevation -> NaN
        {"FID": 5, "STATION": "X2", "LATITUDE": 41, "LONGITUDE": -100, "ELEVATION": 10, "NAME": "B, NE US",
         "month": 12, "day": 25, "Percent_da": None, "Years": 30},        # no probability -> dropped
        {"FID": 6, "STATION": "X3", "LATITUDE": 41, "LONGITUDE": -100, "ELEVATION": 10, "NAME": "C, NE US",
         "month": 11, "day": 25, "Percent_da": 80.0, "Years": 30},        # wrong day -> dropped
    ]
    df = SA.to_frame(REAL_RECORDS + junk)
    assert df.id.tolist() == ["USC00144665", "USC00286055", "USC00331890", "X1"]
    assert df.loc[0, ["name", "state"]].tolist() == ["Leoti", "KS"]
    assert df.elev_m.iloc[0] == pytest.approx(1008.6) and np.isnan(df.elev_m.iloc[3])
    assert df.prob.iloc[0] == 16.5 and df.years.iloc[1] == 30


def test_fetch_attributes_pages_until_exhausted(monkeypatch):
    allrecs = [dict(REAL_RECORDS[0], FID=i, STATION=f"S{i}") for i in range(1, 4504)]
    calls = []

    class Resp:
        def __init__(self, payload): self.p = payload
        def raise_for_status(self): pass
        def json(self): return self.p

    class Sess:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, params=None, timeout=None):
            calls.append(params["resultOffset"])
            o, n = params["resultOffset"], params["resultRecordCount"]
            feats = [{"attributes": a} for a in allrecs[o:o + n]]
            return Resp({"features": feats, "exceededTransferLimit": o + n < len(allrecs)})

    monkeypatch.setattr(SA.requests, "Session", Sess)
    attrs = SA.fetch_attributes(page=2000)
    assert len(attrs) == 4503 and calls[:3] == [0, 2000, 4000]
    assert len({a["STATION"] for a in attrs}) == 4503                  # no dupes across pages


def test_fetch_attributes_surfaces_arcgis_errors(monkeypatch):
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"error": {"code": 400, "message": "bad"}}

    class Sess:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return Resp()

    monkeypatch.setattr(SA.requests, "Session", Sess)
    monkeypatch.setattr(SA.time, "sleep", lambda s: None)
    with pytest.raises(SA.FetchError, match="bad"):
        SA.fetch_attributes(retries=2)


def test_load_uses_cache_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "s.json"
    cache.write_text(json.dumps(REAL_RECORDS))
    monkeypatch.setattr(SA, "fetch_attributes", lambda *a, **k: pytest.fail("should not fetch"))
    assert len(SA.load(cache)) == 3


# ============================================================ xlsx fallback
def write_xlsx(path, columns: dict, title_rows=2):
    df = pd.DataFrame(columns)
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame([["Christmas Day snow statistics (synthetic test fixture)"], ["notes..."]][:title_rows]) \
            .to_excel(xw, header=False, index=False, startrow=0)
        df.to_excel(xw, index=False, startrow=title_rows + 1)


def base_cols(**extra):
    cols = {"Station ID": ["A1", "B2", "C3", "D4"], "Station Name": ["Alpha", "Bravo", "Charlie", "Delta"],
            "State": ["MN", "CO", "TX", "ME"], "Latitude": [45.0, 39.7, 31.0, 46.8],
            "Longitude": [-93.2, -105.0, -97.0, -68.0], "Elevation (ft)": [830, 5280, 500, 600]}
    cols.update(extra)
    return cols


def test_xlsx_picks_depth_ge1_column_and_converts_feet(tmp_path):
    p = tmp_path / "s.xlsx"
    write_xlsx(p, base_cols(**{
        "Probability of Snowfall >= 0.1 inch (%)": [30, 20, 1, 40],
        "Probability of Snow Depth >= 5 inch (%)": [20, 10, 0, 60],
        "Probability of Snow Depth >= 1 inch (%)": [55, 53, 1, 87]}))
    df = load_stations(p)
    assert list(df.prob) == [55, 53, 1, 87]
    assert df.elev_m.iloc[1] == pytest.approx(5280 * 0.3048)


def test_xlsx_fractions_blanks_and_ambiguity(tmp_path):
    p = tmp_path / "a.xlsx"
    write_xlsx(p, base_cols(**{"Prob. of snow depth >= 1 in": [0.55, 0.53, 0.01, 0.87]}))
    assert load_stations(p).prob.round(1).tolist() == [55.0, 53.0, 1.0, 87.0]
    p2 = tmp_path / "b.xlsx"
    write_xlsx(p2, base_cols(**{"Prob. snow depth >= 1 in (%)": [55, None, 1, "N/A"]}))
    assert len(load_stations(p2)) == 2
    p3 = tmp_path / "c.xlsx"
    write_xlsx(p3, base_cols(**{"Prob. snow depth >= 1 in (A)": [55, 53, 1, 87], "Prob. snow depth >= 1 in (B)": [50, 50, 2, 80]}))
    with pytest.raises(StationFormatError, match="--prob-col"):
        load_stations(p3)
    assert load_stations(p3, prob_col="Prob. snow depth >= 1 in (B)").prob.tolist() == [50, 50, 2, 80]
    p4 = tmp_path / "d.xlsx"
    write_xlsx(p4, base_cols(**{"Snowfall prob (%)": [1, 2, 3, 4]}))
    with pytest.raises(StationFormatError, match="Columns found"):
        load_stations(p4)


# ============================================================ stats / vectors
def test_area_stats_uses_true_area_weights():
    g = LccGrid.from_extent(-1_500_000, -1_000_000, 1_500_000, 1_000_000, 10_000)
    cov = np.ones((g.ny, g.nx), np.float32)
    values = np.tile(np.linspace(0, 100, g.nx, dtype=np.float32), (g.ny, 1))
    s = area_stats(values, classify(values, C.CLASS_EDGES), cov, g, 9)
    assert sum(s["share_by_class"]) == pytest.approx(1.0, abs=1e-3)
    assert s["share_ge_50"] == pytest.approx(0.5, abs=0.03)
    assert s["land_area_km2"] > 0


def test_rdp_simplifies_but_keeps_shape():
    t = np.linspace(0, 10, 200)
    line = np.column_stack([t, np.zeros_like(t)])
    assert len(rdp(line, 0.01)) == 2
    zig = np.column_stack([t, np.where(np.arange(200) % 2, 1.0, -1.0)])
    assert len(rdp(zig, 0.01)) == 200


# ============================================================ end to end
def _stations(n=120, seed=1):
    rng = np.random.default_rng(seed)
    lon, lat = rng.uniform(-103.5, -90.5, n), rng.uniform(35.5, 44.5, n)
    return pd.DataFrame({"id": [f"S{i}" for i in range(n)], "name": [f"N{i}" for i in range(n)], "state": "XX",
                         "lat": lat, "lon": lon, "elev_m": rng.uniform(100, 1500, n),
                         "prob": rng.uniform(0, 100, n).round(), "years": 30})


def test_build_writes_consistent_assets(tmp_path):
    meta = build(_stations(), RECT, tmp_path, res=10_000, synthetic=True, source={"name": "t"})
    for f in ["overlay_light.png", "overlay_dark.png", "values.bin.gz", "stations.json", "stations.csv", "meta.json"]:
        assert (tmp_path / f).stat().st_size > 0, f
    assert not list(tmp_path.glob("relief*"))                          # no terrain layer any more
    g = meta["grid"]
    for f in ("overlay_light.png", "overlay_dark.png"):
        assert Image.open(tmp_path / f).size == (g["nx"], g["ny"])
    assert len(gzip.decompress((tmp_path / "values.bin.gz").read_bytes())) == g["nx"] * g["ny"]
    assert meta["estimation"]["method"] == "idw" and meta["estimation"]["cv"]["n"] == 120
    assert "relief" not in meta
    assert meta["crs"]["name"] == "ESRI:102009"
    assert len(meta["classes"]) == 9 and set(meta["classes"][0]["colors"]) == {"light", "dark"}
    assert sum(meta["stats"]["share_by_class"]) == pytest.approx(1.0, abs=1e-3)
    assert len(json.loads((tmp_path / "stations.json").read_text())) == 120


def test_build_tolerates_stations_without_elevation(tmp_path):
    meta = build(_stations().drop(columns="elev_m"), RECT, tmp_path, res=10_000, synthetic=True)
    recs = json.loads((tmp_path / "stations.json").read_text())
    assert meta["n_stations"] == 120 and all(r["e"] is None for r in recs)


def test_build_overlays_are_per_theme_recolours_of_the_same_classes(tmp_path):
    build(_stations(), RECT, tmp_path, res=10_000, synthetic=True)
    light = np.asarray(Image.open(tmp_path / "overlay_light.png"))
    dark = np.asarray(Image.open(tmp_path / "overlay_dark.png"))
    assert (light[..., 3] == dark[..., 3]).all()                       # identical footprint
    lut = {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)): k for k, h in enumerate(C.THEME_COLORS["light"])}
    inv = {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)): k for k, h in enumerate(C.THEME_COLORS["dark"])}
    m = light[..., 3] == 255
    cl = np.array([lut[tuple(p)] for p in light[m][::97, :3]])
    cd = np.array([inv[tuple(p)] for p in dark[m][::97, :3]])
    assert (cl == cd).all()                                            # same class in both themes


# ============================================================ bad spreadsheet download
def test_html_posing_as_xlsx_is_rejected_clearly(tmp_path):
    bad = tmp_path / "x.xlsx"
    bad.write_bytes(b"<!DOCTYPE html><html><body>Not found</body></html>" * 40)
    with pytest.raises(StationFormatError, match="HTML page"):
        load_stations(bad)


def test_ensure_file_never_caches_a_failed_validation_and_replaces_bad_cache(tmp_path, monkeypatch):
    from pipeline import fetch as F
    from pipeline.stations import check_is_workbook

    class Resp:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield self.body

    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    dest = tmp_path / "a.xlsx"
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: Resp(b"<html>" + b"x" * 2000))
    with pytest.raises(F.FetchError):
        F.ensure_file("http://x", dest, validate=check_is_workbook, retries=2)
    assert not dest.exists() and not (tmp_path / "a.xlsx.part").exists()

    dest.write_bytes(b"<html>" + b"x" * 2000)                           # a bad file already in the cache
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: Resp(b"PK\x03\x04" + b"x" * 2000))
    out = F.ensure_file("http://x", dest, validate=check_is_workbook)
    assert out.read_bytes().startswith(b"PK\x03\x04")
