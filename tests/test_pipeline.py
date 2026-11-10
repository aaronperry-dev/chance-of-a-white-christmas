"""Pipeline tests.

Fixtures are SYNTHETIC. The ArcGIS records mirror the live layer's shape (three real
records captured from it) and the Excel files only mimic a plausible layout. Nothing here is
NOAA data and no test touches the network.
"""
import json

import numpy as np
import pandas as pd
import pytest

from pipeline import config as C
from pipeline import stations_arcgis as SA
from pipeline.build import build, station_stats
from pipeline.stations import StationFormatError, load_stations
from pipeline.webvec import rdp


# ============================================================ legend
def test_legend_matches_noaa_classes():
    assert C.class_labels() == ["0\u201310%", "11\u201325%", "26\u201340%", "41\u201350%",
                                "51\u201360%", "61\u201375%", "76\u201390%", "91\u2013100%"]
    assert len(C.NOAA_COLORS) == 8 and all(c.startswith("#") and len(c) == 7 for c in C.NOAA_COLORS)


def test_class_index_upper_edge_is_inclusive():
    cases = {0: 0, 10: 0, 10.5: 1, 11: 1, 25: 1, 25.1: 2, 40: 2, 50: 3, 60: 4, 75: 5, 90: 6, 91: 7, 100: 7}
    assert {p: C.class_index(p) for p in cases} == cases


def test_station_stats_shares():
    s = station_stats(np.array([0, 10, 11, 50, 50, 95, 100, 20.0]))
    assert s["basis"] == "stations" and s["n"] == 8
    assert sum(s["share_by_class"]) == pytest.approx(1.0)
    assert s["share_ge_50"] == pytest.approx(4 / 8) and s["share_le_10"] == pytest.approx(2 / 8)


# ============================================================ NOAA tile grid / projection
def test_tile_grid_matches_service_metadata():
    # ArcGIS: resolution = scale * 0.0254 / 96 (96 dpi) to within ~2 ppm (it uses a slightly different inch);
    # the scales are the service's published LOD scales
    scales = [2.6e7, 1.3e7, 7.5e6, 3.75e6, 1.875e6]
    assert C.TILE_RESOLUTIONS == pytest.approx([s * 0.0254 / 96 for s in scales], rel=1e-5)
    assert C.TILE_SIZE == 256 and len(C.TILE_RESOLUTIONS) == 5
    # the app's view extent sits inside what the service actually caches
    x0, y0, x1, y1 = C.NOAA_EXTENT
    fx0, fy0, fx1, fy1 = C.TILE_FULL_EXTENT
    assert fx0 <= x0 and fy0 <= y0 and x1 <= fx1 and y1 <= fy1
    # and every cached tile row/col is positive in the service's grid (origin is top-left)
    span = C.TILE_RESOLUTIONS[0] * C.TILE_SIZE
    assert (fx0 - C.TILE_ORIGIN[0]) / span > 0 and (C.TILE_ORIGIN[1] - fy1) / span > 0
    assert "{z}" in C.TILES_URL and "{y}" in C.TILES_URL and "{x}" in C.TILES_URL


def test_lcc_matches_noaa_projection_definition():
    from pyproj import Transformer
    assert "+lat_0=40" in C.LCC_PROJ and "+lon_0=-96" in C.LCC_PROJ
    assert "+lat_1=20" in C.LCC_PROJ and "+lat_2=60" in C.LCC_PROJ
    t = Transformer.from_crs("EPSG:4326", C.LCC_PROJ, always_xy=True)
    assert t.transform(-96.0, 40.0) == pytest.approx((0, 0), abs=1e-6)
    assert t.transform(-122.3, 47.6) == pytest.approx((-1841985, 1074564), abs=3)   # Seattle (also checked in the browser)


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
    meta = build(_stations(), tmp_path, synthetic=True, source={"name": "t"})
    for f in ["stations.json", "stations.csv", "meta.json"]:
        assert (tmp_path / f).stat().st_size > 0, f
    assert not list(tmp_path.glob("*.png")) and not list(tmp_path.glob("*.bin.gz"))   # NOAA's tiles are the map
    assert meta["crs"]["name"] == "ESRI:102009" and meta["tiles"]["url"] == C.TILES_URL
    assert meta["tiles"]["origin"] == list(C.TILE_ORIGIN) and len(meta["tiles"]["resolutions"]) == 5
    assert len(meta["classes"]) == 8 and meta["classes"][0]["hi"] == 10 and meta["classes"][-1]["hi"] == 100
    assert sum(meta["stats"]["share_by_class"]) == pytest.approx(1.0, abs=1e-9)
    recs = json.loads((tmp_path / "stations.json").read_text())
    assert len(recs) == 120 and meta == json.loads((tmp_path / "meta.json").read_text())


def test_build_tolerates_stations_without_elevation(tmp_path):
    meta = build(_stations().drop(columns="elev_m"), tmp_path, synthetic=True)
    recs = json.loads((tmp_path / "stations.json").read_text())
    assert meta["n_stations"] == 120 and all(r["e"] is None for r in recs)


def test_build_drops_stations_outside_conus(tmp_path):
    st = _stations(10)
    st.loc[0, ["lat", "lon"]] = [61.2, -149.9]          # Anchorage
    assert build(st, tmp_path)["n_stations"] == 9


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
