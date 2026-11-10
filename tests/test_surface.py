"""Surface tests: projection grid, land mask, terrain tiles, estimators, rendering, full build.

Everything is SYNTHETIC and offline (terrain tiles are monkeypatched).
"""
import gzip
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pipeline import config as C
from pipeline import dem as D
from pipeline import geo, grid as G, interp as I, render as R
from pipeline.build import build, make_grid

BOX = [np.array([[-103.5, 35.0], [-90.0, 35.0], [-90.0, 45.0], [-103.5, 45.0], [-103.5, 35.0]])]   # one polygon, no holes


def box_polys():
    return [BOX]


# ============================================================ geo / grid
def test_lcc_roundtrip_and_scale_factor():
    x, y = geo.lonlat_to_xy(np.array([-122.33, -80.0]), np.array([47.6, 25.8]))
    lon, lat = geo.xy_to_lonlat(x, y)
    assert lon == pytest.approx([-122.33, -80.0], abs=1e-6) and lat == pytest.approx([47.6, 25.8], abs=1e-6)
    assert geo.scale_factor(20.0) == pytest.approx(1.0, abs=1e-9) and geo.scale_factor(60.0) == pytest.approx(1.0, abs=1e-9)
    assert 0.9 < float(geo.scale_factor(40.0)) < 0.95          # between the standard parallels the map is shrunk


def test_grid_geometry_roundtrip():
    g = geo.LccGrid.from_extent(-2_000_050, -1_000_020, 1_000_010, 1_000_030, 1000)
    assert (g.x0, g.y1, g.nx, g.ny) == (-2_001_000, 1_001_000, 3002, 2002)
    assert g.x1 == g.x0 + g.nx * g.res and g.y0 == g.y1 - g.ny * g.res
    c, r = g.to_pixel(g.x0 + 10.5 * g.res, g.y1 - 20.5 * g.res)
    assert (int(c), int(r)) == (10, 20)
    x, y = g.pixel_xy(20, 10)
    assert (x, y) == (g.x0 + 10.5 * g.res, g.y1 - 20.5 * g.res)
    assert g.x_centers()[0] == g.x0 + 500 and g.y_centers()[0] == g.y1 - 500   # rows run north -> south


def test_classify_uses_inclusive_upper_edges():
    v = np.array([0, 10, 10.4, 25, 25.1, 50, 90, 91, 100, np.nan])
    assert G.classify(v).tolist() == [0, 0, 1, 1, 2, 3, 6, 7, 7, -1]


def test_rasterize_mask_covers_polygon_with_antialiased_edge():
    pr = G.project_polygons(box_polys())
    grid, _ = make_grid(box_polys(), 10_000)
    m = G.rasterize_mask(grid, pr)
    cx, cy = geo.lonlat_to_xy(-96.7, 40.0)
    px, py = grid.to_pixel(cx, cy)
    assert m[int(py), int(px)] == pytest.approx(1.0) and m[0, 0] == 0.0
    assert 0.0 < m[(m > 0) & (m < 1)].mean() < 1.0                    # partial-coverage coast pixels exist
    area = m.sum() * 10_000 ** 2
    assert area == pytest.approx(1.05e6 * 1e6, rel=0.5)               # roughly 13.5 x 10 degrees of land


# ============================================================ terrain tiles
def test_terrarium_encode_decode_roundtrip():
    e = np.array([[-200.0, 0.0, 1.5], [1234.25, 4421.0, 8848.0]], np.float32)
    assert D.decode_terrarium(D.encode_terrarium(e)) == pytest.approx(e, abs=1 / 256 + 1e-6)


def test_tile_coords_known_points():
    x, y = D.tile_coords(0.0, 0.0, 1)
    assert (float(x), float(y)) == (1.0, 1.0)
    x, y = D.tile_coords(-180.0, 85.0511287798, 3)
    assert float(x) == pytest.approx(0.0) and float(y) == pytest.approx(0.0, abs=1e-6)


def test_dem_on_grid_samples_a_known_surface(tmp_path, monkeypatch):
    z = 4

    def fake(zz, x, y, cache, retries=4):                          # elevation = 10 m per pixel of mosaic column
        cols = (x * D.TILE + np.arange(D.TILE)).astype(np.float32)
        return np.tile(cols * 10.0, (D.TILE, 1))

    monkeypatch.setattr(D, "_get_tile", fake)
    grid, _ = make_grid(box_polys(), 20_000)
    dem = D.dem_on_grid(grid, tmp_path, z)
    assert dem.shape == (grid.ny, grid.nx)
    cx, cy = geo.lonlat_to_xy(-96.0, 40.0)
    c, r = grid.to_pixel(cx, cy)
    lon, lat = geo.xy_to_lonlat(cx, cy)
    fx, _ = D.tile_coords(lon, lat, z)
    assert dem[int(r), int(c)] == pytest.approx((float(fx) * D.TILE - 0.5) * 10.0, abs=15.0)


def test_get_tile_uses_cache_and_treats_404_as_no_data(tmp_path, monkeypatch):
    calls = []

    class Resp:
        def __init__(self, code, body=b""): self.status_code, self.content = code, body
        def raise_for_status(self): pass

    png = tmp_path / "enc.png"
    Image.fromarray(D.encode_terrarium(np.full((256, 256), 321.0, np.float32))).save(png)
    body = png.read_bytes()
    monkeypatch.setattr(D.requests, "get", lambda url, **k: (calls.append(url), Resp(200, body))[1])
    t = D._get_tile(5, 7, 11, tmp_path / "c")
    assert t.shape == (256, 256) and t[0, 0] == pytest.approx(321.0, abs=0.01)
    D._get_tile(5, 7, 11, tmp_path / "c")
    assert len(calls) == 1                                            # second read came from the cache
    monkeypatch.setattr(D.requests, "get", lambda url, **k: Resp(404))
    assert not D._get_tile(5, 1, 1, tmp_path / "c").any()


def test_hillshade_flat_is_mid_grey_and_responds_to_aspect():
    flat = D.hillshade(np.full((40, 40), 500.0, np.float32), 1000.0)
    assert (flat == 128).all()
    ramp_east_up = np.tile(np.arange(40, dtype=np.float32) * 40.0, (40, 1))     # rises to the east: faces west
    assert D.hillshade(ramp_east_up, 1000.0, azimuth=270)[20, 20] > 128         # sun in the west lights it
    assert D.hillshade(ramp_east_up, 1000.0, azimuth=90)[20, 20] < 128          # sun in the east shades it
    ramp_north_up = ramp_east_up.T[::-1].copy()                                 # rows run south: top row = highest
    assert D.hillshade(ramp_north_up, 1000.0, azimuth=0)[20, 20] < 128          # faces south, sun in the north


# ============================================================ estimators
def _field(n=800, seed=5):
    """Climate-like truth: colder (higher p) to the north and with elevation, plus a local lump."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 600_000, n), rng.uniform(0, 600_000, n)
    z = 300 + 700 * np.exp(-(((x - 300_000) / 120_000) ** 2)) + rng.normal(0, 30, n)
    return x, y, z, lambda x, y, z: 100 / (1 + np.exp(-(-2.5 + 5e-6 * y + 0.004 * (z - 300))))


def test_idw_and_terrain_reproduce_station_values():
    x, y, z, f = _field()
    p = f(x, y, z)
    for m in ("idw", "terrain"):
        est = I.Estimator(x, y, z, p, m)
        got = est.predict(x[:100] + 1.0, y[:100] + 1.0, z[:100])
        assert np.abs(got - p[:100]).mean() < (3.0 if m == "idw" else 1.0), m


def test_terrain_beats_idw_when_elevation_matters_and_never_goes_out_of_range():
    x, y, z, f = _field()
    p = f(x, y, z)
    cv_idw, cv_ter = I.loo_cv(x, y, z, p, "idw"), I.loo_cv(x, y, z, p, "terrain")
    assert cv_ter["rmse"] < cv_idw["rmse"] and cv_ter["n"] == len(p)
    est = I.Estimator(x, y, z, p, "terrain")
    wild = est.predict(np.array([10.0, 3e5]), np.array([10.0, 3e5]), np.array([-5000.0, 20000.0]))      # absurd elevations
    assert ((wild >= 0) & (wild <= 100)).all()


def test_terrain_slope_is_never_negative():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 1e5, 300), rng.uniform(0, 1e5, 300)
    z = rng.uniform(0, 1500, 300)
    p = np.clip(80 - 0.03 * z + rng.normal(0, 3, 300), 0, 100)          # WRONG-signed truth: snow falls with elevation
    est = I.Estimator(x, y, z, p, "terrain")
    lo = est.predict(np.array([5e4]), np.array([5e4]), np.array([0.0]))
    hi = est.predict(np.array([5e4]), np.array([5e4]), np.array([1500.0]))
    assert hi[0] >= lo[0] - 1e-6                                          # constrained: higher is never less snowy


def test_terrain_needs_elevations():
    with pytest.raises(ValueError):
        I.Estimator([0, 1], [0, 1], [np.nan, 1.0], [10, 20], "terrain")


# ============================================================ rendering
def test_value_grid_roundtrip_and_nodata(tmp_path):
    v = np.array([[0.0, 49.6, np.nan], [100.0, 7.2, 30.0]], np.float32)
    cov = np.array([[1, 1, 1], [1, 0.3, 1]], np.float32)
    R.write_value_grid(v, cov, tmp_path / "v.bin.gz")
    got = np.frombuffer(gzip.decompress((tmp_path / "v.bin.gz").read_bytes()), np.uint8).reshape(2, 3)
    assert got.tolist() == [[0, 50, 255], [100, 255, 30]]                 # NaN and <50 % coverage -> nodata


def test_overlay_uses_class_colours_and_coverage_alpha(tmp_path):
    cls = np.array([[0, 7, -1]], np.int8)
    R.render_overlay(cls, np.array([[1.0, 0.5, 1.0]], np.float32), C.NOAA_COLORS, tmp_path / "o.png")
    px = np.asarray(Image.open(tmp_path / "o.png"))[0]
    assert tuple(px[0]) == (0x3b, 0x3b, 0x3b, 255) and tuple(px[1]) == (255, 255, 255, 128) and px[2, 3] == 0


def test_relief_is_transparent_on_flat_ground_and_visible_on_slopes(tmp_path):
    shade = np.array([[128, 128, 40, 230]], np.uint8)
    R.render_relief(np.repeat(shade, 8, 0), np.ones((8, 4), np.float32), tmp_path / "r.webp")
    a = np.asarray(Image.open(tmp_path / "r.webp").convert("RGBA"))[4]
    assert a[0, 3] <= 6 and a[1, 3] <= 6 and a[2, 3] > 100 and a[3, 3] > 100
    assert a[2, :3].max() < 60 and a[3, :3].min() > 200                   # shadow black, highlight white


def test_area_stats_weights_by_ground_area():
    g = geo.LccGrid(-1_000_000, 1_000_000, 100_000, 20, 20)
    cls = np.zeros((20, 20), np.int8); cls[:, 10:] = 7
    vals = np.where(cls == 0, 5.0, 95.0).astype(np.float32)
    s = R.area_stats(vals, cls, np.ones((20, 20), np.float32), g, 8)
    assert sum(s["share_by_class"]) == pytest.approx(1.0) and s["basis"] == "land area"
    assert s["share_ge_50"] == pytest.approx(s["share_by_class"][7]) and s["share_le_10"] == pytest.approx(s["share_by_class"][0])
    assert 0.3 < s["share_ge_50"] < 0.7


# ============================================================ full build
def _stations(n=150, seed=1):
    rng = np.random.default_rng(seed)
    lon, lat = rng.uniform(-103.0, -90.5, n), rng.uniform(35.5, 44.5, n)
    return pd.DataFrame({"id": [f"S{i}" for i in range(n)], "name": [f"N{i}" for i in range(n)], "state": "XX",
                         "lat": lat, "lon": lon, "elev_m": rng.uniform(100, 1500, n),
                         "prob": np.clip(50 + 3.5 * (lat - 40) + rng.normal(0, 8, n), 0, 100).round(), "years": 30})


def _fake_dem(grid):
    xx, yy = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny))
    return (400 + 300 * np.sin(xx / 7.0) * np.cos(yy / 9.0)).astype(np.float32)


def test_build_writes_consistent_assets(tmp_path):
    grid, _ = make_grid(box_polys(), 15_000)
    meta = build(_stations(), box_polys(), tmp_path, grid=grid, dem=_fake_dem(grid), dem_label="fake", synthetic=True, source={"name": "t"})
    for f in ["stations.json", "stations.csv", "meta.json", "overlay.png", "relief.webp", "values.bin.gz"]:
        assert (tmp_path / f).stat().st_size > 0, f
    assert meta == json.loads((tmp_path / "meta.json").read_text())
    assert meta["crs"]["name"] == "ESRI:102009" and meta["tiles"]["url"] == C.TILES_URL and len(meta["tiles"]["resolutions"]) == 5
    assert len(meta["classes"]) == 8 and meta["classes"][0]["hi"] == 10 and meta["classes"][-1]["hi"] == 100
    g = meta["grid"]
    assert (g["nx"], g["ny"], g["res"]) == (grid.nx, grid.ny, 15_000)
    assert Image.open(tmp_path / "overlay.png").size == (grid.nx, grid.ny) == Image.open(tmp_path / "relief.webp").size
    vals = np.frombuffer(gzip.decompress((tmp_path / "values.bin.gz").read_bytes()), np.uint8)
    assert vals.size == grid.nx * grid.ny and (vals == 255).any() and (vals < 255).any()
    s = meta["stats"]
    assert sum(s["share_by_class"]) == pytest.approx(1.0, abs=1e-9) and meta["station_stats"]["basis"] == "stations"
    assert meta["surface"]["method"] in ("terrain", "idw") and meta["surface"]["cv"]["idw"]["n"] == 150
    # the grid values (what the page looks up) reproduce each station to within a few points
    recs = json.loads((tmp_path / "stations.json").read_text())
    arr = vals.reshape(grid.ny, grid.nx)
    errs = []
    for r in recs:
        x, y = geo.lonlat_to_xy(r["lo"], r["la"]); c, ro = grid.to_pixel(x, y)
        v = arr[int(ro), int(c)]
        if v != 255: errs.append(abs(int(v) - r["p"]))
    assert len(errs) > 100 and np.mean(errs) < 6


def test_build_without_terrain_is_plain_idw_with_no_relief(tmp_path):
    meta = build(_stations(), box_polys(), tmp_path, res=20_000, use_terrain=False)
    assert meta["surface"]["method"] == "idw" and meta["surface"]["relief"] is None and not (tmp_path / "relief.webp").exists()


def test_build_falls_back_to_idw_if_terrain_model_loses(tmp_path, monkeypatch):
    real = I.loo_cv
    monkeypatch.setattr("pipeline.build.loo_cv", lambda *a, **k: {**real(*a, **k), "rmse": 1.0 if a[4] == "idw" else 9.0})
    grid, _ = make_grid(box_polys(), 20_000)
    meta = build(_stations(), box_polys(), tmp_path, grid=grid, dem=_fake_dem(grid))
    assert meta["surface"]["method"] == "idw" and (tmp_path / "relief.webp").exists()       # relief still drawn


def test_build_tolerates_stations_without_elevation(tmp_path):
    meta = build(_stations().drop(columns="elev_m"), box_polys(), tmp_path, res=20_000, use_terrain=False)
    recs = json.loads((tmp_path / "stations.json").read_text())
    assert meta["n_stations"] == 150 and all(r["e"] is None for r in recs)


def test_build_drops_stations_outside_conus(tmp_path):
    st = _stations(12)
    st.loc[0, ["lat", "lon"]] = [61.2, -149.9]          # Anchorage
    assert build(st, box_polys(), tmp_path, res=20_000, use_terrain=False)["n_stations"] == 11
