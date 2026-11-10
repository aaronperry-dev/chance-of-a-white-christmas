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
from pipeline import basemap as BM
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
    c0 = tuple(int(C.NOAA_COLORS[0][i:i + 2], 16) for i in (1, 3, 5))
    assert tuple(px[0]) == (*c0, 255) and tuple(px[1]) == (255, 255, 255, 128) and px[2, 3] == 0


def test_overlay_is_transparent_at_or_below_the_threshold(tmp_path):
    vals = np.array([[0.0, 0.9, 1.4, 1.6, 2.0, 95.0, np.nan]], np.float32)
    R.render_overlay(G.classify(vals), np.ones((1, 7), np.float32), C.NOAA_COLORS, tmp_path / "o.png",
                     values=vals, transparent_at_or_below=C.TRANSPARENT_AT_OR_BELOW_PCT)
    a = np.asarray(Image.open(tmp_path / "o.png"))[0, :, 3]
    assert C.TRANSPARENT_AT_OR_BELOW_PCT == 1.0
    assert a.tolist() == [0, 0, 0, 255, 255, 255, 0]            # the page reports whole percents: 1 % and below is clear, 2 % and up is drawn


def test_two_darkest_classes_are_lighter_than_noaa_originals():
    lum = lambda h: sum(int(h[i:i + 2], 16) for i in (1, 3, 5)) / 3
    assert lum(C.NOAA_COLORS[0]) > lum("#3b3b3b") and lum(C.NOAA_COLORS[1]) > lum("#304469")

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


# ============================================================ base map (land elevation + bathymetry)
def test_ramp_interpolates_and_clamps():
    stops = [(0, "#000000"), (100, "#ff8000")]
    got = BM.ramp(np.array([-50.0, 0.0, 50.0, 100.0, 9999.0]), stops)
    assert np.abs(got.astype(int) - [[0, 0, 0], [0, 0, 0], [128, 64, 0], [255, 128, 0], [255, 128, 0]]).max() <= 1
    for t in ("light", "dark"):
        for table in (C.LAND_RAMP, C.WATER_RAMP):
            xs = [x for x, _ in table[t]]
            assert xs == sorted(xs) and xs[0] == 0 and all(c.startswith("#") and len(c) == 7 for _, c in table[t])


def test_rasterize_labels_numbers_polygons_and_punches_holes():
    grid = geo.LccGrid(0, 100, 1, 100, 100)
    outer = np.array([[10, 90], [60, 90], [60, 40], [10, 40], [10, 90]], float)
    hole = np.array([[25, 75], [40, 75], [40, 55], [25, 55], [25, 75]], float)
    other = np.array([[70, 30], [90, 30], [90, 10], [70, 10], [70, 30]], float)
    lab = BM.rasterize_labels(grid, [[outer, hole], [other]])
    assert lab[20, 15] == 1 and lab[35, 30] == 0 and lab[80, 80] == 2 and lab[5, 5] == 0


def test_water_depth_ocean_and_lakes():
    dem = np.full((4, 6), 100.0, np.float32)
    dem[0, :2] = [-30.0, -400.0]                    # ocean
    dem[2:, 3:] = [[180.0, 150.0, 120.0], [180.0, 180.0, 60.0]]       # a lake with 180 m surface and bathymetry
    labels = np.zeros((4, 6), np.int32); labels[2:, 3:] = 1
    d = BM.water_depth(dem, np.ones((4, 6), np.float32), labels)
    assert d[0, :2].tolist() == [30.0, 400.0] and d[1, 0] == 0.0
    assert d[2, 3:].tolist() == pytest.approx([0.0, 30.0, 60.0], abs=6.0) or d[3, 5] > d[3, 3]
    assert d[3, 5] > d[2, 5] > 0 and d[2, 3] < d[2, 5]


def test_colorize_uses_land_ramp_on_land_and_water_ramp_for_lakes_and_sea():
    dem = np.array([[500.0, 500.0, -2000.0]], np.float32)
    land_cov = np.array([[1.0, 1.0, 0.0]], np.float32)
    lake = np.array([[False, True, False]])
    depth = np.array([[0.0, 100.0, 2000.0]], np.float32)
    rgb = BM.colorize(dem, land_cov, lake, depth, "light")[0]
    ft = C.M_TO_FT                                                   # inputs are metres, the ramps are in feet
    assert tuple(rgb[0]) == tuple(BM.ramp(np.array([500.0 * ft]), C.LAND_RAMP["light"])[0])
    assert tuple(rgb[1]) == tuple(BM.ramp(np.array([100.0 * ft]), C.WATER_RAMP["light"])[0])      # lake inside land = water
    assert tuple(rgb[2]) == tuple(BM.ramp(np.array([2000.0 * ft]), C.WATER_RAMP["light"])[0])
    half = BM.colorize(dem[:, :1], np.array([[0.5]], np.float32), np.array([[False]]), depth[:, :1], "light")[0, 0]
    assert half.min() > 0 and not (tuple(half) == tuple(rgb[0]))                                # coast pixel is a blend


def test_all_displayed_elevations_are_imperial(tmp_path):
    assert max(x for x, _ in C.LAND_RAMP["light"]) == 14400 and max(x for x, _ in C.WATER_RAMP["light"]) > 20000   # feet, not metres
    st = _stations(20)
    st["elev_m"] = 1000.0
    build(st, box_polys(), tmp_path, res=20_000, use_terrain=False)
    assert json.loads((tmp_path / "stations.json").read_text())[0]["e"] == 3281
    head = (tmp_path / "stations.csv").read_text().splitlines()
    assert "elevation_ft" in head[0] and "elevation_m" not in head[0] and head[1].split(",")[5] == "3281"


def test_meta_has_conus_extent_for_the_opening_view(tmp_path):
    meta = build(_stations(), box_polys(), tmp_path, res=20_000, use_terrain=False)
    x0, y0, x1, y1 = meta["conus_extent"]
    assert x0 < x1 and y0 < y1
    assert all(g0 <= e0 and e1 <= g1 for g0, e0, e1, g1 in
               [(meta["grid"]["x0"], x0, x1, meta["grid"]["x1"]), (meta["grid"]["y0"], y0, y1, meta["grid"]["y1"])])


def test_land_and_water_ramps_do_not_share_hues_at_the_shore():
    for t in ("light", "dark"):
        land0, water0 = BM.ramp(np.array([0.0]), C.LAND_RAMP[t])[0], BM.ramp(np.array([0.0]), C.WATER_RAMP[t])[0]
        assert int(np.abs(land0.astype(int) - water0.astype(int)).sum()) > 40                   # shoreline stays visible


def test_build_with_base_map_writes_images_and_meta(tmp_path):
    grid, projected = make_grid(box_polys(), 20_000)
    bgrid = geo.LccGrid.from_extent(-2_500_000, -2_000_000, 2_500_000, 1_500_000, 50_000)
    bdem = np.where(np.add.outer(np.arange(bgrid.ny), np.arange(bgrid.nx)) % 9 < 5, 300.0, -600.0).astype(np.float32)
    lake = [np.array([[-98.0, 40.0], [-96.0, 40.0], [-96.0, 41.5], [-98.0, 41.5], [-98.0, 40.0]])]
    world = [np.array([[-130.0, 20.0], [-60.0, 20.0], [-60.0, 55.0], [-130.0, 55.0], [-130.0, 20.0]])]
    st = _stations(300)
    st["prob"] = np.clip(16 * (st.lat - 36.0), 0, 100).round()                 # the south of the box is ~0 %
    meta = build(st, box_polys(), tmp_path, grid=grid, dem=_fake_dem(grid), synthetic=True,
                 base_inputs={"land": [world], "lakes": [lake], "dem": bdem, "grid": bgrid})
    b = meta["base"]
    for f in (b["light"], b["dark"], b["relief"]):
        assert Image.open(tmp_path / f).size == (bgrid.nx, bgrid.ny), f
    assert b["grid"]["res"] == 50_000 and set(b["ramps"]) == {"light", "dark"} and meta["surface"]["transparent_at_or_below"] == 1.0
    ov = np.asarray(Image.open(tmp_path / "overlay.png"))
    vals = np.frombuffer(gzip.decompress((tmp_path / "values.bin.gz").read_bytes()), np.uint8).reshape(grid.ny, grid.nx)
    low = (vals <= 1)
    assert low.any() and (ov[..., 3][low] == 0).all()                           # nothing painted where the chance is 1 % or less
    assert (ov[..., 3][(vals > 1) & (vals < 255)] > 0).mean() > 0.95


def test_build_without_base_removes_stale_base_files(tmp_path):
    for f in ("base_light.webp", "base_dark.webp", "relief_ext.webp"):
        (tmp_path / f).write_bytes(b"x")
    meta = build(_stations(), box_polys(), tmp_path, res=20_000, use_terrain=False)
    assert meta["base"] is None and not (tmp_path / "base_light.webp").exists()
