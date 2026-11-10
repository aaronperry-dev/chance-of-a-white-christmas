"""Central configuration: data sources, projection, grid, terrain, estimation and legend.

Everything the front end needs (tile grid, projection, legend classes and colours)
is written to ``site/data/meta.json`` from here, so this file is the single
source of truth for the map and the legend.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "site" / "data"

# --- Stations ----------------------------------------------------------------
# NOAA's public hosted feature layer behind the Climate.gov / NOAA "White
# Christmas" web app. One point per station with Percent_da (probability of
# >= 1 in snow depth on Dec 25, 1991-2020 normals), ELEVATION (m) and Years.
FEATURE_URL = (
    "https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/"
    "WhiteChristmas_cover1inch_Dec25_2020normals_CONUS/FeatureServer/0"
)
STATIONS_CACHE = "stations_arcgis.json"

# Alternative / fallback: NCEI's downloadable spreadsheet of the same table.
STATIONS_URL = "https://www.ncei.noaa.gov/media/3501"
STATIONS_FILE = "Christmas_day_snow_statistics_1991-2020_updated.xlsx"

# --- Terrain -----------------------------------------------------------------
# AWS Open Data "Terrain Tiles" (Terrarium PNG encoding, Web Mercator, no key).
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
DEM_ZOOM = 7  # ~0.8-1.1 km/px over CONUS; 8 is ~2x finer (4x the download)
TERRAIN_CACHE = "terrarium"

# Hillshade (flat terrain renders mid-grey 128 so it can be blended over colour).
HILLSHADE_AZIMUTH = 315.0
HILLSHADE_ALTITUDE = 45.0
HILLSHADE_Z = 4.0     # vertical exaggeration
HILLSHADE_GAIN = 1.0

# --- Boundaries / context (Natural Earth, via GitHub) ------------------------
_NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
BOUNDARY_URL = _NE + "ne_50m_admin_1_states_provinces.geojson"
BOUNDARY_FILE = "ne_50m_admin_1_states_provinces.geojson"
COUNTRIES_URL = _NE + "ne_50m_admin_0_countries.geojson"
COUNTRIES_FILE = "ne_50m_admin_0_countries.geojson"
PLACES_URL = _NE + "ne_50m_populated_places_simple.geojson"
PLACES_FILE = "ne_50m_populated_places_simple.geojson"

NON_CONUS = {
    "Alaska", "Hawaii", "Puerto Rico", "Guam", "American Samoa",
    "United States Virgin Islands", "Northern Mariana Islands",
}
# Station / polygon sanity box (west, south, east, north), degrees.
CONUS_BBOX = (-125.0, 24.0, -66.5, 49.8)

# --- Projection: the same one NOAA's web app and tile service use -----------
# ESRI:102009 "North America Lambert Conformal Conic" (WKID 102009).
LCC_NAME = "ESRI:102009"
LCC_PROJ = ("+proj=lcc +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 "
            "+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")
# NOAA app's constraint extent in LCC metres (xmin, ymin, xmax, ymax). Used to
# frame the view and limit panning.
NOAA_EXTENT = (-3100465.28, -1958797.37, 2987609.40, 1618376.45)

# --- NOAA's cached map service (the map you see is NOAA's own rendering) ------
# ArcGIS "Christmas_Snow_2020normals_Lambert_update" MapServer (TilesOnly, 256 px PNG).
# Values below are copied from the service's own metadata page (tileInfo / fullExtent).
TILES_URL = ("https://tiles.arcgis.com/tiles/C8EMgrsFcRFL6LrL/arcgis/rest/services/"
             "Christmas_Snow_2020normals_Lambert_update/MapServer/tile/{z}/{y}/{x}")
TILES_SERVICE_PAGE = ("https://tiles.arcgis.com/tiles/C8EMgrsFcRFL6LrL/arcgis/rest/services/"
                      "Christmas_Snow_2020normals_Lambert_update/MapServer")
TILE_SIZE = 256
TILE_ORIGIN = (-3.873557170374695e7, 4.6658285798503935e7)   # top-left of the tile grid (LCC m)
TILE_RESOLUTIONS = [6879.180425027517, 3439.5902125137586, 1984.3789687579376,
                    992.1894843789688, 496.0947421894844]    # metres/pixel for LOD 0..4
TILE_FULL_EXTENT = (-8918789.413932372, -6520590.090497846, 9364286.214718886, 4874706.13759461)
TILE_ATTRIBUTION = ("NOAA, Oceanic and Atmospheric Research (OAR), Climate Program Office (CPO), "
                    "Climate.gov; NOAA National Centers for Environmental Information (NCEI)")

# --- Grid --------------------------------------------------------------------
GRID_RES_M = 1000    # pixel size in metres (NOAA's finest service tiles are ~500 m/px)
GRID_PAD_M = 20_000  # margin around CONUS

# Cells with this chance or less are left transparent so the terrain / bathymetry map shows through
# (the map shows every chance greater than 1 %).
TRANSPARENT_AT_OR_BELOW_PCT = 1.0
M_TO_FT = 3.280839895

# --- Base map: land elevation + bathymetry colour ramps -----------------------
# A coarse raster over NOAA's whole view extent (US, Canada, Mexico, oceans, lakes) drawn under the
# probability surface; it is what shows through where the chance is TRANSPARENT_AT_OR_BELOW_PCT or less.
BASE_RES_M = 2000
BASE_PAD_M = 400_000
BASE_DEM_ZOOM = 6          # ~2 km/px at these latitudes; the 1 km DEM is only needed for the surface
LAKES_URL = _NE + "ne_50m_lakes.geojson"
LAKES_FILE = "ne_50m_lakes.geojson"
# (elevation ft, colour) for land; (depth ft, colour) for ocean and lakes. Light and dark themes.
LAND_RAMP = {
    "light": [(0, "#a8c98f"), (500, "#c3dba0"), (1300, "#e0e6a9"), (2600, "#ecdfa2"), (4300, "#e2c690"),
              (6200, "#cfa77a"), (8500, "#b58863"), (11200, "#9a7160"), (14400, "#d8cfca")],
    "dark": [(0, "#33422d"), (500, "#3d4d33"), (1300, "#4b5338"), (2600, "#575238"), (4300, "#5a4d38"),
             (6200, "#5c4b3c"), (8500, "#5a4a40"), (11200, "#574a47"), (14400, "#6a6260")],
}
WATER_RAMP = {
    "light": [(0, "#d7e9f1"), (300, "#c3dcea"), (1600, "#a9c9de"), (4900, "#8bb0cb"), (9800, "#6f94b4"),
              (16400, "#58799b"), (26200, "#46637f")],
    "dark": [(0, "#26414f"), (300, "#223b49"), (1600, "#1d3340"), (4900, "#182b37"), (9800, "#142330"),
             (16400, "#101d28"), (26200, "#0c1620")],
}

# --- Estimation --------------------------------------------------------------
# Terrain-aware: local regression of logit(probability) on elevation (tricube
# weights over the K nearest stations) + a smooth correction from the regression residuals.
TERRAIN_K = 24
RESID_POWER = 2.0
# Residual correction: Gaussian-weighted mean of the neighbouring residuals with this length scale (m).
# A 1/distance kernel (RESID_SMOOTH_M = 0, the old behaviour) reproduces stations almost exactly but draws
# small bull's-eye patches around any unusual station; 10 km removes about half of the small closed
# patches in flat terrain in tests and still reproduces stations to ~2 percentage points. The terrain
# trend is not touched. RESID_SHRINK fades the correction to zero far from any station.
RESID_SMOOTH_M = 10_000.0
RESID_SHRINK = 0.05
RESID_EPS_M = 1000.0   # only used when RESID_SMOOTH_M = 0 (IDW correction): station values reproduced to <0.5 points
RIDGE_M = 100.0        # shrinks the elevation slope toward 0 where stations barely differ in height
SLOPE_MAX = 0.01       # max logit units per metre of elevation
EXTRAPOLATE_MARGIN_M = 300.0
LOGIT_EPS = 0.005
# Plain IDW baseline (used when no DEM is available, and as the CV benchmark).
IDW_K = 16
IDW_POWER = 2.0
IDW_EPS_M = 5000.0

# --- Legend (NOAA's 8 classes) -------------------------------------------------
# Labels and breaks follow NOAA's maps: 0-10, 11-25, 26-40, 41-50, 51-60, 61-75, 76-90, 91-100 %.
# Swatch colours were sampled from NOAA's interactive-map legend and checked against the colours
# of the rendered tiles (a screenshot of the live page). The hillshade darkens/lightens tiles a bit,
# so tile pixels are close to, not identical to, these values. If they ever differ, edit this list.
CLASS_UPPER_EDGES = [10, 25, 40, 50, 60, 75, 90]          # a class includes its upper edge
# The two darkest classes are lightened from NOAA's #3b3b3b / #304469 (they read as near-black on screen).
NOAA_COLORS = ["#6b6b6b", "#3d5a8c", "#1e50a9", "#1d71d4", "#4799e5", "#80b7ed", "#d0e5f8", "#ffffff"]
assert len(CLASS_UPPER_EDGES) + 1 == len(NOAA_COLORS)


def class_labels() -> list[str]:
    e = CLASS_UPPER_EDGES
    lo = [0] + [x + 1 for x in e]
    hi = e + [100]
    return [f"{a}\u2013{b}%" for a, b in zip(lo, hi)]


def class_index(p: float) -> int:
    """0..7. 10 -> class 0, 10.5 -> class 1, 25 -> class 1, 91 -> class 7."""
    return sum(p > e for e in CLASS_UPPER_EDGES)
