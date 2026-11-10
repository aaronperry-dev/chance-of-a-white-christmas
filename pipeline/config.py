"""Central configuration: data sources, projection, grid, estimation and colour ramps.

Everything the front end needs (projection, class edges, colours, labels) is
written to ``site/data/meta.json`` from here, so this file is the single source
of truth for the map and the legend.
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

# --- Projection: the same one NOAA's web app uses -----------------------------
# ESRI:102009 "North America Lambert Conformal Conic" (WKID 102009).
LCC_NAME = "ESRI:102009"
LCC_PROJ = ("+proj=lcc +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 "
            "+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")
LCC_STD_PARALLELS = (20.0, 60.0)
# NOAA app's constraint extent in LCC metres (xmin, ymin, xmax, ymax). Used to
# frame and limit panning so the view matches theirs.
NOAA_EXTENT = (-3100465.28, -1958797.37, 2987609.40, 1618376.45)
# Zoom ladder for the web map: metres per screen pixel at zoom 0..N.
CRS_RESOLUTIONS = [16384 / 2 ** i for i in range(9)]  # 16384 ... 64

# --- Grid --------------------------------------------------------------------
GRID_RES_M = 1000    # pixel size in metres (NOAA's finest service tiles are ~500 m/px)
GRID_PAD_M = 20_000  # margin around CONUS

# --- Estimation --------------------------------------------------------------
# Plain inverse-distance weighting over the K nearest stations.
IDW_K = 16
IDW_POWER = 2.0
IDW_EPS_M = 5000.0

# --- Colour ramp -------------------------------------------------------------
# ColorBrewer "Blues", 9-class sequential (colorbrewer2.org; Brewer, Harrower
# and The Pennsylvania State University; Apache-2.0).
PALETTE_NAME = "ColorBrewer Blues (9-class, sequential)"
BLUES_9 = [
    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
    "#4292c6", "#2171b5", "#08519c", "#08306b",
]
# Colours are listed LOW -> HIGH probability (class 0 = lowest chance).
#   light theme: textbook direction - low chance = palest, high chance = darkest.
#   dark theme : anchor flipped so high chance glows near-white on a dark map
#                (also NOAA's own white = >90 % convention) and low chance
#                recedes into the dark base.
THEME_COLORS = {"light": list(BLUES_9), "dark": BLUES_9[::-1]}

CLASS_EDGES = [10, 20, 30, 40, 50, 60, 70, 80]  # 9 classes, lower edge inclusive
assert len(CLASS_EDGES) + 1 == len(BLUES_9)


def class_labels() -> list[str]:
    e = CLASS_EDGES
    labels = [f"<{e[0]}%"]
    labels += [f"{lo}–{hi}%" for lo, hi in zip(e[:-1], e[1:])]
    labels.append(f"≥{e[-1]}%")
    return labels
