"""Central configuration: data sources, grid geometry, and the colour ramp.

Everything the front end needs (class edges, colours, labels) is written to
``site/data/meta.json`` from here, so the Python side is the single source of
truth for the legend and the map colours.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "site" / "data"

# --- Sources ---------------------------------------------------------------
# NOAA NCEI: "Downloadable Spreadsheet of Snowfall Probabilities by Weather
# Station from 1991-2020 Climate Normals". This is the station table behind
# the Climate.gov "Historical Probability of a White Christmas" map.
STATIONS_URL = "https://www.ncei.noaa.gov/media/3501"
STATIONS_FILE = "Christmas_day_snow_statistics_1991-2020_updated.xlsx"

# State polygons used only to mask the interpolated surface to land.
BOUNDARY_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_admin_1_states_provinces.geojson"
)
BOUNDARY_FILE = "ne_50m_admin_1_states_provinces.geojson"

NON_CONUS = {
    "Alaska", "Hawaii", "Puerto Rico", "Guam", "American Samoa",
    "United States Virgin Islands", "Northern Mariana Islands",
}

# --- Grid ------------------------------------------------------------------
# (west, south, east, north) in degrees. Slightly generous around CONUS.
CONUS_BBOX = (-125.0, 24.0, -66.5, 49.8)
# Pixel size in degrees of longitude (pixels are square in Web Mercator, which
# is what Leaflet's ImageOverlay expects). 0.04 deg ~ 3.5 km.
GRID_STEP_DEG = 0.04

# Inverse-distance weighting
IDW_K = 16          # nearest stations used per pixel
IDW_POWER = 2.0
IDW_EPS_KM = 5.0    # softening so a pixel on top of a station ~= its value

# --- Colour ramp -----------------------------------------------------------
# ColorBrewer "Blues", 9-class sequential (colorbrewer2.org; Brewer, Harrower
# and The Pennsylvania State University; Apache-2.0). Single hue, colour-blind
# safe, print friendly - a natural fit for snow/ice.
PALETTE_NAME = "ColorBrewer Blues (9-class, sequential)"
BLUES_9 = [
    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
    "#4292c6", "#2171b5", "#08519c", "#08306b",
]
# The map sits on a DARK basemap, so the ramp anchor is flipped: low chance
# recedes into the base (darkest blue) and high chance glows (near-white),
# which also mirrors NOAA's own convention of white = >90 %.
CLASS_COLORS = BLUES_9[::-1]

# Class edges in percent -> len(edges)+1 classes. Must equal len(CLASS_COLORS).
CLASS_EDGES = [10, 20, 30, 40, 50, 60, 70, 80]
assert len(CLASS_EDGES) + 1 == len(CLASS_COLORS)


def class_labels() -> list[str]:
    e = CLASS_EDGES
    labels = [f"<{e[0]}%"]
    labels += [f"{lo}–{hi}%" for lo, hi in zip(e[:-1], e[1:])]
    labels.append(f"≥{e[-1]}%")
    return labels
