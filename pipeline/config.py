"""Central configuration: data sources, NOAA's tile service, projection and legend.

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

# --- Legend (NOAA's 8 classes) -------------------------------------------------
# Labels and breaks follow NOAA's maps: 0-10, 11-25, 26-40, 41-50, 51-60, 61-75, 76-90, 91-100 %.
# Swatch colours were sampled from NOAA's interactive-map legend and checked against the colours
# of the rendered tiles (a screenshot of the live page). The hillshade darkens/lightens tiles a bit,
# so tile pixels are close to, not identical to, these values. If they ever differ, edit this list.
CLASS_UPPER_EDGES = [10, 25, 40, 50, 60, 75, 90]          # a class includes its upper edge
NOAA_COLORS = ["#3b3b3b", "#304469", "#1e50a9", "#1d71d4", "#4799e5", "#80b7ed", "#d0e5f8", "#ffffff"]
assert len(CLASS_UPPER_EDGES) + 1 == len(NOAA_COLORS)


def class_labels() -> list[str]:
    e = CLASS_UPPER_EDGES
    lo = [0] + [x + 1 for x in e]
    hi = e + [100]
    return [f"{a}\u2013{b}%" for a, b in zip(lo, hi)]


def class_index(p: float) -> int:
    """0..7. 10 -> class 0, 10.5 -> class 1, 25 -> class 1, 91 -> class 7."""
    return sum(p > e for e in CLASS_UPPER_EDGES)
