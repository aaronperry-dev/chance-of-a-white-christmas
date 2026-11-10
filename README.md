# Probability of a White Christmas — CONUS dashboard

A static dashboard for GitHub Pages: the historical probability of **at least 1 inch of snow on the ground on December 25** (1991–2020 U.S. Climate Normals, NOAA NCEI) over the contiguous U.S., drawn in NOAA's map style.

**Why the surface is generated here.** NOAA publishes the station values and a set of cached map tiles, but no raw grid, and those tiles have the station dots, city squares and labels baked into the images (they cannot be hidden). So the pipeline re-creates the surface from the same station values, on a clean 1 km grid in NOAA's projection and classes, and shapes it with terrain.

* **Surface:** a 1 km grid in **ESRI:102009** (North America Lambert Conformal Conic), the projection of NOAA's web app and tile service, drawn by Leaflet in NOAA's tile-grid zoom levels. Estimator: a *geographically weighted quasi-binomial regression* of the station frequency on elevation (the K nearest stations, tricube weights; slope constrained to "higher = snowier" and shrunk where the stations barely differ in height). It is solved on an 8 km node lattice, the fitted level and slope are bicubic-upsampled, and the elevation term is applied from the 1 km terrain grid, so the field is smooth regionally and follows the terrain locally. K (12-48) is picked by leave-one-out error at the stations. It **smooths** the stations instead of honouring them: each station is a 30-year frequency (steps of about 3 points), so at the low end one 0 % or 7 % reading is mostly sampling noise, and pulling the surface toward it (inverse-distance or Gaussian residual kernels, tried earlier) draws a round patch around it, which is what appeared as small circles near the 1 % edge. Method background: the same idea as PRISM (weighted climate-elevation regression) and ANUSPLIN (smoothing spline with an elevation covariate), expressed as a local-likelihood fit for a probability. Elevation comes from AWS Open Data **Terrain Tiles** (Terrarium PNG, zoom 7, ≈1 km). The build runs a leave-one-out check against plain inverse-distance weighting; if terrain does not win, it uses the plain version.
* **Cartographic finishing:** the overlay PNG is drawn at 2x the grid. Class edges are cut from a bicubic upsample of the smooth surface (curves, not pixel stairs), then colour patches smaller than `MIN_REGION_KM2` (150 km²) in flat terrain are merged into the surrounding class; patches on real relief (more than `SIEVE_RELIEF_M`, 120 m, of height change over about 3 miles) are kept, so mountain tops stay. The hover value and station table use the unsieved surface.
* **Land and water base map:** under the surface, a colour map of **land elevation** (green lowlands to brown highlands) and **ocean / lake depth** (light to deep blue), over NOAA's whole view extent including Canada and Mexico, with a light and a dark version. Cells with a chance of **1 % or less** are left transparent so this map shows through, i.e. every chance greater than 1 % is drawn (`TRANSPARENT_AT_OR_BELOW_PCT` in `pipeline/config.py`; compared on the whole-percent value the page reports); hover and click still report their exact value. Ocean depth is from the same AWS terrain tiles (which include bathymetry); lake depth is the lake surface minus the elevation, so a lake with no bathymetry in the tiles reads as uniformly shallow. Lakes and country shapes are Natural Earth. The ramps are the `LAND_RAMP` / `WATER_RAMP` tables in the config (in **feet**). The base map is optional: if its Natural Earth files cannot be downloaded (or with `--no-base`), the page falls back to a flat grey background.
* **Terrain relief:** a hillshade of that elevation, drawn as a translucent black/white layer (so it also shows on the white top class). Toolbar switch and strength slider.
* **Hover card:** moving the pointer over the map shows a card with the estimated chance at that spot and the nearest station (name, distance, its own value). It hides while panning, over map controls and outside the U.S. land area, and has a "Hover info" switch in the toolbar.
* **Click look-up:** the page downloads a one-byte-per-cell value grid and reports the exact modelled value of the cell you click, plus the nearest station.
* **Stations:** NOAA's public station layer (`Percent_da`, `ELEVATION`, `Years`), with the NCEI spreadsheet as a fallback. Dots (toolbar or on-map switch), search, and the "Stations by chance" table, which recalculates for whatever part of the map is in view. The headline tiles are shares of **land area** from the surface.
* **Legend:** NOAA's eight classes (0–10, 11–25, 26–40, 41–50, 51–60, 61–75, 76–90, 91–100 %), in both themes. Colours were sampled from NOAA's legend and tiles, except that the two darkest classes are lightened (NOAA's `#3b3b3b` and `#304469` read as near-black); edit `NOAA_COLORS` in `pipeline/config.py` to change any of them.
* **Light / Dark toggle:** page, water and Canada/Mexico context colours, labels. The class colours stay NOAA's.
* **NOAA tiles (reference):** a switch overlays NOAA's own rendering, for comparing against the generated surface. Its dots and labels are part of NOAA's images.
* **Units:** everything the page shows is imperial: probabilities in percent, snow depth in inches, elevations and depths in feet (station elevations in `stations.json` and `stations.csv` are feet too), distances in miles. Internally the pipeline still works in metres (the projection, the grid, the terrain tiles); that never reaches the page.
* **Opening view:** the map opens framed on the contiguous U.S. (not NOAA's larger extent) and "Wide map" re-frames it the same way if you are still looking at the whole country. You can still pan and zoom out to NOAA's full extent.
* **Wide map**, state lines (on by default) and place labels (off) from Natural Earth.

```
pipeline/            Python: stations + AWS terrain -> surface (overlay.png, relief.webp, values.bin.gz) + stations + meta.json
site/                Static page (index.html, app.js, style.css, vendor/)
site/data/           GENERATED by the pipeline (git-ignored)
data/raw/            Cached inputs (station JSON, boundaries, terrarium/ terrain tiles)
tests/               pytest — synthetic fixtures only, no network
.github/workflows/   Build + deploy to GitHub Pages
```

## Deploy to GitHub Pages

1. Push this folder to a GitHub repo (default branch `main`).
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. Push (or run the workflow manually). It runs the tests, builds `site/data/`, and publishes `site/`.

The page will be at `https://<user>.github.io/<repo>/`. NOAA's inputs change only when new Normals are issued (next ~2031), so there is no schedule; downloads (including the terrain tiles) are cached between runs. A cold build downloads about a thousand terrain tiles (the 1 km surface plus the 2 km base map) and takes a few minutes. Visitors' browsers only request NOAA's tiles if they switch the reference layer on.

## Run locally

```bash
pip install -r requirements.txt
python -m pytest -q
python -m pipeline.build              # downloads stations + terrain, writes site/data/
python -m http.server -d site 8000    # open http://localhost:8000
```

Flags: `--source arcgis|xlsx|auto`, `--stations FILE` (local xlsx/csv), `--prob-col "<header>"`, `--refresh`, `--res METRES` (grid size, default 1000), `--dem-zoom 7|8` (terrain detail; 8 is ~2x finer and 4x the download), `--no-terrain` (plain interpolation, no relief or base map), `--no-base` (skip the land / bathymetry base map), `--no-overlays`, `--demo` (**synthetic** stations and terrain for previewing; the page shows a "DEMO DATA" banner).

### If a download is blocked

* NOAA feature layer unreachable → the build falls back to the NCEI spreadsheet; if both fail it prints both reasons. A download that is not really an Excel file (e.g. an HTML error page) is rejected and never cached. You can also hand-place either input in `data/raw/` (`stations_arcgis.json` or `Christmas_day_snow_statistics_1991-2020_updated.xlsx`); files already there are used without touching the network.
* The spreadsheet loader finds columns by header text. If the layout differs, or two probability columns look equally plausible, it stops and prints every header instead of guessing — re-run with `--prob-col "<exact header>"`.
* The state boundaries (Natural Earth, `ne_50m_admin_1_states_provinces.geojson`) are required: they define the land the surface covers. Hand-place the file in `data/raw/` if a runner cannot download it. The state-line / place-label / Canada-Mexico overlays are optional and skipped if their downloads fail.
* AWS terrain tiles unreachable → the build stops with the tile that failed. `--no-terrain` builds the plain interpolated surface without relief.

## Caveats

* The surface is a **model**, not NOAA's grid: it matches NOAA's station values, but between stations (mountains especially) it can differ from NOAA's own map. The build prints the leave-one-out error, and the page footer shows it. Compare against NOAA's tiles with the reference switch.
* A click reports the modelled value for that 1 km cell and the nearest station's measured value; neither is a forecast.
* Climatology is not a forecast.

## Credits

NOAA, Oceanic and Atmospheric Research (OAR), Climate Program Office (CPO), Climate.gov; NOAA National Centers for Environmental Information (NCEI) · Elevation: AWS Open Data Terrain Tiles (Mapzen / SRTM, NED and other sources; see the registry entry) · Leaflet (BSD-2-Clause), Proj4Leaflet (BSD-2-Clause), proj4js (MIT) · Natural Earth (public domain).
