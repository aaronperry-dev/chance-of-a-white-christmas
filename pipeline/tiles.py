"""Slice the big map images (overlay, hillshade, base maps) into a tile pyramid on NOAA's tile grid.

Serving each layer as ONE image makes the browser decode and scale a picture that is tens of thousands of
pixels wide at the finest zoom (hundreds of MB of textures), which is what made zooming slow and shaky. As tiles,
only the visible part is loaded and drawn.

Tile (z, x, y) covers the same square of the map as NOAA's own tile (Leaflet zoom z, tile size TILE_PX), so a plain
Leaflet tile layer on the page's custom CRS lines up with the grid-aligned images exactly. Zoom levels follow the page:
zoom 0 is an extra coarse level (2x NOAA's level 0), zoom 1..5 are NOAA's levels 0..4, zoom 6 is NOAA's finest / 2
(shown by scaling zoom-`native_zoom` tiles up).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C

log = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None


def zoom_resolutions() -> list[float]:
    """Metres per pixel for each Leaflet zoom level of the page (see site/app.js)."""
    r = C.TILE_RESOLUTIONS
    return [r[0] * 2, *r, r[-1] / 2]


def native_zoom(px_m: float, resolutions: list[float] | None = None) -> int:
    """Finest zoom whose pixels are not (much) smaller than the source pixels."""
    res = resolutions or zoom_resolutions()
    ok = [z for z, r in enumerate(res) if r >= 0.9 * px_m]
    return max(ok) if ok else 0


def write_pyramid(src: Path | Image.Image, extent: tuple[float, float, float, float], out_dir: Path, name: str, *,
                  fmt: str = "webp", quality: int = 85, lossless: bool = False, alpha_quality: int = 90,
                  size: int | None = None, min_zoom: int = 0, max_zoom: int | None = None) -> dict:
    """Write ``out_dir/tiles/<name>/{z}/{x}/{y}.<fmt>`` from an image covering ``extent`` = (x0, y0, x1, y1) in LCC metres.

    Returns the description the page needs: url template, tile size, native zoom and the list of tiles that exist
    (fully transparent tiles are not written).
    """
    size = size or C.WEB_TILE_PX
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img.load()
    has_alpha = img.mode in ("RGBA", "LA")
    img = img.convert("RGBA" if has_alpha else "RGB")
    x0, y0, x1, y1 = extent
    px_m = (x1 - x0) / img.width                                   # source pixel size, metres
    res = zoom_resolutions()
    nz = native_zoom(px_m, res) if max_zoom is None else max_zoom
    ox, oy = C.TILE_ORIGIN
    base = Path(out_dir) / "tiles" / name
    if base.exists():
        shutil.rmtree(base)
    have: list[list[int]] = []
    fill = (0, 0, 0, 0) if has_alpha else tuple(int(v) for v in np.asarray(img)[0, 0])
    for z in range(min_zoom, nz + 1):
        r = res[z]
        span = r * size                                            # tile edge, metres
        scale = px_m / r                                           # level pixels per source pixel
        lvl = img if abs(scale - 1) < 1e-6 else img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS if scale < 1 else Image.BICUBIC)
        lvl_px = (x1 - x0) / lvl.width                             # metres per level pixel (exact after rounding)
        lvl_py = (y1 - y0) / lvl.height
        tx0, tx1 = int(np.floor((x0 - ox) / span)), int(np.floor((x1 - ox) / span))
        ty0, ty1 = int(np.floor((oy - y1) / span)), int(np.floor((oy - y0) / span))
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                left, top = ox + tx * span, oy - ty * span          # tile top-left corner (LCC metres)
                u0, v0 = (left - x0) / lvl_px, (y1 - top) / lvl_py
                box = (u0, v0, u0 + span / lvl_px, v0 + span / lvl_py)
                if box[2] <= 0 or box[3] <= 0 or box[0] >= lvl.width or box[1] >= lvl.height:
                    continue
                t = lvl.transform((size, size), Image.EXTENT, box, Image.BILINEAR, fillcolor=fill)
                if has_alpha and t.getchannel("A").getbbox() is None:
                    continue
                d = base / str(z) / str(tx)
                d.mkdir(parents=True, exist_ok=True)
                kw = {"lossless": True, "method": 4} if lossless else {"quality": quality, "alpha_quality": alpha_quality, "method": 4}
                t.save(d / f"{ty}.{fmt}", fmt.upper(), **kw)
                have.append([z, tx, ty])
    log.info("tiles/%s: %d tiles, zooms %d..%d", name, len(have), min_zoom, nz)
    return {"url": f"tiles/{name}/{{z}}/{{x}}/{{y}}.{fmt}", "size": size, "native_zoom": nz, "have": have}
