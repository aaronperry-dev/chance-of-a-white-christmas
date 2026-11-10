"""Station source #1: NOAA's public hosted feature layer (same table the NOAA
web map draws). Structured JSON, so no column guessing, and it carries each
station's elevation (m) and years of record.

Record shape (from the live layer):
  {"FID":1,"STATION":"USC00144665","LATITUDE":38.49,"LONGITUDE":-101.3669,
   "ELEVATION":1008.6,"NAME":"LEOTI, KS US","month":12,"day":25,
   "Percent_da":16.5,"Years":24}
Output columns: id, name, state, lat, lon, elev_m, prob, years
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C
from .fetch import USER_AGENT, FetchError

log = logging.getLogger(__name__)

OUT_FIELDS = "STATION,LATITUDE,LONGITUDE,ELEVATION,NAME,month,day,Percent_da,Years"
_NAME = re.compile(r"^(?P<city>.*?),\s*(?P<st>[A-Z]{2})\s+US$")
_KEEP_UPPER = {"N", "S", "E", "W", "NE", "NW", "SE", "SW", "NNE", "NNW", "SSE", "SSW", "ENE", "ESE",
               "WNW", "WSW", "AP", "AFB", "WSO", "WSFO", "FAA", "II", "III", "IV", "USAF", "CAA"}


def pretty_name(raw: str) -> tuple[str, str]:
    """'NEW BRUNSWICK 3 SE, NJ US' -> ('New Brunswick 3 SE', 'NJ')."""
    raw = (raw or "").strip()
    m = _NAME.match(raw)
    city, st = (m["city"], m["st"]) if m else (raw, "")

    def word(w: str) -> str:
        if w.upper() in _KEEP_UPPER or any(ch.isdigit() for ch in w):
            return w.upper()
        return "-".join(p.capitalize() for p in w.split("-"))

    return " ".join(word(w) for w in city.split()), st


def fetch_attributes(url: str = C.FEATURE_URL, page: int = 2000, retries: int = 4) -> list[dict]:
    """Page through the layer with resultOffset until it runs dry."""
    attrs: list[dict] = []
    with requests.Session() as s:
        s.headers["User-Agent"] = USER_AGENT
        while True:
            params = {"where": "1=1", "outFields": OUT_FIELDS, "returnGeometry": "false",
                      "orderByFields": "FID", "resultOffset": len(attrs),
                      "resultRecordCount": page, "f": "json"}
            last = None
            for a in range(1, retries + 1):
                try:
                    r = s.get(url + "/query", params=params, timeout=(10, 120))
                    r.raise_for_status()
                    data = r.json()
                    if "error" in data:                      # ArcGIS reports errors with HTTP 200
                        raise FetchError(str(data["error"]))
                    break
                except (requests.RequestException, ValueError, FetchError) as exc:
                    last = exc
                    time.sleep(2 * a)
            else:
                raise FetchError(f"feature layer query failed at offset {len(attrs)}: {last}")
            feats = data.get("features", [])
            if not feats:
                break
            attrs.extend(f["attributes"] for f in feats)
            log.info("fetched %d stations", len(attrs))
    if not attrs:
        raise FetchError("feature layer returned no stations")
    return attrs


def to_frame(attrs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(attrs)
    need = {"LATITUDE", "LONGITUDE", "Percent_da"}
    if not need <= set(df.columns):
        raise ValueError(f"feature layer is missing fields {need - set(df.columns)}; got {list(df.columns)}")
    if {"month", "day"} <= set(df.columns):
        df = df[(df.month == 12) & (df.day == 25)]
    names = df["NAME"].map(pretty_name) if "NAME" in df else pd.Series([("", "")] * len(df), index=df.index)
    elev = pd.to_numeric(df.get("ELEVATION"), errors="coerce")
    out = pd.DataFrame({
        "id": df["STATION"].astype(str) if "STATION" in df else df.index.astype(str),
        "name": [n for n, _ in names], "state": [s for _, s in names],
        "lat": pd.to_numeric(df["LATITUDE"], errors="coerce"),
        "lon": pd.to_numeric(df["LONGITUDE"], errors="coerce"),
        "elev_m": elev.where(elev > -500),            # NaN for sentinel/missing values
        "prob": pd.to_numeric(df["Percent_da"], errors="coerce").clip(0, 100),
        "years": pd.to_numeric(df.get("Years"), errors="coerce"),
    })
    out = out.dropna(subset=["lat", "lon", "prob"]).drop_duplicates(subset=["id"])
    return out.reset_index(drop=True)


def load(cache: Path, refresh: bool = False) -> pd.DataFrame:
    """Use the cached raw JSON if present (also lets you hand-place a copy), else fetch."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        log.info("using cached %s", cache)
        attrs = json.loads(cache.read_text())
    else:
        attrs = fetch_attributes()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(attrs, separators=(",", ":")))
    df = to_frame(attrs)
    log.info("%d stations from NOAA feature layer", len(df))
    return df
