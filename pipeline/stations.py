"""Parse the NCEI station table into a clean DataFrame.

The NCEI spreadsheet is a human-oriented table (title rows, long header text),
so columns are located by header text rather than by position. If the
probability column cannot be identified unambiguously the loader stops and
lists the candidates instead of guessing: pass ``--prob-col "<header>"``.

Output columns: id, name, state, lat, lon, elev, prob   (prob in percent, 0-100)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class StationFormatError(ValueError):
    pass


_LAT = re.compile(r"\blat(itude)?\b", re.I)
_LON = re.compile(r"\blon(g|gitude)?\b", re.I)
_ELEV = re.compile(r"elev", re.I)
_STATE = re.compile(r"^\s*(state|st)\b", re.I)
_ID = re.compile(r"(station.?id|\bid\b|ghcn|coop|wban|\bstn\b)", re.I)
_NAME = re.compile(r"(station.?name|^\s*name\s*$|^\s*station\s*$|location)", re.I)


def _read_sheets(path: Path) -> dict[str, pd.DataFrame]:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    return {"csv": pd.read_csv(path, header=None, dtype=object)}


def _find_header_row(raw: pd.DataFrame, scan: int = 30) -> int | None:
    for i in range(min(scan, len(raw))):
        cells = [str(c) for c in raw.iloc[i].tolist() if pd.notna(c)]
        if any(_LAT.search(c) for c in cells) and any(_LON.search(c) for c in cells):
            return i
    return None


def _table_from_sheet(raw: pd.DataFrame) -> pd.DataFrame | None:
    hdr = _find_header_row(raw)
    if hdr is None:
        return None
    names: list[str] = []
    seen: dict[str, int] = {}
    for c in raw.iloc[hdr].tolist():
        base = re.sub(r"\s+", " ", str(c)).strip() if pd.notna(c) else "unnamed"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}.{seen[base] - 1}")
    df = raw.iloc[hdr + 1:].copy()
    df.columns = names
    return df.reset_index(drop=True)


def _first(cols: list[str], pattern: re.Pattern, exclude: set[str] = frozenset()) -> str | None:
    for c in cols:
        if c not in exclude and pattern.search(c):
            return c
    return None


_OTHER_THRESHOLD = re.compile(r"(?<![\d.])(3|5|10|12|24)\s*(in\b|inch|\")|\b0*(03|05|10)ti\b", re.I)


def _score_prob_col(name: str) -> int:
    n = name.lower()
    if "depth" not in n:
        return -1
    score = 3
    if re.search(r"prob|chance|%|percent|pct", n):
        score += 2
    if re.search(r">=|≥|\bge\b|at least|(?<![\d.])1\s*(in\b|inch|\")|001", n):
        score += 1
    if _OTHER_THRESHOLD.search(n):
        score -= 3
    return score


def _pick_prob_col(cols: list[str], numeric_ok: set[str], override: str | None) -> str:
    if override:
        for c in cols:
            if c.lower() == override.lower():
                return c
        raise StationFormatError(f"--prob-col {override!r} not found. Columns: {cols}")
    scored = sorted(((_score_prob_col(c), c) for c in cols if c in numeric_ok), reverse=True)
    good = [(s, c) for s, c in scored if s >= 3]
    if not good:
        raise StationFormatError(
            "Could not identify the 'probability of snow depth >= 1 inch' column.\n"
            f"Columns found: {cols}\nRe-run with --prob-col \"<exact header>\"."
        )
    if len(good) > 1 and good[0][0] == good[1][0]:
        raise StationFormatError(
            "More than one equally plausible probability column: "
            f"{[c for _, c in good]}. Re-run with --prob-col \"<exact header>\"."
        )
    return good[0][1]


def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def load_stations(path: Path, prob_col: str | None = None) -> pd.DataFrame:
    path = Path(path)
    tables = [t for t in (_table_from_sheet(r) for r in _read_sheets(path).values()) if t is not None]
    if not tables:
        raise StationFormatError(f"No sheet in {path.name} has latitude/longitude headers.")
    df = max(tables, key=len)
    cols = list(df.columns)

    lat_c = _first(cols, _LAT)
    lon_c = _first(cols, _LON)
    if not (lat_c and lon_c):
        raise StationFormatError(f"Latitude/longitude columns not found in {cols}")
    id_c = _first(cols, _ID)
    name_c = _first(cols, _NAME, {id_c} if id_c else set())
    state_c = _first(cols, _STATE)
    elev_c = _first(cols, _ELEV)

    # Header text decides which column is the probability; here we only require that
    # it holds *some* numbers (stations with too little data legitimately have blanks).
    numeric_ok = {c for c in cols if _to_num(df[c]).notna().mean() >= 0.05}
    pcol = _pick_prob_col(cols, numeric_ok, prob_col)
    log.info("columns: lat=%r lon=%r prob=%r id=%r name=%r state=%r elev=%r",
             lat_c, lon_c, pcol, id_c, name_c, state_c, elev_c)

    out = pd.DataFrame({
        "id": df[id_c].astype(str).str.strip() if id_c else pd.RangeIndex(len(df)).astype(str),
        "name": df[name_c].astype(str).str.strip() if name_c else "",
        "state": df[state_c].astype(str).str.strip() if state_c else "",
        "lat": _to_num(df[lat_c]),
        "lon": _to_num(df[lon_c]),
        "elev": _to_num(df[elev_c]) if elev_c else np.nan,
        "prob": _to_num(df[pcol]),
    })

    n0 = len(out)
    out = out.dropna(subset=["lat", "lon", "prob"])
    out = out[out.lat.between(-90, 90) & out.lon.between(-180, 180)]

    # Probabilities stored as fractions (0-1) rather than percent.
    if len(out) and out.prob.max() <= 1.0:
        log.warning("probabilities look like fractions; converting to percent")
        out["prob"] = out.prob * 100.0
    out["prob"] = out.prob.clip(0, 100)

    # Western-hemisphere longitudes stored without a sign.
    if len(out) and (out.lon > 0).mean() > 0.95 and out.lon.between(60, 130).all():
        log.warning("longitudes are all positive; flipping sign")
        out["lon"] = -out.lon

    out = out.drop_duplicates(subset=["id", "lat", "lon"]).reset_index(drop=True)
    log.info("kept %d of %d rows with a valid location and probability", len(out), n0)
    return out
