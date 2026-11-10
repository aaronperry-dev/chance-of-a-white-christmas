"""Download helpers: cached, retried, atomic."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

USER_AGENT = "white-christmas-dashboard/1.0 (static climatology dashboard)"


class FetchError(RuntimeError):
    pass


def ensure_file(url: str, dest: Path, *, min_bytes: int = 1000, retries: int = 3) -> Path:
    """Return ``dest``, downloading ``url`` first if it is not already cached.

    A file already present in ``dest`` is always used as-is, so you can drop a
    manually downloaded copy into ``data/raw/`` (or commit it) and the
    pipeline never touches the network for it.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log.info("using cached %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s (attempt %d/%d)", url, attempt, retries)
            with requests.get(url, stream=True, timeout=(10, 120),
                              headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise FetchError(f"download too small ({tmp.stat().st_size} bytes)")
            tmp.replace(dest)
            return dest
        except (requests.RequestException, FetchError, OSError) as exc:
            last = exc
            log.warning("download failed: %s", exc)
            time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    raise FetchError(
        f"Could not download {url}: {last}\n"
        f"Download it manually and save it as {dest} (or pass --stations)."
    )
