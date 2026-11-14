"""HTTP download with a provenance manifest. Any failure raises."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# The EMS portals answer bursts of requests with HTTP 429.
_RETRY_WAIT_S = (10, 30, 60, 120)


def full_url(url: str, params: dict[str, str] | None = None) -> str:
    return requests.Request("GET", url, params=params).prepare().url


def http_get(url: str, params: dict[str, str] | None = None,
             timeout: int = 600) -> requests.Response:
    """GET that waits and retries on HTTP 429 only. The caller checks the status."""
    for wait in _RETRY_WAIT_S:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 429:
            return resp
        time.sleep(wait)
    return requests.get(url, params=params, timeout=timeout)


def fetch(url: str, dest: Path, params: dict[str, str] | None = None,
          timeout: int = 600,
          check: Callable[[requests.Response], None] | None = None) -> Path:
    """Download ``url`` to ``dest``; an existing ``dest`` is reused as-is.
    ``check`` may raise on a bad response body before anything is cached."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = http_get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    if check is not None:
        check(resp)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(resp.content)
    part.replace(dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, url: str, root: Path) -> dict[str, Any]:
    """Provenance of one downloaded file. The retrieval time is the file's mtime,
    so re-running with a cached file keeps the original download date."""
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "path": path.relative_to(root).as_posix(),
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "retrieved_utc": mtime.isoformat(timespec="seconds"),
    }


def write_manifest(out_dir: Path, source_cfg: dict[str, Any],
                   files: list[dict[str, Any]], **extra: Any) -> Path:
    manifest = {
        "name": source_cfg["name"],
        "provider": source_cfg["provider"],
        "license": source_cfg["license"],
        "files": files,
        **extra,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
