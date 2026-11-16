"""Monthly archives of raw Dukascopy daily files (one .tar per instrument and month).

Raw .bi5 files are already LZMA-compressed (~10-15 KB per day and side), so archiving them
unchanged is lossless and compact. Each tar contains `manifest.json` (per file: bytes, sha256,
status 'ok' | 'empty' | 'closed') plus the files `YYYY-MM-DD_BID.bi5` / `_ASK.bi5`.
"""
from __future__ import annotations

import calendar
import hashlib
import io
import json
import tarfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .dukascopy import decode_bi5

CLOSED_WEEKDAYS = {"crypto": set(), "default": {5}}     # FX/metals/indices: no Saturday data


def month_days(year: int, month: int) -> list[date]:
    return [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def archive_path(root: Path, symbol: str, year: int, month: int) -> Path:
    return Path(root) / symbol / f"{year:04d}-{month:02d}.tar"


def pack_month(root: Path, symbol: str, year: int, month: int, files: dict[str, bytes],
               statuses: dict[str, str]) -> Path:
    """files: name -> raw bytes (b'' for empty / closed days)."""
    manifest = {name: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "status": statuses[name]}
                for name, b in sorted(files.items())}
    path = archive_path(root, symbol, year, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        def add(name, payload):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(payload), 0, 0o644      # deterministic archive
            tar.addfile(info, io.BytesIO(payload))
        add("manifest.json", json.dumps(manifest, indent=1, sort_keys=True).encode())
        for name in sorted(files):
            add(name, files[name])
    tmp = path.with_suffix(".tar.tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)
    return path


def read_month(path: Path, verify: bool = True) -> tuple[dict, dict[str, bytes]]:
    with tarfile.open(path, "r") as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read())
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.name != "manifest.json"}
    if verify:
        for name, meta in manifest.items():
            if hashlib.sha256(files[name]).hexdigest() != meta["sha256"]:
                raise ValueError(f"{path}: checksum mismatch for {name}")
    return manifest, files


def load_archived(root: Path, symbol: str, start: date, end: date, price_divisor: float) -> dict[str, pd.DataFrame]:
    """Decode archived months into {'bid': df, 'ask': df} for [start, end]."""
    out = {"bid": [], "ask": []}
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        path = archive_path(root, symbol, y, m)
        if not path.exists():
            raise FileNotFoundError(f"missing archive {path}")
        _, files = read_month(path)
        for name, raw in files.items():
            day = date.fromisoformat(name[:10])
            if start <= day <= end and raw:
                side = "bid" if name.endswith("_BID.bi5") else "ask"
                out[side].append(decode_bi5(raw, day, price_divisor))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    res = {}
    for side, frames in out.items():
        df = pd.concat(frames) if frames else decode_bi5(b"", start, price_divisor)
        res[side] = df[~df.index.duplicated(keep="last")].sort_index()
    return res
