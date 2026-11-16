"""Dukascopy M1 candles: URL scheme, .bi5 decoding and a polite downloader (run locally).

A daily candle file is LZMA-compressed; each record is 24 bytes, big-endian:
    int32 seconds-from-day-start, int32 open, int32 close, int32 low, int32 high, float32 volume
Prices are integers divided by the instrument's `price_divisor` (configs/data/dukascopy.yaml).
URL: {base}/{SYMBOL}/{YYYY}/{MM0}/{DD}/{BID|ASK}_candles_min_1.bi5 with a ZERO-based month.
"""
from __future__ import annotations

import lzma
import struct
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RECORD = struct.Struct(">5if")
COLUMNS = ("open", "close", "low", "high", "volume")


def day_url(base: str, symbol: str, day: date, side: str) -> str:
    side = side.upper()
    if side not in ("BID", "ASK"):
        raise ValueError("side must be BID or ASK")
    return f"{base}/{symbol}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def decode_bi5(raw: bytes, day: date, price_divisor: float) -> pd.DataFrame:
    """Decode one daily candle file into a DataFrame indexed by UTC minute (int64 minutes)."""
    if not raw:
        return _empty()
    data = lzma.decompress(raw)
    if len(data) % RECORD.size:
        raise ValueError(f"corrupt bi5 payload: {len(data)} bytes is not a multiple of {RECORD.size}")
    arr = np.frombuffer(data, dtype=np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"),
                                              ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]))
    day_min = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() // 60)
    minutes = day_min + arr["t"].astype(np.int64) // 60
    df = pd.DataFrame({"open": arr["o"] / price_divisor, "close": arr["c"] / price_divisor,
                       "low": arr["l"] / price_divisor, "high": arr["h"] / price_divisor,
                       "volume": arr["v"].astype(np.float64)}, index=pd.Index(minutes, name="minute"))
    return df


def encode_bi5(df: pd.DataFrame, day: date, price_divisor: float) -> bytes:
    """Inverse of decode_bi5 (used by tests to fabricate files)."""
    day_min = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() // 60)
    out = bytearray()
    for m, r in df.iterrows():
        out += RECORD.pack(int((m - day_min) * 60), *(int(round(r[c] * price_divisor)) for c in
                                                       ("open", "close", "low", "high")), float(r["volume"]))
    return lzma.compress(bytes(out), format=lzma.FORMAT_ALONE)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in COLUMNS}, index=pd.Index([], dtype=np.int64, name="minute"))


def download_range(symbol: str, source: str, start: date, end: date, out_dir: str | Path,
                   base_url: str, pause: float = 1.0, session=None, max_retries: int = 5) -> list[Path]:
    """Download raw daily BID/ASK files into out_dir/raw/{symbol}/... (resumable; read-only GETs).
    Intended to run on your own machine; Dukascopy rate-limits shared cloud IPs."""
    import requests
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "tradeworld-data/0.1")
    out = Path(out_dir) / "raw" / symbol
    paths = []
    for day in pd.date_range(start, end, freq="D").date:
        for side in ("BID", "ASK"):
            path = out / f"{day.isoformat()}_{side}.bi5"
            paths.append(path)
            if path.exists():
                continue
            url = day_url(base_url, source, day, side)
            for attempt in range(max_retries):
                try:
                    resp = session.get(url, timeout=30)
                except requests.RequestException:          # throttled hosts often just reset
                    time.sleep(pause * 2 ** attempt)
                    continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(pause * 2 ** attempt)
                    continue
                if resp.status_code == 404:
                    payload = b""
                else:
                    resp.raise_for_status()
                    payload = resp.content
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                break
            else:
                raise RuntimeError(f"giving up on {url} after {max_retries} attempts")
            time.sleep(pause)
    return paths


def load_raw_days(raw_dir: str | Path, symbol: str, start: date, end: date, price_divisor: float,
                  valid_range: tuple[float, float] | None = None) -> dict[str, pd.DataFrame]:
    """Decode downloaded daily files into BID and ASK frames, checking prices are plausible."""
    raw_dir = Path(raw_dir) / "raw" / symbol
    out = {}
    for side in ("BID", "ASK"):
        frames = []
        for day in pd.date_range(start, end, freq="D").date:
            path = raw_dir / f"{day.isoformat()}_{side}.bi5"
            if path.exists():
                frames.append(decode_bi5(path.read_bytes(), day, price_divisor))
        df = pd.concat(frames) if frames else _empty()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if valid_range is not None and len(df):
            lo, hi = valid_range
            med = float(df["close"].median())
            if not lo <= med <= hi:
                raise ValueError(f"{symbol} {side}: median price {med} outside {valid_range}; "
                                 "price_divisor is probably wrong")
        out[side.lower()] = df
    return out
