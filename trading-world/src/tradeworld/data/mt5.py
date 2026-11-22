"""Load MT5 M1 history exported by scripts/mt5_export.py (datasets/mt5/<INSTRUMENT>/<YYYY-MM>.parquet) into MarketData.

MT5 bars are bid prices with one spread value per minute; the exporter stored ask = bid + spread x point. A minute is
quoted when the bar exists (MT5 writes no bar for minutes without ticks).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from ..market.clock import to_minute
from .align import align


def months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def load_frames(root: Path, instrument: str, start: date, end: date) -> dict[str, pd.DataFrame]:
    parts = []
    for y, m in months(start, end):
        f = Path(root) / instrument / f"{y:04d}-{m:02d}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing (export it with scripts/mt5_export.py)")
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time").set_index("time")
    out = {}
    for side in ("bid", "ask"):
        out[side] = pd.DataFrame({"open": df[f"{side}_o"], "high": df[f"{side}_h"], "low": df[f"{side}_l"],
                                  "close": df[f"{side}_c"], "volume": df["tick_volume"].astype(float)})
    return out


def load_market(root: Path, instruments, start: date, end: date):
    """MarketData on a regular UTC minute grid from the first minute of `start` to the last minute of `end`."""
    frames = {s: load_frames(root, s, start, end) for s in instruments}
    t0 = to_minute(datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
    t1 = to_minute(datetime(end.year, end.month, end.day, tzinfo=timezone.utc)) + 1440
    return align(frames, list(instruments), t0, t1, zero_volume_is_unquoted=False)


def symbol_info(root: Path, instrument: str) -> dict:
    return json.loads((Path(root) / instrument / "symbol_info.json").read_text())
