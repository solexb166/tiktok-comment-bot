"""Align per-instrument bid/ask M1 frames onto one regular UTC minute grid (MarketData)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market_data import MarketData


def align(frames: dict[str, dict[str, pd.DataFrame]], symbols, start_minute: int, end_minute: int,
          zero_volume_is_unquoted: bool = True) -> MarketData:
    """frames[symbol] = {"bid": df, "ask": df} with columns open/high/low/close/volume indexed by
    UTC minute. A minute is quoted only if BOTH sides have a candle there (and, optionally,
    non-zero volume). The grid is [start_minute, end_minute), hour-aligned."""
    if start_minute % 60:
        raise ValueError("start_minute must be hour-aligned")
    times = np.arange(start_minute, end_minute, dtype=np.int64)
    T, N = times.size, len(symbols)
    arrays = {f: np.full((T, N), np.nan) for f in ("bid_o", "bid_h", "bid_l", "bid_c",
                                                   "ask_o", "ask_h", "ask_l", "ask_c", "volume")}
    quoted = np.zeros((T, N), dtype=bool)
    for j, sym in enumerate(symbols):
        sides = frames[sym]
        present = np.ones(T, dtype=bool)
        for side in ("bid", "ask"):
            df = sides[side].reindex(times)
            for src, dst in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")):
                arrays[f"{side}_{dst}"][:, j] = df[src].to_numpy()
            present &= df["close"].notna().to_numpy()
            if side == "bid":
                arrays["volume"][:, j] = df["volume"].fillna(0.0).to_numpy()
        if zero_volume_is_unquoted:
            present &= arrays["volume"][:, j] > 0
        quoted[:, j] = present
    return MarketData.from_raw(times, tuple(symbols), arrays, quoted)
