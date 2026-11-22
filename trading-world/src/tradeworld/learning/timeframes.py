"""Multi-timeframe view of the chart, given to an agent as visible side features (no rule about how to use them).

At every M1 row d the features use only mid prices up to and including the close of minute d (what any trader sees
at that moment; the observation itself already shows the bar closing at d):
  * the move over the last 15 minutes, hour, 4 hours, day and week (log returns of the mid price);
  * how volatile the last hour and the last day were (log of the root-mean-square one-minute return);
  * the time of day and the day of the week, as points on a circle (sin, cos), so 23:59 sits next to 00:00.
Minutes without a quote carry the last quoted mid forward; windows reaching before the first quote use what exists.
"""
from __future__ import annotations

import math

import numpy as np

WINDOWS = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
VOL_WINDOWS = {"1h": 60, "1d": 1440}
NAMES = ([f"move_{k}" for k in WINDOWS] + [f"vol_{k}" for k in VOL_WINDOWS]
         + ["day_sin", "day_cos", "week_sin", "week_cos"])


def _ffill(a: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = np.argmax(np.isfinite(a)) if np.isfinite(a).any() else 0
    out[:first] = a[first]
    return out


def features(md, instrument: int = 0) -> np.ndarray:
    """[T, len(NAMES)] features aligned with the market data's M1 rows (strictly causal)."""
    bid, ask = md.bid_c[:, instrument], md.ask_c[:, instrument]
    mid = _ffill(np.where(np.isfinite(bid) & np.isfinite(ask), 0.5 * (bid + ask), np.nan))
    lm = np.log(mid)
    T = len(lm)
    cols = []
    for w in WINDOWS.values():
        past = np.r_[np.full(min(w, T), lm[0]), lm[:-w]] if w < T else np.full(T, lm[0])
        cols.append(lm - past[:T])
    r1 = np.r_[0.0, np.diff(lm)]
    c2 = np.r_[0.0, np.cumsum(r1 * r1)]
    for w in VOL_WINDOWS.values():
        hi = np.arange(1, T + 1)
        lo = np.maximum(hi - w, 0)
        ms = (c2[hi] - c2[lo]) / np.maximum(hi - lo, 1)
        cols.append(np.log(np.sqrt(ms) + 1e-12))
    minute = np.asarray(md.times, np.int64)
    day = 2 * math.pi * (minute % 1440) / 1440.0
    week = 2 * math.pi * ((minute // 1440 + 3) % 7) / 7.0          # minute 0 = 1970-01-01, a Thursday
    cols += [np.sin(day), np.cos(day), np.sin(week), np.cos(week)]
    return np.column_stack(cols).astype(np.float64)
