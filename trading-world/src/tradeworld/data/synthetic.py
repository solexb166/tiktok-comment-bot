"""Deterministic synthetic markets (for simulator tests and, in Phase 2, learning-loop validation).

A generator produces a mid-price path per instrument on a regular M1 grid; bid/ask are built
around it with a configurable spread; trading sessions follow the instrument's asset class.
The underlying path keeps evolving while a market is closed, so re-opening shows a real gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..market.clock import minute_of_day_utc, weekday_utc
from ..market.instruments import Universe
from .market_data import MarketData

MINUTES_PER_YEAR = 365.25 * 1440


def session_mask(times: np.ndarray, asset_class: str) -> np.ndarray:
    """Approximate retail CFD sessions in UTC: FX 24/5 (Sun 22:00 -> Fri 21:59),
    metals and indices the same with a daily 21:00-21:59 break, crypto 24/7."""
    if asset_class == "crypto":
        return np.ones(times.shape, dtype=bool)
    wd, mod = weekday_utc(times), minute_of_day_utc(times)
    open_ = ~((wd == 5) | ((wd == 4) & (mod >= 22 * 60)) | ((wd == 6) & (mod < 22 * 60)))
    if asset_class in ("metal", "index"):
        open_ &= ~((mod >= 21 * 60) & (mod < 22 * 60))
    return open_


@dataclass
class PathSpec:
    start_price: float
    annual_vol: float = 0.10
    annual_drift: float = 0.0
    spread_bps: float = 1.0           # full spread as bps of mid
    ar1_phi: float = 0.0              # AR(1) coefficient on M1 log returns (mean reversion < 0)


DEFAULT_PATHS = {
    "XAUUSD": PathSpec(2000.0, 0.15, spread_bps=1.5),
    "EURUSD": PathSpec(1.10, 0.07, spread_bps=0.9),
    "GBPUSD": PathSpec(1.27, 0.08, spread_bps=1.2),
    "USDJPY": PathSpec(150.0, 0.09, spread_bps=1.0),
    "BTCUSD": PathSpec(40000.0, 0.60, spread_bps=5.0),
    "NAS100": PathSpec(16000.0, 0.20, spread_bps=1.0),
}


def generate(universe: Universe, start_minute: int, minutes: int, seed: int,
             paths: dict[str, PathSpec] | None = None, sessions: bool = True,
             volume_mean: float = 100.0) -> MarketData:
    """Generate a deterministic synthetic market. `start_minute` is aligned to the hour."""
    if start_minute % 60:
        raise ValueError("start_minute must be aligned to a full hour")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    paths = {**DEFAULT_PATHS, **(paths or {})}
    times = start_minute + np.arange(minutes, dtype=np.int64)
    T, N = minutes, universe.n
    arrays = {f: np.empty((T, N)) for f in ("bid_o", "bid_h", "bid_l", "bid_c",
                                             "ask_o", "ask_h", "ask_l", "ask_c", "volume")}
    quoted = np.empty((T, N), dtype=bool)
    for j, spec in enumerate(universe.specs):
        p = paths.get(spec.symbol) or PathSpec(100.0)
        sigma = p.annual_vol / np.sqrt(MINUTES_PER_YEAR)
        mu = p.annual_drift / MINUTES_PER_YEAR
        eps = rng.standard_normal(T) * sigma
        if p.ar1_phi:
            r = np.empty(T)
            prev = 0.0
            for k in range(T):           # explicit loop keeps AR(1) exact and deterministic
                prev = p.ar1_phi * prev + eps[k]
                r[k] = prev
        else:
            r = eps
        r = r + mu - 0.5 * sigma ** 2
        log_close = np.log(p.start_price) + np.cumsum(r)
        close = np.exp(log_close)
        open_ = np.r_[p.start_price, close[:-1]]
        wiggle = np.abs(rng.standard_normal((T, 2))) * sigma * 0.5
        high = np.maximum(open_, close) * (1 + wiggle[:, 0])
        low = np.minimum(open_, close) * (1 - wiggle[:, 1])
        half = p.spread_bps / 2e4
        for side, sgn in (("bid", -1), ("ask", 1)):
            arrays[f"{side}_o"][:, j] = open_ * (1 + sgn * half)
            arrays[f"{side}_h"][:, j] = high * (1 + sgn * half)
            arrays[f"{side}_l"][:, j] = low * (1 + sgn * half)
            arrays[f"{side}_c"][:, j] = close * (1 + sgn * half)
        arrays["volume"][:, j] = rng.poisson(volume_mean, T).astype(float)
        quoted[:, j] = session_mask(times, spec.asset_class) if sessions else True
    return MarketData.from_raw(times, universe.symbols, arrays, quoted)


def from_mid_paths(universe: Universe, start_minute: int, mid: np.ndarray, spread_bps=0.0,
                   quoted: np.ndarray | None = None, range_bps: float = 0.0) -> MarketData:
    """Build exact, hand-specified markets for golden tests: `mid[t, i]` is the bar CLOSE mid;
    bar t opens at mid[t-1] (or mid[0]). high/low extend `range_bps` beyond open/close."""
    mid = np.asarray(mid, dtype=float)
    T, N = mid.shape
    times = start_minute + np.arange(T, dtype=np.int64)
    sp = np.broadcast_to(np.asarray(spread_bps, dtype=float), (N,))
    open_ = np.vstack([mid[:1], mid[:-1]])
    high = np.maximum(open_, mid) * (1 + range_bps / 1e4)
    low = np.minimum(open_, mid) * (1 - range_bps / 1e4)
    arrays = {}
    for side, sgn in (("bid", -1), ("ask", 1)):
        f = 1 + sgn * sp / 2e4
        arrays[f"{side}_o"], arrays[f"{side}_h"] = open_ * f, high * f
        arrays[f"{side}_l"], arrays[f"{side}_c"] = low * f, mid * f
    arrays["volume"] = np.ones((T, N))
    quoted = np.ones((T, N), dtype=bool) if quoted is None else quoted
    return MarketData.from_raw(times, universe.symbols, arrays, quoted)
