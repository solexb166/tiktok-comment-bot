"""Market data: BTC candles from Coinbase's public API (no API key needed).

Signals are computed from this feed regardless of which broker executes the trades.
A broker's BTCUSD CFD tracks spot BTC closely; stop distances are applied relative to
the broker's own quote at order time, so small price differences between venues don't matter.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
MAX_CANDLES_PER_REQUEST = 300

# timeframe -> (native Coinbase granularity in seconds, resample factor)
TIMEFRAMES = {
    "1m": (60, 1),
    "5m": (300, 1),
    "15m": (900, 1),
    "1h": (3600, 1),
    "4h": (3600, 4),  # Coinbase has no 4h candles; built from 1h
    "6h": (21600, 1),
    "1d": (86400, 1),
}


def timeframe_seconds(timeframe: str) -> int:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; choose from {sorted(TIMEFRAMES)}")
    gran, factor = TIMEFRAMES[timeframe]
    return gran * factor


def bars_per_year(timeframe: str) -> float:
    return 365 * 24 * 3600 / timeframe_seconds(timeframe)


class CoinbaseFeed:
    def __init__(self, product: str = "BTC-USD", session: requests.Session | None = None,
                 request_pause: float = 0.2, timeout: float = 15.0):
        self.product = product
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "btc-bot/0.1")
        self.request_pause = request_pause
        self.timeout = timeout

    # -- low level -------------------------------------------------------
    def _fetch_chunk(self, granularity: int, start: datetime, end: datetime) -> list[list]:
        params = {"granularity": granularity, "start": start.isoformat(), "end": end.isoformat()}
        url = COINBASE_URL.format(product=self.product)
        for attempt in range(5):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == 4:
                    raise
                log.warning("Coinbase request failed (%s), retrying", exc)
                time.sleep(1.0 * (attempt + 1))
        return []

    def _fetch_native(self, granularity: int, start: datetime, end: datetime) -> pd.DataFrame:
        rows: list[list] = []
        step = timedelta(seconds=granularity * MAX_CANDLES_PER_REQUEST)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + step, end)
            rows.extend(self._fetch_chunk(granularity, cursor, chunk_end))
            cursor = chunk_end
            time.sleep(self.request_pause)
        return _to_frame(rows)

    # -- public ----------------------------------------------------------
    def history(self, timeframe: str, start: datetime, end: datetime | None = None,
                closed_only: bool = True) -> pd.DataFrame:
        """OHLCV candles indexed by UTC open time."""
        gran, factor = TIMEFRAMES[timeframe]
        end = end or datetime.now(timezone.utc)
        df = self._fetch_native(gran, start, end)
        if factor > 1:
            df = resample(df, f"{timeframe_seconds(timeframe)}s", factor)
        if closed_only:
            df = drop_unclosed(df, timeframe, now=end if end else None)
        return df

    def recent(self, timeframe: str, bars: int) -> pd.DataFrame:
        """The most recent `bars` fully closed candles."""
        seconds = timeframe_seconds(timeframe)
        now = datetime.now(timezone.utc)
        start = now - timedelta(seconds=seconds * (bars + 2))
        df = self.history(timeframe, start, now, closed_only=True)
        return df.iloc[-bars:]


def _to_frame(rows: list[list]) -> pd.DataFrame:
    # Coinbase row: [time, low, high, open, close, volume], newest first
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def resample(df: pd.DataFrame, rule: str, expected_count: int) -> pd.DataFrame:
    """Aggregate candles; bins with missing sub-candles are dropped."""
    agg = df.resample(rule, origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = df["close"].resample(rule, origin="epoch", label="left", closed="left").count()
    return agg[counts >= expected_count].dropna()


def drop_unclosed(df: pd.DataFrame, timeframe: str, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    seconds = timeframe_seconds(timeframe)
    closes_at = df.index + pd.Timedelta(seconds=seconds)
    return df[closes_at <= pd.Timestamp(now)]


def load_or_download(feed: CoinbaseFeed, timeframe: str, days: int,
                     cache_dir: str | Path = "data") -> pd.DataFrame:
    """Download `days` of history, caching to CSV so repeated backtests are fast."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    path = cache_dir / f"{feed.product}_{timeframe}_{days}d_{end:%Y%m%d}.csv"
    if path.exists():
        return read_csv(path)
    log.info("Downloading %s %s candles for %d days...", feed.product, timeframe, days)
    df = feed.history(timeframe, start, end)
    df.to_csv(path)
    return df


def read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "time"
    return df[["open", "high", "low", "close", "volume"]].astype(float)
