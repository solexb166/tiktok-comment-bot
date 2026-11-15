"""Technical indicators. All functions take/return pandas Series aligned to the input index."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (0-100). NaN during warmup."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses in the window -> RSI 100 (or 50 if price was completely flat).
    flat = (avg_loss == 0) & avg_gain.notna()
    out = out.mask(flat & (avg_gain > 0), 100.0)
    out = out.mask(flat & (avg_gain == 0), 50.0)
    return out


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Returns (middle, upper, lower) bands."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + num_std * std, mid - num_std * std


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    return true_range(df).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def donchian(df: pd.DataFrame, period: int):
    """Highest high / lowest low of the *previous* `period` bars (excludes the current bar),
    so `close > upper` means a genuine breakout of the prior range."""
    upper = df["high"].rolling(period, min_periods=period).max().shift(1)
    lower = df["low"].rolling(period, min_periods=period).min().shift(1)
    return upper, lower
