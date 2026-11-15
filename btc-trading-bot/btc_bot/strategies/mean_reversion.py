"""RSI + Bollinger Band mean reversion.

Long:  close below the lower band AND RSI oversold  -> expect a bounce back to the middle band.
Short: close above the upper band AND RSI overbought -> expect a drop back to the middle band.
Exit:  price returns to the middle band, or the ATR stop is hit.
Optional trend filter: skip longs below / shorts above a long SMA (avoids fading strong trends).
"""
from __future__ import annotations

import pandas as pd

from .. import indicators as ind
from .base import Action, Decision, Strategy


class MeanReversion(Strategy):
    name = "mean_reversion"
    defaults = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "allow_short": True,
        "trend_filter_period": 0,   # 0 = off; e.g. 200 to only trade with the long-term trend
    }

    @property
    def warmup(self) -> int:
        return max(self.bb_period, self.rsi_period, self.atr_period, self.trend_filter_period) + 1

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bb_mid"], out["bb_up"], out["bb_low"] = ind.bollinger(out["close"], self.bb_period, self.bb_std)
        out["rsi"] = ind.rsi(out["close"], self.rsi_period)
        out["atr"] = ind.atr(out, self.atr_period)
        if self.trend_filter_period:
            out["trend_sma"] = ind.sma(out["close"], self.trend_filter_period)
        return out

    def decide(self, row: pd.Series, position: int) -> Decision:
        needed = ["bb_mid", "bb_up", "bb_low", "rsi", "atr"]
        if row[needed].isna().any():
            return Decision(Action.HOLD, reason="warmup")
        close, stop = row["close"], self.stop_atr_mult * row["atr"]

        if position > 0 and close >= row["bb_mid"]:
            return Decision(Action.EXIT, reason="long target: back to middle band")
        if position < 0 and close <= row["bb_mid"]:
            return Decision(Action.EXIT, reason="short target: back to middle band")

        trend_ok_long = trend_ok_short = True
        if self.trend_filter_period and pd.notna(row.get("trend_sma")):
            trend_ok_long = close > row["trend_sma"]
            trend_ok_short = close < row["trend_sma"]

        if position <= 0 and close < row["bb_low"] and row["rsi"] < self.rsi_oversold and trend_ok_long:
            return Decision(Action.ENTER_LONG, stop, f"oversold: close<{row['bb_low']:.0f}, rsi={row['rsi']:.1f}")
        if (position >= 0 and self.allow_short and close > row["bb_up"]
                and row["rsi"] > self.rsi_overbought and trend_ok_short):
            return Decision(Action.ENTER_SHORT, stop, f"overbought: close>{row['bb_up']:.0f}, rsi={row['rsi']:.1f}")
        return Decision(Action.HOLD)
