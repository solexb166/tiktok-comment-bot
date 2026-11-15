"""Donchian channel breakout (turtle-style).

Long:  close breaks above the highest high of the previous `entry_period` bars.
Short: close breaks below the lowest low of the previous `entry_period` bars.
Exit:  close crosses the opposite `exit_period` channel (a shorter, trailing channel),
       or the ATR stop is hit.
"""
from __future__ import annotations

import pandas as pd

from .. import indicators as ind
from .base import Action, Decision, Strategy


class DonchianBreakout(Strategy):
    name = "breakout"
    defaults = {
        "entry_period": 20,
        "exit_period": 10,
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "allow_short": True,
    }

    @property
    def warmup(self) -> int:
        return max(self.entry_period, self.exit_period, self.atr_period) + 1

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["dc_up"], out["dc_low"] = ind.donchian(out, self.entry_period)
        out["dc_exit_up"], out["dc_exit_low"] = ind.donchian(out, self.exit_period)
        out["atr"] = ind.atr(out, self.atr_period)
        return out

    def decide(self, row: pd.Series, position: int) -> Decision:
        needed = ["dc_up", "dc_low", "dc_exit_up", "dc_exit_low", "atr"]
        if row[needed].isna().any():
            return Decision(Action.HOLD, reason="warmup")
        close, stop = row["close"], self.stop_atr_mult * row["atr"]

        if position <= 0 and close > row["dc_up"]:
            return Decision(Action.ENTER_LONG, stop, f"breakout above {row['dc_up']:.0f}")
        if position >= 0 and self.allow_short and close < row["dc_low"]:
            return Decision(Action.ENTER_SHORT, stop, f"breakdown below {row['dc_low']:.0f}")
        if position > 0 and close < row["dc_exit_low"]:
            return Decision(Action.EXIT, reason=f"long exit: below {row['dc_exit_low']:.0f}")
        if position < 0 and close > row["dc_exit_up"]:
            return Decision(Action.EXIT, reason=f"short exit: above {row['dc_exit_up']:.0f}")
        return Decision(Action.HOLD)
