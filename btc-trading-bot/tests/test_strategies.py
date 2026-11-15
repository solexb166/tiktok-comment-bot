import numpy as np
import pandas as pd
import pytest

from btc_bot.strategies import Action, DonchianBreakout, MeanReversion, make_strategy
from conftest import make_bars


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        MeanReversion(bogus=1)
    with pytest.raises(ValueError):
        make_strategy("nope")


def test_mean_reversion_signals():
    s = MeanReversion()
    base = {"bb_mid": 100.0, "bb_up": 110.0, "bb_low": 90.0, "atr": 5.0}
    long_row = pd.Series({**base, "close": 85.0, "rsi": 20.0})
    d = s.decide(long_row, 0)
    assert d.action == Action.ENTER_LONG and d.stop_distance == 10.0
    assert s.decide(pd.Series({**base, "close": 115.0, "rsi": 80.0}), 0).action == Action.ENTER_SHORT
    assert s.decide(pd.Series({**base, "close": 101.0, "rsi": 55.0}), 1).action == Action.EXIT
    assert s.decide(pd.Series({**base, "close": 99.0, "rsi": 45.0}), -1).action == Action.EXIT
    assert s.decide(pd.Series({**base, "close": 85.0, "rsi": 40.0}), 0).action == Action.HOLD  # rsi not oversold
    no_short = MeanReversion(allow_short=False)
    assert no_short.decide(pd.Series({**base, "close": 115.0, "rsi": 80.0}), 0).action == Action.HOLD


def test_mean_reversion_trend_filter():
    s = MeanReversion(trend_filter_period=200)
    row = pd.Series({"bb_mid": 100.0, "bb_up": 110.0, "bb_low": 90.0, "atr": 5.0,
                     "close": 85.0, "rsi": 20.0, "trend_sma": 120.0})
    assert s.decide(row, 0).action == Action.HOLD      # below trend -> no dip buying


def test_breakout_signals():
    s = DonchianBreakout()
    base = {"dc_up": 110.0, "dc_low": 90.0, "dc_exit_up": 105.0, "dc_exit_low": 95.0, "atr": 4.0}
    assert s.decide(pd.Series({**base, "close": 111.0}), 0).action == Action.ENTER_LONG
    assert s.decide(pd.Series({**base, "close": 89.0}), 0).action == Action.ENTER_SHORT
    assert s.decide(pd.Series({**base, "close": 94.0}), 1).action == Action.EXIT
    assert s.decide(pd.Series({**base, "close": 89.0}), 1).action == Action.ENTER_SHORT  # reversal
    assert s.decide(pd.Series({**base, "close": 100.0}), 1).action == Action.HOLD


def test_breakout_fires_on_real_breakout():
    closes = [100.0] * 30 + [120.0]
    df = DonchianBreakout(entry_period=20, exit_period=10).prepare(make_bars(closes, spread=1.0))
    assert DonchianBreakout().decide(df.iloc[-1], 0).action == Action.ENTER_LONG
