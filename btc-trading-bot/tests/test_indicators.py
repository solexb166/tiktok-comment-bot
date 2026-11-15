import numpy as np
import pandas as pd

from btc_bot import indicators as ind
from conftest import make_bars


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    down = up[::-1].reset_index(drop=True)
    assert ind.rsi(up, 14).iloc[-1] == 100.0
    assert ind.rsi(down, 14).iloc[-1] < 1.0
    assert ind.rsi(up, 14).iloc[:13].isna().all()


def test_rsi_flat_is_50():
    assert ind.rsi(pd.Series([5.0] * 30), 14).iloc[-1] == 50.0


def test_bollinger_matches_manual():
    s = pd.Series(np.random.default_rng(0).normal(100, 5, 50))
    mid, up, low = ind.bollinger(s, 20, 2.0)
    window = s.iloc[-20:]
    assert np.isclose(mid.iloc[-1], window.mean())
    assert np.isclose(up.iloc[-1] - mid.iloc[-1], 2 * window.std(ddof=0))
    assert np.isclose(mid.iloc[-1] - low.iloc[-1], 2 * window.std(ddof=0))


def test_donchian_excludes_current_bar():
    df = make_bars([1, 2, 3, 4, 10])
    upper, lower = ind.donchian(df, 3)
    # prior 3 bars (2,3,4): highs are max(open, close) = 2,3,4 -> 4; current bar's 10 excluded
    assert upper.iloc[-1] == 4
    assert df["close"].iloc[-1] > upper.iloc[-1]


def test_atr_constant_range():
    df = make_bars([100.0] * 40, spread=5.0)
    assert np.isclose(ind.atr(df, 14).iloc[-1], 10.0)
