import numpy as np
import pandas as pd
import pytest

from btc_bot.backtest import run_backtest
from btc_bot.risk import RiskConfig
from btc_bot.strategies.base import Action, Decision, Strategy
from conftest import make_bars

BIG = RiskConfig(risk_per_trade=0.01, max_leverage=100, max_position_btc=100, daily_loss_limit=1.0)


class Scripted(Strategy):
    """Enters/exits on fixed bar numbers so fills can be checked exactly."""
    name = "scripted"
    defaults = {"script": {}, "stop": 10.0}

    warmup = 0

    def prepare(self, df):
        out = df.copy()
        out["i"] = np.arange(len(df))
        return out

    def decide(self, row, position):
        action = self.script.get(int(row["i"]), Action.HOLD)
        return Decision(action, self.stop)


def test_fill_at_next_open_and_pnl():
    closes = [100, 100, 100, 110, 120, 130, 130, 130]
    bars = make_bars(closes)
    s = Scripted(script={2: Action.ENTER_LONG, 4: Action.EXIT}, stop=10.0)
    res = run_backtest(bars, s, BIG, initial_equity=1000, fee_bps=0, slippage_bps=0)
    t = res.trades[0]
    assert t.entry_time == bars.index[3] and t.entry_price == bars["open"].iloc[3] == 100
    assert t.exit_time == bars.index[5] and t.exit_price == bars["open"].iloc[5] == 120
    assert t.qty == pytest.approx(1.0)        # 1% of 1000 = 10 risk / 10 stop
    assert t.pnl == pytest.approx(20.0)
    assert res.stats["final_equity"] == pytest.approx(1020.0)


def test_stop_and_gap_fill():
    # Long entered at 100 with stop 90. Bar 4 opens at 100 and trades down to 80:
    # intrabar stop -> filled at the stop price, 90.
    bars = make_bars([100, 100, 100, 100, 80, 80])
    res = run_backtest(bars, Scripted(script={2: Action.ENTER_LONG}, stop=10.0), BIG,
                       initial_equity=1000, fee_bps=0, slippage_bps=0)
    t = res.trades[0]
    assert t.exit_reason == "stop" and t.exit_price == 90
    # Gap: the bar OPENS at 85, below the 90 stop -> filled at the open (worse), 85.
    bars2 = make_bars([100, 100, 100, 100])
    bars2.loc[bars2.index[3], ["open", "low"]] = [85, 85]
    res2 = run_backtest(bars2, Scripted(script={1: Action.ENTER_LONG}, stop=10.0), BIG,
                        initial_equity=1000, fee_bps=0, slippage_bps=0)
    assert res2.trades[0].exit_reason == "stop" and res2.trades[0].exit_price == 85


def test_short_and_reversal():
    closes = [100, 100, 100, 90, 80, 80, 90, 100, 100]
    bars = make_bars(closes)
    s = Scripted(script={1: Action.ENTER_SHORT, 4: Action.ENTER_LONG}, stop=50.0)
    res = run_backtest(bars, s, BIG, initial_equity=1000, fee_bps=0, slippage_bps=0)
    short, long_ = res.trades
    assert short.side == -1 and short.entry_price == 100 and short.exit_price == 80
    assert long_.side == 1 and long_.entry_price == 80
    assert short.pnl > 0 and long_.pnl > 0


def test_fees_reduce_equity_consistently():
    closes = [100, 100, 100, 100, 100, 100]
    bars = make_bars(closes)
    s = Scripted(script={1: Action.ENTER_LONG, 3: Action.EXIT}, stop=10.0)
    res = run_backtest(bars, s, BIG, initial_equity=1000, fee_bps=10, slippage_bps=0)
    t = res.trades[0]
    assert t.fees == pytest.approx(2 * 1.0 * 100 * 0.001)
    assert res.stats["final_equity"] == pytest.approx(1000 - t.fees)
    assert sum(tr.pnl for tr in res.trades) == pytest.approx(res.stats["final_equity"] - 1000)


def test_daily_loss_limit_blocks_entries():
    closes = [100, 100, 100, 100, 100, 100, 100, 100]
    bars = make_bars(closes)
    bars.loc[bars.index[3], "low"] = 50       # stop out on bar 3
    risk = RiskConfig(risk_per_trade=0.05, max_leverage=100, max_position_btc=100, daily_loss_limit=0.04)
    s = Scripted(script={1: Action.ENTER_LONG, 4: Action.ENTER_LONG}, stop=10.0)
    res = run_backtest(bars, s, risk, initial_equity=1000, fee_bps=0, slippage_bps=0)
    assert len(res.trades) == 1               # second entry blocked: down 5% today


def test_real_strategies_run_without_lookahead():
    from btc_bot.strategies import DonchianBreakout, MeanReversion
    rng = np.random.default_rng(1)
    closes = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000)))
    bars = make_bars(closes, spread=50)
    for strat in (MeanReversion(), DonchianBreakout()):
        full = run_backtest(bars, strat)
        # Truncating future data must not change trades that closed before the cut.
        cut = run_backtest(bars.iloc[:1500], strat)
        done = [t for t in cut.trades if t.exit_reason != "end of data"]
        assert [(t.entry_time, t.exit_time) for t in done] == \
               [(t.entry_time, t.exit_time) for t in full.trades[:len(done)]]
        assert len(full.trades) > 0
