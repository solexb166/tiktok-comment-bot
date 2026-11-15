"""Group 7: 5% death rule, stop-out at the profile level, closing order, gaps, no actions after death."""
import numpy as np
import pytest

from tradeworld.agents.baselines import ConstantPolicy, ScriptedPolicy
from tradeworld.market.instruments import InstrumentSpec, Universe
from conftest import act, fills, make_profile, market, run, single


def test_death_at_five_percent_intrabar():
    u = single()
    mids = [1.1, 1.1, 1.1, 1.1, 1.08, 1.08, 1.08]          # -1.8% at 100x
    pol = ConstantPolicy(act(1, [1.0], 100.0))
    res = run(market(u, mids), make_profile(u), {"a": pol})
    acct = res.accounts["a"]
    assert not acct.alive and acct.death_reason == "equity_floor"
    assert acct.balance == pytest.approx(0.5, abs=1e-6)       # liquidated exactly at the 5% floor
    assert not acct.has_positions
    assert not any(e.t > acct.death_t for e in acct.events)   # no actions after death
    assert res.records["t"].max() < acct.death_t + 1


def test_exness_zero_stopout_means_death_first():
    u = single()
    prof = make_profile(u, leverage=2000.0, stop_out_level=0.0)
    res = run(market(u, [1.1, 1.1, 1.1, 1.1, 1.0, 1.0]), prof, {"a": ConstantPolicy(act(1, [1.0], 1000.0))})
    acct = res.accounts["a"]
    assert acct.death_reason == "equity_floor" and not fills(acct, "stop_out")


def test_stop_out_before_death_when_level_is_high():
    u = single()
    prof = make_profile(u, leverage=10.0, stop_out_level=0.5)       # margin 10% of notional
    # 9x long: stop-out (equity = 0.5 * margin) at ~1.0296, death (equity = $0.50) at ~0.984
    mids = [1.1, 1.1, 1.1, 1.1, 1.02, 1.02, 1.02]
    # decision 3 (flat) fills after the stop-out in bar 4, so the agent does not re-enter
    res = run(market(u, mids), prof, {"a": ScriptedPolicy({0: act(1, [1.0], 9.0), 3: act(1, [0.0])}, 1)})
    acct = res.accounts["a"]
    assert fills(acct, "stop_out") and acct.alive and not acct.has_positions
    f = [e for e in fills(acct) if e.data["reason"] == "stop_out"][0]
    units = 90.0 / 1.1
    equity_at_trigger = 10 + units * (f.data["price"] - 1.1)
    assert equity_at_trigger == pytest.approx(0.5 * 0.1 * units * f.data["price"], rel=1e-6)
    assert f.data["price"] == pytest.approx((1.1 * units - 10) / (0.95 * units), rel=1e-6)


def test_stop_out_closes_most_unprofitable_first():
    u = Universe((InstrumentSpec("EURUSD", "EUR", "USD", "fx"), InstrumentSpec("GBPUSD", "GBP", "USD", "fx")))
    prof = make_profile(u, leverage=10.0, stop_out_level=0.5)
    mids = np.array([[1.1, 1.3]] * 4 + [[0.95, 1.28]] * 3)       # EUR falls much more
    pol = ScriptedPolicy({0: act(2, [0.5, 0.5], 9.0)}, 2, default=np.zeros(3))
    acct = run(market(u, mids), prof, {"a": pol}).accounts["a"]
    so = fills(acct, "stop_out")
    assert so and so[0].data["i"] == 0                          # EURUSD closed first


def test_gap_through_fills_at_open():
    u = single()
    mids = [1.1, 1.1, 1.1, 1.1, 1.0, 1.0]                         # bar 4 opens at 1.1, closes 1.0
    q = np.ones((6, 1), bool)
    md = market(u, mids)
    # Make bar 5 open far below the death level (gap): shift bar 4 close so bar 5 opens at 1.0
    res = run(md, make_profile(u), {"a": ConstantPolicy(act(1, [1.0], 500.0))})
    acct = res.accounts["a"]
    assert not acct.alive
    # From a market where the open itself is past the floor:
    mids2 = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    md2 = market(u, mids2)
    for f in ("bid_o", "ask_o", "bid_l", "ask_l"):
        getattr(md2, f)[4] = 1.09                               # gap down at the open of bar 4
    for f in ("bid_h", "ask_h"):
        getattr(md2, f)[4] = 1.1
    acct2 = run(md2, make_profile(u), {"a": ConstantPolicy(act(1, [1.0], 500.0))}).accounts["a"]
    assert acct2.death_reason == "gap_through"
    assert fills(acct2)[-1].data["price"] == pytest.approx(1.09)


def test_extreme_leverage_intrabar_death_price():
    u = single()
    mids = [1.1, 1.1, 1.1, 1.1, 1.0993, 1.0993]                   # -0.064% at 2000x = -127%
    acct = run(market(u, mids), make_profile(u), {"a": ConstantPolicy(act(1, [1.0], 2000.0))}).accounts["a"]
    assert acct.death_reason == "equity_floor" and acct.balance == pytest.approx(0.5, abs=1e-6)
