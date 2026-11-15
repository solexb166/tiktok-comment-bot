"""Group 10: closed markets reject orders, daily breaks, 24/7 crypto, weekend-gap valuation."""
import numpy as np
import pytest

from tradeworld.agents.baselines import ScriptedPolicy
from tradeworld.data.synthetic import PathSpec, generate
from tradeworld.market.instruments import InstrumentSpec, Universe
from conftest import MON_2024_01_08, act, fills, make_profile, market, run, single


def test_order_rejected_when_market_closed():
    u = single()
    q = np.ones((8, 1), bool)
    q[2:4] = False                                                   # rows 2-3 closed
    md = market(u, np.full(8, 1.1), quoted=q)
    res = run(md, make_profile(u), {"a": ScriptedPolicy({0: act(1, [1.0])}, 1)})
    acct = res.accounts["a"]
    assert "market_closed" in res.records.iloc[0]["rejections"]
    f = fills(acct)
    assert f and f[0].t == md.times[4]                                # first fill once the market reopens


def test_sessions_in_synthetic_data():
    u = Universe((InstrumentSpec("XAUUSD", "XAU", "USD", "metal"), InstrumentSpec("BTCUSD", "BTC", "USD", "crypto")))
    md = generate(u, MON_2024_01_08 - 2 * 1440, 3 * 1440, seed=1)    # starts Saturday
    sat = np.arange(0, 1440)
    assert not md.quoted[sat, 0].any() and md.quoted[sat, 1].all()  # gold closed, BTC trades
    mon_break = 2 * 1440 + 21 * 60 + 15                              # Monday 21:15 UTC
    assert not md.quoted[mon_break, 0] and md.quoted[mon_break, 1]    # gold daily break, BTC open
    assert md.quoted[mon_break - 60, 0]


def test_weekend_gap_valued_at_first_real_quote():
    u = single()
    T = 3 * 1440
    start = MON_2024_01_08 - 3 * 1440 + 18 * 60                       # Friday 18:00 UTC
    md = generate(u, start, T, seed=11, paths={"EURUSD": PathSpec(1.1, 0.5, spread_bps=0.0)})
    closed = ~md.quoted[:, 0]
    reopen = int(np.flatnonzero(closed)[-1]) + 1
    last_quote = int(np.flatnonzero(closed)[0]) - 1
    gap = md.mid_o()[reopen, 0] - md.mid_c()[last_quote, 0]
    assert abs(gap) > 0
    from tradeworld.agents.baselines import HoldPolicy
    from tradeworld.interface.actions import ActionMapper
    res = run(md, make_profile(u), {"a": HoldPolicy(act(1, [1.0], 1.0), ActionMapper.primary(1))},
              interval=60, resolution=60, window=1)
    rec = res.records
    acct = res.accounts["a"]
    assert len(fills(acct)) == 1
    units = fills(acct)[0].data["du"]
    # equity is unchanged throughout the closed period
    during = rec[(rec.t > md.times[last_quote] + 60) & (rec.t <= md.times[reopen - 1])]
    assert len(during) > 20 and during["equity"].nunique() == 1
    # the first decision after the reopen sees the full gap in equity: units * price change
    before = during["equity"].iloc[-1]
    after_row = rec[rec.t > md.times[reopen]].iloc[0]
    k = int(after_row.t - md.times[0]) - 1                            # M1 row of that decision
    expected = before + units * (md.mid_c()[k, 0] - md.mid_c()[last_quote, 0])
    assert after_row["equity"] == pytest.approx(expected, rel=1e-12)
