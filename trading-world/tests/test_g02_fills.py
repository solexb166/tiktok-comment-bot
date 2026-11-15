"""Group 2: fills at the next M1 open; buys at ask, sells at bid; adverse slippage; exact commissions."""
import numpy as np
import pytest

from tradeworld.agents.baselines import ScriptedPolicy
from tradeworld.brokerspec.profile import SlippageModel
from conftest import act, fills, make_profile, market, run, single


def test_buy_at_ask_sell_at_bid_next_open():
    u = single()
    mids = np.r_[np.full(3, 1.1), np.full(3, 1.2), np.full(4, 1.15)]
    md = market(u, mids, spread_bps=2.0)
    pol = ScriptedPolicy({0: act(1, [1.0], 2.0), 3: act(1, [0.0])}, 1)
    res = run(md, make_profile(u), {"a": pol})
    acct = res.accounts["a"]
    f = fills(acct)
    # A constant 2x target re-sizes as equity changes (documented target semantics), so only the
    # opening fill (decision 0 -> row 2) and the closing fill (decision 3 -> row 5) are checked.
    buy, close = f[0], f[-1]
    assert buy.t == md.times[2] and buy.data["price"] == pytest.approx(md.ask_o[2, 0])
    assert close.t == md.times[5] and close.data["price"] == pytest.approx(md.bid_o[5, 0])
    assert buy.data["du"] > 0 > close.data["du"] and not acct.has_positions


def test_slippage_is_adverse():
    u = single()
    md = market(u, np.full(10, 1.1), spread_bps=1.0)
    prof = make_profile(u, slippage=SlippageModel(base_bps=1.0, noise_bps=0.5, range_coef=0.0))
    pol = ScriptedPolicy({0: act(1, [1.0]), 2: act(1, [-1.0]), 4: act(1, [0.0])}, 1)
    res = run(md, prof, {"a": pol})
    for e in fills(res.accounts["a"]):
        t = int(np.flatnonzero(md.times == e.t)[0])
        if e.data["du"] > 0:
            assert e.data["price"] >= md.ask_o[t, 0] * (1 + 1e-4) - 1e-15
        else:
            assert e.data["price"] <= md.bid_o[t, 0] * (1 - 1e-4) + 1e-15


def test_commission_exact():
    u = single(contract_size=100_000.0, commission_per_lot=7.0, commission_bps=0.5)
    md = market(u, np.full(8, 1.1))
    pol = ScriptedPolicy({0: act(1, [1.0], 50.0), 2: act(1, [0.0])}, 1)
    res = run(md, make_profile(u), {"a": pol})
    acct = res.accounts["a"]
    units = 500.0 / 1.1
    per_side = units / 100_000 * 7.0 + units * 1.1 * 0.5e-4
    f = fills(acct)
    assert f[0].data["commission"] == pytest.approx(per_side, rel=1e-12)
    assert acct.balance == pytest.approx(10.0 - 2 * per_side, rel=1e-12)
    assert acct.costs_total["commission"] == pytest.approx(2 * per_side, rel=1e-12)
