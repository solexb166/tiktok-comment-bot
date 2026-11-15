"""Group 6: margin model, equity tiers, Unlimited eligibility, HMR, atomic rejection, max volume."""
import math

import numpy as np
import pytest

from tradeworld.accounts.margin import HMRWindow, MarginModel, UnlimitedEligibility
from tradeworld.agents.baselines import ScriptedPolicy
from tradeworld.brokerspec import load_profile
from tradeworld.market.instruments import InstrumentSpec, Universe
from conftest import MON_2024_01_08, act, fills, make_profile, market, run, single

U2 = Universe((InstrumentSpec("EURUSD", "EUR", "USD", "fx"), InstrumentSpec("GBPUSD", "GBP", "USD", "fx")))


def test_a_reference_and_unlimited_variant_rates():
    a = load_profile("environments/A_fractional.yaml")
    r = a.margin.rates(a.universe, 10.0, 0)
    sym = a.universe.symbols
    assert r[sym.index("EURUSD")] == pytest.approx(1 / 2000) and r[sym.index("XAUUSD")] == pytest.approx(1 / 2000)
    assert r[sym.index("BTCUSD")] == pytest.approx(0.0025) and r[sym.index("NAS100")] == pytest.approx(0.0025)
    unl = load_profile("environments/A_unlimited_margin.yaml")
    assert np.all(unl.margin.rates(unl.universe, 10.0, 0) == 0)


def test_equity_tiers_and_unlimited_eligibility():
    u = single()
    m = MarginModel(account_leverage=math.inf,
                    equity_tiers=((5000, math.inf), (30000, 2000), (100000, 1000), (math.inf, 500)),
                    unlimited=UnlimitedEligibility(equity_below=5000, min_closed_trades=10, fallback_leverage=2000))
    assert m.rates(u, 10.0, 0, closed_trades=0)[0] == pytest.approx(1 / 2000)     # fresh account: no Unlimited
    assert m.rates(u, 10.0, 0, closed_trades=10)[0] == 0.0                        # eligible: Unlimited
    assert m.rates(u, 6000.0, 0, closed_trades=50)[0] == pytest.approx(1 / 2000)  # above threshold
    assert m.rates(u, 40000.0, 0, closed_trades=50)[0] == pytest.approx(1 / 1000)
    assert m.rates(u, 2e5, 0, closed_trades=50)[0] == pytest.approx(1 / 500)


def test_fixed_margin_ignores_account_leverage_and_hmr_raises_margin():
    u = Universe((InstrumentSpec("BTCUSD", "BTC", "USD", "crypto", margin_mode="fixed", fixed_margin_rate=0.0025),
                  InstrumentSpec("XAUUSD", "XAU", "USD", "metal")))
    w = HMRWindow(start=100, end=200, leverage_cap={"metal": 200}, fixed_rate_min={"crypto": 0.005})
    for lev in (100.0, math.inf):
        m = MarginModel(account_leverage=lev, hmr_windows=(w,))
        assert m.rates(u, 10, 50)[0] == pytest.approx(0.0025)
        assert m.rates(u, 10, 150)[0] == pytest.approx(0.005)
        assert m.rates(u, 10, 150)[1] == pytest.approx(1 / min(lev, 200))   # HMR only lowers leverage
    assert MarginModel(account_leverage=100.0, hmr_windows=(w,)).rates(u, 10, 50)[1] == pytest.approx(0.01)


def test_atomic_rejection_all_increases():
    md = market(U2, np.tile([1.1, 1.3], (8, 1)))
    prof = make_profile(U2, leverage=100.0)                        # margin 1% of notional
    too_much = ScriptedPolicy({0: act(2, [0.5, 0.5], 150.0)}, 2)    # 150x gross needs 150% margin
    ok = ScriptedPolicy({0: act(2, [0.5, 0.5], 90.0)}, 2)
    res = run(md, prof, {"too_much": too_much, "ok": ok})
    assert not fills(res.accounts["too_much"])
    assert res.records.query("agent == 'too_much'").iloc[0]["rejections"].count("insufficient_margin") == 2
    assert len(fills(res.accounts["ok"])) == 2


def test_reductions_execute_when_increases_rejected():
    md = market(U2, np.tile([1.1, 1.3], (8, 1)))
    prof = make_profile(U2, leverage=100.0)
    pol = ScriptedPolicy({0: act(2, [1.0, 0.0], 50.0),              # long EURUSD 50x
                          2: np.r_[10 / 300, 290 / 300, ActionMapper_u(300.0)]}, 2)   # EUR -> 10x, GBP 290x
    acct = run(md, prof, {"a": pol}).accounts["a"]
    f = fills(acct)
    assert len(f) == 2 and f[1].data["i"] == 0 and f[1].data["du"] < 0   # reduction executed
    assert acct.units[1] == 0.0                                        # increase rejected
    assert acct.units[0] == pytest.approx(100.0 / 1.1, rel=1e-6)


def ActionMapper_u(G):
    from tradeworld.interface.actions import ActionMapper
    return ActionMapper.primary(2).u_for_gross(G)


def test_rejection_is_order_independent():
    U2r = Universe(tuple(reversed(U2.specs)))
    results = []
    for u, prices in ((U2, [1.1, 1.3]), (U2r, [1.3, 1.1])):
        md = market(u, np.tile(prices, (8, 1)))
        eur = u.index("EURUSD")
        d = np.zeros(2)
        d[eur], d[1 - eur] = 0.6, 0.4
        acct = run(md, make_profile(u, leverage=100.0), {"a": ScriptedPolicy({0: act(2, d, 120.0)}, 2)}).accounts["a"]
        results.append({u.symbols[i]: acct.units[i] for i in range(2)})
    assert results[0] == results[1] == {"EURUSD": 0.0, "GBPUSD": 0.0}


def test_max_volume_rejected():
    u = single(max_lots=50.0)
    pol = ScriptedPolicy({0: act(1, [1.0], 10.0)}, 1)                # 100 USD / 1.1 = 90.9 units > 50
    res = run(market(u, np.full(6, 1.1)), make_profile(u), {"a": pol})
    assert "max_volume" in res.records.iloc[0]["rejections"] and not fills(res.accounts["a"])
