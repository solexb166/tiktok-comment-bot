"""Group 3: accounting invariants under random actions across the full exposure range."""
import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from tradeworld.accounts.account import Prices
from tradeworld.agents.baselines import ScriptedPolicy
from tradeworld.brokerspec.profile import SlippageModel
from tradeworld.data.synthetic import PathSpec, generate
from tradeworld.market.instruments import InstrumentSpec, Universe
from conftest import MON_2024_01_08, make_profile, market, run

U3 = Universe((InstrumentSpec("EURUSD", "EUR", "USD", "fx", swap_long=-0.02, swap_short=0.01),
               InstrumentSpec("USDJPY", "USD", "JPY", "fx", swap_long=0.03, swap_short=-0.04),
               InstrumentSpec("XAUUSD", "XAU", "USD", "metal", commission_bps=0.3)))
actions = st.lists(st.lists(st.floats(-1, 1), min_size=4, max_size=4), min_size=1, max_size=40)


def ledger_balance(acct):
    bal = acct.initial
    for e in acct.events:
        if e.kind == "fill":
            bal += e.data["realized"] - e.data["commission"]
        elif e.kind == "swap":
            bal += e.data["amount"]
        elif e.kind == "nbp_reset":
            bal -= e.data["debt"]
    return bal


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(acts=actions, seed=st.integers(0, 10_000), lev=st.sampled_from([np.inf, 2000.0, 50.0]))
def test_invariants_random_actions(acts, seed, lev):
    md = generate(U3, MON_2024_01_08 + 20 * 60, 3 * 1440, seed=seed,
                  paths={"EURUSD": PathSpec(1.1, 0.3), "USDJPY": PathSpec(150, 0.3), "XAUUSD": PathSpec(2000, 0.4)},
                  sessions=False)
    prof = make_profile(U3, leverage=lev, slippage=SlippageModel(0.3, 0.3, 0.05))
    script = {k * 30: a for k, a in enumerate(acts)}
    res = run(md, prof, {"a": ScriptedPolicy(script, 3)}, seed=seed, interval=5, resolution=5, window=4)
    acct = res.accounts["a"]
    # (1) the balance equals the replayed ledger
    assert np.isclose(acct.balance, ledger_balance(acct), rtol=0, atol=1e-9)
    # (2) positions equal the sum of fills
    units = np.zeros(3)
    for e in acct.events:
        if e.kind == "fill":
            units[e.data["i"]] += e.data["du"]
    assert np.allclose(units, acct.units, atol=1e-9 * max(1, np.abs(units).max()))
    # (3) closed-trade P&L + open-trade P&L == balance change (+ NBP resets)
    nbp = sum(e.data["debt"] for e in acct.events if e.kind == "nbp_reset")
    total = sum(t["pnl"] for t in acct.trades) + acct.trade_pnl.sum()
    assert np.isclose(total, acct.balance - acct.initial + nbp, atol=1e-8)
    # (4) equity = balance + unrealized; dead accounts are flat
    p = Prices(md.bid_c[-1], md.ask_c[-1], U3.quote_to_usd(md.mid_c()[-1]))
    assert np.isclose(acct.equity(p), acct.balance + acct.unrealized(p).sum())
    if not acct.alive:
        assert not acct.has_positions


@settings(max_examples=30, deadline=None)
@given(acts=actions)
def test_zero_cost_round_trips_have_zero_pnl(acts):
    u = Universe((InstrumentSpec("EURUSD", "EUR", "USD", "fx"),))
    md = market(u, np.full(400, 1.1))                          # constant price, zero spread
    script = {k * 5: [a[0], a[3]] for k, a in enumerate(acts)}
    script[len(acts) * 5] = [0.0, 0.0]                         # finish flat
    res = run(md, make_profile(u), {"a": ScriptedPolicy(script, 1)})
    acct = res.accounts["a"]
    assert not acct.has_positions
    assert abs(acct.balance - 10.0) < 1e-9
