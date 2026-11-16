"""Group 13: baseline agents match analytic expectations."""
import numpy as np
import pytest

from tradeworld.agents.baselines import FlatPolicy, HoldPolicy, RandomPolicy
from tradeworld.data.synthetic import PathSpec, generate
from tradeworld.interface.actions import ActionMapper
from conftest import MON_2024_01_08, act, fills, make_profile, market, run, single


def test_flat_agent_equity_is_constant():
    u = single()
    md = generate(u, MON_2024_01_08, 2 * 1440, seed=1)
    res = run(md, make_profile(u, leverage=2000.0), {"f": FlatPolicy(1)}, interval=5, resolution=5, window=12)
    assert (res.records["equity"] == 10.0).all() and res.accounts["f"].balance == 10.0
    assert not res.accounts["f"].events


@pytest.mark.parametrize("gross", [1.0, 3.0])
def test_buy_and_hold_matches_analytic(gross):
    u = single()
    md = generate(u, MON_2024_01_08, 2 * 1440, seed=2, paths={"EURUSD": PathSpec(1.1, 0.2, spread_bps=0.0)},
                  sessions=False)
    res = run(md, make_profile(u), {"h": HoldPolicy(act(1, [1.0], gross), ActionMapper.primary(1))},
              interval=5, resolution=5, window=12)
    acct = res.accounts["h"]
    f = fills(acct)
    assert len(f) == 1                                                # bought once, never rebalanced
    t_fill = int(np.flatnonzero(md.times == f[0].t)[0])
    d0 = t_fill - 1
    units = gross * 10.0 / md.mid_c()[d0, 0]
    final = 10.0 + units * (md.mid_c()[-1, 0] - md.mid_o()[t_fill, 0])
    from tradeworld.accounts.account import Prices
    p = Prices(md.bid_c[-1], md.ask_c[-1], np.ones(1))
    assert acct.equity(p) == pytest.approx(final, rel=1e-12)


def test_random_agent_zero_cost_constant_price_keeps_equity():
    u = single()
    md = market(u, np.full(3000, 1.1))
    res = run(md, make_profile(u), {"r": RandomPolicy(1, 0)}, interval=5, resolution=5, window=4)
    acct = res.accounts["r"]
    assert len(fills(acct)) > 100 and acct.balance == pytest.approx(10.0, abs=1e-9)


def test_random_agent_bleeds_costs_on_zero_drift_market():
    """Expected P&L of random trading on a driftless market is minus its costs. Across many
    seeds the mean P&L must be within a few standard errors of -costs."""
    u = single()
    md = generate(u, MON_2024_01_08, 1440, seed=3, paths={"EURUSD": PathSpec(1.1, 0.05, spread_bps=1.0)},
                  sessions=False)
    pnl, costs = [], []
    for s in range(12):
        res = run(md, make_profile(u, leverage=2000.0), {"r": RandomPolicy(1, 100 + s)},
                  seed=s, interval=5, resolution=5, window=4, mappers={"r": ActionMapper.benchmark(1, 3)})
        a = res.accounts["r"]
        from tradeworld.accounts.account import Prices
        eq = a.equity(Prices(md.bid_c[-1], md.ask_c[-1], np.ones(1)))
        pnl.append(eq - 10.0)
        costs.append(a.costs_total["spread"] + a.costs_total["slippage"])
    pnl, costs = np.array(pnl), np.array(costs)
    gross = pnl + costs                                           # market P&L before costs
    assert abs(gross.mean()) < 4 * gross.std(ddof=1) / np.sqrt(len(gross)) + 1e-9
    assert np.all(pnl < gross)                                    # costs always reduce P&L


def test_hold_policy_waits_for_market_to_open():
    """Buy-and-hold must still buy if its first orders are rejected because the market is closed."""
    u = single()
    q = np.ones(400, dtype=bool)
    q[:100] = False                                                   # closed for the first 100 minutes
    md = market(u, np.full(400, 1.1), quoted=q[:, None])
    res = run(md, make_profile(u), {"h": HoldPolicy(act(1, [1.0], 1.0), ActionMapper.primary(1))},
              interval=5, resolution=5, window=4)
    acct = res.accounts["h"]
    f = fills(acct)
    assert len(f) == 1 and f[0].t >= md.times[100]
    assert any(e.kind == "reject" for e in acct.events)
