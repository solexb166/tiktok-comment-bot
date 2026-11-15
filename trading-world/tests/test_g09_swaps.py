"""Group 9: swaps only across the 17:00 New York rollover, triple days, swap-free instruments."""
import numpy as np
import pytest

from tradeworld.agents.baselines import ScriptedPolicy
from tradeworld.market.clock import rollover_minute
from tradeworld.market.instruments import InstrumentSpec, Universe
from conftest import MON_2024_01_08, act, fills, make_profile, market, run


def u_swap(**kw):
    return Universe((InstrumentSpec("XAUUSD", "XAU", "USD", "metal", swap_long=-0.036, swap_short=0.01, **kw),))


def week_market(u, days=4):
    return market(u, np.full(days * 1440, 2000.0), start=MON_2024_01_08)


def test_swaps_mon_tue_wed_triple():
    u = u_swap(triple_swap_weekday=2)
    md = week_market(u)
    from tradeworld.agents.baselines import HoldPolicy
    from tradeworld.interface.actions import ActionMapper
    acct = run(md, make_profile(u), {"a": HoldPolicy(act(1, [1.0], 1.0), ActionMapper.primary(1))},
               interval=60, resolution=60, window=1).accounts["a"]
    assert len(fills(acct)) == 1                                       # held, never rebalanced
    sw = fills(acct, "swap")
    assert [e.data["mult"] for e in sw] == [1, 1, 3, 1]               # Mon, Tue, Wed (triple), Thu
    notional = 10.0                                                   # 1x of $10 at a constant price
    assert sum(e.data["amount"] for e in sw) == pytest.approx(notional * -0.036 / 360 * 6, rel=1e-6)
    from datetime import date
    assert sw[0].t == rollover_minute(date(2024, 1, 8))                # 22:00 UTC in winter


def test_swap_free_and_short_rate():
    md = week_market(u_swap(swap_free=True))
    acct = run(md, make_profile(u_swap(swap_free=True)), {"a": ScriptedPolicy({0: act(1, [1.0])}, 1)},
               interval=60, resolution=60, window=1).accounts["a"]
    assert not fills(acct, "swap")
    u = u_swap()
    acct = run(week_market(u), make_profile(u), {"a": ScriptedPolicy({0: act(1, [-1.0])}, 1)},
               interval=60, resolution=60, window=1).accounts["a"]
    assert all(e.data["amount"] > 0 for e in fills(acct, "swap"))


def test_no_swap_if_not_held_across_rollover():
    u = u_swap()
    md = market(u, np.full(1440, 2000.0), start=MON_2024_01_08)
    # open at 18:00 decision, close at 21:00 -> before the 22:00 UTC rollover
    roll = rollover_minute(__import__("datetime").date(2024, 1, 8)) - MON_2024_01_08
    script = {17: act(1, [1.0]), 20: act(1, [0.0])}                    # hourly decisions
    acct = run(md, make_profile(u), {"a": ScriptedPolicy(script, 1)}, interval=60, resolution=60, window=1).accounts["a"]
    assert fills(acct) and not fills(acct, "swap")
    assert roll == 22 * 60
