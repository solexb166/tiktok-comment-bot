"""Group 5: action mapping (uncapped X1), benchmark caps, rounding, invalid actions, flags."""
import math

import numpy as np
import pytest

from tradeworld.agents.baselines import ScriptedPolicy
from tradeworld.execution.engine import round_to_lots
from tradeworld.interface.actions import ActionMapper
from tradeworld.market.instruments import InstrumentSpec
from conftest import act, fills, make_profile, market, run, single


def test_gross_mapping():
    m = ActionMapper.primary(3)
    assert m.gross(0.0) == pytest.approx(1.0)
    assert m.gross(1.0) == pytest.approx(1e4) and m.gross(-1.0) == pytest.approx(1e-4)
    assert m.gross(0.25) == pytest.approx(10.0)                    # one decade per 0.25
    for G in (1e-3, 0.5, 3.0, 777.0):
        assert m.gross(m.u_for_gross(G)) == pytest.approx(G, rel=1e-12)


def test_weights_formula():
    m = ActionMapper.primary(3)
    r = m.map([0.2, -0.3, 0.1, 0.25])                              # sum|d| = 0.6 <= 1
    assert np.allclose(r.w, 10.0 * np.array([0.2, -0.3, 0.1]))
    r = m.map([0.8, -0.8, 0.4, 0.0])                               # sum|d| = 2 -> normalized
    assert np.isclose(np.abs(r.w).sum(), 1.0) and np.allclose(r.w, [0.4, -0.4, 0.2])
    assert np.all(m.map([0, 0, 0, 0.9]).w == 0)                    # d = 0 -> flat at any G


def test_no_behavioural_cap_in_primary():
    m = ActionMapper.primary(1)
    assert m.cap is None
    assert np.isclose(m.map([1.0, 1.0]).w[0], 1e4)


@pytest.mark.parametrize("cap", [3, 10, 50, 100])
def test_benchmark_caps(cap):
    m = ActionMapper.benchmark(2, cap)
    assert m.gross(1.0) == pytest.approx(cap) and m.gross(-1.0) == pytest.approx(1e-4)
    us = np.linspace(-1, 1, 101)
    g = [m.gross(x) for x in us]
    assert np.all(np.diff(g) > 0)                                   # rescaled, not clipped


def test_invalid_and_flags():
    m = ActionMapper.primary(2)
    assert not m.map([np.nan, 0, 0]).valid
    assert not m.map([np.inf, 0, 0]).valid
    assert not m.map([0, 0]).valid                                   # wrong shape
    assert "clipped_action" in m.map([2.0, 0, 0]).safeguards
    assert "near_ceiling" in m.map([0.1, 0, 0.99]).safeguards


def test_invalid_action_leaves_positions_unchanged():
    u = single()
    pol = ScriptedPolicy({0: act(1, [1.0], 2.0), 2: [np.nan, np.nan]}, 1)
    res = run(market(u, np.full(8, 1.1)), make_profile(u), {"a": pol})
    acct = res.accounts["a"]
    assert len(fills(acct)) == 1 and acct.units[0] > 0
    assert any(e.kind == "invalid_action" for e in acct.events)


def test_reversal_closes_and_reopens():
    u = single()
    pol = ScriptedPolicy({0: act(1, [1.0], 3.0), 2: act(1, [-1.0], 3.0)}, 1)
    acct = run(market(u, np.full(8, 1.1)), make_profile(u), {"a": pol}).accounts["a"]
    assert acct.units[0] < 0 and acct.closed_trades == 1
    assert acct.units[0] == pytest.approx(-30.0 / 1.1, rel=1e-9)


def test_lot_rounding_toward_zero_and_min_lot():
    s = InstrumentSpec("X", "EUR", "USD", "fx", contract_size=1000.0, lot_step=0.01, min_lot=0.01)
    assert round_to_lots(s, 19.9) == pytest.approx(10.0)            # 0.0199 lots -> 0.01
    assert round_to_lots(s, -19.9) == pytest.approx(-10.0)
    assert round_to_lots(s, 9.9) == 0.0                            # below min lot
    assert round_to_lots(s, 1234.5) == pytest.approx(1230.0)


def test_below_min_lot_is_reported():
    u = single("EURUSDc", contract_size=1000.0, lot_step=0.01, min_lot=0.01)
    pol = ScriptedPolicy({0: act(1, [1.0], 0.5)}, 1)                  # 0.5x of $10 = $5 < 0.01 lot
    res = run(market(u, np.full(6, 1.1)), make_profile(u), {"a": pol})
    assert "below_min_lot" in res.records.iloc[0]["rejections"]
    assert not fills(res.accounts["a"])
