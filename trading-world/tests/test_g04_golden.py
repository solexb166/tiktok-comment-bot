"""Group 4: hand-computed golden scenarios (fractional, JPY conversion, Cent USC factor, extreme leverage)."""
from fractions import Fraction

import numpy as np
import pytest

from tradeworld.agents.baselines import ScriptedPolicy
from tradeworld.market.instruments import InstrumentSpec, Universe
from conftest import act, fills, make_profile, market, run, single


def test_eurusd_fractional_two_x():
    u = single()
    mids = [1.1, 1.1, 1.1, 1.111, 1.111]
    pol = ScriptedPolicy({0: act(1, [1.0], 2.0), 2: act(1, [0.0])}, 1)
    acct = run(market(u, mids), make_profile(u), {"a": pol}).accounts["a"]
    # units = 2 * 10 / 1.1 ; pnl = units * 0.011 = 0.2
    assert acct.balance == pytest.approx(10.2, abs=1e-12)


def test_usdjpy_pnl_converted_at_exit_rate():
    u = Universe((InstrumentSpec("USDJPY", "USD", "JPY", "fx"),))
    mids = [150.0, 150.0, 150.0, 151.0, 151.0]
    pol = ScriptedPolicy({0: act(1, [1.0], 1.0), 2: act(1, [0.0])}, 1)
    acct = run(market(u, mids), make_profile(u, initial_equity_usd=1000.0), {"a": pol}).accounts["a"]
    f = fills(acct)
    assert f[0].data["du"] == pytest.approx(1000.0)              # 1x of $1,000 = 1,000 USD
    assert acct.balance == pytest.approx(1000.0 + 1000.0 * 1.0 / 151.0, rel=1e-12)


def cent(symbol, base, quote, asset, contract):
    return InstrumentSpec(symbol, base, quote, asset, contract_size=contract, lot_step=0.01,
                          min_lot=0.01, max_lots=200, account_type="cent_test")


def test_cent_eurusd_usc_factor_of_100():
    u = Universe((cent("EURUSDc", "EUR", "USD", "fx", 1000.0),))
    prof = make_profile(u, environment="B", account_currency="USC", currency_per_usd=100.0)
    # 1.2x of $10 at 1.1 -> 10.909 EUR -> 0.0109 lots -> rounds to 0.01 lot = 10 EUR
    pol = ScriptedPolicy({0: act(1, [1.0], 1.2), 2: act(1, [0.0])}, 1)
    acct = run(market(u, [1.1, 1.1, 1.1, 1.101, 1.101]), prof, {"a": pol}).accounts["a"]
    assert fills(acct)[0].data["du"] == pytest.approx(10.0)
    usc_pnl = (acct.balance - 10.0) * prof.currency_per_usd
    assert usc_pnl == pytest.approx(1.0, abs=1e-9)                 # 10 pips on 0.01 lot = 1 USC


def test_cent_xau_one_dollar_move():
    u = Universe((cent("XAUUSDc", "XAU", "USD", "metal", 1.0),))
    prof = make_profile(u, environment="B", currency_per_usd=100.0)
    pol = ScriptedPolicy({0: act(1, [1.0], 50.0), 2: act(1, [0.0])}, 1)   # $500 -> 0.25 oz -> 0.25 lots
    acct = run(market(u, [2000.0, 2000.0, 2000.0, 2001.0, 2001.0]), prof, {"a": pol}).accounts["a"]
    assert fills(acct)[0].data["du"] == pytest.approx(0.25)
    assert (acct.balance - 10.0) * 100 == pytest.approx(25.0)      # 0.25 oz * $1 = $0.25 = 25 USC


@pytest.mark.parametrize("lev", [2000.0, 10000.0])
def test_extreme_leverage_precision(lev):
    u = single()
    pol = ScriptedPolicy({0: act(1, [1.0], lev), 2: act(1, [0.0])}, 1)
    mids = [1.1, 1.1, 1.1, 1.1001, 1.1001]
    acct = run(market(u, mids), make_profile(u), {"a": pol}).accounts["a"]
    units = Fraction(lev) * 10 / Fraction("1.1")
    expected = 10 + units * (Fraction("1.1001") - Fraction("1.1"))
    assert acct.balance == pytest.approx(float(expected), rel=1e-12)
    assert acct.alive
