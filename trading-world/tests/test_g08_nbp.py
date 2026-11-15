"""Group 8: negative balance protection (profile B behaviour) vs. no-NBP (A and stress)."""
import math

import numpy as np
import pytest

from tradeworld.accounts.account import Account
from tradeworld.agents.baselines import ConstantPolicy
from tradeworld.execution.engine import MarketView
from tradeworld.interface.observation import build_observation
from tradeworld.replay.replay import Replay
from conftest import act, fills, make_profile, market, run, single


def gap_market(u):
    md = market(u, np.full(7, 1.1))
    for f in ("bid_o", "ask_o", "bid_l", "ask_l", "bid_h", "ask_h", "bid_c", "ask_c"):
        getattr(md, f)[4:] = 1.05                                  # -4.5% gap at bar 4
    return md


def test_nbp_resets_negative_balance_to_zero():
    u = single()
    acct = run(gap_market(u), make_profile(u, nbp=True), {"a": ConstantPolicy(act(1, [1.0], 1000.0))}).accounts["a"]
    assert acct.death_reason == "gap_through"
    assert acct.balance == 0.0
    ev = fills(acct, "nbp_reset")
    assert len(ev) == 1 and ev[0].data["debt"] < -400


def test_without_nbp_debt_is_kept():
    u = single()
    acct = run(gap_market(u), make_profile(u, nbp=False), {"a": ConstantPolicy(act(1, [1.0], 1000.0))}).accounts["a"]
    assert acct.balance < -400 and not fills(acct, "nbp_reset")
    assert fills(acct, "negative_balance")


def test_negative_equity_observation_is_finite_and_floored():
    u = single()
    md = market(u, np.full(5, 1.1))
    prof = make_profile(u)
    acct = Account("a", prof, np.random.default_rng(0))
    acct.balance = -3.0
    obs = build_observation(Replay(md, 1, 1, 1), MarketView(md, prof), acct, 2)
    assert np.all(np.isfinite(obs.account))
    assert obs.account[0] == pytest.approx(math.log(0.01))
