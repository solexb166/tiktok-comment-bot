from datetime import datetime, timezone

import numpy as np
import pytest

from tradeworld.market.clock import to_minute
from tradeworld.market.instruments import InstrumentSpec, Universe, default_specs

MON_2024_01_08 = to_minute(datetime(2024, 1, 8, tzinfo=timezone.utc))   # a Monday, winter


@pytest.fixture
def universe6():
    return Universe(default_specs())


def single(symbol="EURUSD", base="EUR", quote="USD", asset="fx", **kw):
    return Universe((InstrumentSpec(symbol, base, quote, asset, **kw),))


# ----------------------------------------------------------------------------- helpers
import math

from tradeworld.accounts.margin import MarginModel
from tradeworld.brokerspec.profile import EnvProfile, SlippageModel
from tradeworld.data.synthetic import from_mid_paths
from tradeworld.world.world import AgentSpec, World

ZERO_SLIP = SlippageModel(0.0, 0.0, 0.0)


def make_profile(universe, *, leverage=math.inf, slippage=ZERO_SLIP, **kw):
    kw.setdefault("margin", MarginModel(account_leverage=leverage))
    return EnvProfile(name="test", environment=kw.pop("environment", "A"), universe=universe,
                      slippage=slippage, **kw)


def market(universe, mids, spread_bps=0.0, range_bps=0.0, start=MON_2024_01_08, quoted=None):
    mids = np.asarray(mids, dtype=float)
    if mids.ndim == 1:
        mids = mids[:, None]
    return from_mid_paths(universe, start, mids, spread_bps=spread_bps, quoted=quoted, range_bps=range_bps)


def run(md, profile, policies: dict, *, seed=0, interval=1, resolution=1, window=1, mappers=None, **kw):
    """Run a small world. With interval=resolution=window=1 the first decision is at M1 row 1,
    decision k happens at row 1 + k and fills at the open of row 2 + k."""
    mappers = mappers or {}
    agents = [AgentSpec(k, p, mappers.get(k)) for k, p in policies.items()]
    return World(md, profile, agents, seed=seed, decision_interval=interval,
                 obs_resolution=resolution, window=window, **kw).run()


def act(n, weights, gross=1.0):
    """Action that targets exposure G*d with d = weights (sum|d| <= 1) and gross level G."""
    from tradeworld.interface.actions import ActionMapper
    return np.r_[np.asarray(weights, float), ActionMapper.primary(n).u_for_gross(gross)]


def fills(acct, kind="fill"):
    return [e for e in acct.events if e.kind == kind]
