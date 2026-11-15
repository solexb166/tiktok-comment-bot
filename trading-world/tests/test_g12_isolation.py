"""Group 12: an agent's results are identical whether it runs alone or alongside others."""
import numpy as np

from tradeworld.agents.baselines import ConstantPolicy, RandomPolicy
from tradeworld.brokerspec import load_profile
from tradeworld.data.synthetic import generate
from conftest import MON_2024_01_08, act, run


def test_alone_vs_crowd_identical():
    prof = load_profile("environments/A_fractional.yaml")
    md = generate(prof.universe, MON_2024_01_08, 2 * 1440, seed=9)
    kw = dict(seed=4, interval=5, resolution=5, window=12)
    alone = run(md, prof, {"b": RandomPolicy(6, 3)}, **kw)
    crowd = run(md, prof, {"a": RandomPolicy(6, 1), "b": RandomPolicy(6, 3),
                           "c": ConstantPolicy(act(6, [1, 0, 0, 0, 0, 0], 3000.0))}, **kw)
    assert alone.event_hash("b") == crowd.event_hash("b")
    ra = alone.records.query("agent == 'b'").reset_index(drop=True).drop(columns=["action_raw"])
    rc = crowd.records.query("agent == 'b'").reset_index(drop=True).drop(columns=["action_raw"])
    assert ra.equals(rc)


def test_observations_are_read_only():
    import pytest
    from tradeworld.accounts.account import Account
    from tradeworld.execution.engine import MarketView
    from tradeworld.interface.observation import build_observation
    from tradeworld.replay.replay import Replay
    prof = load_profile("environments/A_fractional.yaml")
    md = generate(prof.universe, MON_2024_01_08, 1440, seed=1)
    rp = Replay(md, 5, 5, 12)
    obs = build_observation(rp, MarketView(md, prof), Account("x", prof, np.random.default_rng(0)),
                            int(rp.decision_indices[0]))
    for arr in (obs.market, obs.account, obs.exposure):
        with pytest.raises(ValueError):
            arr[...] = 0
