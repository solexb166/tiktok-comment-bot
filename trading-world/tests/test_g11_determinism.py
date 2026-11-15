"""Group 11: same inputs and seed => bit-identical event log; the world seed changes slippage only."""
import numpy as np

from tradeworld.agents.baselines import ConstantPolicy, RandomPolicy
from tradeworld.brokerspec import load_profile
from tradeworld.data.synthetic import generate
from conftest import MON_2024_01_08, act, run


def _world(seed_world, seed_policy=7):
    prof = load_profile("environments/A_fractional.yaml")
    md = generate(prof.universe, MON_2024_01_08, 2 * 1440, seed=5)
    return run(md, prof, {"r": RandomPolicy(6, seed_policy), "c": ConstantPolicy(act(6, [0.2, -0.2, 0.1, 0, 0.1, 0], 20.0))},
               seed=seed_world, interval=5, resolution=5, window=12)


def test_same_seed_identical_hash():
    a, b = _world(1), _world(1)
    for agent in ("r", "c"):
        assert a.event_hash(agent) == b.event_hash(agent)
    assert a.records.drop(columns=["action_raw"]).equals(b.records.drop(columns=["action_raw"]))


def test_world_seed_changes_only_slippage():
    a, b = _world(1), _world(2)
    assert a.event_hash("c") != b.event_hash("c")
    ea = [e for e in a.accounts["c"].events if e.kind == "fill"]
    eb = [e for e in b.accounts["c"].events if e.kind == "fill"]
    # same decisions -> same fill times and instruments; prices differ only by slippage
    assert [(e.t, e.data["i"]) for e in ea[:20]] == [(e.t, e.data["i"]) for e in eb[:20]]
    assert any(x.data["price"] != y.data["price"] for x, y in zip(ea, eb))
