"""Experiment R-FL: the new full-liability reward R6, protection of existing rewards and agents, and the
pre-registered analysis rules."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from tradeworld.learning import evaluate as ev
from tradeworld.learning import rfl
from tradeworld.learning import rfl_analysis as an
from tradeworld.learning.rewards import REWARDS, make_reward
from tradeworld.registry.registry import ROOT, Registry
from tradeworld.registry.store import LocalBlobStore


def _r(name, e0_init=10.0):
    r = make_reward(name)
    r.reset(e0_init)
    return r


def test_r6_equals_r1_on_every_bar_that_does_not_end_the_account():
    rng = np.random.default_rng(0)
    r1, r6 = _r("R1"), _r("R6")
    for _ in range(5000):
        e_prev = 10.0 * rng.uniform(0.06, 20.0)
        e_now = e_prev * rng.uniform(0.01, 3.0)
        if e_now <= 0.05 * 10.0:                    # the account dies on this bar: excluded
            continue
        assert r6(e_prev, e_now, False) == pytest.approx(r1(e_prev, e_now, False), abs=1e-12)


@pytest.mark.parametrize("name,c", [("R6", 0.01), ("R6-c0.05", 0.05)])
def test_r6_tangent_extension(name, c):
    r = _r(name)
    assert r(10.0, 10.0 * c, False) == pytest.approx(math.log(c))            # continuous at c
    for m in (c * 0.5, 0.0, -1.0, -4.5):
        assert r(10.0, 10.0 * m, False) == pytest.approx(math.log(c) + (m - c) / c)
    ms = np.linspace(-5, 2, 400)
    vals = [r(10.0, 10.0 * m, False) for m in ms]
    assert all(np.isfinite(vals)) and np.all(np.diff(vals) > 0)              # finite, strictly increasing
    for m in (c * 0.9, 0.0, -2.0):                                            # never harsher than true log
        assert r(10.0, 10.0 * m, False) >= (math.log(m) if m > 0 else -math.inf)


def test_r6_charges_more_for_a_larger_overshoot_and_r1_does_not():
    r1, r6 = _r("R1"), _r("R6")
    crash = lambda G: 10.0 * (1 - 0.10 * G)                                   # 10% crash at size G
    assert r1(10.0, crash(15), False) == pytest.approx(r1(10.0, crash(55), False))   # R1: flat beyond ruin
    assert r6(10.0, crash(55), False) < r6(10.0, crash(15), False) < r6(10.0, crash(9.95), False)


def test_existing_rewards_unchanged():
    assert {"R1", "R2", "R3", "R4", "R5"} <= set(REWARDS)
    r1 = _r("R1")
    assert r1(10.0, 12.0, False) == pytest.approx(math.log(1.2))
    assert r1(10.0, -5.0, False) == pytest.approx(math.log(0.1) - math.log(10.0))
    r2 = _r("R2")
    assert r2(10.0, 12.0, False) == pytest.approx(0.2)


def test_env_runs_with_r6():
    from tradeworld.learning import synthetic_tasks as st
    from tradeworld.learning.env import TradeWorldEnv
    t = st.build("S6", 0, 3000)
    env = TradeWorldEnv(t.md, t.profile, reward="R6", episode_decisions=50, **t.env_kwargs)
    env.reset(seed=0)
    for _ in range(20):
        _, r, te, tr, info = env.step(np.array([1.0, 0.1]))
        assert math.isfinite(r)


def test_groups_differ_only_in_reward():
    g = rfl.groups()
    assert [c["reward"] for c in g.values()] == ["R1", "R6", "R6-c0.05"]
    strip = lambda c: {k: v for k, v in c.items() if k not in ("reward", "description")}
    assert strip(g["R1"]) == strip(g["R6"]) == strip(g["R6-c0.05"])


@pytest.mark.parametrize("root,key", [("registry", ("S6", "R1", 2)), ("registry-phase3b", "C2-up")])
def test_earlier_agents_regenerate_identical_decisions(tmp_path, root, key):
    reg = Registry(ROOT / root)
    reg.cache = LocalBlobStore(tmp_path / "cache")
    if root == "registry":
        a = reg.find(*key)
        e = next(x for x in reg.evaluations(a) if x["round_id"] == "gate-1")
    else:
        m = json.loads((ROOT / root / "experiments" / "phase3-exp1b" / "manifest.json").read_text())
        a = next(x["agent_id"] for x in m["agents"].values() if x["group"] == key)
        e = next(x for x in reg.evaluations(a) if x["round_id"] == "exp1b-final-a")
    assert len(ev.decision_log(reg, e)) == e["decisions"]


# ---------------------------------------------------------------------------- analysis rules
def _seed(i, **kw):
    s = dict(seed_index=i, ruinous=0.0, deaths=0, median_size=2.9, q2=0.9, growth_ok=True, tilt=1.1, w7=True,
             train_gap=0.0, boundary=0.0, rejected=0.0, safeguard_share=0.0, nonruinous_median=2.8,
             recovery_after=2.8, recovery_else=2.9, recovery_n=5, stability=dict(finite=True))
    s.update(kw)
    return s


def test_interpretation_and_verdicts():
    seeds = {"R1": [_seed(i, ruinous=0.05, q2=-2.0) for i in range(1, 7)],
             "R6": [_seed(i) for i in range(1, 7)],
             "R6-c0.05": [_seed(i, median_size=0.6, q2=0.3, nonruinous_median=0.6) for i in range(1, 7)]}
    v = an.verdicts(seeds, 2.93)
    assert v["groups"]["R1"]["interpretation"] == "still enters the ruin zone"
    assert v["groups"]["R6"]["interpretation"] == "safe growth"
    assert v["groups"]["R6-c0.05"]["interpretation"] == "safe but timid (learning problem)"
    assert v["truncation_confirmed_as_cause"]
    assert v["groups"]["R6"]["T"]["ok"] and not v["groups"]["R6-c0.05"]["T"]["ok"]
    assert v["groups"]["R6"]["GR"]["ok"] is True                      # 30 decisions after big losses, ratio ~1
    assert all("p_holm" in c for g in an.NEW for c in v["groups"][g]["comparisons"].values())


def test_recovery_sizes():
    logs = pd.DataFrame(dict(round="a", episode=0, pnl=[0.0, -3.0, 0.1, 0.0], equity_before=[10.0, 10.0, 7.0, 7.1],
                             leverage_after_fill=[1.0, 1.0, 5.0, 1.0]))
    after, other, n = an.recovery_sizes(logs)
    assert n == 1 and after == pytest.approx(5.0) and other == pytest.approx(1.0)
