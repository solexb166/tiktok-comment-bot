"""Experiment M3: the cost-aware model-based agent (model_kelly3), its reference, ablation and registry integration."""
import math

import numpy as np
import pytest
import torch

from tradeworld.learning import evaluate as ev
from tradeworld.learning import m3
from tradeworld.learning import m3_analysis as an
from tradeworld.learning import model_kelly2 as mk2
from tradeworld.learning import model_kelly3 as mk3
from tradeworld.learning import synthetic_tasks as st
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.registry.registry import Registry
from tradeworld.registry.store import LocalBlobStore

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
Z, W = np.polynomial.hermite_e.hermegauss(32)
WQ = W / math.sqrt(2 * math.pi)

def test_earlier_markets_unchanged():
    import json
    import subprocess
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    old = subprocess.run(["git", "-C", str(root), "show", "origin/claude/phase3-m2:trading-world/src/tradeworld/learning/synthetic_tasks.py"],
                         capture_output=True, text=True)
    if old.returncode != 0:
        pytest.skip("M2 branch not available")
    ns = {}
    exec(compile(old.stdout.replace("from ..", "from tradeworld."), "old_tasks", "exec"), ns)
    for t in st.TASKS:
        assert st.build(t, 11, 3000).md.content_hash() == ns["build"](t, 11, 3000).md.content_hash(), t


def test_new_markets_have_the_planned_costs():
    a, b = st.build("S1C", 3, 2000), st.build("S2C", 3, 2000)
    assert a.params["spread"] == pytest.approx(0.4 * st.SIGMA) and a.params["phi"] == -0.3
    assert b.params["spread"] == pytest.approx(1.0 * st.SIGMA) and b.params["mean_regime_len"] == 400
    assert a.md.content_hash() != st.build("S1", 3, 2000).md.content_hash()


@pytest.mark.parametrize("task", ["S1C", "S2C", "S3"])
def test_observation_reader_recovers_spread_position_and_last_return(task):
    t = st.build(task, 5, 3000)
    env = TradeWorldEnv(t.md, t.profile, episode_decisions=50, extra=t.extra, random_start=False, **ENV)
    rd = mk3.ObsReader(env.encoder)
    o, _ = env.reset(seed=0, options={"start_decision": 0})
    mid = t.md.mid_c()[:, 0]
    assert rd.half_spread(o) == pytest.approx(t.params["spread"] / 2, rel=1e-3)
    assert rd.exposure(o) == 0.0
    shim = type("S", (), {"mapper": env.mapper})()
    for G in (37.5, -12.0, 80.0):
        o, *_ = env.step(mk2.Agent.action(shim, G))
        d = int(env.dec[env.k])
        p = env.view.prices_close(d)
        true_x = float(env.account.exposure(p)[0])
        assert rd.exposure(o) == pytest.approx(true_x, rel=2e-3)
        std = 10.0 ** (4.0 * (float(o[rd.vol_col]) - 1.0))
        last = float(o[(env.encoder.W - 1) * env.encoder.F]) * std
        assert last == pytest.approx(math.log(mid[d] / mid[d - 1]), rel=2e-3, abs=1e-9)


def test_decision_band():
    s, mu = 1e-3, 2e-5
    pi, m, sd = np.ones((5, 1)), np.full((5, 1), mu), np.full((5, 1), s)
    G, lo, hi = mk3.decide(pi, m, sd, np.array([0.0, 18.0, 25.0, 30.0, -5.0]), 5e-6, Z, WQ, 2000.0, 100)
    assert G == pytest.approx([15.5, 18.0, 25.0, 25.5, 15.5], abs=0.05)       # move to band edge or stay inside
    G0, _, _ = mk3.decide(pi, m, sd, np.zeros(5), 0.0, Z, WQ, 2000.0, 100)
    G2 = mk2.kelly_sizes(pi, m, sd, Z, WQ, 2000.0, 100)
    assert G0 == pytest.approx(G2, rel=1e-6)                                   # zero cost: exactly version 2's rule


def test_calibration_shrinks_noise_and_keeps_signal():
    rng = np.random.default_rng(0)
    n = 40000
    sig = rng.standard_normal(n) * 1e-4
    real = sig + rng.standard_normal(n) * 1e-3
    good = mk3.calibration(sig, real)
    assert 0.8 < good["slope"] < 1.2 and good["k"] > 0.75
    noise = mk3.calibration(rng.standard_normal(n) * 1e-4, real)
    assert noise["k"] < 0.3
    over = mk3.calibration(3 * sig, real)                                      # predictions three times too strong
    assert 0.25 < over["k"] < 0.45


def test_paired_death_test():
    a = [True] * 6 + [False] * 114
    r = [True] * 2 + [False] * 118
    d = an.paired_death_test(a, r)
    assert d["agent_only"] == 4 and d["reference_only"] == 0 and d["p"] == pytest.approx(1 / 16)
    assert an.paired_death_test([False] * 5, [False] * 5)["p"] == 1.0
    with pytest.raises(ValueError):
        an.paired_death_test([True], [True, False])


def test_reference_trades_s1c_against_the_last_move_and_stays_flat_in_s3():
    for task in ("S1C", "S3"):
        ident = dict(task=task, frozen=dict(reward="R1", env_settings=ENV, action_space=LOG10, task=task, episode_decisions=50),
                     action_space=LOG10)
        pol = mk3.reference_policy(ident)
        t = st.build(task, 7, 3000)
        env = TradeWorldEnv(t.md, t.profile, episode_decisions=200, extra=t.extra, random_start=False, **ENV)
        rd = mk3.ObsReader(env.encoder)
        o, _ = env.reset(seed=0, options={"start_decision": 0})
        mid = t.md.mid_c()[:, 0]
        strong, side = 0, 0
        for _ in range(150):
            d = int(env.dec[env.k])
            last = math.log(mid[d] / mid[d - 1])
            o, *_ = env.step(pol(o))
            x = float(env.account.exposure(env.view.prices_close(d))[0])
            if task == "S3":
                assert x == 0.0
            elif abs(-0.3 * last) > t.params["spread"] / 2 * 1.3:
                strong += 1
                side += np.sign(x) == -np.sign(last)
        if task == "S1C":
            assert strong > 20 and side == strong


def test_train_evaluate_ablation_reference_and_regenerate(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(m3, "TRAIN_DATA_STEPS", 4000)
    monkeypatch.setattr(m3, "DATA_POINTS", (1500, 4000 - 40))
    monkeypatch.setattr(mk3, "CONFIG", dict(mk3.CONFIG, max_epochs=5))
    monkeypatch.setattr(ev, "TEST_STEPS_DATA", 3000)
    monkeypatch.setattr(ev, "TEST_EPISODES", 2)
    reg = Registry(tmp_path / "reg")
    a = reg.create_agent(task="S2C", reward="R1", seed=5, env_settings=ENV, action_space=LOG10, profile="A_synthetic",
                         episode_decisions=200, extra=dict(agent_type="model-kelly-v3"))
    v = m3.train_agent(reg, a, 5)
    assert v["final_checkpoint"]["step"] == 4000 - 40 and "persistence" in v["final_checkpoint"]
    data = [dict(task="S2C", seed=777, data_hash=st.build("S2C", 777, 3000).md.content_hash())]
    reg.record_round("r1", dict(kind="fresh", data=data, test_steps=3000))
    e = ev.evaluate(reg, a, v["version"], "r1")
    assert e["decisions"] > 0 and e["checkpoint_sha256"] == v["final_checkpoint"]["model_sha256"]
    # the cost-blind ablation decides from the same model
    b = reg.create_agent(task="S2C", reward="R1", seed=5, env_settings=ENV, action_space=LOG10, profile="A_synthetic",
                         episode_decisions=200, extra=dict(agent_type="model-kelly-v3-costblind", ablation_of=a))
    n = reg.start_training(b, None, None, {}, info=dict(ablation=True))
    reg.record_version(b, dict(version=n, parent_version=None, kind="initial", reason="ablation", changes={},
                               training=dict(algorithm="none"), checkpoints=[], final_checkpoint=dict(v["final_checkpoint"]),
                               history={}, code_version="test"))
    eb = ev.evaluate(reg, b, 1, "r1")
    assert eb["checkpoint_sha256"] == e["checkpoint_sha256"]
    assert sum(ep["costs"] for ep in eb["episodes"]) >= sum(ep["costs"] for ep in e["episodes"]) - 1e-9
    # the true-parameter reference
    r = reg.create_agent(task="S2C", reward="R1", seed=0, env_settings=ENV, action_space=LOG10, profile="A_synthetic",
                         episode_decisions=200, extra=dict(agent_type="reference-cost-aware"))
    n = reg.start_training(r, None, None, {}, info=dict(reference=True))
    sha = reg.blobs.put(b"{}", "json")
    reg.record_version(r, dict(version=n, parent_version=None, kind="initial", reason="reference", changes={},
                               training=dict(algorithm="none"), checkpoints=[],
                               final_checkpoint=dict(step=0, model_sha256=sha, state_sha256=None, kind="final"),
                               history={}, code_version="test"))
    er = ev.evaluate(reg, r, 1, "r1")
    assert np.mean([ep["flag_agreement"] for ep in er["episodes"]]) > 0.9
    reg.cache = LocalBlobStore(tmp_path / "fresh-cache")
    for x in (e, eb, er):
        assert len(ev.decision_log(reg, x)) == x["decisions"]                 # regenerates every decision exactly
