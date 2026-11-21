"""Experiment M4: switching markets, the regime-tracking agent (model_kelly4), its references, ablation and registry."""
import math

import numpy as np
import pytest
import torch

from tradeworld.learning import evaluate as ev
from tradeworld.learning import m4
from tradeworld.learning import model_kelly2 as mk2
from tradeworld.learning import model_kelly4 as mk4
from tradeworld.learning import synthetic_tasks as st
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.registry.registry import Registry
from tradeworld.registry.store import LocalBlobStore

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")


def _ident(task, episodes=1000):
    return dict(task=task, frozen=dict(task=task, reward="R1", env_settings=ENV, action_space=LOG10, episode_decisions=episodes,
                                       profile="A_synthetic"), action_space=LOG10)


def test_earlier_markets_unchanged():
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    old = subprocess.run(["git", "-C", str(root), "show", "origin/claude/phase3-m3:trading-world/src/tradeworld/learning/synthetic_tasks.py"],
                         capture_output=True, text=True)
    if old.returncode != 0:
        pytest.skip("M3 branch not available")
    ns = {}
    exec(compile(old.stdout.replace("from ..", "from tradeworld."), "old_tasks", "exec"), ns)
    for t in st.TASKS + st.COST_TASKS:
        assert st.build(t, 11, 3000).md.content_hash() == ns["build"](t, 11, 3000).md.content_hash(), t


def test_switching_markets_switch_and_hide_the_regime():
    for t, vals in (("S1X", (-0.3, 0.3)), ("S2X", (-1.0, 1.0)), ("S6X", (1 / 300, 1 / 50))):
        k = st.build(t, 4, 40000)
        h = k.hidden[:, 0]
        assert set(np.round(np.unique(h), 6)) == set(np.round(vals, 6))
        switches = int((np.diff(h) != 0).sum())
        assert 3 <= switches <= 40, (t, switches)
    assert st.build("S2X", 4, 2000).extra is not None and st.build("S1X", 4, 2000).extra is None


@pytest.mark.parametrize("task", ["S1X", "S2X"])
def test_features_read_last_returns_and_flag_from_the_observation(task):
    t = st.build(task, 5, 3000)
    env = TradeWorldEnv(t.md, t.profile, episode_decisions=50, extra=t.extra, random_start=False, **ENV)
    f = mk4.Features(env.encoder)
    o, _ = env.reset(seed=0, options={"start_decision": 0})
    mid = t.md.mid_c()[:, 0]
    for _ in range(5):
        o, *_ = env.step(np.array([0.0, -1.0]))
    d = int(env.dec[env.k])
    x = f.from_full(o)
    assert x[0] == 1.0
    for lag in (1, 2, 3):
        assert x[lag] == pytest.approx(math.log(mid[d - lag + 1] / mid[d - lag]), rel=2e-3, abs=1e-9)
    if task == "S2X":
        assert x[-1] == t.extra[d, 0]


def test_regime_fit_recovers_two_patterns():
    rng = np.random.default_rng(0)
    n, s = 40000, 1e-3
    phi = np.where((np.arange(n) // 4000) % 2 == 0, -0.3, 0.3)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi[i] * r[i - 1] + s * rng.standard_normal()
    y = r[1:]
    X = np.column_stack([np.ones(n - 1), r[:-1], np.zeros(n - 1), np.zeros(n - 1)])
    pi0, m, sd = mk2.em(y, 1, 100, 1e-9)
    best = max((mk4.fit_regimes(y, X, m, sd, pi0, 2, mk4.CONFIG, seed=k) for k in range(3)), key=lambda f: f["loglik"])
    assert sorted(np.round(best["beta"][:, 1], 1)) == [-0.3, 0.3]
    assert all(1 / (1 - best["A"][k, k]) > 1000 for k in range(2))


def test_static_shrinkage_removes_chance_patterns():
    rng = np.random.default_rng(1)
    n = 30000
    y = 1e-3 * rng.standard_normal(n)
    X = np.column_stack([np.ones(n), 1e-3 * rng.standard_normal((n, 3))])
    art, _ = mk4.learn(X, y, 0, dict(mk4.CONFIG, max_regimes=2, regime_starts=2, regime_iters=10))
    assert art["K"] == 1
    assert np.all(np.abs(art["model"]["beta"][0][1:]) < 0.02)


def test_agent_starts_each_run_afresh_and_is_deterministic():
    t = st.build("S1X", 3, 4000)
    ident = _ident("S1X", 200)
    Xm, y, cols = mk4.collect(t, ident)
    X = mk4.Features(mk4._encoder(ident)).from_market_columns(Xm, cols)
    art, _ = mk4.learn(X, y, 0, dict(mk4.CONFIG, max_regimes=2, regime_starts=1, regime_iters=5))
    agent = mk4.Agent(art, ev.mapper_from(ident), mk4._encoder(ident))
    test = st.build("S1X", 77, 3000)
    a, _ = ev.run_recorded(agent, ident, test, [0, 1000])
    b, _ = ev.run_recorded(agent, ident, test, [0, 1000])
    assert [r["target_exposure"] for r in a] == [r["target_exposure"] for r in b]


def test_train_evaluate_references_ablation_and_regenerate(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(m4, "TRAIN_DATA_STEPS", 6000)
    monkeypatch.setattr(m4, "DATA_POINTS", (2000, 6000 - 40))
    monkeypatch.setattr(mk4, "CONFIG", dict(mk4.CONFIG, max_regimes=2, regime_starts=1, regime_iters=5))
    monkeypatch.setattr(ev, "TEST_STEPS_DATA", 3000)
    monkeypatch.setattr(ev, "TEST_EPISODES", 2)
    reg = Registry(tmp_path / "reg")
    mk = lambda **e: reg.create_agent(task="S2X", reward="R1", seed=5, env_settings=ENV, action_space=LOG10, profile="A_synthetic",
                                      episode_decisions=200, extra=e)
    a = mk(agent_type="model-kelly-v4")
    v = m4.train_agent(reg, a, 5)
    assert v["final_checkpoint"]["step"] == 6000 - 40 and "regimes" in v["final_checkpoint"]
    data = [dict(task="S2X", seed=777, data_hash=st.build("S2X", 777, 3000).md.content_hash())]
    reg.record_round("r1", dict(kind="fresh", data=data, test_steps=3000))
    evs = [ev.evaluate(reg, a, v["version"], "r1")]
    b = mk(agent_type="model-kelly-v4-static", ablation_of=a)
    for ident_id, ck in ((b, dict(v["final_checkpoint"])), ):
        n = reg.start_training(ident_id, None, None, {}, info=dict(ablation=True))
        reg.record_version(ident_id, dict(version=n, parent_version=None, kind="initial", reason="ablation", changes={},
                                          training=dict(algorithm="none"), checkpoints=[], final_checkpoint=ck, history={},
                                          code_version="test"))
    evs.append(ev.evaluate(reg, b, 1, "r1"))
    for kind in ("reference-true-filter", "reference-oracle"):
        r = mk(agent_type=kind)
        n = reg.start_training(r, None, None, {}, info=dict(reference=True))
        sha = reg.blobs.put(b"{}", "json")
        reg.record_version(r, dict(version=n, parent_version=None, kind="initial", reason="reference", changes={},
                                   training=dict(algorithm="none"), checkpoints=[],
                                   final_checkpoint=dict(step=0, model_sha256=sha, state_sha256=None, kind="final"),
                                   history={}, code_version="test"))
        evs.append(ev.evaluate(reg, r, 1, "r1"))
    assert all(e["decisions"] > 0 for e in evs)
    reg.cache = LocalBlobStore(tmp_path / "fresh-cache")
    for e in evs:
        assert len(ev.decision_log(reg, e)) == e["decisions"]           # regenerates every decision exactly
