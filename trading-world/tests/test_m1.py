"""Experiment M1: the model-based growth-optimal agent (model_kelly) and its registry integration."""
import math

import numpy as np
import pytest
import torch

from tradeworld.learning import evaluate as ev
from tradeworld.learning import m1
from tradeworld.learning import model_kelly as mk
from tradeworld.learning import synthetic_tasks as st
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.registry.registry import Registry
from tradeworld.registry.store import LocalBlobStore

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")


def test_market_columns_exclude_account_features():
    t = st.build("S6", 0, 2000)
    enc = TradeWorldEnv(t.md, t.profile, episode_decisions=10, **t.env_kwargs).encoder
    cols = mk.market_columns(enc)
    m = enc.n * enc.W * enc.F + enc.n
    assert len(cols) == enc.dim - enc.A and cols.max() < enc.dim
    assert not set(range(m, m + enc.A)) & set(cols.tolist())


def test_learns_a_crash_distribution_and_sizes_near_the_optimum():
    """Pure-noise features, S6-like returns: the learned sizes must be safe and near the growth optimum."""
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    n = 60_000
    crash = rng.random(n) < 1 / 300
    y = np.where(crash, math.log(0.9), 5e-4 - 0.5 * 3e-3 ** 2 + 3e-3 * rng.standard_normal(n)).astype(np.float32)
    X = rng.standard_normal((n, 8)).astype(np.float32)
    art, hist = mk.train(X, y, seed=0, config=dict(mk.CONFIG, max_epochs=40))
    assert hist and art["epochs"] == len(hist)

    class Map:
        def u_for_gross(self, G):
            return G
    agent = mk.Agent(art, np.arange(8), Map())
    G = agent.sizes(rng.standard_normal((500, 8)).astype(np.float32))
    assert np.all(G < 9.5)                                   # never a ruinous size
    assert 1.2 < np.median(G) < 4.5                          # near the growth optimum (2.93x)
    pi, mu, _ = agent.predictive(X[:200])
    low = mu.argmin(1)
    assert np.median(mu[np.arange(200), low]) < -0.03        # a crash component was learned


def test_action_maps_to_the_chosen_position():
    from tradeworld.interface.actions import ActionMapper
    mapper = ActionMapper(1)
    agent = mk.Agent.__new__(mk.Agent)
    agent.mapper = mapper
    for G in (2.93, -1.5, 0.3):
        a = agent.action(G)
        assert mapper.map(a).w[0] == pytest.approx(G, rel=1e-6)
    assert mapper.map(agent.action(0.0)).w[0] == pytest.approx(0.0, abs=1e-3)


def test_train_evaluate_and_regenerate(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(m1, "TRAIN_DATA_STEPS", 4000)
    monkeypatch.setattr(m1, "DATA_POINTS", (1500, 4000 - 40))
    monkeypatch.setattr(mk, "CONFIG", dict(mk.CONFIG, max_epochs=5))
    monkeypatch.setattr(ev, "TEST_STEPS_DATA", 3000)
    monkeypatch.setattr(ev, "TEST_EPISODES", 2)
    reg = Registry(tmp_path / "reg")
    a = reg.create_agent(task="S6", reward="R1", seed=5, env_settings=ENV, action_space=LOG10, profile="A_synthetic",
                         episode_decisions=200, extra=dict(agent_type="model-kelly"))
    v = m1.train_agent(reg, a, 5)
    assert v["kind"] == "initial" and v["final_checkpoint"]["step"] == 4000 - 40 and len(v["checkpoints"]) == 1
    data = [dict(task="S6", seed=777, data_hash=st.build("S6", 777, 3000).md.content_hash())]
    reg.record_round("r1", dict(kind="fresh", data=data, test_steps=3000))
    e = ev.evaluate(reg, a, v["version"], "r1")
    assert e["decisions"] > 0 and e["checkpoint_sha256"] == v["final_checkpoint"]["model_sha256"]
    reg.cache = LocalBlobStore(tmp_path / "fresh-cache")                    # force regeneration from the blob
    log = ev.decision_log(reg, e)                                           # raises if any decision differs
    assert len(log) == e["decisions"]
