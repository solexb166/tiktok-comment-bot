"""Experiment M2: model-based agent v2 (model_kelly2), benchmark policy and registry integration."""
import math

import numpy as np
import pytest
import torch

from tradeworld.learning import evaluate as ev
from tradeworld.learning import m2
from tradeworld.learning import model_kelly2 as mk2
from tradeworld.learning import synthetic_tasks as st
from tradeworld.registry.registry import Registry
from tradeworld.registry.store import LocalBlobStore

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")


class Map:
    size = 2

    def u_for_gross(self, G):
        return G


def _s6_returns(n, seed):
    rng = np.random.default_rng(seed)
    crash = rng.random(n) < 1 / 300
    return np.where(crash, math.log(0.9), 5e-4 - 0.5 * 3e-3 ** 2 + 3e-3 * rng.standard_normal(n))


def test_em_recovers_a_crash_mixture():
    y = _s6_returns(150_000, 1)
    pi, mu, sd = mk2.em(y, 3, 500, 1e-9)
    k = int(np.argmin(mu))
    assert mu[k] == pytest.approx(math.log(0.9), abs=2e-3) and pi[k] == pytest.approx(1 / 300, rel=0.2)


def test_noise_features_choose_unconditional_model_and_near_optimal_size():
    torch.set_num_threads(2)
    y = _s6_returns(120_000, 2)
    X = np.random.default_rng(3).standard_normal((len(y), 6)).astype(np.float32)
    art, hist = mk2.learn(X, y, 0, dict(mk2.CONFIG, max_epochs=15))
    assert art["kind"] == "unconditional"
    G = mk2.Agent(art, np.arange(6), Map()).sizes(X[:10])
    assert np.all(G == G[0]) and 2.0 < G[0] < 3.9 and G[0] < 9.5


def test_informative_feature_chooses_conditional_model():
    torch.set_num_threads(2)
    rng = np.random.default_rng(4)
    n = 60_000
    flag = np.sign(rng.standard_normal(n))
    y = flag * 4e-4 + 1e-3 * rng.standard_normal(n)
    X = np.c_[flag, rng.standard_normal((n, 3))].astype(np.float32)
    art, _ = mk2.learn(X, y, 0, dict(mk2.CONFIG, max_epochs=30))
    assert art["kind"] == "conditional"
    G = mk2.Agent(art, np.arange(4), Map()).sizes(np.array([[1, 0, 0, 0], [-1, 0, 0, 0]], np.float32))
    assert G[0] > 50 and G[1] < -50                                     # follows the informative feature (optimum +-400)


def test_shrinkage_keeps_a_clear_edge_and_shrinks_noise():
    y = _s6_returns(150_000, 5)
    pi, mu, sd = mk2.em(y, 3, 500, 1e-9)
    _, info = mk2.shrink(y, pi, mu, sd)
    assert info["t"] > 5 and info["shrink"] > 0.95
    noise = 1e-3 * np.random.default_rng(6).standard_normal(150_000)
    pi, mu, sd = mk2.em(noise, 1, 500, 1e-9)
    _, info = mk2.shrink(noise, pi, mu, sd)
    assert info["shrink"] < 1 and (abs(info["t"]) >= 1 or info["shrink"] == 0)


def test_benchmark_constant_policy():
    ident = dict(frozen=dict(action_space=LOG10), benchmark_size=2.93)
    from tradeworld.interface.actions import ActionMapper
    a = ev.constant_policy(ident)(None)
    assert ActionMapper(1).map(a).w[0] == pytest.approx(2.93, rel=1e-6)
    a0 = ev.constant_policy(dict(ident, benchmark_size=0.0))(None)
    assert abs(ActionMapper(1).map(a0).w[0]) < 1e-3


def test_train_evaluate_and_regenerate(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(m2, "TRAIN_DATA_STEPS", 5000)
    monkeypatch.setattr(m2, "DATA_POINTS", (2000, 4900))
    monkeypatch.setattr(mk2, "CONFIG", dict(mk2.CONFIG, max_epochs=4))
    monkeypatch.setattr(ev, "TEST_STEPS_DATA", 3000)
    monkeypatch.setattr(ev, "TEST_EPISODES", 2)
    reg = Registry(tmp_path / "reg")
    a = reg.create_agent(task="S6", reward="R1", seed=5, env_settings=ENV, action_space=LOG10, profile="A_synthetic",
                         episode_decisions=200, extra=dict(agent_type="model-kelly-v2"))
    v = m2.train_agent(reg, a, 5)
    assert v["final_checkpoint"]["selection"]["chosen"] in ("unconditional", "conditional")
    data = [dict(task="S6", seed=777, data_hash=st.build("S6", 777, 3000).md.content_hash())]
    reg.record_round("r1", dict(kind="fresh", data=data, test_steps=3000))
    e = ev.evaluate(reg, a, v["version"], "r1")
    reg.cache = LocalBlobStore(tmp_path / "fresh-cache")
    assert len(ev.decision_log(reg, e)) == e["decisions"]               # regenerated exactly from the stored model
