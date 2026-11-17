"""Phase 3 Experiment 1: optional configurations for new agents, separate registry, protection
of the original system, and the pre-registered statistics."""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from tradeworld.interface.actions import ActionMapper
from tradeworld.learning import continual as ct
from tradeworld.learning import evaluate as ev
from tradeworld.learning import phase3_analysis as an
from tradeworld.registry.datasets import DataLedger
from tradeworld.registry.registry import ROOT, Registry, RegistryError
from tradeworld.registry.store import LocalBlobStore, RegistryError as StoreError, read_json

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None)


# ---------------------------------------------------------------------------- size representations
def test_default_mapper_is_unchanged():
    m = ActionMapper.primary(1)
    assert m.size_repr == "log10" and list(m.low) == [-1, -1] and list(m.high) == [1, 1]
    rng = np.random.default_rng(0)
    for a in rng.uniform(-1.3, 1.3, (200, 2)):
        x = m.map(a)
        u = float(np.clip(a[1], -1, 1))
        assert x.G == pytest.approx(10 ** (-4 + (u + 1) / 2 * 8))
        assert x.w[0] == pytest.approx(x.G * np.clip(a[0], -1, 1) / max(1.0, abs(np.clip(a[0], -1, 1))))


@pytest.mark.parametrize("repr_", ["exp", "softplus"])
def test_new_size_representations_reach_any_positive_size(repr_):
    m = ActionMapper(1, size_repr=repr_)
    for G in (1e-4, 0.01, 0.5, 2.93, 9.5, 17.0, 100.0, 2000.0, 10_000.0):
        z = m.u_for_gross(G)
        assert m.gross(z) == pytest.approx(G, rel=1e-6)
        x = m.map([1.0, z])
        assert x.valid and x.w[0] == pytest.approx(G, rel=1e-6) and "clipped_action" not in x.safeguards


def test_new_representations_are_less_sensitive():
    ratio = {}
    for r in ("log10", "exp", "softplus"):
        m = ActionMapper(1, size_repr=r)
        z = m.u_for_gross(2.0)
        ratio[r] = m.gross(z + 0.25) / m.gross(z)
    assert ratio["log10"] == pytest.approx(10.0, rel=1e-6)          # x10 per 0.25 (the diagnosed problem)
    assert ratio["exp"] == pytest.approx(math.exp(0.25), rel=1e-6)  # x1.28
    assert ratio["softplus"] < 1.2                                   # about +0.25x at 2x


def test_unknown_representation_refused():
    with pytest.raises(ValueError):
        ActionMapper(1, size_repr="cap10")


def test_env_action_space_follows_representation():
    from tradeworld.learning import synthetic_tasks as st
    from tradeworld.learning.env import TradeWorldEnv
    t = st.build("S6", 0, 3000)
    e0 = TradeWorldEnv(t.md, t.profile, episode_decisions=50, **t.env_kwargs)
    assert float(e0.action_space.low[1]) == -1.0 and float(e0.action_space.high[1]) == 1.0
    m = ActionMapper(1, size_repr="exp")
    e = TradeWorldEnv(t.md, t.profile, episode_decisions=50, mapper=m, **t.env_kwargs)
    assert float(e.action_space.low[1]) == -25.0 and float(e.action_space.high[1]) == 25.0
    e.reset(seed=0)
    _, _, _, _, info = e.step(np.array([1.0, math.log(2.93)]))
    assert info["net_leverage"] == pytest.approx(2.93, rel=0.02)


def test_mapper_from_identity_defaults_to_existing_behaviour():
    assert ev.mapper_from({"frozen": {"action_space": LOG10}}).size_repr == "log10"
    assert ev.mapper_from({"frozen": {"action_space": dict(LOG10, size_repr="softplus")}}).size_repr == "softplus"


# ---------------------------------------------------------------------------- separate registry
@pytest.fixture
def p3reg(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ct, "TRAIN_STEPS_DATA", 3000)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "n_steps", 128)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "batch_size", 64)
    orig = tmp_path / "registry"
    (orig / "data").mkdir(parents=True)
    (orig / "data" / "ledger.jsonl").write_text("".join(
        json.dumps(dict(task="S6", seed=s, purpose=p, round_id=None, note="", at="x")) + "\n"
        for s, p in ((1000, "train"), (1001, "train"), (20000, "test"))))
    root = tmp_path / "registry-phase3"
    (root / "data").mkdir(parents=True)
    (root / "data" / "inherits.json").write_text(json.dumps(dict(ledgers=["../../registry/data/ledger.jsonl"])))
    (root / "storage.json").write_text(json.dumps(dict(path="../artifacts/blobs")))
    return Registry(root)


def test_inherited_ledger_makes_original_seeds_used(p3reg):
    led = DataLedger(p3reg.root)
    assert led.allocate("S6", "train") == 1002
    assert led.allocate("S6", "test") == 20001
    with pytest.raises(StoreError):
        led.record("S6", 20000, "test")
    orig = (p3reg.root.parent / "registry" / "data" / "ledger.jsonl").read_text()
    assert orig.count("\n") == 3                                  # the original ledger was not written


def test_missing_inherited_ledger_is_refused(p3reg):
    (p3reg.root.parent / "registry" / "data" / "ledger.jsonl").unlink()
    with pytest.raises(StoreError):
        DataLedger(p3reg.root).fresh_seed("S6", "train")


def test_blobs_go_to_the_separate_store(p3reg):
    sha = p3reg.blobs.put(b"checkpoint")
    assert (p3reg.root.parent / "artifacts" / "blobs").exists()
    assert LocalBlobStore(p3reg.root.parent / "artifacts" / "blobs").exists(sha)
    assert not (p3reg.root / "blobs").exists()


def test_identity_extra_fields(p3reg):
    a = p3reg.create_agent(task="S6", reward="R1", seed=1002, env_settings=ENV, action_space=dict(LOG10, size_repr="exp"),
                           profile="A_synthetic", episode_decisions=50, extra=dict(experiment="x", group="C1"))
    i = p3reg.identity(a)
    assert i["group"] == "C1" and i["frozen"]["action_space"]["size_repr"] == "exp"
    with pytest.raises(StoreError):
        p3reg.create_agent(task="S6", reward="R1", seed=1, env_settings=ENV, action_space=LOG10,
                           profile="A_synthetic", episode_decisions=50, extra=dict(frozen={}))


@pytest.mark.parametrize("group,settings,repr_", [
    ("D", dict(learning_rate=1e-4, n_epochs=4, target_kl=0.02), "log10"),
    ("A", dict(gamma=0.999), "log10"),
    ("C2", {}, "softplus"),
])
def test_new_agent_initial_settings_are_applied_and_recorded(p3reg, group, settings, repr_):
    a = p3reg.create_agent(task="S6", reward="R1", seed=1002, env_settings=ENV, action_space=dict(LOG10, size_repr=repr_),
                           profile="A_synthetic", episode_decisions=50)
    v = ct.train_version(p3reg, a, steps=512, n_envs=2, checkpoint_every=256, seed=1002, initial_settings=settings,
                         log=lambda m: None)
    h = v["training"]["hyperparameters"]
    for k, x in settings.items():
        assert h[k] == pytest.approx(x)
    assert v.get("initial_settings", {}) == settings
    assert h["log_std"][-1] == pytest.approx(ct.U_LOG_STD_INIT, abs=0.1)          # same initial noise in output units
    model = ev.load_model(p3reg, v["final_checkpoint"]["model_sha256"])            # resumes with the same settings
    for k, x in settings.items():
        got = float(model.lr_schedule(1.0)) if k == "learning_rate" else getattr(model, k)
        assert got == pytest.approx(x)


def test_initial_settings_restricted(p3reg):
    a = p3reg.create_agent(task="S6", reward="R1", seed=1002, env_settings=ENV, action_space=LOG10,
                           profile="A_synthetic", episode_decisions=50)
    with pytest.raises(RegistryError):
        ct.train_version(p3reg, a, steps=256, n_envs=2, initial_settings=dict(clip_range=0.1), log=lambda m: None)


# ---------------------------------------------------------------------------- protection
def test_existing_agent_decisions_regenerate_identically_with_new_code(tmp_path):
    """An original S6 agent, re-run from its checkpoint under the modified code, makes exactly the
    decisions recorded before the change (fingerprint of the full decision log)."""
    reg = Registry(ROOT / "registry")
    reg.cache = LocalBlobStore(tmp_path / "cache")                  # never touch the original cache
    a = reg.find("S6", "R1", 0)
    e = next(x for x in reg.evaluations(a) if x["round_id"] == "gate-1")
    log = ev.decision_log(reg, e)                                    # raises if the fingerprint differs
    assert len(log) == e["decisions"]


# ---------------------------------------------------------------------------- statistics
def test_permutation_and_holm_and_signflip():
    assert an.permutation_p([10] * 6, [0] * 6) == pytest.approx(1 / 924)
    assert an.permutation_p([0] * 6, [0] * 6) == 1.0
    adj = an.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj == {"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.5}
    assert an.signflip_p([-1, -2, -3]) == pytest.approx(1 / 8)
    lo, hi = an.bootstrap_diff_ci([1, 2, 3], [0, 0, 0])
    assert 0 < lo < 2 < hi <= 3


def test_lineage_follows_interrupted_attempts():
    class R:
        def versions(self, a):
            return [dict(version=1, kind="interrupted", parent_version=None, final_checkpoint=dict(step=1000),
                         checkpoints=[dict(step=500, model_sha256="a"), dict(step=1000, model_sha256="b")]),
                    dict(version=2, kind="continued", parent_version=1, final_checkpoint=dict(step=3000, model_sha256="d"),
                         checkpoints=[dict(step=2000, model_sha256="c")])]
    chain = an.lineage(R(), "x", 3000)
    assert [v["version"] for v in chain] == [1, 2]
    assert an.checkpoint_at(chain, 1500)[1]["model_sha256"] == "c"
    assert an.lineage(R(), "x", 4000) == []
