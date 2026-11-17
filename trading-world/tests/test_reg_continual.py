"""Continued training keeps previous learning; every attempt is a new immutable version."""
import io

import pytest
import torch

from tradeworld.learning import continual as ct
from tradeworld.learning import evaluate as ev
from tradeworld.learning.compare import compare_seeds, compare_versions
from tradeworld.registry.registry import FrozenRuleError, Registry
from tradeworld.registry.store import sha256_file

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
ACT = dict(log10_min=-4.0, log10_max=4.0, cap=None)


@pytest.fixture
def reg(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ct, "TRAIN_STEPS_DATA", 3000)
    monkeypatch.setattr(ev, "TEST_STEPS_DATA", 3000)
    monkeypatch.setattr(ev, "TEST_EPISODES", 2)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "n_steps", 128)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "batch_size", 64)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "n_epochs", 2)
    return Registry(tmp_path / "reg")


def agent(reg, reward="R1", seed=0):
    return reg.create_agent(task="S7", reward=reward, seed=seed, env_settings=ENV, action_space=ACT,
                            profile="A_synthetic", episode_decisions=50)


def params(reg, ckpt):
    from stable_baselines3 import PPO
    m = PPO.load(io.BytesIO(reg.blobs.get(ckpt["model_sha256"])), device="cpu")
    return torch.cat([p.detach().flatten() for p in m.policy.parameters()])


def test_new_agent_version_has_checkpoints_history_and_provenance(reg):
    a = agent(reg)
    v1 = ct.train_version(reg, a, steps=512, n_envs=2, checkpoint_every=256, log=lambda m: None)
    assert v1["version"] == 1 and v1["kind"] == "initial" and v1["parent_version"] is None
    assert len(v1["checkpoints"]) >= 3 and v1["checkpoints"][-1]["kind"] == "final"
    assert all(reg.blobs.exists(c["model_sha256"]) and reg.blobs.exists(c["state_sha256"]) for c in v1["checkpoints"])
    assert v1["history"]["complete"] and v1["history"]["episodes"] > 0 and v1["history"]["updates"] > 0
    assert v1["training"]["train_data"][0]["data_hash"] and v1["code_version"]["git_commit"]
    assert reg.state(a)["status"] == "awaiting_evaluation"


def test_continued_version_starts_from_parent_learning_and_records_changes(reg):
    a = agent(reg)
    v1 = ct.train_version(reg, a, steps=512, n_envs=2, checkpoint_every=256, log=lambda m: None)
    p1 = reg.root / "agents" / a / "versions" / "v0001.json"
    h1 = sha256_file(p1)
    v2 = ct.train_version(reg, a, steps=512, parent=1, n_envs=2, checkpoint_every=256, log=lambda m: None,
                          changes={"learning_rate": 1e-4, "log_std": {"exposure": -3.0}}, reason="failed gate")
    assert v2["parent_version"] == 1 and v2["parent_checkpoint"]["model_sha256"] == v1["final_checkpoint"]["model_sha256"]
    assert v2["training"]["step_start"] == v1["training"]["step_end"]          # experience accumulates
    assert v2["training"]["rng_restored"] is True
    ch = v2["training"]["settings_changes"]
    assert ch["learning_rate"]["to"] == pytest.approx(1e-4) and ch["log_std"]["to"][-1] == pytest.approx(-3.0)
    assert not torch.equal(params(reg, v1["final_checkpoint"]), params(reg, v2["final_checkpoint"]))
    assert sha256_file(p1) == h1                                               # v1 untouched
    assert reg.state(a)["current_version"] == 2 and reg.state(a)["training_attempts"] == 2


def test_frozen_rules_block_a_version_before_any_training(reg):
    a = agent(reg)
    ct.train_version(reg, a, steps=256, n_envs=2, log=lambda m: None)
    with pytest.raises(FrozenRuleError):
        ct.train_version(reg, a, steps=256, parent=1, changes={"reward": "R2"}, log=lambda m: None)
    assert reg.state(a)["current_version"] == 1 and reg.state(a)["status"] == "awaiting_evaluation"


def test_fresh_training_data_is_drawn_from_the_ledger(reg):
    a = agent(reg)
    ct.train_version(reg, a, steps=256, n_envs=2, log=lambda m: None)
    v2 = ct.train_version(reg, a, steps=256, parent=1, n_envs=2, changes={"train_data_seeds": ["fresh"]},
                          log=lambda m: None)
    s = v2["training"]["train_data"][0]["seed"]
    assert s >= 1000 and v2["training"]["rng_restored"] == "global_only"
    v3 = ct.train_version(reg, a, steps=256, parent=2, n_envs=2, changes={"train_data_seeds": ["fresh"]},
                          log=lambda m: None)
    assert v3["training"]["train_data"][0]["seed"] != s


def test_interrupted_training_resumes_from_its_last_checkpoint(reg, monkeypatch):
    a = agent(reg)
    ct.train_version(reg, a, steps=256, n_envs=2, log=lambda m: None)
    real = ct.checkpoint_callback

    def crashing(*args, **kw):
        cb = real(*args, **kw)
        orig = cb.save

        def save(kind):
            rec = orig(kind)
            if kind == "periodic":
                raise KeyboardInterrupt("container restarted")
            return rec
        cb.save = save
        return cb
    monkeypatch.setattr(ct, "checkpoint_callback", crashing)
    with pytest.raises(KeyboardInterrupt):
        ct.train_version(reg, a, steps=1024, parent=1, n_envs=2, checkpoint_every=256, log=lambda m: None)
    monkeypatch.setattr(ct, "checkpoint_callback", real)
    v2 = reg.version(a, 2)                                    # the interrupted attempt is kept as a version
    assert v2["kind"] == "interrupted" and v2["history"]["complete"] is False and len(v2["checkpoints"]) == 1
    v = ct.recover(reg, a, 2, steps=256, log=lambda m: None)
    assert v["version"] == 3 and v["parent_version"] == 2 and v["recovered_from"]["version"] == 2
    assert v["parent_checkpoint"]["model_sha256"] == v2["final_checkpoint"]["model_sha256"]
    assert v["training"]["step_start"] == v2["final_checkpoint"]["step"] and v["training"]["rng_restored"] is True


def test_versions_compared_on_identical_fresh_data(reg):
    a = agent(reg)
    ct.train_version(reg, a, steps=256, n_envs=2, log=lambda m: None)
    r0 = ev.create_round(reg, ["S7"])
    ev.evaluate(reg, a, 1, r0["round_id"])
    ct.train_version(reg, a, steps=256, parent=1, n_envs=2, log=lambda m: None)
    r1 = ev.create_round(reg, ["S7"])
    c = compare_versions(reg, a, r1["round_id"])
    sr = c["same_round"]
    assert list(sr["version"]) == [1, 2] and set(sr["round_id"]) == {r1["round_id"]}
    assert list(c["original"]["round_id"]) == [r0["round_id"], r1["round_id"]]   # v1's original kept
    assert len(c["training"]) > 0
    b = agent(reg, seed=1)
    ct.train_version(reg, b, steps=256, n_envs=2, log=lambda m: None)
    ev.evaluate(reg, b, 1, r1["round_id"])
    assert set(compare_seeds(reg, "S7", "R1", r1["round_id"])["seed"]) == {0, 1}
