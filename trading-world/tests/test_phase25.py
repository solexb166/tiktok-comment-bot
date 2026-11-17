"""Phase 2.5 runner: schedule, stages, rounds, locked candidate selection, guards."""
import pytest
import torch

from tradeworld.learning import continual as ct
from tradeworld.learning import evaluate as ev
from tradeworld.learning import phase25 as p25
from tradeworld.registry.registry import Registry, RegistryError
from tradeworld.registry.store import WriteOnceError, sha256_file

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
ACT = dict(log10_min=-4.0, log10_max=4.0, cap=None)


def test_schedule_has_33_versions_and_138_evaluations():
    c = p25.evaluation_count()
    assert c["versions"] == 33 and c["total"] == 138
    assert c["per_round"] == {1: 15, 2: 24, 3: 33, 4: 33, 5: 33}
    s = p25.schedule()
    assert s[1]["R1"] == ["v1", "A1", "B1"] and s[1]["R2"] == ["v1", "B1"]
    assert s[5]["R1"] == ["v1", "A1", "A2", "A3", "B1", "B2", "B3"]
    assert [p25.parent_label(x) for x in ("A1", "A2", "B3")] == ["v1", "A1", "B2"]
    assert p25.matched_control_label("A2") == "B2" and p25.matched_control_label("B3") == "B3"


@pytest.fixture
def exp(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ct, "TRAIN_STEPS_DATA", 3000)
    monkeypatch.setattr(ev, "TEST_STEPS_DATA", 3000)
    monkeypatch.setattr(ev, "TEST_EPISODES", 3)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "n_steps", 64)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "batch_size", 64)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "n_epochs", 1)
    reg = Registry(tmp_path / "reg")
    for reward in ("R1", "R2"):
        for seed in (0, 1, 2):
            a = reg.create_agent(task="S6", reward=reward, seed=seed, env_settings=ENV, action_space=ACT,
                                 profile="A_synthetic", episode_decisions=50)
            ct.train_version(reg, a, steps=128, n_envs=2, log=lambda m: None)
    e = p25.Experiment(reg)
    with pytest.raises(RegistryError):
        e.plan(["test conditions"], steps=128, checkpoint_every=64)        # Q2 must be frozen first
    from tradeworld.learning.metrics import freeze_q2
    freeze_q2(reg, approved_by="test")
    e.plan(["test conditions"], steps=128, checkpoint_every=64)
    return reg, e


def test_full_flow_with_guards(exp, monkeypatch):
    reg, e = exp
    monkeypatch.setattr(ct, "build_venv", _small_venv(ct.build_venv))
    v1_files = {a: sha256_file(reg.root / "agents" / a / "versions" / "v0001.json") for a in reg.agents()}
    with pytest.raises(WriteOnceError):
        e.plan(["again"])
    with pytest.raises(RegistryError):
        e.run_round(1, workers=2, log=lambda m: None)                  # stage 1 not trained yet
    for k in (1, 2, 3):
        e.run_stage(k, workers=2, log=lambda m: None)
        e.run_round(k, workers=2, log=lambda m: None)
    assert e.stage_complete(3) and all(e.round_complete(n) for n in (1, 2, 3))
    with pytest.raises(RegistryError):
        e.run_round(4, workers=2, log=lambda m: None)                  # no locked candidates yet
    cand = e.select()
    assert set(cand["seeds"]) == {"R1-0", "R1-1", "R1-2"}
    assert all("closer_to_optimum_than_v1" in r for r in cand["seeds"]["R1-1"]["table"])
    assert not any("closer_to_optimum_than_v1" in r for r in cand["seeds"]["R1-0"]["table"])
    assert all(c["candidate"] in ("A1", "A2", "A3", "B1", "B2", "B3") for c in cand["seeds"].values())
    with pytest.raises(RegistryError):
        e.select()                                                     # locked: cannot reselect
    e.run_round(4, workers=2, log=lambda m: None)
    e.run_round(5, workers=2, log=lambda m: None)
    counts = {n: sum(1 for x in e.progress() if x["kind"] == "evaluated" and x["round"] == n) for n in range(1, 6)}
    assert counts == {1: 15, 2: 24, 3: 33, 4: 33, 5: 33}
    a = e.manifest()["agents"]["R1-0"]["agent_id"]
    vs = reg.versions(a)
    assert len(vs) == 7 and {v["version"]: v["parent_version"] for v in vs}[e.version_of("R1-0", "A2")] == \
        e.version_of("R1-0", "A1")
    assert all(sha256_file(reg.root / "agents" / x / "versions" / "v0001.json") == h for x, h in v1_files.items())
    b_seeds = [reg.version(a, e.version_of("R1-0", f"B{k}"))["training"]["train_data"][0]["seed"] for k in (1, 2, 3)]
    assert len(set(b_seeds)) == 3 and min(b_seeds) >= 1000                  # fresh, never-used paths
    r = e.criteria("R1-0", cand["seeds"]["R1-0"]["candidate"])
    assert r["round4"] is not None and r["round5"] is not None and r["passed"] in (True, False)
    text = p25.report(e)
    assert "Every version in every round" in text and "Pre-registered conclusion" in text
    assert text.count("| R1-0 |") >= 7


def _small_venv(real):
    def f(identity, tasks, n_envs, seed):
        return real(identity, tasks, 2, seed)
    return f
