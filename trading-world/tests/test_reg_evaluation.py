"""Decision recording, fresh test data, immutable evaluations and failure diagnosis."""
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch

from tradeworld.learning import evaluate as ev
from tradeworld.learning import synthetic_tasks as st
from tradeworld.learning.diagnosis import diagnose
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.learning.recorder import DecisionRecorder, classify_change
from tradeworld.learning.train import run_episodes
from tradeworld.registry.datasets import DataLedger, SealedDataError, check_period, load_splits
from tradeworld.registry.registry import FrozenRuleError, Registry
from tradeworld.registry.store import RegistryError, WriteOnceError

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
ACT = dict(log10_min=-4.0, log10_max=4.0, cap=None)


def ident(task="S6", reward="R1", seed=0, ep=200):
    return dict(task=task, reward=reward, original_seed=seed,
                frozen=dict(task=task, reward=reward, env_settings=ENV, action_space=ACT, profile="A_synthetic",
                            episode_decisions=ep))


def test_classify_position_changes():
    assert [classify_change(*x) for x in [(0, 0), (0, 1), (1, 0), (1, -1), (1, 2), (2, 1), (1, 1)]] == \
        ["stay_flat", "open", "close", "reverse", "increase", "reduce", "hold"]


def test_recorder_is_read_only_and_records_consequences():
    t = st.build("S6", 0, 3000)
    script = [[1, 0.0], [1, 0.0], [1, 0.1], [1, 0.05], [-1, 0.05], [0, 0.0], [0, 0.0]] + [[1, 0.7]] * 150   # ~630x: allowed by 1:2000 margin, fatal here
    plain = TradeWorldEnv(t.md, t.profile, episode_decisions=150, random_start=False, **t.env_kwargs)
    rec = DecisionRecorder(TradeWorldEnv(t.md, t.profile, episode_decisions=150, random_start=False, **t.env_kwargs))
    o1, _ = plain.reset(seed=3)
    o2, _ = rec.reset(seed=3)
    assert np.array_equal(o1, o2)
    for a in script:
        r1 = plain.step(np.array(a, float))
        r2 = rec.step(np.array(a, float))
        assert np.array_equal(r1[0], r2[0]) and r1[1] == r2[1]
        if r2[2] or r2[3]:
            break
    rows = pd.DataFrame(rec.rows)
    assert list(rows["position_change"][:7]) == ["open", "hold", "increase", "reduce", "reverse", "close", "stay_flat"]
    assert np.allclose(rows["pnl"], rows["equity_after"] - rows["equity_before"])
    assert (rows[["cost_spread", "cost_slippage"]] >= 0).all().all()
    assert rows["direction"].iloc[4] == -1 and rows["gross_selected"].iloc[0] == pytest.approx(1.0)
    died = rows[rows["died"]]
    assert len(died) == 1 and died["death_reason"].iloc[0] != "" and not died["alive_after"].iloc[0]


def test_recorded_episode_stats_match_the_original_gate_definitions():
    t = st.build("S2", 0, 4000)
    i = ident("S2", ep=150)
    policy = lambda o: np.array([np.tanh(o[-1] * 3), 0.05])
    rows, stats = ev.run_recorded(policy, i, t, [0, 500, 1000])
    ref = run_episodes(lambda o, env: policy(o), t, [0, 500, 1000], 150)
    for a, b in zip(stats, ref):
        for k in ("log_growth", "died", "steps", "mean_gross", "mean_net", "turnover", "costs", "flag_agreement"):
            assert a[k] == pytest.approx(getattr(b, k)), k


def test_data_ledger_gives_fresh_seeds_and_protects_sealed_data(tmp_path):
    led = DataLedger(tmp_path)
    led.record("S6", 10_000, "test", round_id="gate-1")
    s1 = led.fresh_seed("S6", "test")
    led.record("S6", s1, "test", round_id="round-0001")
    s2 = led.fresh_seed("S6", "test")
    assert s1 != s2 and s1 >= 20_000
    with pytest.raises(RegistryError):
        led.record("S6", s1, "train")
    with pytest.raises(SealedDataError):
        led.record("S6", 900_000, "test")
    with pytest.raises(SealedDataError):
        led.sealed_seed("S6", reason="", approved_by="")
    assert led.sealed_seed("S6", "final assessment", "owner") >= 900_000
    assert len(led.sealed_accesses()) == 1


def test_real_data_splits_seal_the_final_holdout():
    sp = load_splits()
    check_period(date(2019, 1, 1), date(2019, 12, 31), "train", sp)
    with pytest.raises(SealedDataError):
        check_period(date(2025, 6, 1), date(2025, 8, 1), "validation", sp)
    with pytest.raises(FrozenRuleError):
        check_period(date(2023, 3, 1), date(2023, 4, 1), "train", sp)


def test_diagnosis_separates_observations_from_explanations():
    t = st.build("S6", 0, 3000)
    n = 400
    lev = np.full(n, 0.5)
    lev[[50, 120]] = 150.0
    log = pd.DataFrame(dict(
        episode=np.repeat([0, 1], n // 2), decision_row=np.arange(n) + 31, leverage_after_fill=lev,
        exposure_after_fill=lev, market_return=np.where(np.arange(n) % 97 == 0, np.log(0.9), 0.001),
        expected_return=np.full(n, 1e-4), position_change="hold", pnl=np.r_[-np.ones(10), np.ones(n - 10)],
        died=np.isin(np.arange(n), [50, 120]), death_reason=np.where(np.isin(np.arange(n), [50, 120]), "equity_floor", ""),
        cost_spread=0.0, cost_slippage=0.0, cost_commission=0.0, cost_swap=0.0, equity_before=10.0))
    d = diagnose(log, t, ident(), None)
    codes = {o["code"] for o in d["observations"]}
    assert "deaths_after_excessive_leverage" in codes and "leverage_spikes" in codes
    assert all(w["text"].startswith("Possible:") for w in d["possible_explanations"])
    assert d["metrics"]["fatal_leverage"] == pytest.approx(0.95 / -np.log(0.9))


@pytest.fixture
def small_registry(tmp_path, monkeypatch):
    from stable_baselines3 import PPO
    from tradeworld.learning.checkpoint import save_checkpoint
    from tradeworld.learning.train import make_env_fn
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    monkeypatch.setattr(ev, "TEST_STEPS_DATA", 5000)
    monkeypatch.setattr(ev, "TEST_EPISODES", 3)
    reg = Registry(tmp_path / "reg")
    t = st.build("S3", 0, 4000)
    venv = DummyVecEnv([make_env_fn(t, "R1", 200, 0, 0)])
    m = PPO("MlpPolicy", venv, n_steps=64, batch_size=64, seed=0, device="cpu", policy_kwargs=dict(net_arch=[8]))
    m.learn(128)
    a = reg.create_agent(task="S3", reward="R1", seed=0, env_settings=ENV, action_space=ACT, profile="A_synthetic",
                         episode_decisions=200)
    c = save_checkpoint(reg.blobs, m, venv, {})
    reg.start_training(a, 1, None, {})
    reg.record_version(a, dict(version=1, parent_version=None, checkpoints=[c], final_checkpoint=c))
    return reg, a


def test_evaluation_is_immutable_uses_fresh_data_and_pins_criteria(small_registry, monkeypatch):
    reg, a = small_registry
    r1 = ev.create_round(reg, ["S3"])
    e = ev.evaluate(reg, a, 1, r1["round_id"])
    assert e["result"] in ("pass", "fail") and e["decisions"] == 3 * 200
    assert e["test_data"]["seed"] >= 20_000 and e["criteria"]["criteria_id"] == ev.CRITERIA_ID
    log = ev.decision_log(reg, e)
    assert len(log) == 600 and {"reward", "equity_before", "equity_after", "position_change"} <= set(log.columns)
    assert len(ev.decision_samples(reg, e)) in (200, 400)                      # permanent sample episodes
    import shutil
    shutil.rmtree(reg.root / "cache")                                           # lose the cache:
    assert ev.log_fingerprint(ev.decision_log(reg, e)) == e["decision_log_fingerprint"]   # regenerated exactly
    with pytest.raises(WriteOnceError):
        ev.evaluate(reg, a, 1, r1["round_id"])
    r2 = ev.create_round(reg, ["S3"])
    assert r2["data"][0]["seed"] != r1["data"][0]["seed"]
    monkeypatch.setattr(ev, "criteria_fingerprint", lambda: {"changed": "x"})
    with pytest.raises(FrozenRuleError):
        ev.evaluate(reg, a, 1, r2["round_id"])
