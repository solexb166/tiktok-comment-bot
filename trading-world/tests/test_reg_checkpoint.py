"""Checkpoints preserve everything needed to continue learning; training history is complete."""
import pickle

import numpy as np
import pytest
import torch

from tradeworld.learning import synthetic_tasks as st
from tradeworld.learning.checkpoint import capture_state, load_checkpoint, save_checkpoint
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.learning.history import EpisodeStatsWrapper, checkpoint_callback, history_callback
from tradeworld.registry.store import LocalBlobStore, read_jsonl

EP = 20


def make_venv(task, n=2, seed=0):
    from stable_baselines3.common.vec_env import DummyVecEnv

    def fn(rank):
        def _f():
            env = EpisodeStatsWrapper(TradeWorldEnv(task.md, task.profile, reward="R1", episode_decisions=EP,
                                                    **task.env_kwargs))
            env.reset(seed=seed * 1000 + rank)
            return env
        return _f
    v = DummyVecEnv([fn(r) for r in range(n)])
    v.seed(seed)
    return v


def new_model(venv, seed=0):
    from stable_baselines3 import PPO
    torch.manual_seed(seed)
    return PPO("MlpPolicy", venv, n_steps=64, batch_size=64, n_epochs=2, seed=seed, device="cpu",
               policy_kwargs=dict(net_arch=[16, 16]), verbose=0)


@pytest.fixture(scope="module")
def task():
    torch.set_num_threads(1)
    return st.build("S7", 0, 3000)


def test_checkpoint_round_trip_restores_weights_optimizer_and_random_streams(task, tmp_path):
    store = LocalBlobStore(tmp_path / "b")
    v = make_venv(task)
    m = new_model(v)
    m.learn(512)
    rec = save_checkpoint(store, m, v, dict(version=1))
    saved = pickle.loads(store.get(rec["state_sha256"]))
    assert rec["step"] == m.num_timesteps and store.get(rec["model_sha256"])
    m2 = load_checkpoint(store, rec, make_venv(task, seed=99))
    for a, b in zip(m.policy.parameters(), m2.policy.parameters()):
        assert torch.equal(a, b)
    o1, o2 = m.policy.optimizer.state_dict()["state"], m2.policy.optimizer.state_dict()["state"]
    assert all(torch.equal(o1[k]["exp_avg"], o2[k]["exp_avg"]) for k in o1)
    assert m2.num_timesteps == m.num_timesteps and torch.equal(m2.policy.log_std, m.policy.log_std)
    assert set(saved) >= {"torch_rng", "numpy_global", "python_random", "envs", "sb3"}
    assert all(e["np_random"] is not None and e["account_rng"] is not None for e in saved["envs"])


def test_resuming_twice_from_one_checkpoint_gives_identical_learning(task, tmp_path):
    store = LocalBlobStore(tmp_path / "b")
    v = make_venv(task)
    m = new_model(v)
    m.learn(256)
    rec = save_checkpoint(store, m, v, {})
    finals = []
    for _ in range(2):
        r = load_checkpoint(store, rec, make_venv(task, seed=7))
        r.learn(256, reset_num_timesteps=False)
        finals.append(torch.cat([p.detach().flatten() for p in r.policy.parameters()]))
    assert torch.equal(finals[0], finals[1])
    assert not torch.equal(finals[0], torch.cat([p.detach().flatten() for p in m.policy.parameters()]))


def test_history_keeps_every_episode_and_every_update(task, tmp_path):
    v = make_venv(task)
    m = new_model(v)
    h = history_callback()
    m.learn(64 * 2 * 20, callback=h)                       # 2,560 steps = 128 episodes of 20
    ep, up = h.frames()
    assert len(ep) >= 120                                  # far more than SB3's last-100 buffer
    assert {"episode_reward", "growth", "died", "max_drawdown", "mean_gross", "max_gross", "turnover",
            "trades", "costs", "time_in_market", "share_gross_over_10x"} <= set(ep.columns)
    assert len(up) >= 15 and {"train/value_loss", "train/policy_gradient_loss", "train/entropy_loss",
                              "explore/log_std_exposure"} <= set(up.columns)


def test_periodic_checkpoints_are_written_and_logged_as_they_happen(task, tmp_path):
    store = LocalBlobStore(tmp_path / "b")
    v = make_venv(task)
    m = new_model(v)
    h = history_callback()
    log = tmp_path / "inflight.jsonl"
    cb = checkpoint_callback(store, v, every=256, inflight_log=log, meta=dict(version=1), history=h)
    from stable_baselines3.common.callbacks import CallbackList
    m.learn(1024, callback=CallbackList([h, cb]))
    recs = read_jsonl(log)
    assert len(recs) >= 3 and [r["step"] for r in recs] == sorted(r["step"] for r in recs)
    assert all(store.exists(r["model_sha256"]) and store.exists(r["state_sha256"]) for r in recs)
    assert recs[-1]["history_episodes_sha256"]
