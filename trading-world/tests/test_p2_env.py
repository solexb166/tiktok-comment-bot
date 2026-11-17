"""Phase 2: Gymnasium environment."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from tradeworld.agents.baselines import ScriptedPolicy
from tradeworld.data.synthetic import PathSpec, generate
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.tracking.hashing import event_log_hash
from tradeworld.world.world import AgentSpec, World
from conftest import MON_2024_01_08, make_profile, single

L = 300


@pytest.fixture(scope="module")
def setup():
    u = single()
    md = generate(u, MON_2024_01_08, 6 * 1440, seed=7,
                  paths={"EURUSD": PathSpec(1.1, 0.10, spread_bps=1.0)}, sessions=False)
    return u, md, make_profile(u, leverage=2000.0)


def test_check_env(setup):
    u, md, prof = setup
    check_env(TradeWorldEnv(md, prof, episode_decisions=L), skip_render_check=True)


def test_env_matches_world_exactly(setup):
    """The env must produce exactly the event log of the tested World for the same actions."""
    u, md, prof = setup
    rng = np.random.default_rng(3)
    actions = rng.uniform(-1, 1, (L, 2)) * [1.0, 0.5]
    env = TradeWorldEnv(md, prof, episode_decisions=L, random_start=False)
    env.reset(seed=11)
    for a in actions:
        _, _, te, tr, _ = env.step(a)
        if te or tr:
            break
    steps = env.k
    h_env = event_log_hash(env.account.events)

    env2 = TradeWorldEnv(md, prof, episode_decisions=L, random_start=False)
    env2.reset(seed=11)
    dec = env2.dec
    script = {k: actions[k] for k in range(steps)}
    w = World(md, prof, [AgentSpec("agent", ScriptedPolicy(script, 1))], seed=env2._world_seed,
              decision_interval=5, obs_resolution=5, window=120)
    res = w.run(stop=int(dec[steps]))
    assert event_log_hash(res.accounts["agent"].events) == h_env


def test_env_is_deterministic(setup):
    u, md, prof = setup
    out = []
    for _ in range(2):
        env = TradeWorldEnv(md, prof, episode_decisions=50)
        o, _ = env.reset(seed=5)
        env.action_space.seed(5)
        traj = [o]
        for _ in range(50):
            o, r, te, tr, info = env.step(env.action_space.sample())
            traj.append(np.r_[o, r])
            if te or tr:
                break
        out.append(np.concatenate(traj))
    assert np.array_equal(out[0], out[1])


def test_flat_actions_keep_equity_and_reward_zero(setup):
    u, md, prof = setup
    env = TradeWorldEnv(md, prof, episode_decisions=50)
    env.reset(seed=0)
    for _ in range(50):
        _, r, te, tr, info = env.step(np.zeros(2))
        assert r == 0.0 and info["equity"] == 10.0
    assert tr and not te


def test_death_terminates_episode(setup):
    u, md, prof = setup
    env = TradeWorldEnv(md, make_profile(u), episode_decisions=L)   # unlimited margin
    env.reset(seed=0)
    for k in range(L):
        _, r, te, tr, info = env.step(np.array([1.0 if k % 2 else -1.0, 1.0]))   # 10,000x, flipping
        if te:
            break
    assert te and not info["alive"] and info["equity"] <= 0.5 + 1e-9


def test_extra_features_are_past_only(setup):
    """The agent sees extra[d] at decision row d: changing rows after d must not change the obs."""
    u, md, prof = setup
    extra = np.arange(md.T, dtype=float)[:, None]
    env = TradeWorldEnv(md, prof, episode_decisions=10, extra=extra, random_start=False)
    o, _ = env.reset(seed=0)
    d = int(env.dec[0])
    assert o[-1] == d
    extra2 = extra.copy()
    extra2[d + 1:] = -1.0
    env2 = TradeWorldEnv(md, prof, episode_decisions=10, extra=extra2, random_start=False)
    o2, _ = env2.reset(seed=0)
    assert np.array_equal(o, o2)


def test_observation_is_finite_and_bounded(setup):
    u, md, prof = setup
    env = TradeWorldEnv(md, prof, episode_decisions=100)
    o, _ = env.reset(seed=1)
    for _ in range(100):
        assert np.all(np.isfinite(o)) and env.observation_space.contains(o)
        o, r, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            break
