"""Phase 2: synthetic markets have the statistical structure each experiment relies on."""
import numpy as np
import pytest

from tradeworld.learning import synthetic_tasks as st
from tradeworld.learning.env import TradeWorldEnv

N = 60_000


def rets(task):
    m = task.md.mid_c()[:, 0]
    return np.log(m[1:] / m[:-1])


@pytest.mark.parametrize("name", st.TASKS)
def test_tasks_are_deterministic_and_seed_dependent(name):
    a, b, c = st.build(name, 1, 5000), st.build(name, 1, 5000), st.build(name, 2, 5000)
    assert a.md.content_hash() == b.md.content_hash() != c.md.content_hash()


def test_s1_is_mean_reverting():
    r = rets(st.build("S1", 0, N))
    assert np.corrcoef(r[1:], r[:-1])[0, 1] == pytest.approx(-0.3, abs=0.02)


def test_s2_flag_predicts_next_return_sign():
    t = st.build("S2", 0, N)
    r = rets(t)                                   # r[k] = return of bar k+1
    flag_seen = t.extra[:-1, 0]                   # flag visible at close of bar k
    assert np.mean(r * flag_seen) > 0.1 * st.SIGMA


@pytest.mark.parametrize("name", ["S3", "S4", "S5"])
def test_random_walks_have_no_autocorrelation(name):
    r = rets(st.build(name, 0, N))
    assert abs(np.corrcoef(r[1:], r[:-1])[0, 1]) < 0.02


def test_s5_canary_is_future_return_and_never_observed():
    t = st.build("S5", 0, 5000)
    r = rets(t)
    assert np.allclose(t.hidden[:-1, 0], r, atol=1e-12)        # hidden[k] = return of bar k+1
    env = TradeWorldEnv(t.md, t.profile, episode_decisions=100, extra=t.extra, **t.env_kwargs)
    assert t.extra is None and env.observation_space.shape[0] == env.encoder.dim
    o, _ = env.reset(seed=0)
    d = int(env.dec[0])
    assert not np.any(np.isclose(o, t.hidden[d, 0], rtol=0, atol=1e-12) & (o != 0))


def test_s6_has_crashes_and_growth_optimal_leverage():
    t = st.build("S6", 0, N)
    r = rets(t)
    assert np.sum(r < -0.09) > 100
    assert 2.0 < t.params["growth_optimal_G"] < 4.0


def test_s7_matches_known_mu_sigma():
    t = st.build("S7", 0, 200_000)
    r = rets(t)
    assert r.std() == pytest.approx(1e-2, rel=0.01)
    assert t.params["growth_optimal_G"] == pytest.approx(5.0)
