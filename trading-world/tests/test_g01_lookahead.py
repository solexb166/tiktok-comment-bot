"""Group 1: no look-ahead. Observations and all account activity up to time t are independent
of data after t; observation bars are built only from completed M1 bars."""
import numpy as np

from tradeworld.agents.baselines import RandomPolicy
from tradeworld.brokerspec import load_profile
from tradeworld.data.market_data import FIELDS, MarketData
from tradeworld.data.synthetic import generate
from tradeworld.execution.engine import MarketView
from tradeworld.interface.observation import build_observation
from tradeworld.accounts.account import Account
from tradeworld.replay.replay import Replay
from conftest import MON_2024_01_08, run


def perturb_after(md, m, seed=99):
    """Copy of md where every row > m is scaled by a random per-row/instrument factor."""
    rng = np.random.default_rng(seed)
    f = np.ones((md.T, md.N))
    f[m + 1:] = rng.uniform(0.9, 1.1, (md.T - m - 1, md.N))
    arrays = {k: getattr(md, k) * (f if k != "volume" else 1.0) for k in FIELDS}
    arrays["volume"] = np.where(np.arange(md.T)[:, None] > m, 7.0, md.volume)
    return MarketData.from_raw(md.times, md.symbols, arrays, md.quoted)


def test_observation_independent_of_future():
    prof = load_profile("environments/A_fractional.yaml")
    md = generate(prof.universe, MON_2024_01_08, 1440, seed=1)
    rp = Replay(md, 5, 5, 120)
    d = int(rp.decision_indices[10])
    md2 = perturb_after(md, d)
    obs = []
    for data in (md, md2):
        acct = Account("a", prof, np.random.default_rng(0))
        obs.append(build_observation(Replay(data, 5, 5, 120), MarketView(data, prof), acct, d))
    assert np.array_equal(obs[0].market, obs[1].market)
    assert np.array_equal(obs[0].account, obs[1].account)
    # sanity: the perturbation does change later observations
    later = int(rp.decision_indices[40])
    w1, w2 = Replay(md, 5, 5, 120).market_window(later), Replay(md2, 5, 5, 120).market_window(later)
    assert not np.array_equal(w1, w2)


def test_full_run_history_independent_of_future():
    prof = load_profile("environments/A_fractional.yaml")
    md = generate(prof.universe, MON_2024_01_08, 2 * 1440, seed=2)
    m = 1500
    md2 = perturb_after(md, m)
    cut = int(md.times[m])
    logs = []
    for data in (md, md2):
        res = run(data, prof, {"r": RandomPolicy(6, seed=5)}, seed=3, interval=5, resolution=5, window=12)
        logs.append([e.canonical() for e in res.accounts["r"].events if e.t <= cut])
    assert logs[0] == logs[1] and len(logs[0]) > 10


def test_obs_bars_only_from_completed_m1_bars():
    prof = load_profile("environments/A_fractional.yaml")
    md = generate(prof.universe, MON_2024_01_08, 600, seed=3)
    rp = Replay(md, 1, 5, 2)
    assert rp.obs_bar(8) == 0           # rows 5..9 (bar 1) not complete at the close of row 8
    assert rp.obs_bar(9) == 1           # bar 1 complete at the close of row 9
    mid_c = md.mid_c()
    j = rp.obs_bar(37)
    # last row aggregated into bar j is <= 37
    assert 5 * j + 4 <= 37
    expected_ret = np.log(mid_c[5 * j + 4] / mid_c[5 * j - 1])
    assert np.allclose(rp.bar_features[j, :, 0], expected_ret, rtol=1e-5)


def test_decisions_aligned_and_fillable():
    prof = load_profile("environments/A_fractional.yaml")
    md = generate(prof.universe, MON_2024_01_08, 3000, seed=4)
    rp = Replay(md, 5, 5, 120)
    assert np.all((md.times[rp.decision_indices] + 1) % 5 == 0)
    assert rp.decision_indices.max() + 1 < md.T
    assert rp.obs_bar(int(rp.decision_indices[0])) >= 120
