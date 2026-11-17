"""Decision-quality score for markets with a known return process (S6, S7)."""
import math

import numpy as np
import pytest

from tradeworld.learning import synthetic_tasks as st
from tradeworld.learning.quality import KnownMarket, episode_expected_growth, instability, quality_metrics


@pytest.fixture(scope="module")
def s6():
    return KnownMarket.from_task(st.build("S6", 0, 1000))


def test_staying_out_scores_zero(s6):
    assert s6.growth(0.0)[0] == 0.0
    m = quality_metrics(np.zeros(1000), s6)
    assert m["quality"] == 0.0 and m["ruinous_share"] == 0.0 and m["episode_quality"] == 0.0


def test_optimum_is_about_2_9x_and_scores_one(s6):
    G, g = s6.optimal()
    assert 2.85 < G < 3.0 and g > 0
    assert quality_metrics(np.full(1000, G), s6)["quality"] == pytest.approx(1.0)
    for other in (0.5 * G, 1.5 * G):
        assert s6.growth(other)[0] < g


def test_s7_optimum_matches_theory():
    mk = KnownMarket.from_task(st.build("S7", 0, 1000))
    assert mk.optimal()[0] == pytest.approx(5.0, abs=0.05)                  # mu / sigma^2


def test_excessive_sizes_are_negative_and_ruinous(s6):
    assert s6.growth(6.0)[0] < 0 and s6.growth(9.0)[0] < s6.growth(6.0)[0]
    assert np.isneginf(s6.growth(10.0)[0])
    assert s6.ruin_long() == pytest.approx(9.5) and bool(s6.ruinous(9.5)) and not bool(s6.ruinous(9.4))
    q = [quality_metrics(np.full(1000, G), s6)["episode_quality"] for G in (2.93, 6.0, 9.0, 12.0)]
    assert q[0] > 0.9 and q[0] > q[1] > q[2] > q[3] and q[3] < -8


def test_formula_matches_monte_carlo(s6):
    rng = np.random.default_rng(0)
    n = 2_000_000
    crash = rng.random(n) < s6.crash_prob
    r = np.where(crash, math.log(1 - s6.crash), s6.m + s6.sigma * rng.standard_normal(n))
    for G in (1.0, 2.93, 5.0):
        x = np.log(1 + G * np.expm1(r))
        assert abs(x.mean() - s6.growth(G)[0]) < 4 * x.std() / math.sqrt(n)


def test_score_ignores_the_realised_path(s6):
    G = np.r_[np.full(300, 2.0), np.full(300, 12.0)]
    assert quality_metrics(G, s6) == quality_metrics(G, s6)                 # depends only on decisions
    q = quality_metrics(np.full(1000, 12.0), s6)
    assert q["ruinous_share"] == 1.0 and q["episode_quality"] < -8          # a lucky path cannot help


def test_instability_measures_size_jumps():
    steady = instability(np.full(100, 2.0))
    jumpy = instability(np.tile([1.0, 20.0], 50))
    assert steady["median_size_change"] == pytest.approx(1.0) and steady["share_size_jumps_10x"] == 0
    assert jumpy["median_size_change"] == pytest.approx(20.0) and jumpy["share_size_jumps_10x"] == 1.0
    assert instability([1.0, 50.0, 1.0], episodes=[0, 1, 1])["share_size_jumps_10x"] == 1.0   # not across episodes


def test_episode_growth_is_deterministic_and_death_is_permanent(s6):
    a = episode_expected_growth(np.full(1000, 12.0), s6, 200, seed=3)
    b = episode_expected_growth(np.full(1000, 12.0), s6, 200, seed=3)
    assert np.array_equal(a, b) and np.mean(a <= math.log(0.05) + 1e-12) > 0.9
