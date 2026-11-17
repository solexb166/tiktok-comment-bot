"""Synthetic learning-validation markets S1-S7 (architecture §10.3, Phase 2 gate).

Each task is ONE synthetic instrument on an always-open M1 grid with known statistical structure,
so we know in advance what a working learner should do. Per-step returns are defined on the
decision grid (one decision per M1 bar in these tasks), which keeps the signal on the time scale
the agent acts on. Train and test markets come from different seeds (out-of-sample evaluation).

| Task | Market                                                  | Expected (gate)                                  |
|------|---------------------------------------------------------|--------------------------------------------------|
| S1   | AR(1) mean-reverting returns, with spread               | profitable; beats leverage-matched random agents |
| S2   | drift whose sign switches; regime flag visible          | position follows the flag                        |
| S3   | random walk, with spread                                | converges to ~flat                               |
| S4   | random walk, no costs                                   | no systematic profit                             |
| S5   | random walk + hidden future-return canary (not observed)| no profit (profit would mean a leak)             |
| S6   | positive drift + fat-tailed crashes                     | R1 moderate leverage survives; R2 high, dies     |
| S7   | GBM, known mu/sigma, no costs                           | R1 leverage -> mu/sigma^2 = 5; R2 does not       |
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..accounts.margin import MarginModel
from ..brokerspec.profile import EnvProfile, SlippageModel
from ..data.market_data import MarketData
from ..data.synthetic import from_mid_paths
from ..market.instruments import InstrumentSpec, Universe
from ..tracking.hashing import stable_int

START = 28_401_120          # 2024-01-01 00:00 UTC in minutes (hour-aligned; any value works)
SIGMA = 1e-3                # per-step return volatility used by S1-S5


@dataclass
class Task:
    name: str
    md: MarketData
    profile: EnvProfile
    extra: np.ndarray | None = None          # visible side features [T, E] (S2 regime flag)
    hidden: np.ndarray | None = None         # NOT given to the agent (S5 canary)
    params: dict = field(default_factory=dict)
    env_kwargs: dict = field(default_factory=dict)


def universe() -> Universe:
    return Universe((InstrumentSpec("SYN", "SYN", "USD", "fx", swap_free=True),))


def profile() -> EnvProfile:
    """A-reference margin (1:2000), no slippage: costs come only from each task's spread."""
    return EnvProfile(name="A_synthetic", environment="A", universe=universe(),
                      margin=MarginModel(account_leverage=2000.0),
                      slippage=SlippageModel(0.0, 0.0, 0.0), tags=("synthetic",))


def _market(log_ret: np.ndarray, spread_frac: float, range_frac: float) -> MarketData:
    mid = 100.0 * np.exp(np.cumsum(log_ret))
    return from_mid_paths(universe(), START, mid[:, None], spread_bps=spread_frac * 1e4,
                          range_bps=range_frac * 1e4)


def build(name: str, seed: int, steps: int = 200_000) -> Task:
    rng = np.random.default_rng(np.random.SeedSequence([seed, stable_int(name) % (2 ** 31)]))
    eps = rng.standard_normal(steps)
    env_kwargs = dict(decision_interval=1, obs_resolution=1, window=30)
    if name == "S1":
        phi, spread = -0.3, 0.1 * SIGMA
        r = np.empty(steps)
        prev = 0.0
        for k in range(steps):
            prev = phi * prev + SIGMA * eps[k]
            r[k] = prev
        return Task(name, _market(r, spread, 0.3 * SIGMA), profile(),
                    params=dict(phi=phi, sigma=SIGMA, spread=spread), env_kwargs=env_kwargs)
    if name == "S2":
        mu, spread, mean_len = 0.15 * SIGMA, 0.05 * SIGMA, 400
        flag = np.empty(steps)
        state, k = 1.0, 0
        while k < steps:
            n = int(rng.geometric(1.0 / mean_len))
            flag[k:k + n] = state
            state, k = -state, k + n
        r = flag * mu + SIGMA * eps
        # the flag in effect for bar t+1 is visible at the close of bar t (a known regime, not a forecast)
        visible = np.r_[flag[1:], flag[-1]][:, None]
        return Task(name, _market(r, spread, 0.3 * SIGMA), profile(), extra=visible,
                    params=dict(mu=mu, sigma=SIGMA, spread=spread, mean_regime_len=mean_len),
                    env_kwargs=env_kwargs)
    if name in ("S3", "S4", "S5"):
        spread = 0.1 * SIGMA if name == "S3" else 0.0
        r = SIGMA * eps
        hidden = None
        if name == "S5":
            hidden = np.r_[r[1:], 0.0][:, None]          # next bar's return: must never be observable
        return Task(name, _market(r, spread, 0.3 * SIGMA), profile(), hidden=hidden,
                    params=dict(sigma=SIGMA, spread=spread), env_kwargs=env_kwargs)
    if name == "S6":
        mu, sigma, p, J = 5e-4, 3e-3, 1 / 300, 0.10
        crash = rng.random(steps) < p
        r = np.where(crash, math.log(1 - J), mu - 0.5 * sigma ** 2 + sigma * eps)
        G = np.linspace(0.01, 1 / J - 1e-3, 20000)
        g = (1 - p) * (G * mu - G ** 2 * sigma ** 2 / 2) + p * np.log(1 - G * J)
        return Task(name, _market(r, 0.0, 0.3 * sigma), profile(),
                    params=dict(mu=mu, sigma=sigma, crash_prob=p, crash=J, growth_optimal_G=float(G[g.argmax()])),
                    env_kwargs=env_kwargs)
    if name == "S7":
        mu, sigma = 5e-4, 1e-2
        r = mu - 0.5 * sigma ** 2 + sigma * eps
        return Task(name, _market(r, 0.0, 0.3 * sigma), profile(),
                    params=dict(mu=mu, sigma=sigma, growth_optimal_G=mu / sigma ** 2), env_kwargs=env_kwargs)
    raise ValueError(f"unknown task {name!r}")


TASKS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
