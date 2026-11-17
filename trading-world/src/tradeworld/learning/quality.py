"""Decision quality for markets whose return process is known exactly (S6, S7).

S6 bar return (one decision = one M1 bar; synthetic_tasks.build):
    r = ln(1 - J)                      with probability p        (crash)
    r = m + sigma * Z,  Z ~ N(0, 1)    with probability 1 - p    (m = mu - sigma^2 / 2)
S7 is the same with p = 0.

A decision holds net exposure G (position value / equity, signed) over the next bar. S6 and
S7 have no spread, slippage, commission or swap, and the fill happens at the previous close, so
the account's equity multiplier over the bar is exactly  M(G) = 1 + G * (e^r - 1).

Expected log growth of a decision (Kelly growth rate, the expectation of the R1 reward):
    g(G) = (1 - p) * E_Z[ ln(1 + G (e^(m + sigma Z) - 1)) ] + p * ln(1 - G J)
g depends only on the decision G and the market's known parameters, never on the realised
path, so a lucky outcome cannot raise it and an unlucky one cannot lower it.

  g(0) = 0 (staying out); g is maximised at the growth-optimal size G*; g(G*) > 0.
  Ruinous decision: one ordinary adverse move would end the account (equity <= 5% death line):
      long  G >= (1 - 0.05) / J                    (a crash; 9.5x in S6)
      short |G| >= (1 - 0.05) / (e^(m + 5 sigma) - 1)    (a 5-sigma up move)
  For G >= 1/J, g is minus infinity (a crash takes equity to zero or below).

Scores:
  quality  q = mean of g(G) / g(G*) over non-ruinous decisions: 1 = optimal sizing, 0 = no better
           than staying out, negative = sizes that lose on average.
  ruinous share = fraction of ruinous decisions (reported separately).
  episode quality Q = expected final log growth of an episode's recorded decision sequence,
           replayed on K simulated market paths drawn from the known process, with death at 5%
           permanent (no growth after it), divided by H * g(G*) for H decisions. It weighs good
           sizing against ruin on one scale, uses no realised returns (luck-independent), and is
           1 for optimal sizing throughout, 0 for staying out, very negative for ruinous play.
           (A per-decision score with death as a one-step floor was rejected: it treats death as a
           bounded loss and so rises again above 10x, rewarding larger ruinous bets.)
The Z-expectation is computed with 96-point Gauss-Hermite quadrature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEATH = 0.05
_Z, _W = np.polynomial.hermite_e.hermegauss(96)
_W = _W / math.sqrt(2 * math.pi)


@dataclass(frozen=True)
class KnownMarket:
    mu: float
    sigma: float
    crash_prob: float = 0.0
    crash: float = 0.0

    @classmethod
    def from_task(cls, task) -> "KnownMarket | None":
        p = task.params
        if task.name == "S6":
            return cls(p["mu"], p["sigma"], p["crash_prob"], p["crash"])
        if task.name == "S7":
            return cls(p["mu"], p["sigma"])
        return None

    @property
    def m(self) -> float:
        return self.mu - 0.5 * self.sigma ** 2

    def _branches(self, G, floor):
        G = np.atleast_1d(np.asarray(G, dtype=float))
        x = np.expm1(self.m + self.sigma * _Z)                      # e^r - 1 at the quadrature nodes
        mult = 1.0 + G[:, None] * x[None, :]
        normal = (np.log(np.maximum(mult, floor)) * _W).sum(axis=1)
        crash_mult = 1.0 - G * self.crash
        crash = np.log(np.maximum(crash_mult, floor)) if self.crash_prob else np.zeros_like(G)
        return normal, crash, crash_mult

    def growth(self, G) -> np.ndarray:
        """Expected log growth per decision (minus infinity where a crash wipes out the account)."""
        normal, crash, crash_mult = self._branches(G, 1e-300)
        g = (1 - self.crash_prob) * normal + self.crash_prob * crash
        if self.crash_prob:
            g = np.where(crash_mult <= 0, -np.inf, g)
        return g

    def optimal(self) -> tuple[float, float]:
        hi = (1 / self.crash - 1e-9) if self.crash_prob else 4 * self.mu / self.sigma ** 2 + 1
        grid = np.linspace(0, hi, 4001)
        k = int(np.argmax(self.growth(grid)))
        lo, up = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        for _ in range(60):                                           # golden-section refinement
            a = up - (up - lo) / 1.618033988749895
            b = lo + (up - lo) / 1.618033988749895
            if self.growth(a)[0] > self.growth(b)[0]:
                up = b
            else:
                lo = a
        G = (lo + up) / 2
        return float(G), float(self.growth(G)[0])

    def ruin_long(self) -> float:
        return (1 - DEATH) / self.crash if self.crash_prob else math.inf

    def ruin_short(self) -> float:
        return (1 - DEATH) / math.expm1(self.m + 5 * self.sigma)

    def ruinous(self, G) -> np.ndarray:
        G = np.asarray(G, dtype=float)
        return (G >= self.ruin_long()) | (-G >= self.ruin_short())


def episode_expected_growth(G, market: KnownMarket, paths: int = 500, seed: int = 0) -> np.ndarray:
    """Final log growth of one episode's decision sequence G on `paths` simulated market paths
    (death at 5% is permanent). Deterministic for a given seed; uses no realised returns."""
    G = np.asarray(G, dtype=float)
    rng = np.random.default_rng(seed)
    H = len(G)
    crash = rng.random((paths, H)) < market.crash_prob
    r = np.where(crash, math.log(1 - market.crash) if market.crash_prob else 0.0,
                 market.m + market.sigma * rng.standard_normal((paths, H)))
    mult = 1.0 + G[None, :] * np.expm1(r)
    logm = np.log(np.maximum(mult, 1e-300))
    cum = np.cumsum(logm, axis=1)
    dead = (cum <= math.log(DEATH)) | (mult <= 0).cumsum(axis=1).astype(bool)
    first = np.where(dead.any(axis=1), dead.argmax(axis=1), H)
    final = np.where(first < H, math.log(DEATH), cum[:, -1] if H else 0.0)
    return final


def quality_metrics(exposure, market: KnownMarket, episodes=None, paths: int = 2000) -> dict:
    """Decision-quality summary for recorded decisions (signed net exposures)."""
    G = np.asarray(exposure, dtype=float)
    g_star_G, g_star = market.optimal()
    ruin = market.ruinous(G)
    q = market.growth(G[~ruin]) / g_star if (~ruin).any() else np.array([])
    ep = np.zeros(len(G), dtype=int) if episodes is None else np.asarray(episodes)
    per_ep = []
    for k, e in enumerate(np.unique(ep)):
        g = G[ep == e]
        # common random numbers: episode k of every version is scored on the same simulated paths,
        # so differences between versions are not blurred by simulation noise
        per_ep.append(float(np.mean(episode_expected_growth(g, market, paths, seed=k))) / (len(g) * g_star))
    return dict(optimal_size=g_star_G, optimal_growth_per_decision=g_star,
                quality=float(np.mean(q)) if len(q) else None,
                ruinous_share=float(np.mean(ruin)) if len(G) else None,
                episode_quality=float(np.mean(per_ep)) if per_ep else None, episode_quality_by_episode=per_ep,
                ruin_size_long=market.ruin_long(), ruin_size_short=market.ruin_short())


def instability(exposure, episodes=None) -> dict:
    """How erratically position size changes between consecutive decisions."""
    G = np.abs(np.asarray(exposure, dtype=float))
    ep = np.zeros(len(G)) if episodes is None else np.asarray(episodes)
    a, b = G[:-1], G[1:]
    ok = (ep[:-1] == ep[1:]) & (a > 1e-6) & (b > 1e-6)
    if not ok.any():
        return dict(median_size_change=None, share_size_jumps_10x=None)
    r = np.abs(np.log(b[ok] / a[ok]))
    return dict(median_size_change=float(np.exp(np.median(r))),            # typical factor between decisions
                share_size_jumps_10x=float(np.mean(r > math.log(10))))


# ---------------------------------------------------------------------------- Q2 (proposed final)
# Version history of the decision-quality score (kept, never overwritten):
#   q0  per-decision g with death as a one-step floor ......... rejected: rises above 10x
#   Q1  episode replay on simulated paths, permanent death ..... rejected: 20x scored above 10x
#       (once death is near certain, the rare crash-free paths decide, and they favour size)
#   Q2  clamped Kelly score, below .............................. proposed
Q2_VERSION = "Q2.0"


def q2_floor(market: KnownMarket) -> float:
    """The score of the smallest ruinous long position: g(G_ruin) / g(G*)."""
    G_star, g_star = market.optimal()
    G_ruin = market.ruin_long() if market.crash_prob else (1 - DEATH) / -math.expm1(market.m - 5 * market.sigma)
    return float(market.growth(G_ruin)[0] / g_star)


def q2_scores(exposure, market: KnownMarket) -> np.ndarray:
    """Q2 per decision:  s(G) = max(g(G) / g(G*), F)  and  s(G) = F for a ruinous decision,
    where F = g(G_ruin) / g(G*) is the score of the smallest ruinous long position.
    Because g is concave, s rises to 1 at G* and never increases again as the position grows
    beyond it; every ruinous size scores exactly F, the worst possible value."""
    G = np.asarray(exposure, dtype=float)
    _, g_star = market.optimal()
    F = q2_floor(market)
    if market.crash_prob:
        ruin = market.ruinous(G)
    else:
        ruin = (G >= (1 - DEATH) / -math.expm1(market.m - 5 * market.sigma)) | (-G >= market.ruin_short())
    s = np.full(G.shape, F)
    ok = ~ruin
    if ok.any():
        s[ok] = np.maximum(market.growth(G[ok]) / g_star, F)
    return s


def q2_metrics(exposure, market: KnownMarket, episodes=None) -> dict:
    """Q2 for recorded decisions: mean per episode (for paired comparisons) and overall."""
    s = q2_scores(exposure, market)
    ep = np.zeros(len(s), dtype=int) if episodes is None else np.asarray(episodes)
    per_ep = [float(s[ep == e].mean()) for e in np.unique(ep)]
    return dict(version=Q2_VERSION, q2=float(np.mean(per_ep)) if per_ep else None, q2_by_episode=per_ep,
                floor=q2_floor(market), optimal_size=market.optimal()[0])
