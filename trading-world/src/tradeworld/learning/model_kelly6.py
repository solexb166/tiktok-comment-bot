"""Model-based growth-optimal agent, version 6 ("model-kelly-v6"): estimation risk measured robustly, for every model.

The first look at real data (2022, development only; reports/dev/real_2022_first_look.md) showed version 5 is not
safe on real prices: it sized 15-28x on effects that were mostly noise and lost on three of four instruments. Two causes:
per-regime trends and patterns were never corrected for estimation error, and the single-regime model's significance
measure assumed its return mixture was exact (a Wald statistic of about 10^6 on heavy-tailed, clustered returns).

Version 6 keeps version 5's world model and adds one step, learned from the agent's own experience: it trusts a
predicted edge only as far as such edges held up on experience it did not fit.
  * Held-out check: on the chronologically last 20% of experience (never used to fit the model), the realised simple
    return is regressed on the model's predicted edge (filtered one step ahead, exactly as when trading): slope b and
    the average realised return (level).
  * Robust uncertainty: the standard errors of b and of the level are measured by a moving-block bootstrap (blocks of
    BLOCK bars, BOOT resamples, fixed seed), which respects volatility clustering, fat tails and serial dependence;
    nothing assumes the model is exact.
  * Shrinkage (positive-part James-Stein): k = max(0, 1 - SE_b^2 / b^2) * min(b, 1); level kept in proportion
    max(0, 1 - SE_level^2 / level^2). Every prediction's edge becomes level_kept + k * (edge - average predicted edge),
    applied by shifting its predictive mixture (all regimes, all components) by one common amount.
  Effects that are real and large (the synthetic markets) give b near 1 with a small error, so almost nothing changes;
  effects that do not hold up out of sample are shrunk towards no edge, so the agent bets little or stays out.
Persistence is measured on the corrected edges. Decision, regime tracking and everything else are version 5's.
"""
from __future__ import annotations

import math

import numpy as np

from . import model_kelly2 as mk2
from . import model_kelly3 as mk3
from . import model_kelly5 as mk5
from .model_kelly import collect, market_columns  # noqa: F401
from .model_kelly5 import Features, _encoder, from_bytes, mixture, filter_predictive, stationary, to_bytes  # noqa: F401
from .model_kelly5 import oracle_policy, true_model  # noqa: F401  (references)

BLOCK = 100
BOOT = 400
CONFIG = dict(mk5.CONFIG, check_block=BLOCK, check_boot=BOOT)


def _edges(model: dict, pred: np.ndarray, X: np.ndarray) -> np.ndarray:
    out = np.empty(len(X))
    for t in range(len(X)):
        w, mu, sd = mixture(model, pred[t], X[t])
        out[t] = float(np.sum(w * (np.exp(mu + 0.5 * sd ** 2) - 1.0)))
    return out


def block_bootstrap(stat, n: int, block: int, boot: int, seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap: `stat(idx)` evaluated on `boot` resamples of whole blocks of consecutive indices."""
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = np.arange(0, max(n - block, 0) + 1)
    out = np.empty(boot)
    for b in range(boot):
        s = rng.choice(starts, nb)
        idx = (s[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        out[b] = stat(idx)
    return out


def held_out_check(edge: np.ndarray, realised: np.ndarray, block: int, boot: int) -> dict:
    e = edge - edge.mean()
    v = float((e ** 2).sum())
    level = float(realised.mean())

    def slope(idx):
        ee = edge[idx] - edge[idx].mean()
        vv = float((ee ** 2).sum())
        return float((ee * (realised[idx] - realised[idx].mean())).sum() / vv) if vv > 0 else 0.0

    b = float((e * (realised - realised.mean())).sum() / v) if v > 0 else 0.0
    se_b = float(np.std(block_bootstrap(slope, len(edge), block, boot, 1), ddof=1)) if v > 0 else math.inf
    se_l = float(np.std(block_bootstrap(lambda idx: float(realised[idx].mean()), len(edge), block, boot, 2), ddof=1))
    k = (max(0.0, 1.0 - se_b ** 2 / b ** 2) * min(b, 1.0)) if (b > 0 and math.isfinite(se_b)) else 0.0
    kl = max(0.0, 1.0 - se_l ** 2 / level ** 2) if level != 0 else 0.0
    return dict(slope=b, slope_se=se_b, slope_t=b / se_b if se_b > 0 else math.inf, k=k, level=level, level_se=se_l,
                level_t=level / se_l if se_l > 0 else math.inf, level_kept=kl, n=int(len(edge)), block=block, boot=boot)


def learn(X: np.ndarray, y: np.ndarray, seed: int, config: dict = CONFIG) -> tuple[dict, list[dict]]:
    c = dict(config)
    art, hist = mk5.learn(X, y, seed, {k: v for k, v in c.items() if k in mk5.CONFIG})
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64)
    cut = int(len(y) * (1 - c["val_frac"]))
    for key, pkey in (("model", "persistence"), ("static", "static_persistence")):
        mod = art[key]
        pred, _ = filter_predictive(mod, y, X)
        e = _edges(mod, pred, X)
        chk = held_out_check(e[cut:], np.expm1(y[cut:]), c["check_block"], c["check_boot"])
        chk["mean_edge"] = float(e[:cut].mean())
        art[f"{key}_check"] = chk
        ec = chk["level_kept"] * chk["level"] + chk["k"] * (e[cut:] - chk["mean_edge"])
        rho = float(np.corrcoef(ec[:-1], ec[1:])[0, 1]) if ec.std() > 0 else 0.0
        art[pkey] = dict(rho=rho if math.isfinite(rho) else 0.0)
    art["config"] = dict(art["config"], check_block=c["check_block"], check_boot=c["check_boot"])
    return art, hist


class Agent(mk5.Agent):
    """Version 5's regime-tracking agent; every predictive mixture is corrected by the held-out check."""

    def __init__(self, art: dict, mapper, encoder, static_only: bool = False):
        super().__init__(art, mapper, encoder, static_only=static_only)
        self.check = art["static_check" if static_only else "model_check"]

    def target(self, obs) -> float:
        obs = np.asarray(obs, np.float64)
        x = self.feat.from_full(obs)
        if self.x_prev is not None:
            self._observe(float(x[1]))
        self.x_prev = x
        w, mu, sd = mixture(self.model, self.p, x)
        edge = float(np.sum(w * (np.exp(mu + 0.5 * sd ** 2) - 1.0)))
        c = self.check
        target = c["level_kept"] * c["level"] + c["k"] * (edge - c["mean_edge"])
        mu = mk3.shift_to(w[None, :], mu[None, :], sd[None, :], np.array([target]))[0]
        G, _, _ = mk3.decide(w[None, :], mu[None, :], sd[None, :], np.array([self.reader.exposure(obs)]),
                             self.reader.half_spread(obs) * (1.0 - self.rho), self.z, self.wq, self.gmax, self.iters)
        return float(G[0])


def load_policy(blobs, sha: str, identity: dict, static_only: bool = False):
    from .evaluate import mapper_from
    return Agent(from_bytes(blobs.get(sha)), mapper_from(identity), _encoder(identity), static_only=static_only)


def true_filter_policy(identity: dict):
    return mk5.true_filter_policy(identity)
