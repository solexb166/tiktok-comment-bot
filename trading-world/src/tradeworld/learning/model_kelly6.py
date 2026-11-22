"""Model-based growth-optimal agent, version 6 ("model-kelly-v6"): estimation risk measured robustly, for every model.

The first look at real data (2022, development only; reports/dev/real_2022_first_look.md) showed version 5 is not
safe on real prices: it sized 15-28x on effects that were mostly noise and lost on three of four instruments. Two causes:
per-regime trends and patterns were never corrected for estimation error, and the single-regime model's significance
measure assumed its return mixture was exact (a Wald statistic of about 10^6 on heavy-tailed, clustered returns).

Version 6 keeps version 5's world model and adds one step, learned from the agent's own experience: it trusts a
predicted edge only as far as such edges held up on experience it did not fit.
  * Forward check: the agent replays its own history as if live - for each of several stretches (from 55%, 70% and 85%
    of its experience onward) it learns from everything before the stretch, exactly as it learns, and predicts the
    stretch one step ahead. On all these predictions pooled, the realised simple return is regressed THROUGH THE ORIGIN on the model's whole predicted edge, trend included (filtered one step
    ahead, exactly as when trading): slope b. (A first development version checked the trend separately, by whether the
    held-out average return was significant; a rolling walk-forward on real 2022 data showed that lets a past trend
    through even when the learned trend did not continue, with 25-90x bets. The through-origin check asks the right
    question: did what the model predicted, trend included, continue? A single last-20% slice was also too little
    evidence: weak edges passed partly and were bet at 11-26x; pooling three forward stretches roughly triples it.)
  * Robust uncertainty: the standard error of b is measured by a moving-block bootstrap (blocks of
    BLOCK bars, BOOT resamples, fixed seed), which respects volatility clustering, fat tails and serial dependence;
    nothing assumes the model is exact.
  * Shrinkage (positive-part James-Stein): k = max(0, 1 - SE_b^2 / b^2) * min(b, 1). Every prediction's edge becomes
    k * edge, applied by shifting its predictive mixture (all regimes, all components) by one common amount.
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
FOLDS = (0.55, 0.70, 0.85)
CONFIG = dict(mk5.CONFIG, check_block=BLOCK, check_boot=BOOT, check_folds=FOLDS)


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
    """Does the model's WHOLE predicted edge (trend included) hold up on experience it did not fit? Through-origin slope
    of realised simple return on predicted edge, b = sum(e r) / sum(e^2), with a moving-block-bootstrap standard error;
    kept fraction k = max(0, 1 - SE^2 / b^2) * min(b, 1) (0 if b <= 0). A trend learned in the fitted period that did not
    continue in the held-out period gives b near 0, so it is not bet on."""
    v = float((edge ** 2).sum())

    def slope(idx):
        vv = float((edge[idx] ** 2).sum())
        return float((edge[idx] * realised[idx]).sum() / vv) if vv > 0 else 0.0

    b = float((edge * realised).sum() / v) if v > 0 else 0.0
    se = float(np.std(block_bootstrap(slope, len(edge), block, boot, 1), ddof=1)) if v > 0 else math.inf
    k = (max(0.0, 1.0 - se ** 2 / b ** 2) * min(b, 1.0)) if (b > 0 and math.isfinite(se) and se > 0) else 0.0
    return dict(slope=b, slope_se=se, slope_t=b / se if se > 0 else math.inf, k=k, n=int(len(edge)), block=block, boot=boot,
                mean_edge=float(edge.mean()), realised_mean=float(realised.mean()))


def forward_folds(X: np.ndarray, y: np.ndarray, seed: int, c: dict) -> dict:
    """Replay the agent's own history as if live: for each fold start a, learn from everything before a (exactly as the
    agent learns), then predict the stretch up to the next fold start, filtering one step ahead as when trading.
    Returns, for the regime model and for the single-regime model, pooled predicted edges, realised simple returns and
    per-fold summaries."""
    starts = [int(len(y) * f) for f in c["check_folds"]] + [len(y)]
    out = {key: ([], [], []) for key in ("model", "static")}
    sub = {k: v for k, v in c.items() if k in mk5.CONFIG}
    for a, b in zip(starts[:-1], starts[1:]):
        art_f, _ = mk5.learn(X[:a], y[:a], seed, sub)
        r = np.expm1(y[a:b])
        for key in ("model", "static"):
            mod = art_f[key]
            pred, _ = filter_predictive(mod, y[:b], X[:b])
            e = _edges(mod, pred[a:b], X[a:b])
            out[key][0].append(e)
            out[key][1].append(r)
            out[key][2].append(dict(start=a, end=b, regimes=int(mod["K"]),
                                    corr=float(np.corrcoef(e, r)[0, 1]) if e.std() > 0 else 0.0))
    return {key: (np.concatenate(E), np.concatenate(R), info) for key, (E, R, info) in out.items()}


def learn(X: np.ndarray, y: np.ndarray, seed: int, config: dict = CONFIG) -> tuple[dict, list[dict]]:
    """Version 5's learning on all experience, then the forward check: the same learning procedure replayed on the
    agent's own history (several stretches, each predicted by a model that never saw it); the kept fraction applies
    to the final model's predictions."""
    c = dict(config)
    art, hist = mk5.learn(X, y, seed, {k: v for k, v in c.items() if k in mk5.CONFIG})
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64)
    folds = forward_folds(X, y, seed, c)
    for key, pkey in (("model", "persistence"), ("static", "static_persistence")):
        e, r, info = folds[key]
        chk = held_out_check(e, r, c["check_block"], c["check_boot"])
        chk["folds"] = info
        art[f"{key}_check"] = chk
        pred, _ = filter_predictive(art[key], y, X)
        tail = slice(int(len(y) * c["check_folds"][-1]), len(y))
        ec = chk["k"] * _edges(art[key], pred[tail], X[tail])
        rho = float(np.corrcoef(ec[:-1], ec[1:])[0, 1]) if ec.std() > 0 else 0.0
        art[pkey] = dict(rho=rho if math.isfinite(rho) else 0.0)
    art["config"] = dict(art["config"], check_block=c["check_block"], check_boot=c["check_boot"], check_folds=list(c["check_folds"]))
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
        target = self.check["k"] * edge
        mu = mk3.shift_to(w[None, :], mu[None, :], sd[None, :], np.array([target]))[0]
        G, _, _ = mk3.decide(w[None, :], mu[None, :], sd[None, :], np.array([self.reader.exposure(obs)]),
                             self.reader.half_spread(obs) * (1.0 - self.rho), self.z, self.wq, self.gmax, self.iters)
        return float(G[0])


def load_policy(blobs, sha: str, identity: dict, static_only: bool = False):
    from .evaluate import mapper_from
    return Agent(from_bytes(blobs.get(sha)), mapper_from(identity), _encoder(identity), static_only=static_only)


def true_filter_policy(identity: dict):
    return mk5.true_filter_policy(identity)
