"""Model-based growth-optimal agent, version 3 ("model-kelly-v3"): trading costs and how long an edge lasts.

Version 2 (Experiment M2) sized every decision for maximum expected log growth under a world model learned from
the agent's own experience, and passed the crash, known-optimum and no-edge markets, all without trading costs.
Version 3 keeps everything version 2 learns and adds what costs require. Everything new is learned from the
agent's own experience or read from its own observation; nothing about the market is given to it:

1. Calibration of the conditional model (estimation risk). Version 2 corrected the unconditional edge for
   estimation error (James-Stein) but trusted the conditional model's predictions as they were. Version 3 checks
   them on held-out experience (the chronologically last 20%, not used to fit the model): it regresses the realised
   simple return on the predicted edge, giving a slope b and its standard error. A slope below 1 means the
   predictions are noisier than they look. Each prediction's deviation from the average edge is multiplied by
   k = max(0, 1 - SE^2 / b^2) * min(b, 1) (positive-part James-Stein on the slope; the slope is not extrapolated
   above 1), and the average edge itself is corrected exactly as in version 2. Predictions that do not hold up out
   of sample are thereby shrunk towards no edge.
2. Persistence. rho = the lag-1 autocorrelation of the agent's own calibrated predicted edge along its experience.
   A position entered on an edge that persists for about 1 / (1 - rho) bars earns that edge for that long, so the
   cost of entering is weighed per bar as cost * (1 - rho). (A signal that flips sign, rho < 0, is charged more,
   because the position will soon have to be reversed.)
3. Cost-aware decision. The quoted spread and the agent's current position are part of its observation. A market
   order fills at the ask or the bid, so changing the position by dG (in multiples of equity) costs half the
   quoted spread times |dG|. The agent chooses
       G = argmax  E[ln(1 + G R)] - h_eff |G - G_now|,   h_eff = (spread / 2) * (1 - rho),
   a concave problem solved exactly: inside the no-trade band (|f'(G_now)| <= h_eff) the agent keeps its position;
   outside it, it moves just to the band's edge (f'(G) = +-h_eff). With zero cost this is exactly version 2's rule.

As in version 2, the expectation is evaluated by 32-node Gauss-Hermite quadrature per mixture component, which
treats moves up to about 10 standard deviations of each learned component as possible: a size at which such a
move would wipe out the account is never chosen. Where the unconstrained optimum lies beyond that (S1C/S2C at
about 150x and more), this binds; it is the agent's belief about the largest possible move, reported in every
evaluation as the share of decisions at that bound, and the reference policy uses the same rule. No style, holding
rule, exposure cap or penalty is imposed.
"""
from __future__ import annotations

import math

import numpy as np

from . import model_kelly2 as mk2
from .model_kelly import collect, market_columns  # noqa: F401  (same experience and observation columns)

CONFIG = dict(mk2.CONFIG, calibrate=True, persistence=True)


def edge_of(pi, mu, sd) -> np.ndarray:
    """E[e^x - 1] under each row's Gaussian mixture of the log return."""
    return (pi * (np.exp(mu + 0.5 * sd ** 2) - 1.0)).sum(1)


def shift_to(pi, mu, sd, target) -> np.ndarray:
    """Shift each row's components by one common amount so that the row's edge becomes `target`."""
    base = (pi * np.exp(mu + 0.5 * sd ** 2)).sum(1)
    return mu + np.log((1.0 + target) / base)[:, None]


def calibration(edge_hat: np.ndarray, realised: np.ndarray) -> dict:
    """Out-of-sample slope of the realised simple return on the predicted edge, with positive-part James-Stein."""
    e = edge_hat - edge_hat.mean()
    v = float((e ** 2).sum())
    if v <= 0:
        return dict(slope=0.0, slope_se=math.inf, t=0.0, k=0.0, n=int(len(e)))
    b = float((e * (realised - realised.mean())).sum() / v)
    resid = realised - realised.mean() - b * e
    se = float(math.sqrt((resid ** 2).sum() / max(len(e) - 2, 1) / v))
    js = max(0.0, 1.0 - se ** 2 / b ** 2) if b > 0 else 0.0
    return dict(slope=b, slope_se=se, t=b / se if se > 0 else math.inf, k=js * min(max(b, 0.0), 1.0), n=int(len(e)))


class Model:
    """The learned world model (version 2's artifact) plus version 3's calibration."""

    def __init__(self, art: dict):
        self.art = art
        self.v2 = mk2.Agent(art, None, None)
        self.cal = art.get("calibration")

    def predictive(self, X: np.ndarray):
        pi, mu, sd = self.v2.predictive(X)
        if self.art["kind"] == "conditional" and self.cal is not None:
            e = edge_of(pi, mu, sd)
            target = self.cal["level"] + self.cal["k"] * (e - self.cal["mean_edge"])
            mu = shift_to(pi, mu, sd, target)
        return pi, mu, sd


def learn(X: np.ndarray, y: np.ndarray, seed: int, config: dict = CONFIG) -> tuple[dict, list[dict]]:
    """Version 2's learning, then calibration (conditional model) and persistence, all from the same experience."""
    c = dict(config)
    art, hist = mk2.learn(X, y, seed, {k: v for k, v in c.items() if k in mk2.CONFIG})
    y = np.asarray(y, np.float64)
    cut = int(len(y) * (1 - c["val_frac"]))
    if art["kind"] == "conditional" and c["calibrate"]:
        pi, mu, sd = mk2.Agent(art, None, None).predictive(np.asarray(X, np.float32))
        e = edge_of(pi, mu, sd)
        cal = calibration(e[cut:], np.expm1(y[cut:]))
        # the average edge, corrected for estimation error exactly as version 2 corrects the unconditional edge
        simple = np.expm1(y)
        level, se = float(simple.mean()), float(simple.std(ddof=1) / math.sqrt(len(simple)))
        lk = max(0.0, 1.0 - se ** 2 / level ** 2) if level != 0 else 0.0
        cal.update(mean_edge=float(e[:cut].mean()), level=lk * level, level_raw=level, level_se=se, level_shrink=lk)
        art["calibration"] = cal
    model = Model(art)
    if c["persistence"]:
        pi, mu, sd = model.predictive(np.asarray(X, np.float32))
        e = edge_of(pi, mu, sd)
        rho = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if e.std() > 0 else 0.0
        art["persistence"] = dict(rho=rho if math.isfinite(rho) else 0.0)
    art["config"] = dict(art["config"], **{k: c[k] for k in ("calibrate", "persistence")})
    return art, hist


to_bytes, from_bytes = mk2.to_bytes, mk2.from_bytes


# ---------------------------------------------------------------------------- decision
def _grid(pi, mu, sd, z, wq):
    x = mu[:, :, None] + sd[:, :, None] * z[None, None, :]
    w = (pi[:, :, None] * wq[None, None, :]).reshape(len(pi), -1)
    return w, np.expm1(x).reshape(len(pi), -1)


def _deriv(w, R, g):
    return (w * R / (1 + g[:, None] * R)).sum(1)


def decide(pi, mu, sd, G_now, h, z, wq, gmax, iters) -> np.ndarray:
    """argmax_G E[ln(1 + G R)] - h |G - G_now| for each row (concave; exact no-trade band)."""
    w, R = _grid(pi, mu, sd, z, wq)
    neg = np.where(R < 0, -R, 0.0).max(1)
    pos = np.where(R > 0, R, 0.0).max(1)
    hi = np.minimum(np.where(neg > 0, (1 - 1e-9) / np.maximum(neg, 1e-300), gmax), gmax)
    lo = np.maximum(np.where(pos > 0, -(1 - 1e-9) / np.maximum(pos, 1e-300), -gmax), -gmax)
    G0 = np.clip(np.asarray(G_now, np.float64), lo, hi)
    h = np.broadcast_to(np.asarray(h, np.float64), G0.shape)
    d0 = _deriv(w, R, G0)
    up, dn = d0 > h, d0 < -h
    a, b = np.where(up, G0, lo), np.where(up, hi, G0)
    tgt = np.where(up, h, -h)
    for _ in range(iters):
        g = 0.5 * (a + b)
        dv = _deriv(w, R, g) - tgt
        a, b = np.where(dv > 0, g, a), np.where(dv > 0, b, g)
    return np.where(up | dn, 0.5 * (a + b), G0), lo, hi


class ObsReader:
    """Reads the quoted spread and the current position from the agent's own observation vector."""

    def __init__(self, encoder):
        from .features import PRICE_SHAPED  # noqa: F401
        n, W, F = encoder.n, encoder.W, encoder.F
        assert n == 1, "single-instrument agent"
        self.spread_col = (W - 1) * F + 3                  # last bar's quoted spread / window std
        self.vol_col = n * W * F                            # log10(window std) / 4 + 1
        self.expo_col = n * W * F + n + 7                   # slog(exposure) of the instrument

    def half_spread(self, obs) -> float:
        std = 10.0 ** (4.0 * (float(obs[self.vol_col]) - 1.0))
        return 0.5 * max(float(obs[self.spread_col]), 0.0) * std

    def exposure(self, obs) -> float:
        v = float(obs[self.expo_col])
        return math.copysign(math.expm1(abs(v)), v)


class Agent:
    def __init__(self, art: dict, cols: np.ndarray, mapper, encoder, cost_blind: bool = False):
        c = art["config"]
        self.cost_blind = cost_blind            # ablation only: decide as if trading were free
        self.art, self.cols, self.mapper = art, cols, mapper
        self.model = Model(art)
        self.reader = ObsReader(encoder)
        z, w = np.polynomial.hermite_e.hermegauss(c["quad_nodes"])
        self.z, self.wq = z, w / math.sqrt(2 * math.pi)
        self.gmax, self.iters = c["max_abs_size"], c["bisection_iters"]
        self.rho = float(art.get("persistence", {}).get("rho", 0.0))
        self.last = dict(G=0.0, at_bound=False, h=0.0, G_now=0.0)

    def target(self, x_market: np.ndarray, G_now: float, half_spread: float) -> float:
        pi, mu, sd = self.model.predictive(x_market[None, :])
        h = 0.0 if self.cost_blind else half_spread * (1.0 - self.rho)
        G, lo, hi = decide(pi, mu, sd, np.array([G_now]), h, self.z, self.wq, self.gmax, self.iters)
        g = float(G[0])
        self.last = dict(G=g, h=h, G_now=G_now, at_bound=bool(g >= hi[0] * (1 - 1e-6) or g <= lo[0] * (1 - 1e-6)))
        return g

    def action(self, G: float) -> np.ndarray:
        return mk2.Agent.action(self, G)

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, np.float32)
        return self.action(self.target(obs[self.cols], self.reader.exposure(obs), self.reader.half_spread(obs)))


def load_policy(blobs, sha: str, identity: dict, cost_blind: bool = False):
    from .env import TradeWorldEnv
    from .evaluate import mapper_from
    from . import synthetic_tasks as st
    art = from_bytes(blobs.get(sha))
    f = identity["frozen"]
    t = st.build(f["task"], 0, 2000)
    enc = TradeWorldEnv(t.md, t.profile, reward=f["reward"], episode_decisions=1, extra=t.extra, random_start=False,
                        **f["env_settings"]).encoder
    return Agent(art, market_columns(enc), mapper_from(identity), enc, cost_blind=cost_blind)


# ---------------------------------------------------------------------------- reference (not an agent)
def true_rho(task: str, params: dict) -> float:
    """Persistence of the true edge: S1C's return autocorrelation phi; S2C's regime flag (switching probability
    1 / mean length per bar, so autocorrelation 1 - 2 / mean length); zero for markets without a signal."""
    if task == "S1C":
        return float(params["phi"])
    if task == "S2C":
        return 1.0 - 2.0 / float(params["mean_regime_len"])
    return 0.0


def reference_policy(identity: dict):
    """Reference (not an agent; it learns nothing): the same cost-aware decision as version 3, but with the market's
    TRUE parameters in place of a learned model: S1C next log return ~ N(phi * last return, sigma); S2C ~ N(flag * mu,
    sigma); S3 no edge (flat); persistence from the true process. It reads the same observation as the agent (last
    return, regime flag, quoted spread, current position), so the comparison isolates what the agent had to learn."""
    from .env import TradeWorldEnv
    from .evaluate import mapper_from
    from . import synthetic_tasks as st
    f = identity["frozen"]
    t = st.build(f["task"], 0, 2000)
    enc = TradeWorldEnv(t.md, t.profile, reward=f["reward"], episode_decisions=1, extra=t.extra, random_start=False,
                        **f["env_settings"]).encoder
    reader = ObsReader(enc)
    p, task = t.params, f["task"]
    sigma = float(p.get("sigma", st.SIGMA))
    rho = true_rho(task, p)
    ret_col = (enc.W - 1) * enc.F + 0                      # last bar's return / window std
    flag_col = enc.dim - 1                                  # S2C: the visible regime flag (last extra feature)
    z, w = np.polynomial.hermite_e.hermegauss(CONFIG["quad_nodes"])
    wq = w / math.sqrt(2 * math.pi)
    shim = type("Shim", (), {})()
    shim.mapper = mapper_from(identity)

    def mean(obs) -> float:
        if task == "S1C":
            std = 10.0 ** (4.0 * (float(obs[reader.vol_col]) - 1.0))
            return float(p["phi"]) * float(obs[ret_col]) * std
        if task == "S2C":
            return float(obs[flag_col]) * float(p["mu"])
        return None

    def policy(obs):
        obs = np.asarray(obs, np.float32)
        m = mean(obs)
        if m is None:
            return mk2.Agent.action(shim, 0.0)
        G, _, _ = decide(np.ones((1, 1)), np.array([[m]]), np.array([[sigma]]), np.array([reader.exposure(obs)]),
                         reader.half_spread(obs) * (1.0 - rho), z, wq, CONFIG["max_abs_size"], CONFIG["bisection_iters"])
        return mk2.Agent.action(shim, float(G[0]))
    return policy
