"""Model-based growth-optimal agent, version 4 ("model-kelly-v4"): markets that change.

Versions 2 and 3 learn one world model from their experience and keep it. Version 4 learns that a market may have a
few hidden moods (regimes) and keeps track, while trading, of which one is in force. Everything is learned from the
agent's own experience or read from its own observation; nothing about the market is given to it.

World model (fitted by expectation-maximisation over the hidden regimes, Baum-Welch):
    next log return y | regime k, features x  ~  sum_j pi[k, j] N( x . beta[k] + m[j], (s[j] c[k])^2 )
  * x = (1, the last three bar returns, the visible side features), read from the observation;
  * shared return shapes (m, s): version 2's unconditional mixture of experienced returns (EM, number of components
    by held-out likelihood), so crashes are a component whose weight can differ by regime;
  * per regime: component weights pi[k] (e.g. crash frequency), dependence on the features beta[k] (e.g. the
    direction of a pattern), volatility scale c[k];
  * a Markov chain between regimes (how long each lasts).
  Number of regimes K in {1, 2, 3}: each candidate is fitted on the first 80% of experience (K > 1 from four starting
  points, keeping the best training fit); a larger K is kept only if it predicts the held-out last 20% significantly
  better (paired t > 2 on per-bar log likelihood). K = 1 is a market that does not change.
  Estimation risk (K = 1): the average edge is corrected as in version 2 (positive-part James-Stein) and the feature
  coefficients by positive-part James-Stein on their joint Wald statistic, so chance patterns are not traded.
  Persistence: the lag-1 autocorrelation of the agent's own predicted edge on held-out experience (as version 3).
Decision: at every bar the agent updates its belief about the current regime with the bar just observed (forward
  filter; each test run starts from the regimes' long-run frequencies), forms the predictive mixture and chooses its
  position exactly as version 3: maximum expected log growth after the cost of getting there (half the quoted spread
  per unit traded, weighed per bar by persistence; 32-node quadrature, so a size at which a ~10-sd move of a learned
  component would wipe out the account is never chosen). No style, holding rule, exposure cap or penalty is imposed.
"""
from __future__ import annotations

import io
import math

import numpy as np

from . import model_kelly2 as mk2
from . import model_kelly3 as mk3
from .model_kelly import collect, market_columns  # noqa: F401

CONFIG = dict(max_components=4, em_iters=300, em_tol=1e-9, val_frac=0.2, max_regimes=3, regime_starts=4,
              regime_iters=40, regime_tol=1e-7, init_block=500, select_t=2.0, quad_nodes=32, max_abs_size=2000.0,
              bisection_iters=100)


# ---------------------------------------------------------------------------- features from the observation
class Features:
    """x = (1, last three bar log returns, visible side features), recovered from the observation vector."""

    def __init__(self, encoder):
        n, W, F = encoder.n, encoder.W, encoder.F
        assert n == 1, "single-instrument agent"
        self.ret_cols = [(W - 1) * F, (W - 2) * F, (W - 3) * F]
        self.vol_col = n * W * F
        self.extra_cols = list(range(encoder.dim - encoder.E, encoder.dim))
        self.d = 4 + len(self.extra_cols)

    def from_full(self, v) -> np.ndarray:
        v = np.asarray(v, np.float64)
        std = 10.0 ** (4.0 * (v[..., self.vol_col] - 1.0))
        r = np.stack([v[..., c] * std for c in self.ret_cols], -1)
        ones = np.ones(v.shape[:-1] + (1,))
        parts = [ones, r] + ([v[..., self.extra_cols]] if self.extra_cols else [])
        return np.concatenate(parts, -1)

    def from_market_columns(self, Xm: np.ndarray, cols: np.ndarray) -> np.ndarray:
        full = np.zeros((len(Xm), int(cols.max()) + 1))
        full[:, cols] = Xm
        return self.from_full(full)


# ---------------------------------------------------------------------------- regime model
def _comp(y, mean, m, s, c):
    sd = s[None, None, :] * c[None, :, None]
    z = (y[:, None, None] - mean[:, :, None] - m[None, None, :]) / sd
    return np.exp(-0.5 * z * z) / (sd * math.sqrt(2 * math.pi))


def _forward(E, A, p0):
    T, K = E.shape
    a = np.empty((T, K))
    pred = np.empty((T, K))
    c = np.empty(T)
    p = p0.copy()
    for t in range(T):
        pred[t] = p
        v = p * E[t]
        s = v.sum()
        s = s if s > 0 else 1e-300
        c[t] = s
        a[t] = v / s
        p = a[t] @ A
    return a, pred, c


def _backward(E, A, a, c):
    T, K = E.shape
    g = np.empty((T, K))
    g[-1] = a[-1]
    b = np.ones(K)
    xi = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        nb = (A * (E[t + 1] * b)[None, :]) / c[t + 1]
        xi += a[t][:, None] * nb
        b = nb.sum(1)
        gt = a[t] * b
        g[t] = gt / gt.sum()
    return g, xi


def _init_labels(y, X, K, block, seed):
    rng = np.random.default_rng(seed)
    n = len(y) // block
    scale = y.std() / (X[:, 1:].std(0) + 1e-12)
    feats = []
    for i in range(n):
        sl = slice(i * block, (i + 1) * block)
        Xi, yi = X[sl], y[sl]
        b = np.linalg.lstsq(Xi.T @ Xi + 1e-12 * np.eye(X.shape[1]), Xi.T @ yi, rcond=None)[0]
        r = yi - Xi @ b
        feats.append(np.r_[b[1:] / scale, math.log(r.std() + 1e-12), np.mean(np.abs(r) > 4 * y.std())])
    F = np.array(feats)
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    cent = F[rng.choice(len(F), K, replace=False)]
    for _ in range(50):
        lab = ((F[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
        cent = np.array([F[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(K)])
    labels = np.repeat(lab, block)
    return np.r_[labels, np.full(len(y) - len(labels), lab[-1])]


def stationary(A: np.ndarray) -> np.ndarray:
    if len(A) == 1:
        return np.ones(1)
    w, v = np.linalg.eig(A.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1))])
    p = np.abs(p)
    return p / p.sum()


def fit_regimes(y, X, m, s, pi0, K, c, seed=0) -> dict:
    T, d = X.shape
    labels = np.zeros(T, int) if K == 1 else _init_labels(y, X, K, c["init_block"], seed)
    beta = np.zeros((K, d))
    cs = np.ones(K)
    pi = np.tile(pi0, (K, 1))
    for k in range(K):
        sel = labels == k
        if sel.sum() > d + 10:
            beta[k] = np.linalg.lstsq(X[sel], y[sel] - (pi0 * m).sum(), rcond=None)[0]
    A = np.ones((1, 1)) if K == 1 else (np.full((K, K), 0.001 / (K - 1)) + np.eye(K) * (1 - 0.001 - 0.001 / (K - 1)))
    p0 = np.full(K, 1.0 / K)
    prev, ll, it = -np.inf, -np.inf, 0
    info = None
    for it in range(c["regime_iters"]):
        mean = X @ beta.T
        D = _comp(y, mean, m, s, cs) * pi[None]
        E = D.sum(2) + 1e-300
        a, _, cc = _forward(E, A, p0)
        ll = float(np.log(cc).sum())
        g, xi = _backward(E, A, a, cc) if K > 1 else (np.ones((T, 1)), np.ones((1, 1)))
        r = D / E[:, :, None]
        w = g[:, :, None] * r
        pi = np.maximum(w.sum(0) / w.sum(0).sum(1, keepdims=True), 1e-8)
        pi /= pi.sum(1, keepdims=True)
        for k in range(K):
            wk = w[:, k, :] / (s[None, :] ** 2 * cs[k] ** 2)
            sw = wk.sum(1)
            tgt = (wk * (y[:, None] - m[None, :])).sum(1)
            M = (X * sw[:, None]).T @ X + 1e-9 * np.eye(d)
            beta[k] = np.linalg.solve(M, X.T @ tgt)
            if K == 1:
                info = M
            res = y[:, None] - (X @ beta[k])[:, None] - m[None, :]
            cs[k] = math.sqrt(max((w[:, k, :] * res ** 2 / s[None, :] ** 2).sum() / max(w[:, k, :].sum(), 1e-12), 1e-6))
        if K > 1:
            A = xi / xi.sum(1, keepdims=True)
            p0 = g[0]
        if ll - prev < c["regime_tol"] * abs(ll):
            break
        prev = ll
    return dict(K=K, beta=beta, c=cs, pi=pi, A=A, p0=p0, m=m, s=s, loglik=ll, iters=it + 1, info=info)


def filter_predictive(model: dict, y, X, p_start=None):
    mean = X @ model["beta"].T
    D = _comp(y, mean, model["m"], model["s"], model["c"]) * model["pi"][None]
    E = D.sum(2) + 1e-300
    _, pred, cc = _forward(E, model["A"], stationary(model["A"]) if p_start is None else p_start)
    return pred, np.log(cc)


def mixture(model: dict, prob: np.ndarray, x: np.ndarray):
    w = (prob[:, None] * model["pi"]).reshape(-1)
    mu = ((model["beta"] @ x)[:, None] + model["m"][None, :]).reshape(-1)
    sd = (model["c"][:, None] * model["s"][None, :]).reshape(-1)
    return w, mu, sd


def _edge(w, mu, sd) -> float:
    return float(np.sum(w * (np.exp(mu + 0.5 * sd ** 2) - 1.0)))


def shrink_static(model: dict, y_fit: np.ndarray) -> dict:
    """K = 1 estimation risk: James-Stein on the feature coefficients (joint Wald statistic) and on the average edge."""
    out = dict(model)
    beta = model["beta"][0].copy()
    slopes = beta[1:]
    p = len(slopes)
    info = model["info"]
    wald = float(slopes @ np.linalg.inv(np.linalg.inv(info)[1:, 1:]) @ slopes) if p else 0.0
    ks = max(0.0, 1.0 - max(p - 2, 1) / wald) if wald > 0 else 0.0
    beta[1:] = ks * slopes
    simple = np.expm1(y_fit)
    level, se = float(simple.mean()), float(simple.std(ddof=1) / math.sqrt(len(simple)))
    kl = max(0.0, 1.0 - se ** 2 / level ** 2) if level != 0 else 0.0
    out["beta"] = beta[None, :]
    # shift the shared components so that the average predicted edge becomes kl * level
    X0 = np.zeros_like(beta)
    X0[0] = 1.0
    base = _edge(*mixture(out, np.ones(1), X0))
    delta = math.log((1.0 + kl * level) / (1.0 + base)) if (1.0 + base) > 0 else 0.0
    out["m"] = model["m"] + delta
    out["shrinkage"] = dict(slope_wald=wald, slope_shrink=ks, level=level, level_se=se, level_shrink=kl, shift=delta)
    return out


def learn(X: np.ndarray, y: np.ndarray, seed: int, config: dict = CONFIG) -> tuple[dict, list[dict]]:
    c = dict(config)
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64)
    cut = int(len(y) * (1 - c["val_frac"]))
    hist = []
    best = None
    for J in range(1, c["max_components"] + 1):
        pj = mk2.em(y[:cut], J, c["em_iters"], c["em_tol"])
        v = mk2.mixture_nll(y[cut:], *pj)
        hist.append(dict(stage="components", J=J, val_nll=v))
        if best is None or v < best[0] - 1e-9:
            best = (v, J, pj)
    pi0, m, s = best[2]
    cands = {}
    for K in range(1, c["max_regimes"] + 1):
        fits = [fit_regimes(y[:cut], X[:cut], m, s, pi0, K, c, seed=seed * 10 + st)
                for st in (range(1) if K == 1 else range(c["regime_starts"]))]
        mod = max(fits, key=lambda f: f["loglik"])
        _, lp = filter_predictive(mod, y, X)
        cands[K] = (mod, lp[cut:])
        hist.append(dict(stage="regimes", K=K, train_loglik=mod["loglik"], heldout_loglik=float(lp[cut:].mean()),
                         starts=[f["loglik"] for f in fits]))
    K, tests = 1, {}
    for k2 in range(2, c["max_regimes"] + 1):
        dlt = cands[k2][1] - cands[K][1]
        t = float(dlt.mean() / (dlt.std(ddof=1) / math.sqrt(len(dlt)))) if dlt.std() > 0 else 0.0
        tests[f"K{k2} vs K{K}"] = t
        if t > c["select_t"]:
            K = k2
    static = shrink_static(cands[1][0], y[:cut])
    chosen = static if K == 1 else cands[K][0]
    # persistence of the agent's own predicted edge on held-out experience
    pred, _ = filter_predictive(chosen, y, X)
    e = np.array([_edge(*mixture(chosen, pred[t], X[t])) for t in range(cut, len(y))])
    rho = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if e.std() > 0 else 0.0
    pred_s, _ = filter_predictive(static, y, X)
    es = np.array([_edge(*mixture(static, pred_s[t], X[t])) for t in range(cut, len(y))])
    rho_s = float(np.corrcoef(es[:-1], es[1:])[0, 1]) if es.std() > 0 else 0.0
    clean = lambda d: {k: v for k, v in d.items() if k != "info"}
    art = dict(kind="regimes", model=clean(chosen), static=clean(static), K=K, components=best[1],
               persistence=dict(rho=rho if math.isfinite(rho) else 0.0),
               static_persistence=dict(rho=rho_s if math.isfinite(rho_s) else 0.0),
               selection=dict(chosen=f"K={K}", regimes=K, components=best[1], tests=tests,
                              heldout_loglik={f"K={k}": float(v[1].mean()) for k, v in cands.items()}),
               shrinkage=static.get("shrinkage"), config=c, samples=int(len(y)))
    return art, hist


def to_bytes(art: dict) -> bytes:
    import pickle
    return pickle.dumps(art, protocol=4)


def from_bytes(data: bytes) -> dict:
    import pickle
    return pickle.loads(data)


# ---------------------------------------------------------------------------- the agent
class Agent:
    """Stateful within a test run: belief about the current regime, updated with every bar the agent observes."""

    def __init__(self, art: dict, mapper, encoder, static_only: bool = False):
        c = art["config"]
        self.art, self.mapper = art, mapper
        self.model = art["static"] if static_only else art["model"]
        self.rho = float((art["static_persistence"] if static_only else art["persistence"])["rho"])
        self.feat = Features(encoder)
        self.reader = mk3.ObsReader(encoder)
        z, w = np.polynomial.hermite_e.hermegauss(c["quad_nodes"])
        self.z, self.wq = z, w / math.sqrt(2 * math.pi)
        self.gmax, self.iters = c["max_abs_size"], c["bisection_iters"]
        self.reset()

    def reset(self) -> None:
        self.p = stationary(self.model["A"])
        self.x_prev = None

    def _observe(self, y_prev: float) -> None:
        mod = self.model
        mean = mod["beta"] @ self.x_prev
        sd = mod["c"][:, None] * mod["s"][None, :]
        dens = (mod["pi"] * np.exp(-0.5 * ((y_prev - mean[:, None] - mod["m"][None, :]) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))).sum(1)
        v = self.p * (dens + 1e-300)
        self.p = (v / v.sum()) @ mod["A"]

    def target(self, obs) -> float:
        obs = np.asarray(obs, np.float64)
        x = self.feat.from_full(obs)
        if self.x_prev is not None:
            self._observe(float(x[1]))                     # the bar just completed is the previous decision's outcome
        self.x_prev = x
        w, mu, sd = mixture(self.model, self.p, x)
        G, _, _ = mk3.decide(w[None, :], mu[None, :], sd[None, :], np.array([self.reader.exposure(obs)]),
                             self.reader.half_spread(obs) * (1.0 - self.rho), self.z, self.wq, self.gmax, self.iters)
        return float(G[0])

    def action(self, G: float) -> np.ndarray:
        return mk2.Agent.action(self, G)

    def __call__(self, obs) -> np.ndarray:
        return self.action(self.target(obs))


def _encoder(identity):
    from .env import TradeWorldEnv
    from . import synthetic_tasks as st
    f = identity["frozen"]
    t = st.build(f["task"], 0, 2000)
    return TradeWorldEnv(t.md, t.profile, reward=f["reward"], episode_decisions=1, extra=t.extra, random_start=False,
                         **f["env_settings"]).encoder


def load_policy(blobs, sha: str, identity: dict, static_only: bool = False):
    from .evaluate import mapper_from
    return Agent(from_bytes(blobs.get(sha)), mapper_from(identity), _encoder(identity), static_only=static_only)


# ---------------------------------------------------------------------------- oracle reference (not an agent)
def oracle_policy(identity: dict, task):
    """Reference, not an agent: version 4's decision with the TRUE regime of the test data at every bar (from the
    task's hidden regime path, which no agent ever sees) and the market's true parameters."""
    from .evaluate import mapper_from
    enc = _encoder(identity)
    reader = mk3.ObsReader(enc)
    feat = Features(enc)
    p, name = task.params, task.name
    z, w = np.polynomial.hermite_e.hermegauss(CONFIG["quad_nodes"])
    wq = w / math.sqrt(2 * math.pi)
    shim = type("Shim", (), {})()
    shim.mapper = mapper_from(identity)

    class Oracle:
        reference_needs_row = True

        def __call__(self, obs, row):
            obs = np.asarray(obs, np.float64)
            x = feat.from_full(obs)
            h = float(task.hidden[row, 0])
            if name == "S1X":
                ws, mus, sds, rho = np.ones(1), np.array([h * x[1]]), np.array([p["sigma"]]), h
            elif name == "S2X":
                ws, mus, sds = np.ones(1), np.array([h * x[-1] * p["mu"]]), np.array([p["sigma"]])
                rho = 1.0 - 2.0 / p["mean_regime_len"]
            else:  # S6X
                ws = np.array([1 - h, h])
                mus = np.array([p["mu"] - 0.5 * p["sigma"] ** 2, math.log(1 - p["crash"])])
                sds = np.array([p["sigma"], 1e-6])
                rho = 0.0
            G, _, _ = mk3.decide(ws[None, :], mus[None, :], sds[None, :], np.array([reader.exposure(obs)]),
                                 reader.half_spread(obs) * (1.0 - rho), z, wq, CONFIG["max_abs_size"], CONFIG["bisection_iters"])
            return mk2.Agent.action(shim, float(G[0]))
    return Oracle()


# ---------------------------------------------------------------------------- true-model filter reference (not an agent)
def true_model(task_name: str, params: dict) -> dict:
    """The market's TRUE regime model in version 4's form (true regimes, their parameters and switching rates). A
    reference that knows the rules but, like the agent, must infer the current regime from the prices it sees."""
    if task_name == "S1X":
        L, sig = params["mean_spell"], params["sigma"]
        beta = np.array([[0.0, params["phi_values"][0], 0.0, 0.0], [0.0, params["phi_values"][1], 0.0, 0.0]])
        pi, m, s = np.ones((2, 1)), np.zeros(1), np.array([sig])
    elif task_name == "S2X":
        L, sig, mu = params["mean_spell"], params["sigma"], params["mu"]
        beta = np.array([[0.0, 0.0, 0.0, 0.0, mu], [0.0, 0.0, 0.0, 0.0, -mu]])
        pi, m, s = np.ones((2, 1)), np.zeros(1), np.array([sig])
    elif task_name == "S6X":
        (pc, ps), (Lc, Ls) = params["crash_probs"], params["mean_spells"]
        beta = np.zeros((2, 4))
        pi = np.array([[1 - pc, pc], [1 - ps, ps]])
        m = np.array([params["mu"] - 0.5 * params["sigma"] ** 2, math.log(1 - params["crash"])])
        s = np.array([params["sigma"], 1e-4])
        A = np.array([[1 - 1 / Lc, 1 / Lc], [1 / Ls, 1 - 1 / Ls]])
        return dict(K=2, beta=beta, c=np.ones(2), pi=pi, A=A, p0=np.full(2, 0.5), m=m, s=s)
    else:
        raise ValueError(task_name)
    A = np.array([[1 - 1 / L, 1 / L], [1 / L, 1 - 1 / L]])
    return dict(K=2, beta=beta, c=np.ones(2), pi=pi, A=A, p0=np.full(2, 0.5), m=m, s=s)


def true_filter_policy(identity: dict):
    """Reference, not an agent: version 4's decision with the market's TRUE regime model (true_model) and the same
    persistence rule, measured on 40,000 bars of the market (seed 0) exactly as the agent measures its own."""
    from .evaluate import mapper_from
    from . import synthetic_tasks as st
    name = identity["frozen"]["task"]
    task = st.build(name, 0, 40_000)
    enc = _encoder(identity)
    feat = Features(enc)
    Xm, y, cols = collect(task, identity)
    X = feat.from_market_columns(Xm, cols)
    mod = true_model(name, task.params)
    pred, _ = filter_predictive(mod, y, X)
    e = np.array([_edge(*mixture(mod, pred[t], X[t])) for t in range(len(y))])
    rho = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if e.std() > 0 else 0.0
    art = dict(model=mod, static=mod, persistence=dict(rho=rho), static_persistence=dict(rho=rho), config=CONFIG)
    return Agent(art, mapper_from(identity), enc)
