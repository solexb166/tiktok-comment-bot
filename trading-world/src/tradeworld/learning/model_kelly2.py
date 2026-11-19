"""Model-based growth-optimal agent, version 2 ("model-kelly-v2").

Lessons from Experiment M1 (model_kelly, version 1): a mixture-density network trained by gradient descent
learned the crash frequency but spread the crash into a broad component (its components start near zero and
widened to reach a rare cluster far away), so it sized ~20% below the optimum; sharpening it exposed noisy,
state-dependent estimates. Version 2 learns from the same experience, with three standard statistical tools:

1. Two candidate world models, both fitted only to the agent's own experienced bars:
     unconditional  a Gaussian mixture of the next bar's log return, fitted exactly by expectation-maximisation
                    (components initialised at quantiles of the observed returns), with the number of
                    components (1-6) chosen by held-out likelihood;
     conditional    a mixture-density network on the agent's market observation (as version 1, but with the
                    components initialised at quantiles of the observed returns, and a wider numerical range for
                    their spread so it cannot trap a component).
   The agent uses whichever predicts held-out experience better (chronological 80/20 split; on a tie, the
   simpler unconditional model). Features that carry no information are thus discovered and ignored.
2. Estimation risk: the edge E[e^x - 1] estimated from finite data is corrected by positive-part James-Stein
   shrinkage, k = max(0, 1 - SE^2 / edge^2), with SE the edge's standard error from the same data: an edge
   the data has determined well is kept; an edge indistinguishable from zero is shrunk towards zero, so the
   agent does not trade sampling noise. (Unconditional model; the conditional model is protected against
   over-fitting by its held-out selection.)
3. Decision: as version 1, the position maximising expected log growth under the learned distribution
   (no floor: a predicted wipe-out is minus infinity), within +-2,000x (the 1:2000 margin limit).

Nothing about the market is given to the agent. No style, holding rule, exposure cap or penalty is imposed.
"""
from __future__ import annotations

import io
import math

import numpy as np

from .model_kelly import collect, market_columns  # noqa: F401  (same experience and observation columns)

CONFIG = dict(max_components=6, em_iters=500, em_tol=1e-9, val_frac=0.2, mdn_components=5, mdn_hidden=64, lr=1e-3,
              weight_decay=1e-5, batch=1024, max_epochs=100, patience=8, sigma_log_max=6.0, quad_nodes=32,
              max_abs_size=2000.0, bisection_iters=100)
LOG2PI = math.log(2 * math.pi)


# ---------------------------------------------------------------------------- unconditional mixture (EM)
def em(y: np.ndarray, K: int, iters: int, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, np.float64)
    mu = np.quantile(y, np.linspace(0.001, 0.999, K)) if K > 1 else np.array([y.mean()])
    sd = np.full(K, y.std() + 1e-12)
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(iters):
        lg = -0.5 * ((y[:, None] - mu) / sd) ** 2 - np.log(sd) - 0.5 * LOG2PI + np.log(pi)
        mx = lg.max(1, keepdims=True)
        r = np.exp(lg - mx)
        s = r.sum(1, keepdims=True)
        r /= s
        ll = float((mx[:, 0] + np.log(s[:, 0])).mean())
        nk = r.sum(0) + 1e-12
        pi = nk / len(y)
        mu = (r * y[:, None]).sum(0) / nk
        sd = np.sqrt((r * (y[:, None] - mu) ** 2).sum(0) / nk) + 1e-7
        if ll - prev < tol:
            break
        prev = ll
    return pi, mu, sd


def mixture_nll(y, pi, mu, sd) -> float:
    lg = -0.5 * ((np.asarray(y, np.float64)[:, None] - mu) / sd) ** 2 - np.log(sd) - 0.5 * LOG2PI + np.log(pi)
    mx = lg.max(1, keepdims=True)
    return -float((mx[:, 0] + np.log(np.exp(lg - mx).sum(1))).mean())


def shrink(y: np.ndarray, pi, mu, sd) -> tuple[np.ndarray, dict]:
    """Positive-part James-Stein correction of the edge E[e^x - 1] for estimation risk; returns shifted means."""
    simple = np.expm1(np.asarray(y, np.float64))
    edge = float((pi * (np.exp(mu + 0.5 * sd ** 2) - 1)).sum())
    se = float(simple.std(ddof=1) / math.sqrt(len(simple)))
    k = max(0.0, 1.0 - se ** 2 / edge ** 2) if edge != 0 else 0.0
    target = k * edge
    # shift every component by the same delta so that the mixture's edge becomes k * edge
    base = (pi * np.exp(mu + 0.5 * sd ** 2)).sum()
    delta = math.log((1.0 + target) / base)
    return mu + delta, dict(edge=edge, edge_se=se, t=edge / se if se > 0 else math.inf, shrink=k, shift=delta)


# ---------------------------------------------------------------------------- conditional mixture-density network
def _mdn(d, K, h, sigma_log_max):
    import torch

    class MDN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh())
            self.pi, self.mu, self.ls = torch.nn.Linear(h, K), torch.nn.Linear(h, K), torch.nn.Linear(h, K)

        def forward(self, x):
            z = self.f(x)
            return torch.log_softmax(self.pi(z), -1), self.mu(z), self.ls(z).clamp(-12, sigma_log_max)
    return MDN()


def _nll_t(model, x, y):
    import torch
    lp, mu, ls = model(x)
    comp = -0.5 * ((y[:, None] - mu) / ls.exp()) ** 2 - ls - 0.5 * LOG2PI
    return -(torch.logsumexp(lp + comp, -1)).mean()


def fit_conditional(X, y, cut, seed, c) -> tuple[dict, list[dict], float]:
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu_x, sd_x = X[:cut].mean(0), X[:cut].std(0) + 1e-6
    scale = float(y[:cut].std()) or 1.0
    Xn = torch.tensor((X - mu_x) / sd_x, dtype=torch.float32)
    yn = torch.tensor(y / scale, dtype=torch.float32)
    K = c["mdn_components"]
    model = _mdn(X.shape[1], K, c["mdn_hidden"], c["sigma_log_max"])
    with torch.no_grad():
        model.mu.weight.mul_(0.01)
        model.mu.bias.copy_(torch.tensor(np.quantile(y[:cut] / scale, np.linspace(0.001, 0.999, K)), dtype=torch.float32))
        model.ls.weight.mul_(0.01)
        model.ls.bias.fill_(-1.0)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    best, best_state, bad, hist = math.inf, None, 0, []
    for ep in range(c["max_epochs"]):
        perm = torch.tensor(rng.permutation(cut))
        tot = 0.0
        for i in range(0, cut, c["batch"]):
            b = perm[i:i + c["batch"]]
            opt.zero_grad()
            loss = _nll_t(model, Xn[b], yn[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        with torch.no_grad():
            v = float(_nll_t(model, Xn[cut:], yn[cut:]))
        hist.append(dict(epoch=ep + 1, train_nll=tot / cut, val_nll=v))
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    model.load_state_dict(best_state)
    # held-out NLL in the original return units, comparable with the unconditional model
    val_nll = best + math.log(scale)
    art = dict(state_dict={k: t.cpu() for k, t in model.state_dict().items()}, mu_x=mu_x, sd_x=sd_x, scale=scale,
               d=int(X.shape[1]))
    return art, hist, val_nll


# ---------------------------------------------------------------------------- learning from experience
def learn(X: np.ndarray, y: np.ndarray, seed: int, config: dict = CONFIG) -> tuple[dict, list[dict]]:
    """Fit both candidate world models on experienced (observation, next return) pairs; keep the one that
    predicts held-out experience better. Returns (artifact, history)."""
    c = dict(config)
    y = np.asarray(y, np.float64)
    cut = int(len(y) * (1 - c["val_frac"]))
    # unconditional: components chosen by held-out likelihood, then refitted on all experience
    sel = []
    for K in range(1, c["max_components"] + 1):
        p = em(y[:cut], K, c["em_iters"], c["em_tol"])
        sel.append(dict(K=K, val_nll=mixture_nll(y[cut:], *p)))
    bestK = min(sel, key=lambda r: (r["val_nll"], r["K"]))
    unc_val = bestK["val_nll"]
    cond_art, hist, cond_val = fit_conditional(np.asarray(X, np.float32), y, cut, seed, c)
    history = [dict(stage="unconditional", **r) for r in sel] + [dict(stage="conditional", **h) for h in hist]
    if cond_val < unc_val:
        art = dict(kind="conditional", **cond_art)
    else:
        pi, mu, sd = em(y, bestK["K"], c["em_iters"], c["em_tol"])
        mu_s, info = shrink(y, pi, mu, sd)
        art = dict(kind="unconditional", pi=pi, mu=mu_s, mu_raw=mu, sd=sd, K=bestK["K"], shrinkage=info)
    art.update(config=c, samples=int(len(y)), selection=dict(unconditional_val_nll=unc_val, conditional_val_nll=cond_val,
                                                              chosen=art["kind"], components=bestK["K"]))
    return art, history


def to_bytes(art: dict) -> bytes:
    import torch
    buf = io.BytesIO()
    torch.save(art, buf)
    return buf.getvalue()


def from_bytes(data: bytes) -> dict:
    import torch
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------- decision
def kelly_sizes(pi, mu, sd, z, wq, gmax, iters) -> np.ndarray:
    """Growth-optimal signed position for each row of mixture parameters [N, K]."""
    x = mu[:, :, None] + sd[:, :, None] * z[None, None, :]
    w = (pi[:, :, None] * wq[None, None, :]).reshape(len(pi), -1)
    y = np.expm1(x).reshape(len(pi), -1)
    neg = np.where(y < 0, -y, 0.0).max(1)
    pos = np.where(y > 0, y, 0.0).max(1)
    hi = np.minimum(np.where(neg > 0, (1 - 1e-9) / np.maximum(neg, 1e-300), gmax), gmax)
    lo = np.maximum(np.where(pos > 0, -(1 - 1e-9) / np.maximum(pos, 1e-300), -gmax), -gmax)
    a, b = lo.copy(), hi.copy()
    for _ in range(iters):
        g = 0.5 * (a + b)
        dv = (w * y / (1 + g[:, None] * y)).sum(1)
        a = np.where(dv > 0, g, a)
        b = np.where(dv > 0, b, g)
    return 0.5 * (a + b)


class Agent:
    def __init__(self, art: dict, cols: np.ndarray, mapper):
        c = art["config"]
        self.art, self.cols, self.mapper = art, cols, mapper
        z, w = np.polynomial.hermite_e.hermegauss(c["quad_nodes"])
        self.z, self.wq = z, w / math.sqrt(2 * math.pi)
        self.gmax, self.iters = c["max_abs_size"], c["bisection_iters"]
        if art["kind"] == "unconditional":
            self.constant = float(kelly_sizes(art["pi"][None], art["mu"][None], art["sd"][None], self.z, self.wq,
                                              self.gmax, self.iters)[0])
        else:
            import torch
            self.torch = torch
            self.model = _mdn(art["d"], c["mdn_components"], c["mdn_hidden"], c["sigma_log_max"])
            self.model.load_state_dict(art["state_dict"])
            self.model.eval()

    def predictive(self, X: np.ndarray):
        if self.art["kind"] == "unconditional":
            n = len(X)
            return (np.repeat(self.art["pi"][None], n, 0), np.repeat(self.art["mu"][None], n, 0),
                    np.repeat(self.art["sd"][None], n, 0))
        a = self.art
        with self.torch.no_grad():
            lp, mu, ls = self.model(self.torch.tensor((X - a["mu_x"]) / a["sd_x"], dtype=self.torch.float32))
        return lp.exp().numpy().astype(np.float64), mu.numpy() * a["scale"], ls.exp().numpy() * a["scale"]

    def sizes(self, X: np.ndarray) -> np.ndarray:
        if self.art["kind"] == "unconditional":
            return np.full(len(X), self.constant)
        return kelly_sizes(*self.predictive(X), self.z, self.wq, self.gmax, self.iters)

    def action(self, G: float) -> np.ndarray:
        if abs(G) < 1e-6:
            a = np.zeros(self.mapper.size)
            a[-1] = -1.0
            return a
        return np.array([float(np.sign(G)), self.mapper.u_for_gross(abs(G))])

    def __call__(self, obs) -> np.ndarray:
        if self.art["kind"] == "unconditional":
            return self.action(self.constant)
        return self.action(float(self.sizes(np.asarray(obs, np.float32)[self.cols][None, :])[0]))


def load_policy(blobs, sha: str, identity: dict):
    from .env import TradeWorldEnv
    from .evaluate import mapper_from
    from . import synthetic_tasks as st
    art = from_bytes(blobs.get(sha))
    f = identity["frozen"]
    t = st.build(f["task"], 0, 2000)
    enc = TradeWorldEnv(t.md, t.profile, reward=f["reward"], episode_decisions=1, extra=t.extra, random_start=False,
                        **f["env_settings"]).encoder
    return Agent(art, market_columns(enc), mapper_from(identity))
