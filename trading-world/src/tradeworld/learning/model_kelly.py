"""Model-based growth-optimal agent ("model-kelly"): learns the market from experience, then sizes.

1. World model (learned): a mixture-density network predicts the full distribution of the next bar's
   log return from the agent's MARKET observation (the same encoded window the PPO agents see, without
   the account features, which cannot influence the market). It is fitted by maximum likelihood on the
   bars the agent has experienced, with a chronological validation split and early stopping.
2. Decision: each bar, the position G (signed exposure / equity) that maximises the expected log growth
   of equity under the learned distribution, E[ln(1 + G (e^x - 1))] (the objective of reward R1, without
   any floor: a predicted wipe-out counts as minus infinity). Concave in G; solved by bisection on the
   derivative over the region where every predicted outcome leaves positive equity, intersected with
   +-2,000x (the account's 1:2000 margin limit; not a behavioural cap).

Nothing about the market (drift, volatility, crash frequency or size) is given to the agent; it is all
estimated from experience. No trading style, holding rule, indicator, exposure cap or penalty is imposed.
Expectations use Gauss-Hermite quadrature per mixture component.
"""
from __future__ import annotations

import io
import math

import numpy as np

CONFIG = dict(components=5, hidden=64, lr=1e-3, weight_decay=1e-5, batch=1024, max_epochs=100, patience=8,
              val_frac=0.2, quad_nodes=32, max_abs_size=2000.0, bisection_iters=80)


# ---------------------------------------------------------------------------- data
def market_columns(encoder) -> np.ndarray:
    """Indices of the market part of the encoded observation (window, volatility level, extra features)."""
    m = encoder.n * encoder.W * encoder.F + encoder.n
    return np.r_[np.arange(m), np.arange(m + encoder.A, encoder.dim)]


def collect(task, identity) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observations at every decision of the task's market and the next bar's log return of the
    instrument (what the agent experiences by watching the market), plus the market columns. The market
    is replayed once while holding no position; the account columns are not used by the model."""
    from .env import TradeWorldEnv
    f = identity["frozen"]
    probe = TradeWorldEnv(task.md, task.profile, reward=f["reward"], episode_decisions=1, extra=task.extra,
                          random_start=False, **f["env_settings"])
    L = len(probe.dec_all) - 2
    env = TradeWorldEnv(task.md, task.profile, reward=f["reward"], episode_decisions=L, extra=task.extra,
                        random_start=False, **f["env_settings"])
    cols = market_columns(env.encoder)
    mid = task.md.mid_c()[:, 0]
    o, _ = env.reset(seed=0, options={"start_decision": 0})
    obs = np.empty((L, len(cols)), np.float32)
    rets = np.empty(L, np.float32)
    flat = np.zeros(env.mapper.size)
    flat[-1] = -1.0
    for k in range(L):
        d, d1 = int(env.dec[env.k]), int(env.dec[env.k + 1])
        obs[k] = o[cols]
        rets[k] = math.log(mid[d1] / mid[d])
        o, *_ = env.step(flat)
    return obs, rets, cols


# ---------------------------------------------------------------------------- model
def _mdn(d, K, h):
    import torch

    class MDN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh())
            self.pi, self.mu, self.ls = torch.nn.Linear(h, K), torch.nn.Linear(h, K), torch.nn.Linear(h, K)

        def forward(self, x):
            z = self.f(x)
            return torch.log_softmax(self.pi(z), -1), self.mu(z), self.ls(z).clamp(-12, 2)
    return MDN()


def _nll(model, x, y):
    import torch
    lp, mu, ls = model(x)
    comp = -0.5 * ((y[:, None] - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
    return -(torch.logsumexp(lp + comp, -1)).mean()


def train(X: np.ndarray, y: np.ndarray, seed: int, config: dict = CONFIG) -> tuple[dict, list[dict]]:
    """Fit the world model on experienced (observation, next return) pairs. Returns (artifact, history)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-6
    scale = float(y.std()) or 1.0
    Xn = torch.tensor((X - mu_x) / sd_x, dtype=torch.float32)
    yn = torch.tensor(y / scale, dtype=torch.float32)
    cut = int(len(y) * (1 - config["val_frac"]))
    model = _mdn(X.shape[1], config["components"], config["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best, best_state, bad, hist = math.inf, None, 0, []
    for ep in range(config["max_epochs"]):
        perm = torch.tensor(rng.permutation(cut))
        tot = 0.0
        for i in range(0, cut, config["batch"]):
            b = perm[i:i + config["batch"]]
            opt.zero_grad()
            loss = _nll(model, Xn[b], yn[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        with torch.no_grad():
            v = float(_nll(model, Xn[cut:], yn[cut:]))
        hist.append(dict(epoch=ep + 1, train_nll=tot / cut, val_nll=v))
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= config["patience"]:
                break
    model.load_state_dict(best_state)
    art = dict(state_dict={k: t.cpu() for k, t in model.state_dict().items()}, mu_x=mu_x, sd_x=sd_x, scale=scale,
               d=int(X.shape[1]), config=dict(config), best_val_nll=best, epochs=len(hist), samples=int(len(y)))
    return art, hist


def to_bytes(art: dict) -> bytes:
    import torch
    buf = io.BytesIO()
    torch.save(art, buf)
    return buf.getvalue()


def from_bytes(data: bytes) -> dict:
    import torch
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------- decision
class Agent:
    def __init__(self, art: dict, cols: np.ndarray, mapper):
        import torch
        c = art["config"]
        self.model = _mdn(art["d"], c["components"], c["hidden"])
        self.model.load_state_dict(art["state_dict"])
        self.model.eval()
        self.mu_x, self.sd_x, self.scale = art["mu_x"], art["sd_x"], art["scale"]
        self.cols, self.mapper = cols, mapper
        z, w = np.polynomial.hermite_e.hermegauss(c["quad_nodes"])
        self.z, self.w = z, w / math.sqrt(2 * math.pi)
        self.gmax, self.iters = c["max_abs_size"], c["bisection_iters"]
        self.torch = torch

    def predictive(self, X: np.ndarray):
        with self.torch.no_grad():
            lp, mu, ls = self.model(self.torch.tensor((X - self.mu_x) / self.sd_x, dtype=self.torch.float32))
        return lp.exp().numpy().astype(np.float64), mu.numpy() * self.scale, ls.exp().numpy() * self.scale

    def sizes(self, X: np.ndarray) -> np.ndarray:
        """Growth-optimal signed position for each market observation row."""
        pi, mu, sg = self.predictive(X)
        x = mu[:, :, None] + sg[:, :, None] * self.z[None, None, :]
        w = (pi[:, :, None] * self.w[None, None, :]).reshape(len(X), -1)
        y = np.expm1(x).reshape(len(X), -1)
        neg = np.where(y < 0, -y, 0.0).max(1)
        pos = np.where(y > 0, y, 0.0).max(1)
        hi = np.minimum(np.where(neg > 0, (1 - 1e-9) / np.maximum(neg, 1e-300), self.gmax), self.gmax)
        lo = np.maximum(np.where(pos > 0, -(1 - 1e-9) / np.maximum(pos, 1e-300), -self.gmax), -self.gmax)
        a, b = lo.copy(), hi.copy()
        for _ in range(self.iters):
            g = 0.5 * (a + b)
            dv = (w * y / (1 + g[:, None] * y)).sum(1)
            a = np.where(dv > 0, g, a)
            b = np.where(dv > 0, b, g)
        return 0.5 * (a + b)

    def action(self, G: float) -> np.ndarray:
        if abs(G) < 1e-6:
            return np.array([0.0, -1.0])
        return np.array([float(np.sign(G)), self.mapper.u_for_gross(abs(G))])

    def __call__(self, obs) -> np.ndarray:
        G = float(self.sizes(np.asarray(obs, np.float32)[self.cols][None, :])[0])
        return self.action(G)


def load_policy(blobs, sha: str, identity: dict):
    from .evaluate import mapper_from
    from .env import TradeWorldEnv
    from . import synthetic_tasks as st
    art = from_bytes(blobs.get(sha))
    f = identity["frozen"]
    t = st.build(f["task"], 0, 2000)
    enc = TradeWorldEnv(t.md, t.profile, reward=f["reward"], episode_decisions=1, extra=t.extra, random_start=False,
                        **f["env_settings"]).encoder
    return Agent(art, market_columns(enc), mapper_from(identity))
