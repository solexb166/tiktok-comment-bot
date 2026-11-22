"""v7 development (frozen train/validation splits only). Each month: the agent itself chooses its decision speed and
learns its true trading cost, from its own replay of recent history as if live; then trades the month once.
  For every candidate speed: learn from history up to REPLAY months before the month, trade the replay stretch in the
  simulator (full costs: spread + slippage), measure growth and the cost actually paid per unit traded.
  Keep the speed with the best replay growth (if none grew, the agent stays out: its own conclusion, not a rule).
  Learn from all history at that speed and trade the month, charging the decision the cost it learned.
No rule about how often to trade: the cost-aware hold band decides that."""
import dataclasses, math, sys, time
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from tradeworld.brokerspec.profile import load_profile
from tradeworld.data.dukascopy_archive import month_days
from tradeworld.data.mt5 import load_market, months
from tradeworld.learning import model_kelly6 as mk, model_kelly3 as mk3, timeframes as tf
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.learning.evaluate import run_recorded, mapper_from
from tradeworld.learning.synthetic_tasks import Task
from tradeworld.market.clock import to_minute
from tradeworld.market.instruments import Universe
DATA = Path("/home/user/tiktok-comment-bot/trading-world")
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
sym = sys.argv[1]
SPEEDS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [15, 60, 240, 1440]
FIRST, LAST = (2023, int(__import__('os').environ.get('V7_FROM', '1'))), (2023, 12)
REPLAY = 3                                             # months of the agent's own recent history replayed as if live
md = load_market(DATA / "datasets" / "mt5", [sym], date(2022, 1, 1), date(2023, 12, 31))
full = load_profile("environments/A_fractional.yaml")
prof = dataclasses.replace(full, universe=Universe((full.universe.specs[full.universe.index(sym)],)))
FE = tf.features(md)

def sub(a, b):
    return dataclasses.replace(md, **{f.name: getattr(md, f.name)[a:b] for f in dataclasses.fields(md)
                                      if isinstance(getattr(md, f.name), np.ndarray) and getattr(md, f.name).shape[:1] == md.times.shape})

def row_at(y, m):
    return int(np.searchsorted(md.times, to_minute(datetime(y, m, 1, tzinfo=timezone.utc))))

class Prep:
    """Inputs as the agent learned them: columns that never varied in its own history are dropped (they duplicate the
    constant), the rest are put on a common scale, and every input is held inside the range the agent has seen
    (a fitted formula is not applied beyond its experience)."""
    def __init__(self, X):
        sd = X[:, 1:].std(0)
        self.keep = np.r_[0, 1 + np.where(sd > 1e-9 * (1 + np.abs(X[:, 1:]).max(0)))[0]]
        Z = X[:, self.keep]
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.mu[0], self.sd[0] = 0.0, 1.0
        Z = (Z - self.mu) / self.sd
        self.lo, self.hi = Z.min(0), Z.max(0)
    def __call__(self, X):
        Z = (np.asarray(X)[..., self.keep] - self.mu) / self.sd
        return np.clip(Z, self.lo, self.hi)

class CostAgent(mk.Agent):
    """v6 agent whose decision is charged the cost per unit traded it measured on its own replay (spread + slippage),
    instead of the quoted half spread only."""
    def __init__(self, art, mapper, encoder, cost=None, prep=None):
        super().__init__(art, mapper, encoder)
        self.cost = cost
        self.prep = prep if prep is not None else (lambda X: X)
    def target(self, obs):
        obs = np.asarray(obs, np.float64)
        raw = self.feat.from_full(obs)
        if self.x_prev is not None:
            self._observe(float(raw[1]))          # the bar's actual log return updates the regime belief
        x = self.prep(raw)
        self.x_prev = x
        w, mu, sd = mk.mixture(self.model, self.p, x)
        edge = float(np.sum(w * (np.exp(mu + 0.5 * sd ** 2) - 1.0)))
        mu = mk3.shift_to(w[None, :], mu[None, :], sd[None, :], np.array([self.check["k"] * edge]))[0]
        h = self.reader.half_spread(obs) if self.cost is None else max(self.cost, self.reader.half_spread(obs))
        G, _, _ = mk3.decide(w[None, :], mu[None, :], sd[None, :], np.array([self.reader.exposure(obs)]),
                             h * (1.0 - self.rho), self.z, self.wq, self.gmax, self.iters)
        return float(G[0])

def setup(k):
    env_set = dict(decision_interval=k, obs_resolution=k, window=30)
    ident = dict(task=sym, frozen=dict(task=sym, reward="R1", env_settings=env_set, action_space=LOG10,
                 episode_decisions=500, profile="A_fractional"), action_space=LOG10)
    return env_set, ident

def learn(k, fit_end):
    env_set, ident = setup(k)
    tr = Task(sym, sub(0, fit_end), prof, extra=FE[:fit_end], params=dict(real=True), env_kwargs=env_set)
    Xm, y, cols = mk.collect(tr, ident)
    enc = TradeWorldEnv(tr.md, prof, episode_decisions=10, extra=tr.extra, random_start=False, **env_set).encoder
    X = mk.Features(enc).from_market_columns(Xm, cols)
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    prep = Prep(X[ok])
    art, _ = mk.learn(prep(X[ok]), y[ok], 0)
    return art, prep

def trade(k, art, prep, a, b, cost=None):
    """Trade rows [a, b) once. Returns growth, the cost actually paid per unit traded, and stats."""
    env_set, ident = setup(k)
    lo = max(a - (-(-31 * k // 60)) * 60, 0)     # 30 bars of history, starting on a full hour
    te = Task(sym, sub(lo, b), prof, extra=FE[lo:b], params=dict(real=True), env_kwargs=env_set)
    ident["frozen"]["episode_decisions"] = (b - a) // k - 2
    agent = CostAgent(art, mapper_from(ident), TradeWorldEnv(te.md, prof, episode_decisions=10, extra=te.extra,
                      random_start=False, **env_set).encoder, cost, prep)
    rows, st = run_recorded(agent, ident, te, [0])
    traded = sum(abs(r["exposure_after_fill"] - p["net_exposure_end"]) * r["equity_before"]
                 for p, r in zip([dict(net_exposure_end=0.0)] + rows[:-1], rows))
    paid = sum(r["cost_spread"] + r["cost_slippage"] + r["cost_commission"] for r in rows)
    lev = np.array([abs(r["exposure_after_fill"]) for r in rows])
    return dict(g=st[0]["log_growth"], died=st[0]["died"], cost=paid / traded if traded > 0 else None, traded=traded,
                paid=paid, kept=art["model_check"]["k"], slope=art["model_check"]["slope"], se=art["model_check"]["slope_se"],
                K=art["K"], lev=float(np.median(lev)), inmkt=float(np.mean(lev > 1e-6)))

def replay(k, rcut, cut):
    """The agent's own replay at speed k: trade once to measure its true cost per unit traded, then again charging
    that cost (as it would trade), which is the growth that counts."""
    art, prep = learn(k, rcut)
    first = trade(k, art, prep, rcut, cut)
    if first["cost"] is None:
        return first
    second = trade(k, art, prep, rcut, cut, first["cost"])
    second["cost"] = second["cost"] if second["cost"] is not None else first["cost"]
    return second

res = []
for yy, m in months(date(*FIRST, 1), date(*LAST, 1)):
    t0 = time.time()
    cut = row_at(yy, m)
    end = int(np.searchsorted(md.times, to_minute(datetime(yy, m, month_days(yy, m)[-1].day, tzinfo=timezone.utc)) + 1440))
    ry, rm = (yy, m - REPLAY) if m > REPLAY else (yy - 1, m - REPLAY + 12)
    rcut = row_at(ry, rm)
    replays = {}
    for k in SPEEDS:
        replays[k] = replay(k, rcut, cut)
    best = max(SPEEDS, key=lambda k: replays[k]["g"])
    note = " | ".join(f"{k}m {replays[k]['g']:+.3f} (kept {replays[k]['kept']:.2f}, cost {1e4*(replays[k]['cost'] or 0):.2f}bp)" for k in SPEEDS)
    if replays[best]["g"] <= 0:
        g, out = 0.0, None
        choice = "stay out"
    else:
        cost = replays[best]["cost"]
        art, prep = learn(best, cut)
        out = trade(best, art, prep, cut, end, cost)
        g = out["g"]
        choice = f"{best}m, cost {1e4*(cost or 0):.2f}bp -> growth {g:+.3f} died {out['died']} kept {out['kept']:.2f} size {out['lev']:.1f}x in market {out['inmkt']:.0%} paid {out['paid']:.2f}"
    res.append(g)
    print(f"{sym} v7 {yy}-{m:02d}: replay [{note}] => {choice}  ({time.time()-t0:.0f}s)", flush=True)
r = np.array(res)
print(f"{sym} v7 SUMMARY: months {len(r)}, total log growth {r.sum():+.3f}, up {int((r > 1e-9).sum())}, down {int((r < -1e-9).sum())}, flat {int((np.abs(r) <= 1e-9).sum())}", flush=True)
