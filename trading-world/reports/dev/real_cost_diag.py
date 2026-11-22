"""Rolling walk-forward on real 2022 data (development; inside the frozen train split). For each test month April-October:
learn from everything before it (version 6 + multi-timeframe view), then trade that month once, as live trading would."""
import dataclasses, math, sys
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, "src")
from tradeworld.brokerspec.profile import load_profile
from tradeworld.data.align import align
from tradeworld.data.dukascopy_archive import load_archived, month_days
from tradeworld.learning import model_kelly6 as mk, timeframes as tf
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.learning.evaluate import run_recorded, mapper_from
from tradeworld.learning.synthetic_tasks import Task
from tradeworld.market.clock import to_minute
from tradeworld.market.instruments import Universe
DATA = Path("/home/user/tiktok-comment-bot/trading-world")
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
sym, k, use_tf = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "tf"
from tradeworld.data.mt5 import load_market as load_mt5
MT5 = DATA / "datasets" / "mt5"
FIRST, LAST = (2022, 4), (2023, 12)      # development only: frozen 'train' (2022) and 'validation' (2023) splits
md = load_mt5(MT5, [sym], date(2022, 1, 1), date(2023, 12, 31))
full = load_profile("environments/A_fractional.yaml")
prof = dataclasses.replace(full, universe=Universe((full.universe.specs[full.universe.index(sym)],)))
env_set = dict(decision_interval=k, obs_resolution=k, window=30)
FE = tf.features(md) if use_tf else None
def sub(a, b):
    return dataclasses.replace(md, **{f.name: getattr(md, f.name)[a:b] for f in dataclasses.fields(md)
                                      if isinstance(getattr(md, f.name), np.ndarray) and getattr(md, f.name).shape[:1] == md.times.shape})
def task(a, b):
    return Task(sym, sub(a, b), prof, extra=None if FE is None else FE[a:b], params=dict(real=True), env_kwargs=env_set)
ident = dict(task=sym, frozen=dict(task=sym, reward="R1", env_settings=env_set, action_space=LOG10, episode_decisions=500,
             profile="A_fractional"), action_space=LOG10)
res = []
from tradeworld.data.mt5 import months as month_list
for yy, m in [(int(sys.argv[4]), int(sys.argv[5]))]:
    cut = int(np.searchsorted(md.times, to_minute(datetime(yy, m, 1, tzinfo=timezone.utc))))
    end = int(np.searchsorted(md.times, to_minute(datetime(yy, m, month_days(yy, m)[-1].day, tzinfo=timezone.utc)) + 1440))
    tr = task(0, cut)
    Xm, y, cols = mk.collect(tr, ident)
    enc = TradeWorldEnv(tr.md, prof, episode_decisions=10, extra=tr.extra, random_start=False, **env_set).encoder
    X = mk.Features(enc).from_market_columns(Xm, cols)
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    art, _ = mk.learn(X[ok], y[ok], 0)
    te = task(cut - 31 * k, end)                       # 30 bars of history before the month for the first observation
    n_dec = (end - cut) // k - 2
    ident["frozen"]["episode_decisions"] = n_dec
    agent = mk.Agent(art, mapper_from(ident), TradeWorldEnv(te.md, prof, episode_decisions=10, extra=te.extra, random_start=False, **env_set).encoder)
    rows, stats = run_recorded(agent, ident, te, [0])
    g = stats[0]["log_growth"]; lev = np.array([abs(r["exposure_after_fill"]) for r in rows])
    mc = art["model_check"]
    res.append(g)
    cs = sum(r["cost_spread"] for r in rows); cl = sum(r["cost_slippage"] for r in rows); cc = sum(r["cost_commission"] for r in rows)
    pnl = sum(r["pnl"] for r in rows)
    eq0 = rows[0]["equity_before"]
    print(f"  COSTS: spread ${cs:.2f}  slippage ${cl:.2f}  commission ${cc:.2f}  | total P&L ${pnl:+.2f} on ${eq0:.2f} start | "
          f"P&L before costs ${pnl + cs + cl + cc:+.2f}", flush=True)
    print(f"{sym} {k}m {'tf' if use_tf else 'base'} MT5 {yy}-{m:02d}: K={art['K']} check slope {mc['slope']:+.2f}±{mc['slope_se']:.2f} kept {mc['k']:.2f} | "
          f"growth {g:+.3f} died {stats[0]['died']} median size {np.median(lev):.1f}x in market {np.mean(lev > 1e-6):.0%}", flush=True)
r = np.array(res)
print(f"{sym} {k}m {'tf' if use_tf else 'base'} MT5 SUMMARY: months {len(r)}, total log growth {r.sum():+.3f}, months up {int((r > 1e-9).sum())}, down {int((r < -1e-9).sum())}, flat {int((np.abs(r) <= 1e-9).sum())}", flush=True)
