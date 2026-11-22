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
src = yaml.safe_load((DATA / "configs/data/dukascopy.yaml").read_text())
plan = yaml.safe_load((DATA / "configs/data/download_plan.yaml").read_text())
last = 9 if sym == "NAS100" else 10
frames = {sym: load_archived(DATA / plan["archive_dir"], sym, date(2022, 1, 1), month_days(2022, last)[-1], float(src["instruments"][sym]["price_divisor"]["value"]))}
t0 = to_minute(datetime(2022, 1, 1, tzinfo=timezone.utc)); t1 = to_minute(datetime(2022, last, month_days(2022, last)[-1].day, tzinfo=timezone.utc)) + 1440
md = align(frames, [sym], t0, t1, zero_volume_is_unquoted=True)
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
for m in range(4, last + 1):
    cut = int(np.searchsorted(md.times, to_minute(datetime(2022, m, 1, tzinfo=timezone.utc))))
    end = int(np.searchsorted(md.times, to_minute(datetime(2022, m, month_days(2022, m)[-1].day, tzinfo=timezone.utc)) + 1440))
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
    print(f"{sym} {k}m {'tf' if use_tf else 'base'} 2022-{m:02d}: K={art['K']} check slope {mc['slope']:+.2f}±{mc['slope_se']:.2f} kept {mc['k']:.2f} | "
          f"growth {g:+.3f} died {stats[0]['died']} median size {np.median(lev):.1f}x in market {np.mean(lev > 1e-6):.0%}", flush=True)
r = np.array(res)
print(f"{sym} {k}m {'tf' if use_tf else 'base'} SUMMARY: months {len(r)}, total log growth {r.sum():+.3f}, months up {int((r > 1e-9).sum())}, down {int((r < -1e-9).sum())}, flat {int((np.abs(r) <= 1e-9).sum())}", flush=True)
