"""Development walk-forward on real 2022 data (inside the frozen train split): a v5 agent learns from Jan-Jul and
trades Aug-Oct in the real simulator (Environment A costs: real Dukascopy spreads, slippage, swaps). Not a stage test."""
import dataclasses, math, sys, time
from datetime import date
import numpy as np
sys.path.insert(0, "src")
from datetime import datetime, timezone
from pathlib import Path
import yaml
from tradeworld.data.align import align
from tradeworld.data.dukascopy_archive import load_archived
from tradeworld.market.clock import to_minute
DATA = Path("/home/user/tiktok-comment-bot/trading-world")          # the full download (read only)


def load_market(symbols, start, end):
    src = yaml.safe_load((DATA / "configs/data/dukascopy.yaml").read_text())
    plan = yaml.safe_load((DATA / "configs/data/download_plan.yaml").read_text())
    frames = {s: load_archived(DATA / plan["archive_dir"], s, start, end, float(src["instruments"][s]["price_divisor"]["value"])) for s in symbols}
    t0 = to_minute(datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
    t1 = to_minute(datetime(end.year, end.month, end.day, tzinfo=timezone.utc)) + 1440
    zv = src.get("zero_volume_is_unquoted", {})
    zv = zv.get("value", True) if isinstance(zv, dict) else bool(zv)
    return align(frames, symbols, t0, t1, zero_volume_is_unquoted=zv)
from tradeworld.brokerspec.profile import load_profile
from tradeworld.data.dukascopy_archive import month_days
from tradeworld.learning import model_kelly6 as mk
from tradeworld.learning.env import TradeWorldEnv
from tradeworld.learning.evaluate import run_recorded, mapper_from
from tradeworld.learning.synthetic_tasks import Task
from tradeworld.market.instruments import Universe
from tradeworld.learning import timeframes as tf
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
sym, k = sys.argv[1], int(sys.argv[2])
full = load_profile("environments/A_fractional.yaml")
spec = full.universe.specs[full.universe.index(sym)]
prof = dataclasses.replace(full, universe=Universe((spec,)))
end = month_days(2022, 9 if sym == "NAS100" else 10)[-1]
md = load_market([sym], date(2022, 1, 1), end)
env_set = dict(decision_interval=k, obs_resolution=k, window=30)
cut_t = (int(md.times[0]) + int(0.7 * (int(md.times[-1]) - int(md.times[0])))) // 1440 * 1440
cut = int(np.searchsorted(md.times, cut_t))
def sub(a, b):
    return dataclasses.replace(md, **{f.name: getattr(md, f.name)[a:b] for f in dataclasses.fields(md)
                                      if isinstance(getattr(md, f.name), np.ndarray) and getattr(md, f.name).shape[:1] == md.times.shape})
train_md, test_md = sub(0, cut), sub(cut, len(md.times))
FE = tf.features(md)                                   # computed on the whole history: each row uses only its past
EX = {id(train_md): FE[:cut], id(test_md): FE[cut:]}
mk_task = lambda m: Task(sym, m, prof, extra=EX[id(m)], params=dict(real=True), env_kwargs=env_set)
ident = dict(task=sym, frozen=dict(task=sym, reward="R1", env_settings=env_set, action_space=LOG10, episode_decisions=500,
             profile="A_fractional"), action_space=LOG10)
t0 = time.time()
tr = mk_task(train_md)
Xm, y, cols = mk.collect(tr, ident)
env = TradeWorldEnv(tr.md, prof, episode_decisions=10, extra=tr.extra, random_start=False, **env_set)
X = mk.Features(env.encoder).from_market_columns(Xm, cols)
ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
art, _ = mk.learn(X[ok], y[ok], 0)
sh = art.get("shrinkage") or {}
mc = art["model_check"]
print(f"{sym} {k}min [multi-timeframe]: held-out check slope {mc['slope']:+.2f} (se {mc['slope_se']:.2f}) kept {mc['k']:.2f}; level t {mc['level_t']:+.2f} kept {mc['level_kept']:.2f}", flush=True)
print(f"{sym} {k}min: {ok.sum()} examples, regimes K={art['K']}, drift t={sh.get('level', 0) / max(sh.get('level_se', 1e-12), 1e-12):+.2f} "
      f"kept {sh.get('level_shrink', 0):.2f}, pattern Wald {sh.get('slope_wald', 0):.1f} kept {sh.get('slope_shrink', 0):.2f}, rho {art['persistence']['rho']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
te = mk_task(test_md)
n_dec = (len(test_md.times) - 31 * k) // k
ep = min(500, max(50, n_dec // 4))
ident["frozen"]["episode_decisions"] = ep
starts = list(range(0, n_dec - ep - 5, ep))
agent = mk.Agent(art, mapper_from(ident), TradeWorldEnv(te.md, prof, episode_decisions=10, extra=te.extra, random_start=False, **env_set).encoder)
rows, stats = run_recorded(agent, ident, te, starts)
g = np.array([s["log_growth"] for s in stats]); lev = np.array([abs(r["exposure_after_fill"]) for r in rows])
print(f"  Aug-Oct: {len(stats)} runs x {ep} decisions: growth/run {g.mean():+.3f} (each: {np.round(g, 2).tolist()}), "
      f"deaths {sum(s['died'] for s in stats)}, median size {np.median(lev):.2f}x, max size {lev.max():.1f}x, in market {np.mean(lev > 1e-6):.0%}, "
      f"costs/run ${np.mean([s['costs'] for s in stats]):.2f}", flush=True)
