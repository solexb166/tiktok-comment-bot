"""Development (train 2022 -> validation 2023 only): is there a pooled edge across all six markets?
Every market's returns and inputs are put on its own volatility scale (causal: the last day's realised volatility),
so one model can learn from all of them. Fit on 2022, predict 2023 one bar ahead; the question is whether the predicted
edge holds up out of sample (through-origin slope, block-bootstrap error) and whether it beats the cost of trading."""
import sys
from datetime import date
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from tradeworld.data.mt5 import load_market
from tradeworld.learning import timeframes as tf
from tradeworld.learning.model_kelly6 import block_bootstrap
DATA = Path("/home/user/tiktok-comment-bot/trading-world/datasets/mt5")
SYMS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "NAS100"]
SPEEDS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [60, 240, 1440]
SPLIT = 1672531200 // 60    # 2023-01-01 00:00 UTC in minutes since 1970
rows = {k: [] for k in SPEEDS}
for s in SYMS:
    md = load_market(DATA, [s], date(2022, 1, 1), date(2023, 12, 31))
    F = tf.features(md)
    bid, ask = md.bid_c[:, 0], md.ask_c[:, 0]
    ok = np.isfinite(bid) & np.isfinite(ask)
    mid = tf._ffill(np.where(ok, 0.5 * (bid + ask), np.nan))
    hs = tf._ffill(np.where(ok, 0.5 * (ask - bid) / mid, np.nan))
    # rms one-minute log return over the last 1440 QUOTED minutes (closed-market minutes carry no information)
    q = np.where(ok)[0]
    lq = np.log(0.5 * (bid[q] + ask[q]))
    r1 = np.r_[0.0, np.diff(lq)]
    c2 = np.r_[0.0, np.cumsum(r1 * r1)]
    hi = np.arange(1, len(q) + 1); lo = np.maximum(hi - 1440, 0)
    rq = np.sqrt((c2[hi] - c2[lo]) / np.maximum(hi - lo, 1))
    rms1 = np.full(len(bid), np.nan); rms1[q] = rq; rms1 = tf._ffill(rms1)
    t = np.asarray(md.times, np.int64)
    for k in SPEEDS:
        d = np.arange(1440 * 7, len(t) - k, k)                    # decision rows on the k grid, after a week of history
        d = d[ok[d]]
        lm = np.log(mid)
        r_now = lm[d] - lm[d - k]
        sig = rms1[d] * np.sqrt(k)                                # this market's typical move over one bar
        y = (lm[d + k] - lm[d]) / sig                             # next bar's move, in units of its own volatility
        lags = [(lm[d - j * k] - lm[d - (j + 1) * k]) / sig for j in range(3)]
        moves = [F[d, tf.NAMES.index(f"move_{w}")] / (rms1[d] * np.sqrt(m)) for w, m in tf.WINDOWS.items()]
        vols = [F[d, tf.NAMES.index("vol_1h")] - F[d, tf.NAMES.index("vol_1d")]]   # recent vs daily volatility
        clock = [F[d, tf.NAMES.index(n)] for n in ("day_sin", "day_cos", "week_sin", "week_cos")]
        X = np.column_stack(lags + moves + vols + clock)
        good = np.isfinite(y) & np.all(np.isfinite(X), 1) & (sig > 0)
        rows[k].append(dict(sym=s, t=t[d][good], X=X[good], y=y[good], cost=(hs[d] / sig)[good]))
for k in SPEEDS:
    R = rows[k]
    t = np.concatenate([r["t"] for r in R]); X = np.concatenate([r["X"] for r in R]); y = np.concatenate([r["y"] for r in R])
    who = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(R)]); cost = np.concatenate([r["cost"] for r in R])
    o = np.argsort(t, kind="stable"); t, X, y, who, cost = t[o], X[o], y[o], who[o], cost[o]
    tr, te = t < SPLIT, t >= SPLIT
    keep = X[tr].std(0) > 1e-9
    Xs = X[:, keep]; mu, sd = Xs[tr].mean(0), Xs[tr].std(0); Z = np.clip((Xs - mu) / sd, -5, 5)
    A = np.column_stack([np.ones(len(Z)), Z])
    lam = 1.0 * tr.sum() / 100                                    # mild ridge
    beta = np.linalg.solve(A[tr].T @ A[tr] + lam * np.diag(np.r_[0, np.ones(A.shape[1] - 1)]), A[tr].T @ y[tr])
    p = A @ beta
    e, r = p[te], y[te]
    v = float((e ** 2).sum()); b = float((e * r).sum() / v)
    se = float(np.std(block_bootstrap(lambda i: float((e[i] * r[i]).sum() / max((e[i] ** 2).sum(), 1e-300)), len(e), 200, 400, 1), ddof=1))
    corr = float(np.corrcoef(e, r)[0, 1])
    print(f"speed {k:5d}m: train {tr.sum():6d} test {te.sum():6d} bars | out-of-sample slope {b:+.3f} +- {se:.3f} (t {b/se:+.2f}), corr {corr:+.4f} | "
          f"median |pred| {np.median(np.abs(e)):.4f} sd vs median half-spread {np.median(cost[te]):.4f} sd", flush=True)
    for i, rr in enumerate(R):
        m = te & (who == i)
        if m.sum() > 50:
            ee, yy = p[m], y[m]
            print(f"      {rr['sym']:7s} corr {np.corrcoef(ee, yy)[0,1]:+.4f}  slope {(ee*yy).sum()/(ee**2).sum():+.3f}  n {m.sum()}", flush=True)
