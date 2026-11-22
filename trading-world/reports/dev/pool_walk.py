"""Development: pooled edge check on the owner's long MT5 history (H1/H4/D1 bars). Learn on everything up to
2022-12-31 (frozen train split), predict 2023 (validation) one bar ahead. 2024+ is never read.
Each market's moves are in units of its own recent volatility (causal: rms bar return over the previous N bars).
Inputs (all causal, at the close of bar t): the last 3 bar moves, moves over longer windows, recent vs longer volatility,
and the clock. Question: does the pooled prediction hold up out of sample, and is it bigger than the cost of trading?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from tradeworld.learning.model_kelly6 import block_bootstrap
H = Path("/home/user/tiktok-comment-bot/trading-world/datasets/mt5_history")
SYMS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "NAS100"]
SPLIT, END = 1672531200 // 60, 1672531200 // 60   # walk-forward inside the train split only (<= 2022)          # 2023-01-01, 2024-01-01 (minutes)
WIN = {"H1": [6, 24, 120, 480], "H4": [6, 30, 120, 250], "D1": [5, 20, 60, 250]}     # windows in bars
VOLN = {"H1": 120, "H4": 60, "D1": 20}
LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SPREAD_BP = dict(EURUSD=0.75, GBPUSD=0.74, USDJPY=0.67, XAUUSD=0.85, NAS100=2.9, BTCUSD=4.4)   # median M1 spreads (bp)
for fr in sys.argv[1].split(","):
    R = []
    for s in SYMS:
        df = pd.read_parquet(H / s / f"{fr}.parquet").sort_values("time")
        df = df[df.time < END]
        lm = np.log(0.5 * (df.bid_c.values + df.ask_c.values)); t = df.time.values
        r = np.r_[np.nan, np.diff(lm)]
        n = VOLN[fr]
        c2 = np.r_[0, np.nancumsum(r ** 2)]
        idx = np.arange(len(r))
        vol = np.sqrt((c2[idx + 1] - c2[np.maximum(idx + 1 - n, 0)]) / np.maximum(np.minimum(idx + 1, n), 1))
        volL = np.sqrt((c2[idx + 1] - c2[np.maximum(idx + 1 - 5 * n, 0)]) / np.maximum(np.minimum(idx + 1, 5 * n), 1))
        T = len(r); L = max(WIN[fr]) + 5 * n
        d = np.arange(L, T - 1)
        sig = vol[d]
        y = r[d + 1] / sig
        cols = [r[d - j] / sig for j in range(3)]
        cols += [(lm[d] - lm[d - w]) / (sig * np.sqrt(w)) for w in WIN[fr]]
        cols += [np.log(vol[d] / volL[d])]
        mins = t[d] % 1440; days = (t[d] // 1440 + 3) % 7
        cols += [np.sin(2 * np.pi * mins / 1440), np.cos(2 * np.pi * mins / 1440), np.sin(2 * np.pi * days / 7), np.cos(2 * np.pi * days / 7)]
        X = np.column_stack(cols)
        good = np.isfinite(y) & np.all(np.isfinite(X), 1) & (sig > 0)
        R.append(dict(sym=s, t=t[d][good], X=X[good], y=y[good], cost=(0.5 * SPREAD_BP[s] * 1e-4 / sig)[good]))
    t = np.concatenate([q["t"] for q in R]); X = np.concatenate([q["X"] for q in R]); y = np.concatenate([q["y"] for q in R])
    who = np.concatenate([[i] * len(q["y"]) for i, q in enumerate(R)]); cost = np.concatenate([q["cost"] for q in R])
    o = np.argsort(t, kind="stable"); t, X, y, who, cost = t[o], X[o], y[o], who[o], cost[o]
    import datetime as _dt
    yr = lambda Y: int(_dt.datetime(Y, 1, 1, tzinfo=_dt.timezone.utc).timestamp()) // 60
    tot = []
    for Y in range(2017, 2023):
        tr, te = t < yr(Y), (t >= yr(Y)) & (t < yr(Y + 1))
        if tr.sum() < 500:
            print(f"{fr} {Y}: too little history before this year ({tr.sum()} bars)"); continue
        keep = X[tr].std(0) > 1e-9
        Xs = X[:, keep]; mu, sd = Xs[tr].mean(0), Xs[tr].std(0)
        Z = np.clip((Xs - mu) / sd, *np.percentile((Xs[tr] - mu) / sd, [0, 100], axis=0))
        A = np.column_stack([np.ones(len(Z)), Z]); yc = np.clip(y, -10, 10)
        beta = np.linalg.solve(A[tr].T @ A[tr] + LAM * np.diag(np.r_[0, np.ones(A.shape[1] - 1)]), A[tr].T @ yc[tr])
        p = A @ beta; e, rr = p[te], y[te]
        def slope(i):
            return float((e[i] * rr[i]).sum() / max((e[i] ** 2).sum(), 1e-300))
        b = slope(np.arange(len(e))); se = float(np.std(block_bootstrap(slope, len(e), 50, 200, 1), ddof=1))
        per = " ".join(f"{q['sym'][:3]} {np.corrcoef(p[te & (who == i)], y[te & (who == i)])[0,1]:+.3f}" if (te & (who == i)).sum() > 50 else f"{q['sym'][:3]}   -   " for i, q in enumerate(R))
        print(f"{fr} {Y}: train {tr.sum():6d} test {te.sum():5d} | slope {b:+.2f} +- {se:.2f} (t {b/se:+.2f}) corr {np.corrcoef(e, rr)[0,1]:+.4f} | by market corr: {per}", flush=True)
