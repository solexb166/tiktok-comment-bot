"""Real 2022 data probe (development only, inside the frozen 'train' split): is there predictability worth its cost?"""
import math, sys
from datetime import date
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from baseline_report import load_market
from tradeworld.data.dukascopy_archive import month_days

def bars(md, k):
    """mid close and mean relative spread sampled every k minutes over quoted minutes."""
    bid, ask = md.bid_c[:, 0], md.ask_c[:, 0]
    q = np.isfinite(bid) & np.isfinite(ask)
    t = md.times[q]; mid = ((bid + ask) / 2)[q]; sp = ((ask - bid) / ((ask + bid) / 2))[q]
    grp = t // k
    last = np.r_[np.nonzero(np.diff(grp))[0], len(grp) - 1]
    return grp[last], mid[last], sp[last], t[last]

for sym in sys.argv[1:]:
    end = month_days(2022, 9 if sym == "NAS100" else 10)[-1]
    md = load_market([sym], date(2022, 1, 1), end)
    split_t = md.times[0] + (np.searchsorted(md.times, md.times[0]) + 0)  # placeholder
    for k, lab in ((1, "M1"), (15, "M15"), (60, "H1")):
        g, mid, sp, t = bars(md, k)
        r = np.diff(np.log(mid)); gap = np.diff(g) != 1          # returns across a gap (weekend/closure) are excluded
        r[gap] = np.nan
        hour = ((t[1:] // 60) % 24)
        cut = int(len(r) * 0.7)                                   # Jan-Jul learn / Aug-Oct check (approx.)
        ok = np.isfinite(r)
        sd = np.nanstd(r)
        cost = np.nanmedian(sp) / 2
        # lag-1..3 linear predictability, fitted on first 70%, checked out of sample
        X = np.column_stack([np.r_[np.nan, r[:-1]], np.r_[np.nan, np.nan, r[:-2]], np.r_[np.nan] * 3 if False else np.r_[[np.nan] * 3, r[:-3]]])
        m = ok & np.all(np.isfinite(X), 1)
        tr, te = m.copy(), m.copy(); tr[cut:] = False; te[:cut] = False
        b = np.linalg.lstsq(np.c_[np.ones(tr.sum()), X[tr]], r[tr], rcond=None)[0]
        pred = np.c_[np.ones(te.sum()), X[te]] @ b
        oos_r2 = 1 - np.sum((r[te] - pred) ** 2) / np.sum((r[te] - r[tr].mean()) ** 2)
        corr = np.corrcoef(pred, r[te])[0, 1]
        # hour-of-day: mean return by hour on first 70%, sign agreement out of sample
        hm_tr = np.array([np.nanmean(r[:cut][hour[:cut] == h]) if (hour[:cut] == h).any() else 0 for h in range(24)])
        hm_te = np.array([np.nanmean(r[cut:][hour[cut:] == h]) if (hour[cut:] == h).any() else 0 for h in range(24)])
        hc = np.corrcoef(np.nan_to_num(hm_tr), np.nan_to_num(hm_te))[0, 1]
        print(f"{sym:7s} {lab:4s} bars {ok.sum():7d}  sd/bar {sd:.2e}  half-spread/sd {cost / sd:5.2f}  "
              f"lags OOS R2 {oos_r2:+.5f} corr {corr:+.3f}  |pred|/cost {np.median(np.abs(pred)) / cost:5.2f}  hour-effect OOS corr {hc:+.2f}", flush=True)
