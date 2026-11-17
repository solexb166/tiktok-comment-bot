"""Phase 2 gate: per-run summary and the pre-registered pass/fail criteria
(docs/decisions/0002-phase2-gate-criteria.md). Shared by the report and the dashboard."""
from __future__ import annotations

import math

import numpy as np


def summarise(run):
    eps = run["episodes"]
    lg = np.array([e["log_growth"] for e in eps])
    t = lg.mean() / (lg.std(ddof=1) / math.sqrt(len(lg))) if len(lg) > 1 and lg.std(ddof=1) > 0 else 0.0
    agree = [e["flag_agreement"] for e in eps if e["flag_agreement"] is not None]
    return dict(
        lg=float(lg.mean()), t=float(t), died=float(np.mean([e["died"] for e in eps])),
        gross=float(np.mean([e["mean_gross"] for e in eps])), net=float(np.mean([e["mean_net"] for e in eps])),
        turnover=float(np.mean([e["turnover"] for e in eps])), agree=float(np.mean(agree)) if agree else None,
        null95=float(np.percentile(run["null"]["mean_log_growth_per_agent"], 95)) if "null" in run else None,
    )


def verdict(task, reward, s, ref=None):
    """Returns (passed, reason). `ref` is the R1 summary for the same task/seed (R2 criteria)."""
    if task == "S1":
        ok = s["lg"] > 0 and s["null95"] is not None and s["lg"] > s["null95"]
        return ok, f"growth {s['lg']:+.3f} vs null 95th pct {s['null95']:+.3f}"
    if task == "S2":
        ok = (s["agree"] or 0) >= 0.8 and s["lg"] > 0
        return ok, f"agrees with regime {100 * (s['agree'] or 0):.0f}% of the time, growth {s['lg']:+.3f}"
    if task == "S3":
        return s["gross"] < 0.2, f"mean gross exposure {s['gross']:.3f}x"
    if task in ("S4", "S5"):
        return s["t"] < 2, f"growth {s['lg']:+.4f}, t = {s['t']:.2f}"
    if task == "S6" and reward == "R1":
        ok = s["died"] <= 0.10 and 1.0 <= s["gross"] <= 6.0
        return ok, f"gross {s['gross']:.2f}x (optimal ~2.9x), deaths {100 * s['died']:.0f}%"
    if task == "S6" and reward == "R2":
        if ref is None:
            return False, "no R1 run to compare"
        ok = s["gross"] > 2 * ref["gross"] and s["died"] > ref["died"]
        return ok, f"gross {s['gross']:.2f}x vs R1 {ref['gross']:.2f}x, deaths {100 * s['died']:.0f}% vs {100 * ref['died']:.0f}%"
    if task == "S7" and reward == "R1":
        return 2.5 <= s["net"] <= 10.0, f"net leverage {s['net']:.2f}x (optimal 5x)"
    if task == "S7" and reward == "R2":
        ok = not (2.5 <= s["net"] <= 10.0) or s["died"] >= 0.5
        return ok, f"net leverage {s['net']:.2f}x, deaths {100 * s['died']:.0f}%"
    return False, "no criterion"
