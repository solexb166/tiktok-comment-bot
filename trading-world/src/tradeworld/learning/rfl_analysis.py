"""Experiment R-FL: the pre-registered analysis (proposal 0007, section 11).

Frozen before any R-FL agent is trained: the manifest records this file's SHA-256 and those of every
file it relies on; the report refuses to run if any changed.

Groups: R1 (current reward, concurrent control), R6 (full-liability log, c = 0.01: primary) and
R6-c0.05 (sensitivity). Everything else as the Experiment 1 control (original size output and exploration,
PPO defaults, 3M steps). Six paired seeds (the same fresh training seed for a seed index in every group).
Each agent is judged on its FINAL checkpoint only, on two fresh rounds of 20 test runs.

Per-seed success: the approved Experiment 1b criteria W1-W7 and E, unchanged (phase3b_analysis.seed_flags).
A group supports safe growth if at least 4 of its 6 seeds meet every criterion.

Additional pre-registered checks for unintended incentives (each R6 group against the concurrent R1 control):
  T  timidity: per seed, the median size of NON-ruinous decisions; the ratio of group mean to control mean,
     with a 95% bootstrap interval across seeds; the check fails if the interval's lower bound is below 0.75
     (a fall of more than 25% cannot be excluded)
  GR gambling to recover: per seed, mean size of decisions that follow a bar on which equity fell more than
     20% divided by mean size elsewhere; the check fails if the group median exceeds 1.5 (not measurable if
     fewer than 10 such decisions in the whole group)
  ST training stability: every agent completes; no non-finite training loss; median explained variance and
     value loss over the last 20% of updates reported against the control
Planned comparisons with the control on Q2, ruinous share and distance |ln(median size / G*)|: exact two-sided
permutation tests across seeds (924 relabellings), Holm-corrected over the 6 comparisons, bootstrap intervals.

Interpretation (fixed in advance, per group, checked in this order):
  "safe growth"                        the group supports safe growth
  "still enters the ruin zone"         ruinous share >= 1% in at least 3 of 6 seeds
  "safe but timid (learning problem)"  W1 met in at least 4 seeds and W3 in at most 2
  "mixed"                              anything else
The truncation of R1 is reported as confirmed as a cause only if R6 is "safe growth" and the R1 control is not.
"""
from __future__ import annotations

import math

import numpy as np

from . import phase3_analysis as p3a
from . import phase3b_analysis as w

CONTROL = "R1"
NEW = ("R6", "R6-c0.05")
TIMID_RATIO = 0.75
GR_LOSS, GR_MAX, GR_MIN_N = 0.20, 1.5, 10
METRICS = ("q2", "ruinous", "size_distance")


def interpretation(flags: list[dict], seeds: list[dict]) -> str:
    if sum(f["success"] for f in flags) >= w.SUCCESS_SEEDS:
        return "safe growth"
    if sum(s["ruinous"] >= w.RUIN_MAX for s in seeds) >= w.EXPLORATION_SEEDS:
        return "still enters the ruin zone"
    if sum(f["W1"] for f in flags) >= w.TIMID_SAFE_SEEDS and sum(f["W3"] for f in flags) <= w.TIMID_SIZE_MAX:
        return "safe but timid (learning problem)"
    return "mixed"


def nonruinous_median(logs, market) -> float:
    x = logs["exposure_after_fill"].to_numpy()
    ok = ~market.ruinous(x)
    lev = logs["leverage_after_fill"].to_numpy()
    return float(np.median(lev[ok])) if ok.any() else 0.0


def recovery_sizes(logs) -> tuple[float, float, int]:
    """(mean size after a bar losing > 20% of equity, mean size elsewhere, number of such decisions)."""
    lev = logs["leverage_after_fill"].to_numpy()
    prev_loss = logs.groupby(["round", "episode"])["pnl"].shift(1).to_numpy() / \
        logs.groupby(["round", "episode"])["equity_before"].shift(1).to_numpy()
    after = prev_loss < -GR_LOSS
    n = int(np.nansum(after))
    a = float(lev[after].mean()) if n else math.nan
    b = float(lev[~after].mean()) if (~after).any() else math.nan
    return a, b, n


def stability(up) -> dict:
    import pandas as pd
    out = {}
    if up is None or not len(up):
        return dict(finite=False)
    k = max(1, len(up) // 5)
    tail = up.iloc[-k:]
    for col, key in (("train/explained_variance", "explained_variance"), ("train/value_loss", "value_loss"),
                     ("train/loss", "loss")):
        if col in up:
            v = pd.to_numeric(up[col], errors="coerce")
            out[key + "_last20"] = float(pd.to_numeric(tail[col], errors="coerce").median())
            out[key + "_all_finite"] = bool(np.isfinite(v.dropna()).all())
    out["finite"] = all(v for k_, v in out.items() if k_.endswith("_all_finite"))
    return out


def verdicts(seeds: dict, g_star: float) -> dict:
    out, raw = {}, {}
    ctrl = sorted(seeds[CONTROL], key=lambda s: s["seed_index"])
    for g, ss in seeds.items():
        ss = sorted(ss, key=lambda s: s["seed_index"])
        flags = [w.seed_flags(s, g_star) for s in ss]
        q = [s["q2"] for s in ss]
        r = dict(n=len(ss), flags=flags, successes=sum(f["success"] for f in flags),
                 counts={k: sum(f[k] for f in flags) for k in ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "E")},
                 supports_safe_growth=sum(f["success"] for f in flags) >= w.SUCCESS_SEEDS,
                 interpretation=interpretation(flags, ss), q2_mean=float(np.mean(q)), q2_ci=w.boot_mean(q),
                 ruinous_mean=float(np.mean([s["ruinous"] for s in ss])), ruinous_ci=w.boot_mean([s["ruinous"] for s in ss]),
                 median_size_mean=float(np.mean([s["median_size"] for s in ss])),
                 median_size_ci=w.boot_mean([s["median_size"] for s in ss]),
                 seeds_ruinous_ge_1pct=sum(s["ruinous"] >= w.RUIN_MAX for s in ss),
                 all_finite=all(s.get("stability", {}).get("finite", False) for s in ss),
                 explained_variance_last20=float(np.nanmedian([s.get("stability", {}).get("explained_variance_last20", np.nan) for s in ss])),
                 value_loss_last20=float(np.nanmedian([s.get("stability", {}).get("value_loss_last20", np.nan) for s in ss])))
        n_after = sum(s["recovery_n"] for s in ss)
        ratios = [s["recovery_after"] / s["recovery_else"] for s in ss
                  if s["recovery_n"] > 0 and s["recovery_else"] and math.isfinite(s["recovery_after"])]
        r["GR"] = dict(decisions=n_after, median_ratio=float(np.median(ratios)) if ratios else None,
                       ok=None if n_after < GR_MIN_N or not ratios else bool(np.median(ratios) <= GR_MAX))
        if g != CONTROL:
            a = np.array([s["nonruinous_median"] for s in ss])
            b = np.array([s["nonruinous_median"] for s in ctrl])
            rng = np.random.default_rng(w.BOOT_SEED)
            boot = a[rng.integers(0, len(a), (w.BOOT, len(a)))].mean(1) / np.maximum(
                b[rng.integers(0, len(b), (w.BOOT, len(b)))].mean(1), 1e-12)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            r["T"] = dict(ratio=float(a.mean() / max(b.mean(), 1e-12)), ci=[float(lo), float(hi)], ok=bool(lo >= TIMID_RATIO))
            r["comparisons"] = {}
            for m in METRICS:
                x = [w.metric(s, m, g_star) for s in ss]
                y = [w.metric(s, m, g_star) for s in ctrl]
                key = f"{g} vs R1: {m}"
                raw[key] = w.permutation_p_two_sided(x, y)
                r["comparisons"][key] = dict(difference=float(np.mean(x) - np.mean(y)), ci=w.boot_diff(x, y), p_raw=raw[key])
        out[g] = r
    adj = p3a.holm(raw)
    for g in NEW:
        for key, c in out[g]["comparisons"].items():
            c["p_holm"] = adj[key]
    paired = {}
    for g in NEW:
        sa = {s["seed_index"]: s for s in seeds[g]}
        sb = {s["seed_index"]: s for s in seeds[CONTROL]}
        idx = sorted(set(sa) & set(sb))
        for m in METRICS:
            d = [w.metric(sa[i], m, g_star) - w.metric(sb[i], m, g_star) for i in idx]
            paired[f"{g} - R1 (same seed): {m}"] = dict(mean=float(np.mean(d)), p=w.signflip_p_two_sided(d), n=len(d))
    truncation_confirmed = out["R6"]["interpretation"] == "safe growth" and out[CONTROL]["interpretation"] != "safe growth"
    return dict(groups=out, paired=paired, truncation_confirmed_as_cause=truncation_confirmed)
