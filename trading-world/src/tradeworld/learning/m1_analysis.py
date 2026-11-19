"""Experiment M1: the pre-registered analysis (frozen by checksum in the manifest before any training).

Agents: model-based growth-optimal agents (model_kelly), six seeds per market, each judged on its FINAL
checkpoint (after 200,000 bars of experience) on two fresh rounds of 20 test runs (40 runs).

S6 (group MK-S6): a seed is a successful growth agent only if the approved Experiment 1b criteria hold:
  W1 ruinous decisions < 1%; W2 at most 1 death in 40 test runs; W3 median size within x2 of G* (1.47-5.86x);
  W4 Q2.0 >= 0.50; W5 growth above zero with 95% CI above zero and above the matched-random 95th percentile in
  both rounds; W6 previous-move tilt < 1.5; E exploitation checks clear;
  W7' (replaces W7, which measures PPO training episodes this agent does not have) learning from experience:
  on the development round, Q2 after 200,000 bars of experience is higher than after 20,000 bars, and the
  final ruinous share is below 1%.
  The group supports safe growth if at least 4 of 6 seeds meet every criterion.
  Interpretation, in this order: "safe growth"; "still enters the ruin zone" (ruinous >= 1% in >= 3 seeds);
  "safe but timid (learning problem)" (W1 in >= 4 seeds and W3 in <= 2); otherwise "mixed".
  Planned comparisons on Q2, ruinous share and distance |ln(median / G*)|, against the R-FL R1 control and the
  Experiment 1 control (historical references, different test data; Q2 depends only on decisions): exact
  two-sided permutation tests across seeds (924 relabellings), Holm-corrected over the 6 comparisons.

S7 (group MK-S7, generality check; Q2 is provisional for S7 and is NOT used): a seed succeeds if W1, W2, W3
  (median within x2 of S7's G* = 5x), W5, W6, E and W7'' hold, W7'' being: on the development round the
  distance |ln(median / G*)| after 200,000 bars is smaller than after 20,000 bars, and final ruinous < 1%.
  The group passes if at least 4 of 6 seeds succeed.
"""
from __future__ import annotations

import math

import numpy as np

from . import phase3_analysis as p3a
from . import phase3b_analysis as w

FIRST, LAST = 20_000, 200_000
METRICS = ("q2", "ruinous", "size_distance")


def learning_curve_flag(curve: dict, s: dict) -> dict:
    out = dict(w7=False, w7_s7=False)
    if FIRST in curve and LAST in curve:
        out["w7"] = bool(curve[LAST]["q2"] > curve[FIRST]["q2"] and s["ruinous"] < w.RUIN_MAX)
        g = s.get("g_star") or 1.0
        dist = lambda x: abs(math.log(max(x, 1e-12) / g))
        out["w7_s7"] = bool(dist(curve[LAST]["median"]) < dist(curve[FIRST]["median"]) and s["ruinous"] < w.RUIN_MAX)
    return out


def interpretation(flags: list[dict], seeds: list[dict]) -> str:
    if sum(f["success"] for f in flags) >= w.SUCCESS_SEEDS:
        return "safe growth"
    if sum(s["ruinous"] >= w.RUIN_MAX for s in seeds) >= w.EXPLORATION_SEEDS:
        return "still enters the ruin zone"
    if sum(f["W1"] for f in flags) >= w.TIMID_SAFE_SEEDS and sum(f["W3"] for f in flags) <= w.TIMID_SIZE_MAX:
        return "safe but timid (learning problem)"
    return "mixed"


def _group_summary(ss: list[dict], flags: list[dict]) -> dict:
    return dict(n=len(ss), flags=flags, successes=sum(f["success"] for f in flags),
                q2_mean=float(np.mean([s["q2"] for s in ss])), q2_ci=w.boot_mean([s["q2"] for s in ss]),
                ruinous_mean=float(np.mean([s["ruinous"] for s in ss])), ruinous_ci=w.boot_mean([s["ruinous"] for s in ss]),
                median_size_mean=float(np.mean([s["median_size"] for s in ss])),
                median_size_ci=w.boot_mean([s["median_size"] for s in ss]),
                growth_mean=float(np.mean([s["growth"] for s in ss])),
                deaths_total=int(sum(s["deaths"] for s in ss)),
                seeds_ruinous_ge_1pct=sum(s["ruinous"] >= w.RUIN_MAX for s in ss))


def verdicts(seeds: dict, refs: dict) -> dict:
    out = {}
    # ---- S6
    ss = sorted(seeds["MK-S6"], key=lambda s: s["seed_index"])
    g6 = ss[0]["g_star"]
    flags = []
    for s in ss:
        f = w.seed_flags(dict(s, w7=s["w7"]), g6)
        flags.append(f)
    r = _group_summary(ss, flags)
    r["counts"] = {k: sum(f[k] for f in flags) for k in ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "E")}
    r["supports_safe_growth"] = r["successes"] >= w.SUCCESS_SEEDS
    r["interpretation"] = interpretation(flags, ss)
    ref_sets = {"R-FL R1 control": refs["rfl"]["seeds"]["R1"], "Exp1 control": refs["exp1"]["seeds"]["control"]}
    raw, comps = {}, {}
    for name, rs in ref_sets.items():
        for mname in METRICS:
            a = [w.metric(s, mname, g6) for s in ss]
            b = [w.metric(s, mname, g6) for s in rs]
            key = f"MK-S6 vs {name}: {mname}"
            raw[key] = w.permutation_p_two_sided(a, b)
            comps[key] = dict(difference=float(np.mean(a) - np.mean(b)), ci=w.boot_diff(a, b), p_raw=raw[key],
                              reference_mean=float(np.mean(b)))
    adj = p3a.holm(raw)
    for k in comps:
        comps[k]["p_holm"] = adj[k]
    r["comparisons"] = comps
    out["MK-S6"] = r
    # ---- S7
    ss7 = sorted(seeds["MK-S7"], key=lambda s: s["seed_index"])
    g7 = ss7[0]["g_star"]
    flags7 = []
    for s in ss7:
        f = w.seed_flags(dict(s, w7=s["w7_s7"]), g7)
        f["success"] = all(f[k] for k in ("W1", "W2", "W3", "W5", "W6", "W7", "E"))
        flags7.append(f)
    r7 = _group_summary(ss7, flags7)
    r7["counts"] = {k: sum(f[k] for f in flags7) for k in ("W1", "W2", "W3", "W5", "W6", "W7", "E")}
    r7["passes"] = r7["successes"] >= w.SUCCESS_SEEDS
    out["MK-S7"] = r7
    return out
