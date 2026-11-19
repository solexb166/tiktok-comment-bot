"""Experiment M2: the pre-registered analysis (frozen by checksum in the manifest before any training).

Agents: model-based growth-optimal agents v2 (model_kelly2), six seeds per market, judged on their FINAL
checkpoint (200,000 bars of experience) on six fresh rounds of 20 test runs (120 runs). Test power was set from
Experiment M1's power analysis: with 40 runs the exact growth-optimal policy itself passed the growth test only
9 of 15 times; with 120 runs its standard error falls to about 0.064 against a mean growth of about 0.25 per run.
Benchmark = the exact growth-optimal constant size (computed from the market's known parameters) on the same runs.

S6 (MK2-S6): a seed succeeds only if all hold:
  W1  ruinous decisions < 1%
  W2' test deaths <= the benchmark's deaths on the same 120 runs + 2
  W3  median size within x2 of G* (1.47-5.86x)
  W4  Q2.0 >= 0.50
  W5' mean log growth per run > 0 with its 95% bootstrap interval (120 runs) above 0, and above the matched-random
      95th percentile in at least 5 of the 6 rounds
  W6  previous-move tilt < 1.5
  W7' on the development round, Q2 after 200,000 bars of experience > after 20,000 bars, and final ruinous < 1%
  E   own-training-path Q2 minus fresh Q2 <= 0.5; decisions at 1,000x or more < 0.1%; rejected orders <= 0.1%;
      size output at the upper end of its numerical range (the only end that could cap exposure) < 0.1%
S7 (MK2-S7): as S6 without W4 (Q2 provisional for S7), with G* = 5x (W3: 2.5-10x), W7'' instead of W7' (the
  distance |ln(median / G*)| after 200,000 bars < after 20,000 bars, and final ruinous < 1%), and E without the
  Q2-based training-path gap.
S4 (MK2-S4, no edge): a seed succeeds if it claims no edge: |t| of mean log growth over the 120 runs < 2, no test
  deaths, and the E checks on boundary, rejected orders and numerical bound.
A market passes if at least 4 of its 6 seeds succeed. M2 passes if all three markets pass.
S6 interpretation, in this order: "safe growth"; "still enters the ruin zone" (ruinous >= 1% in >= 3 seeds);
  "safe but timid (learning problem)" (W1 in >= 4 seeds and W3 in <= 2); otherwise "mixed".
Planned comparisons (S6), on Q2, ruinous share and distance |ln(median / G*)|, against Experiment M1's MK-S6 and the
  R-FL R1 control (historical references): exact two-sided permutation tests across seeds, Holm over 6 comparisons.
"""
from __future__ import annotations

import math

import numpy as np

from . import phase3_analysis as p3a
from . import phase3b_analysis as w

FIRST, LAST = 20_000, 200_000
METRICS = ("q2", "ruinous", "size_distance")


def benchmark_metrics(evals: list[dict], size: float) -> dict:
    g = [ep["log_growth"] for e in evals for ep in e["episodes"]]
    return dict(size=size, deaths=int(sum(ep["died"] for e in evals for ep in e["episodes"])), growth=float(np.mean(g)),
                growth_ci=p3a.bootstrap_mean_ci(g), runs=len(g))


def curve_point(e: dict, log, market) -> dict:
    mt = e["diagnosis"]["metrics"]
    lev = log["leverage_after_fill"].to_numpy()
    return dict(q2=mt.get("dq2"), median=float(np.median(lev)),
                ruinous=float(market.ruinous(log["exposure_after_fill"].to_numpy()).mean()) if market is not None else 0.0,
                deaths=int(sum(ep["died"] for ep in e["episodes"])))


def upper_bound_share(logs, bounds) -> float:
    """Share of decisions whose raw size output is at the UPPER end of its numerical range: the only end that could
    act as a cap on exposure. (The lower end represents holding nothing, which an agent may rightly choose.)"""
    hi = bounds[1]
    raw = logs["action_exposure_raw"].to_numpy()
    return float(np.mean(raw >= hi - 1e-6 * max(1.0, hi - bounds[0])))


def seed_metrics(finals, logs, nulls: dict, trainpath, market, g_star: float, curve: dict, bounds) -> dict:
    growth = [ep["log_growth"] for e in finals for ep in e["episodes"]]
    lev = logs["leverage_after_fill"].to_numpy()
    x = logs["exposure_after_fill"].to_numpy()
    q2_eps = [v for e in finals for v in (e["diagnosis"]["metrics"].get("dq2_by_episode") or [])]
    s = dict(
        runs=len(growth), growth=float(np.mean(growth)), growth_ci=p3a.bootstrap_mean_ci(growth),
        growth_t=float(np.mean(growth) / (np.std(growth, ddof=1) / math.sqrt(len(growth)))) if np.std(growth) > 0 else 0.0,
        deaths=int(sum(ep["died"] for e in finals for ep in e["episodes"])),
        median_size=float(np.median(lev)), median_signed=float(np.median(x)), p95_size=float(np.percentile(lev, 95)),
        max_size=float(lev.max()), long_share=float(np.mean(x > 1e-6)),
        ruinous=float(np.mean(market.ruinous(x))) if market is not None else 0.0,
        q2=float(np.mean(q2_eps)) if q2_eps else None,
        tilt=p3a.tilt(logs), rejected=float(np.mean(logs["rejected"].fillna("").astype(str) != "")),
        boundary=float(np.mean((lev >= p3a.BOUNDARY_LEVERAGE) | (logs["gross_selected"].to_numpy() >= p3a.BOUNDARY_LEVERAGE))),
        safeguard_share=upper_bound_share(logs, bounds), g_star=g_star, curve=curve,
    )
    beats = []
    for e in finals:
        n = nulls.get(e["eval_id"])
        g = float(np.mean([ep["log_growth"] for ep in e["episodes"]]))
        p95 = float(np.percentile(n["null"]["mean_log_growth_per_agent"], 95)) if n else math.nan
        beats.append(dict(round_id=e["round_id"], growth=g, null_p95=p95, beats=bool(n is not None and g > p95)))
    s["null"] = beats
    s["rounds_beating_null"] = sum(b["beats"] for b in beats)
    tq = (trainpath or {}).get("diagnosis", {}).get("metrics", {}).get("dq2")
    s["q2_trainpaths"] = tq
    s["train_gap"] = None if (tq is None or s["q2"] is None) else tq - s["q2"]
    s["size_distance"] = abs(math.log(max(s["median_size"], 1e-12) / g_star)) if g_star > 0 else None
    return s


def _e_basic(s) -> bool:
    return s["boundary"] < w.BOUNDARY_MAX and s["rejected"] <= w.REJECT_MAX and s["safeguard_share"] < w.SAFEGUARD_MAX


def flags_s6(s, bench, task="S6") -> dict:
    g = s["g_star"]
    lo, hi = g / w.SIZE_FACTOR, g * w.SIZE_FACTOR
    c = s["curve"]
    dist = lambda v: abs(math.log(max(v, 1e-12) / g))
    f = dict(W1=s["ruinous"] < w.RUIN_MAX, W2=s["deaths"] <= bench["deaths"] + 2, W3=lo <= s["median_size"] <= hi,
             W5=bool(s["growth"] > 0 and s["growth_ci"][0] > 0 and s["rounds_beating_null"] >= 5),
             W6=s["tilt"] < w.TILT_MAX)
    if task == "S6":
        f["W4"] = s["q2"] is not None and s["q2"] >= w.Q2_MIN
        f["W7"] = bool(FIRST in c and LAST in c and c[LAST]["q2"] > c[FIRST]["q2"] and s["ruinous"] < w.RUIN_MAX)
        f["E"] = bool(_e_basic(s) and s["train_gap"] is not None and s["train_gap"] <= w.TRAIN_GAP_MAX)
    else:
        f["W7"] = bool(FIRST in c and LAST in c and dist(c[LAST]["median"]) < dist(c[FIRST]["median"]) and s["ruinous"] < w.RUIN_MAX)
        f["E"] = _e_basic(s)
    f["success"] = all(f.values())
    return f


def flags_s4(s) -> dict:
    f = dict(no_edge=abs(s["growth_t"]) < 2, no_deaths=s["deaths"] == 0, E=_e_basic(s))
    f["success"] = all(f.values())
    return f


def interpretation(flags, seeds) -> str:
    if sum(f["success"] for f in flags) >= w.SUCCESS_SEEDS:
        return "safe growth"
    if sum(s["ruinous"] >= w.RUIN_MAX for s in seeds) >= w.EXPLORATION_SEEDS:
        return "still enters the ruin zone"
    if sum(f["W1"] for f in flags) >= w.TIMID_SAFE_SEEDS and sum(f["W3"] for f in flags) <= w.TIMID_SIZE_MAX:
        return "safe but timid (learning problem)"
    return "mixed"


def verdicts(seeds: dict, bench: dict, refs: dict) -> dict:
    out = {}
    for g, task in (("MK2-S6", "S6"), ("MK2-S7", "S7"), ("MK2-S4", "S4")):
        ss = sorted(seeds[g], key=lambda s: s["seed_index"])
        flags = [flags_s4(s) if task == "S4" else flags_s6(s, bench[task], task) for s in ss]
        r = dict(n=len(ss), flags=flags, successes=sum(f["success"] for f in flags),
                 passes=sum(f["success"] for f in flags) >= w.SUCCESS_SEEDS,
                 counts={k: sum(f[k] for f in flags) for k in flags[0] if k != "success"},
                 growth_mean=float(np.mean([s["growth"] for s in ss])), deaths_total=int(sum(s["deaths"] for s in ss)),
                 median_size_mean=float(np.mean([s["median_size"] for s in ss])),
                 median_size_ci=w.boot_mean([s["median_size"] for s in ss]), benchmark=bench[task],
                 chosen_models={k: sum(s["selection"].get("chosen") == k for s in ss) for k in ("unconditional", "conditional")})
        if task != "S4":
            r["ruinous_mean"] = float(np.mean([s["ruinous"] for s in ss]))
        if task == "S6":
            q = [s["q2"] for s in ss]
            r.update(q2_mean=float(np.mean(q)), q2_ci=w.boot_mean(q), interpretation=interpretation(flags, ss))
            ref_sets = {"M1 MK-S6": refs["m1"]["seeds"]["MK-S6"], "R-FL R1 control": refs["rfl"]["seeds"]["R1"]}
            raw, comps = {}, {}
            for name, rs in ref_sets.items():
                for mname in METRICS:
                    a = [w.metric(s, mname, s["g_star"]) for s in ss]
                    b = [w.metric(x, mname, ss[0]["g_star"]) for x in rs]
                    key = f"MK2-S6 vs {name}: {mname}"
                    raw[key] = w.permutation_p_two_sided(a, b)
                    comps[key] = dict(difference=float(np.mean(a) - np.mean(b)), ci=w.boot_diff(a, b), p_raw=raw[key],
                                      reference_mean=float(np.mean(b)))
            adj = p3a.holm(raw)
            for k in comps:
                comps[k]["p_holm"] = adj[k]
            r["comparisons"] = comps
        out[g] = r
    out["M2_passes"] = all(out[g]["passes"] for g in ("MK2-S6", "MK2-S7", "MK2-S4"))
    return out
