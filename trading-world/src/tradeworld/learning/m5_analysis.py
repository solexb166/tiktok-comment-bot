"""Experiment M5 (same criteria as M4): the pre-registered analysis (frozen by checksum in the manifest before any training).

Agents: model-based growth-optimal agents v5 (model_kelly5, version 4 with a more reliable regime fit; a learned regime-switching world model, tracked while
trading, with version 3's cost-aware decision), six seeds per market, judged on their FINAL checkpoint (200,000 bars of
experience) on six fresh rounds of 20 test runs (120 runs).
References (not agents), on the same test runs:
  S1X, S2X, S6X  true-model filter: version 4's decision with the market's TRUE regime model, inferring the current
                 regime from prices like the agent (the best a trader who knows the rules can do); the ORACLE, which is
                 told the true current regime at every bar, is reported as an upper bound only
  S1C            Experiment M3's true-parameter cost-aware reference;   S6  the exact growth-optimal constant size
Ablation (not agents; one criterion and reported): each switching-market agent's own single-regime model (K = 1,
  version 4's static fallback), deciding on the same runs, to measure what tracking regimes is worth.

S1X, S2X, S6X (markets that switch regime at unannounced times): a seed succeeds if
  A1  mean log growth per run > 0 with its 95% bootstrap interval (120 runs) above 0
  A2  growth above the matched-random 95th percentile in at least 5 of the 6 rounds
  A3  mean growth per run >= 50% of the true-model filter's mean growth on the same 120 runs
  A4  not more deaths than the true-model filter beyond chance: of the runs in which exactly one of the two died, the
      agent's share is not significantly above one half (exact one-sided binomial test, p >= 0.05). (No absolute
      ceiling: in design simulations the true-model filter itself died in 5 of 20 S2X runs; full growth-optimal
      sizing of a weak signal that can silently reverse is inherently risky, which the report states.)
  A5  adaptation is real: the agent's growth minus its own single-regime model's growth on the same runs is > 0 with a
      95% bootstrap interval (paired over the 120 runs) above 0
  E   decisions at 1,000x or more < 0.1%; rejected orders <= 0.1%; size output at the upper end of its range < 0.1%
S1C (no change; regression check against version 3): Experiment M3's S1C criteria (m3_analysis.flags_predictable:
  C1-C5, E) against the M3 true-parameter reference.
S6 (no change, no costs; regression check against version 2): Experiment M2's S6 criteria (m2_analysis.flags_s6:
  W1, W2', W3, W4, W5', W6, W7', E) against the exact optimum.
A market passes if at least 4 of its 6 seeds succeed. M5 passes if all five markets pass.
Reported, not criteria: oracle growth and deaths; regimes found (number, parameters, spell lengths); death rates.
"""
from __future__ import annotations

import numpy as np

from . import m2_analysis as m2a
from . import m3_analysis as m3a
from . import phase3_analysis as p3a
from . import phase3b_analysis as w

GROWTH_SHARE_MIN = 0.5
DEATH_P_MIN = 0.05
TASK_OF = {"MK5-S1X": "S1X", "MK5-S2X": "S2X", "MK5-S6X": "S6X", "MK5-S1C": "S1C", "MK5-S6": "S6"}
SWITCHING = ("S1X", "S2X", "S6X")


def growth_runs(evals: list[dict]) -> list[float]:
    return [ep["log_growth"] for e in sorted(evals, key=lambda e: e["round_id"]) for ep in e["episodes"]]


def paired_diff(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b):
        raise ValueError("not the same test runs")
    d = np.asarray(a) - np.asarray(b)
    return dict(difference=float(d.mean()), ci=p3a.bootstrap_mean_ci(list(d)))


def flags_switching(s, ref, ablation) -> dict:
    dt = m3a.paired_death_test(s["died_runs"], ref["died_runs"])
    ad = paired_diff(s["growth_runs"], ablation["growth_runs"])
    s["death_test"], s["adaptation"] = dt, ad
    f = dict(A1=bool(s["growth"] > 0 and s["growth_ci"][0] > 0), A2=s["rounds_beating_null"] >= 5,
             A3=bool(ref["growth"] > 0 and s["growth"] >= GROWTH_SHARE_MIN * ref["growth"]),
             A4=bool(dt["p"] >= DEATH_P_MIN), A5=bool(ad["difference"] > 0 and ad["ci"][0] > 0), E=m3a._e(s))
    f["success"] = all(f.values())
    return f


def verdicts(seeds: dict, refs: dict, oracle: dict, ablation: dict) -> dict:
    out = {}
    for g, task in TASK_OF.items():
        ss = sorted(seeds[g], key=lambda s: s["seed_index"])
        if task in SWITCHING:
            flags = [flags_switching(s, refs[task], ablation[s["agent_id"]]) for s in ss]
        elif task == "S1C":
            flags = [m3a.flags_predictable(s, refs[task]) for s in ss]
        else:
            flags = [m2a.flags_s6(s, refs[task], "S6") for s in ss]
        r = dict(n=len(ss), flags=flags, successes=sum(f["success"] for f in flags),
                 passes=sum(f["success"] for f in flags) >= w.SUCCESS_SEEDS,
                 counts={k: sum(f[k] for f in flags) for k in flags[0] if k != "success"},
                 growth_mean=float(np.mean([s["growth"] for s in ss])), growth_ci=w.boot_mean([s["growth"] for s in ss]),
                 deaths_total=int(sum(s["deaths"] for s in ss)), reference=refs[task],
                 median_size_mean=float(np.mean([s["median_size"] for s in ss])),
                 regimes_chosen={k: sum(s["selection"].get("regimes") == k for s in ss) for k in (1, 2, 3)})
        if task in SWITCHING:
            r["oracle"] = oracle.get(task)
            r["share_of_true_filter"] = r["growth_mean"] / refs[task]["growth"] if refs[task]["growth"] else None
            ab = [ablation[s["agent_id"]] for s in ss]
            r["ablation"] = dict(growth_mean=float(np.mean([a["growth"] for a in ab])), deaths_total=int(sum(a["deaths"] for a in ab)))
        out[g] = r
    out["M5_passes"] = all(out[g]["passes"] for g in TASK_OF)
    return out
