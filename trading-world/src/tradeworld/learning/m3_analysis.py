"""Experiment M3: the pre-registered analysis (frozen by checksum in the manifest before any training).

Agents: model-based growth-optimal agents v3 (model_kelly3: version 2 plus calibration of the conditional model,
persistence, and the cost-aware decision), six seeds per market, judged on their FINAL checkpoint (200,000 bars
of experience) on six fresh rounds of 20 test runs (120 runs). References (not agents): S1C, S2C, S3 the same
cost-aware decision with the market's TRUE parameters (model_kelly3.reference_policy); S6 the exact growth-optimal
constant size (as in M2). Both run on the same test runs as the agents.

S1C and S2C (predictable, trading costs high enough that trading every signal loses money): a seed succeeds if
  C1  mean log growth per run > 0 with its 95% bootstrap interval (120 runs) above 0
  C2  growth above the matched-random 95th percentile in at least 5 of the 6 rounds
  C3  mean growth per run >= 50% of the reference's mean growth on the same 120 runs
  C4  not more deaths than the reference beyond chance, on the same 120 runs: of the runs in which exactly one of
      the two died, the agent's share is not significantly above one half (exact one-sided binomial test, p >= 0.05),
      and the agent dies in at most 10% of runs
  C5  holds the edge's side when the edge is worth its cost: on decisions where the TRUE expected return exceeds the
      per-bar cost of entering (half spread x (1 - true persistence)), the agent's position is on the side of the
      true expected return in >= 90% of them. (The reference, by the decision's mathematics, always is: 99.96% in
      design simulations. A looser test - side held on every decision - would fail even the reference, 68%, because
      the no-trade band rightly keeps positions through weak opposite signals.)
  E   decisions at 1,000x or more < 0.1%; rejected orders <= 0.1%; size output at the upper end of its numerical
      range < 0.1%
S3 (noise with the same spread as S1; no edge): a seed succeeds if
  N1  claims no edge: |t| of mean log growth over the 120 runs < 2
  N2  stops paying to trade noise: spread costs paid < 0.5% of starting equity per run on average
  N3  no test deaths
  E   as above
S6 (regression check: no costs, so version 3 must keep everything version 2 achieved): Experiment M2's S6 criteria
  (m2_analysis.flags_s6: W1, W2', W3, W4, W5', W6, W7', E) against the exact optimum.
A market passes if at least 4 of its 6 seeds succeed. M3 passes if all four markets pass.
Reported, not criteria: the same learned models deciding as if trading were free ("cost-blind ablation", S1C and
  S2C final rounds), to show what weighing costs is worth; S1C/S2C learning curve (growth on the development round
  after 20,000, 60,000 and 200,000 bars of experience); calibration slopes and persistence; costs and turnover.
Planned comparison (S1C, S2C): growth of v3 minus the cost-blind ablation, same seeds and runs: exact two-sided
  sign-flip test across the six seeds, Holm over the two markets.
"""
from __future__ import annotations

import math

import numpy as np

from . import m2_analysis as m2a
from . import phase3_analysis as p3a
from . import phase3b_analysis as w

GROWTH_SHARE_MIN = 0.5
ALIGN_MIN = 0.90
DEATH_P_MIN = 0.05
DEATH_RATE_MAX = 0.10
NOISE_T_MAX = 2.0
NOISE_COST_MAX = 0.005
TASK_OF = {"MK3-S1C": "S1C", "MK3-S2C": "S2C", "MK3-S3": "S3", "MK3-S6": "S6"}


def growth_stats(evals: list[dict]) -> dict:
    g = [ep["log_growth"] for e in evals for ep in e["episodes"]]
    sd = float(np.std(g, ddof=1)) if len(g) > 1 else 0.0
    return dict(runs=len(g), growth=float(np.mean(g)), growth_ci=p3a.bootstrap_mean_ci(g),
                died_runs=[bool(ep["died"]) for e in sorted(evals, key=lambda e: e["round_id"]) for ep in e["episodes"]],
                growth_t=float(np.mean(g) / (sd / math.sqrt(len(g)))) if sd > 0 else 0.0,
                deaths=int(sum(ep["died"] for e in evals for ep in e["episodes"])),
                costs_per_run=float(np.mean([ep["costs"] for e in evals for ep in e["episodes"]])),
                turnover=float(np.mean([ep["turnover"] for e in evals for ep in e["episodes"]])))


def alignment(task: str, logs, params: dict, rho_true: float) -> dict | None:
    """On decisions where the true expected log return exceeds the per-bar cost of entering, the share whose position
    is on its side. S1C: true mean = phi x last log return (recovered from the observation's normalised return and
    volatility level); S2C: flag x mu (the flag in effect for the next bar, column `flag`, from the test data)."""
    if task == "S1C":
        r = logs["obs_last_return"].to_numpy(float) * 10.0 ** (4.0 * (logs["obs_vol_level"].to_numpy(float) - 1.0))
        m = float(params["phi"]) * r
    elif task == "S2C":
        m = logs["flag"].to_numpy(float) * float(params["mu"])
    else:
        return None
    cost = float(params["spread"]) / 2.0 * (1.0 - rho_true)
    strong = np.abs(m) > cost
    x = logs["exposure_after_fill"].to_numpy(float)
    share = float(np.mean(np.sign(x[strong]) == np.sign(m[strong]))) if strong.any() else 0.0
    return dict(share=share, decisions=int(strong.sum()), of=int(len(m)))


def seed_metrics_cost(task: str, finals, logs, nulls: dict, bounds, params: dict, rho_true: float) -> dict:
    s = growth_stats(finals)
    lev = logs["leverage_after_fill"].to_numpy()
    s.update(median_size=float(np.median(lev)), p95_size=float(np.percentile(lev, 95)), max_size=float(lev.max()),
             time_in_market=float(np.mean(lev > 1e-6)),
             changes=float(np.mean(logs["position_change"].fillna("").astype(str).isin(["open", "increase", "decrease", "close", "reverse"]))),
             rejected=float(np.mean(logs["rejected"].fillna("").astype(str) != "")),
             boundary=float(np.mean((lev >= p3a.BOUNDARY_LEVERAGE) | (logs["gross_selected"].to_numpy() >= p3a.BOUNDARY_LEVERAGE))),
             safeguard_share=m2a.upper_bound_share(logs, bounds), alignment=alignment(task, logs, params, rho_true))
    beats = []
    for e in finals:
        n = nulls.get(e["eval_id"])
        g = float(np.mean([ep["log_growth"] for ep in e["episodes"]]))
        p95 = float(np.percentile(n["null"]["mean_log_growth_per_agent"], 95)) if n else math.nan
        beats.append(dict(round_id=e["round_id"], growth=g, null_p95=p95, beats=bool(n is not None and g > p95)))
    s["null"] = beats
    s["rounds_beating_null"] = sum(b["beats"] for b in beats)
    return s


def _e(s) -> bool:
    return s["boundary"] < w.BOUNDARY_MAX and s["rejected"] <= w.REJECT_MAX and s["safeguard_share"] < w.SAFEGUARD_MAX


def paired_death_test(agent_died: list[bool], ref_died: list[bool]) -> dict:
    """Exact one-sided binomial test on the runs where exactly one of agent and reference died."""
    if len(agent_died) != len(ref_died):
        raise ValueError("agent and reference were not run on the same test runs")
    b = sum(a and not r for a, r in zip(agent_died, ref_died))
    c = sum(r and not a for a, r in zip(agent_died, ref_died))
    n = b + c
    p = sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n if n else 1.0
    return dict(agent_only=b, reference_only=c, p=p)


def flags_predictable(s, ref) -> dict:
    dt = paired_death_test(s["died_runs"], ref["died_runs"])
    s["death_test"] = dt
    f = dict(C1=bool(s["growth"] > 0 and s["growth_ci"][0] > 0), C2=s["rounds_beating_null"] >= 5,
             C3=bool(ref["growth"] > 0 and s["growth"] >= GROWTH_SHARE_MIN * ref["growth"]),
             C4=bool(dt["p"] >= DEATH_P_MIN and s["deaths"] <= DEATH_RATE_MAX * s["runs"]),
             C5=bool(s["alignment"] is not None and s["alignment"]["share"] >= ALIGN_MIN),
             E=_e(s))
    f["success"] = all(f.values())
    return f


def flags_noise(s) -> dict:
    f = dict(N1=abs(s["growth_t"]) < NOISE_T_MAX, N2=s["costs_per_run"] / 10.0 < NOISE_COST_MAX, N3=s["deaths"] == 0, E=_e(s))
    f["success"] = all(f.values())
    return f


def sign_flip_p(d) -> float:
    """Exact two-sided sign-flip test of mean(d) = 0 over all 2^n sign patterns."""
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    cnt = 0
    for m in range(2 ** n):
        sg = np.array([1 if (m >> i) & 1 else -1 for i in range(n)])
        cnt += abs((sg * d).mean()) >= obs - 1e-15
    return cnt / 2 ** n


def verdicts(seeds: dict, refs: dict, ablation: dict) -> dict:
    out = {}
    for g, task in TASK_OF.items():
        ss = sorted(seeds[g], key=lambda s: s["seed_index"])
        if task == "S6":
            flags = [m2a.flags_s6(s, refs[task], "S6") for s in ss]
        elif task == "S3":
            flags = [flags_noise(s) for s in ss]
        else:
            flags = [flags_predictable(s, refs[task]) for s in ss]
        r = dict(n=len(ss), flags=flags, successes=sum(f["success"] for f in flags),
                 passes=sum(f["success"] for f in flags) >= w.SUCCESS_SEEDS,
                 counts={k: sum(f[k] for f in flags) for k in flags[0] if k != "success"},
                 growth_mean=float(np.mean([s["growth"] for s in ss])), growth_ci=w.boot_mean([s["growth"] for s in ss]),
                 deaths_total=int(sum(s["deaths"] for s in ss)), reference=refs[task],
                 median_size_mean=float(np.mean([s["median_size"] for s in ss])),
                 chosen_models={k: sum(s["selection"].get("chosen") == k for s in ss) for k in ("unconditional", "conditional")})
        if task in ("S1C", "S2C"):
            r["growth_share_of_reference"] = r["growth_mean"] / refs[task]["growth"] if refs[task]["growth"] else None
            ab = [ablation.get(s["agent_id"]) for s in ss]
            if all(a is not None for a in ab):
                d = [s["growth"] - a["growth"] for s, a in zip(ss, ab)]
                r["ablation"] = dict(growth_mean=float(np.mean([a["growth"] for a in ab])),
                                     deaths_total=int(sum(a["deaths"] for a in ab)),
                                     difference=float(np.mean(d)), ci=w.boot_mean(d), p_raw=sign_flip_p(d))
        out[g] = r
    raw = {g: out[g]["ablation"]["p_raw"] for g in ("MK3-S1C", "MK3-S2C") if "ablation" in out[g]}
    adj = p3a.holm(raw) if raw else {}
    for g, p in adj.items():
        out[g]["ablation"]["p_holm"] = p
    out["M3_passes"] = all(out[g]["passes"] for g in TASK_OF)
    return out
