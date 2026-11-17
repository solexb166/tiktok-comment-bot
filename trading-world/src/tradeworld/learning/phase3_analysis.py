"""Phase 3 Experiment 1: the pre-registered analysis (proposal 0006, sections 7 and 8).

This file is frozen before any Experiment 1 agent is trained: the experiment manifest records its
SHA-256, and the report refuses to run if the file has changed since. Every threshold and test
used to judge the experiment is defined here and nowhere else.

Unit of analysis: the seed (six independent agents per group). Each agent is judged on its FINAL
checkpoint only (no selection), on two fresh evaluation rounds of 20 test runs each (40 runs).

An intervention group shows genuine improvement over the replicated control only if ALL hold:
  G1  mean decision quality Q2.0 above control: exact one-sided permutation test across seeds
      (all 924 relabellings of 6 + 6), Holm-corrected across the four intervention groups,
      adjusted p < 0.05, and effect (group mean - control mean) at least +0.5
  G2  ruinous-decision share below 1% in at least 5 of 6 seeds
  G3  median position size within 0.5x to 2x of the growth-optimal size in at least 4 of 6 seeds
  G4  in at least 4 of 6 seeds: mean realised log growth over the 40 runs above zero with its 95%
      bootstrap interval above zero, AND in each round above the 95th percentile of ten
      leverage- and turnover-matched random agents
  G5  replication: the Q2 effect over control is positive in each of the two rounds separately
  G6  training history: ruinous share in the last 20% of training episodes below the first 20%
      (or below 0.1% in both: nothing left to fall from) in at least 4 of 6 seeds
Reward-exploitation checks (all must be clear):
  E1  no training-path advantage: mean (Q2 on own training paths - Q2 on fresh data) <= 0.5
  E2  no reliance on simulator boundaries: mean share of decisions at 1,000x or more (half the
      1:2000 margin limit) below 0.1%
  E3  noise-driven sizing: median across seeds of the previous-move tilt below 1.5 (tilt =
      max(r, 1/r), r = mean size after a previous-bar move above +2 sd / after one below -2 sd,
      sd = 1.4826 x median absolute deviation of bar returns, so that crashes do not inflate it;
      S6 bar returns are independent, so the right value is 1)
  E4  rejected orders: mean share of decisions with a rejected order no more than control + 0.1 pp
Learning from catastrophe (K): in training, ruinous share in the 50 training episodes after each
  death minus the 50 before, averaged over the agent's deaths (seeds with 5+ deaths). Must be
  negative (exact one-sided sign-flip test across seeds, p < 0.05). With fewer than 3 such seeds it
  is reported as not measurable and does not block. If the control shows a similar fall, the fall
  is reported as not attributable to the intervention.
"""
from __future__ import annotations

import io
import itertools
import math

import numpy as np
import pandas as pd

ALPHA = 0.05
MIN_EFFECT_Q2 = 0.5
RUIN_MAX, RUIN_SEEDS = 0.01, 5
SIZE_LO, SIZE_HI, SIZE_SEEDS = 0.5, 2.0, 4
GROWTH_SEEDS = 4
HISTORY_SEEDS, HISTORY_FLOOR = 4, 0.001
TRAIN_GAP_MAX = 0.5
TILT_MAX, TILT_SD = 1.5, 2.0
BOUNDARY_LEVERAGE, BOUNDARY_MAX = 1000.0, 0.001
REJECT_MARGIN = 0.001
CATASTROPHE_WINDOW, CATASTROPHE_MIN_DEATHS, CATASTROPHE_MIN_SEEDS = 50, 5, 3
BOOT, BOOT_SEED = 10_000, 20261117
CONTROL = "control"
INTERVENTIONS = ("C1", "C2", "D", "A")


# ---------------------------------------------------------------------------- statistics
def permutation_p(a, b) -> float:
    """Exact one-sided p-value of mean(a) - mean(b) over all relabellings of the pooled seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.r_[a, b]
    obs = a.mean() - b.mean()
    n, hits, total = len(a), 0, 0
    for idx in itertools.combinations(range(len(pooled)), n):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        total += 1
        hits += pooled[m].mean() - pooled[~m].mean() >= obs - 1e-12
    return hits / total


def holm(p: dict) -> dict:
    """Holm step-down adjusted p-values."""
    keys = sorted(p, key=lambda k: p[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def bootstrap_diff_ci(a, b) -> list[float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(BOOT_SEED)
    d = a[rng.integers(0, len(a), (BOOT, len(a)))].mean(1) - b[rng.integers(0, len(b), (BOOT, len(b)))].mean(1)
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def bootstrap_mean_ci(x) -> list[float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(BOOT_SEED)
    return [float(v) for v in np.percentile(x[rng.integers(0, len(x), (BOOT, len(x)))].mean(1), [2.5, 97.5])]


def signflip_p(d) -> float:
    """Exact one-sided p-value that the mean of d is below zero (all 2^n sign assignments)."""
    d = np.asarray(d, float)
    obs = d.mean()
    vals = [np.mean(d * np.array(s)) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean(np.array(vals) <= obs + 1e-12))


# ---------------------------------------------------------------------------- records
def lineage(reg, agent_id: str, total_steps: int) -> list[dict]:
    """The versions that make up an agent's completed training, oldest first (an interrupted
    attempt and its continuation form one lineage). Empty if training is not complete."""
    vs = {v["version"]: v for v in reg.versions(agent_id)}
    done = [v for v in vs.values() if v["kind"] != "interrupted" and v["final_checkpoint"]["step"] >= total_steps]
    if not done:
        return []
    out, v = [], max(done, key=lambda x: x["version"])
    while v is not None:
        out.append(v)
        v = vs.get(v.get("parent_version")) if v.get("parent_version") is not None else None
    return out[::-1]


def checkpoint_at(chain: list[dict], step: int) -> tuple[int, dict]:
    """(version, checkpoint) of the first checkpoint at or after `step` along a lineage."""
    best = None
    for v in chain:
        for c in v.get("checkpoints", []) + [v["final_checkpoint"]]:
            if c["step"] >= step and (best is None or c["step"] < best[1]["step"]):
                best = (v["version"], c)
    if best is None:
        raise ValueError(f"no checkpoint at or after step {step}")
    return best


def history_frames(reg, chain: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps, ups = [], []
    for v in chain:
        h = v.get("history") or {}
        if h.get("episodes_sha256"):
            eps.append(pd.read_parquet(io.BytesIO(reg.blobs.get(h["episodes_sha256"]))).assign(version=v["version"]))
        if h.get("updates_sha256"):
            ups.append(pd.read_parquet(io.BytesIO(reg.blobs.get(h["updates_sha256"]))).assign(version=v["version"]))
    ep = pd.concat(eps, ignore_index=True).sort_values(["step", "env"], kind="stable") if eps else pd.DataFrame()
    up = pd.concat(ups, ignore_index=True).sort_values("step", kind="stable") if ups else pd.DataFrame()
    return ep.reset_index(drop=True), up.reset_index(drop=True)


# ---------------------------------------------------------------------------- per seed
def tilt(log: pd.DataFrame) -> float:
    r = log["market_return"].to_numpy()
    prev = log.groupby(["round", "episode"])["market_return"].shift(1).to_numpy()
    sd = 1.4826 * float(np.median(np.abs(r - np.median(r)))) or float(np.std(r))
    lev = log["leverage_after_fill"].to_numpy()
    up, dn = prev > TILT_SD * sd, prev < -TILT_SD * sd
    mu_up = float(lev[up].mean()) if up.any() else 0.0
    mu_dn = float(lev[dn].mean()) if dn.any() else 0.0
    if mu_up <= 0 and mu_dn <= 0:
        return 1.0
    if mu_up <= 0 or mu_dn <= 0:
        return math.inf
    x = mu_up / mu_dn
    return max(x, 1 / x)


def history_metrics(ep: pd.DataFrame, up: pd.DataFrame) -> dict:
    out = dict(training_episodes=int(len(ep)))
    if len(ep) and "share_ruinous" in ep:
        k = max(1, len(ep) // 5)
        sr = ep["share_ruinous"].fillna(0.0).to_numpy()
        first, last = float(sr[:k].mean()), float(sr[-k:].mean())
        out.update(ruin_first=first, ruin_last=last, history_met=bool(last < first or (first < HISTORY_FLOOR and last < HISTORY_FLOOR)),
                   death_rate_first=float(ep["died"].iloc[:k].mean()), death_rate_last=float(ep["died"].iloc[-k:].mean()),
                   training_deaths=int(ep["died"].sum()))
        died = np.flatnonzero(ep["died"].to_numpy())
        W = CATASTROPHE_WINDOW
        diffs = [sr[i + 1:i + 1 + W].mean() - sr[i - W:i].mean() for i in died if i - W >= 0 and i + W < len(sr)]
        out.update(catastrophe_deaths=len(diffs), catastrophe_change=float(np.mean(diffs)) if diffs else None)
    for col, key in (("train/approx_kl", "median_kl"), ("train/clip_fraction", "median_clip_fraction")):
        if len(up) and col in up:
            out[key] = float(pd.to_numeric(up[col], errors="coerce").median())
    return out


def seed_metrics(finals: list[dict], logs: pd.DataFrame, nulls: list[dict], trainpath: dict | None,
                 market, g_star: float) -> dict:
    q2_eps = [x for e in finals for x in e["diagnosis"]["metrics"]["dq2_by_episode"]]
    growth = [ep["log_growth"] for e in finals for ep in e["episodes"]]
    lev = logs["leverage_after_fill"].to_numpy()
    m = dict(
        q2=float(np.mean(q2_eps)), q2_by_round={e["round_id"]: float(e["diagnosis"]["metrics"]["dq2"]) for e in finals},
        ruinous=float(np.mean(market.ruinous(logs["exposure_after_fill"].to_numpy()))),
        median_size=float(np.median(lev)), p95_size=float(np.percentile(lev, 95)), max_size=float(lev.max()),
        long_share=float(np.mean(logs["exposure_after_fill"].to_numpy() > 1e-6)),
        deaths=int(sum(ep["died"] for e in finals for ep in e["episodes"])),
        growth=float(np.mean(growth)), growth_ci=bootstrap_mean_ci(growth),
        tilt=tilt(logs), rejected=float(np.mean(logs["rejected"].fillna("").astype(str) != "")),
        boundary=float(np.mean((lev >= BOUNDARY_LEVERAGE) | (logs["gross_selected"].to_numpy() >= BOUNDARY_LEVERAGE))),
        gate_results={e["round_id"]: e["result"] for e in finals},
    )
    m["size_ratio"] = m["median_size"] / g_star
    m["size_ok"] = SIZE_LO <= m["size_ratio"] <= SIZE_HI
    m["ruin_ok"] = m["ruinous"] < RUIN_MAX
    beats = []
    for e in finals:
        n = next((x for x in nulls if x["eval_id"] == e["eval_id"]), None)
        g = float(np.mean([ep["log_growth"] for ep in e["episodes"]]))
        p95 = float(np.percentile(n["null"]["mean_log_growth_per_agent"], 95)) if n else math.nan
        beats.append(dict(round_id=e["round_id"], growth=g, null_p95=p95, beats=bool(n is not None and g > p95)))
    m["null"] = beats
    m["growth_ok"] = bool(m["growth"] > 0 and m["growth_ci"][0] > 0 and all(b["beats"] for b in beats))
    m["q2_trainpaths"] = float(trainpath["diagnosis"]["metrics"]["dq2"]) if trainpath else None
    m["train_gap"] = None if trainpath is None else m["q2_trainpaths"] - m["q2"]
    return m


# ---------------------------------------------------------------------------- per group
def group_verdicts(seeds: dict) -> dict:
    """seeds: {group: [per-seed metric dicts]} -> pre-registered verdicts."""
    ctrl = seeds[CONTROL]
    cq = [s["q2"] for s in ctrl]
    raw = {g: permutation_p([s["q2"] for s in seeds[g]], cq) for g in INTERVENTIONS if g in seeds}
    adj = holm(raw)
    rounds = sorted(ctrl[0]["q2_by_round"])
    ctrl_rej = float(np.mean([s["rejected"] for s in ctrl]))
    ctrl_k = [s["catastrophe_change"] for s in ctrl if (s.get("catastrophe_deaths") or 0) >= CATASTROPHE_MIN_DEATHS]
    out = {}
    for g, ss in seeds.items():
        q = [s["q2"] for s in ss]
        r = dict(n=len(ss), q2_mean=float(np.mean(q)), q2_sd=float(np.std(q, ddof=1)) if len(q) > 1 else None,
                 ruin_ok=sum(s["ruin_ok"] for s in ss), size_ok=sum(s["size_ok"] for s in ss),
                 growth_ok=sum(s["growth_ok"] for s in ss), history_ok=sum(bool(s.get("history_met")) for s in ss),
                 ruinous_mean=float(np.mean([s["ruinous"] for s in ss])),
                 median_size_mean=float(np.mean([s["median_size"] for s in ss])),
                 G2=sum(s["ruin_ok"] for s in ss) >= RUIN_SEEDS, G3=sum(s["size_ok"] for s in ss) >= SIZE_SEEDS,
                 G4=sum(s["growth_ok"] for s in ss) >= GROWTH_SEEDS,
                 G6=sum(bool(s.get("history_met")) for s in ss) >= HISTORY_SEEDS)
        gaps = [s["train_gap"] for s in ss if s["train_gap"] is not None]
        r["train_gap_mean"] = float(np.mean(gaps)) if gaps else None
        r["E1"] = bool(gaps) and r["train_gap_mean"] <= TRAIN_GAP_MAX
        r["boundary_mean"] = float(np.mean([s["boundary"] for s in ss]))
        r["E2"] = r["boundary_mean"] < BOUNDARY_MAX
        r["tilt_median"] = float(np.median([s["tilt"] for s in ss]))
        r["E3"] = r["tilt_median"] < TILT_MAX
        r["rejected_mean"] = float(np.mean([s["rejected"] for s in ss]))
        r["E4"] = r["rejected_mean"] <= ctrl_rej + REJECT_MARGIN
        ks = [s["catastrophe_change"] for s in ss if (s.get("catastrophe_deaths") or 0) >= CATASTROPHE_MIN_DEATHS]
        if len(ks) < CATASTROPHE_MIN_SEEDS:
            r["K"], r["K_status"] = None, f"not measurable ({len(ks)} seeds with {CATASTROPHE_MIN_DEATHS}+ training deaths)"
        else:
            p = signflip_p(ks)
            r["K"] = bool(np.mean(ks) < 0 and p < ALPHA)
            r["K_status"] = f"mean change {np.mean(ks):+.4f} over {len(ks)} seeds, sign-flip p = {p:.3f}"
            r["K_mean"], r["K_p"] = float(np.mean(ks)), p
        r["K_control_mean"] = float(np.mean(ctrl_k)) if ctrl_k else None
        if g != CONTROL:
            eff = r["q2_mean"] - float(np.mean(cq))
            per_round = {rid: float(np.mean([s["q2_by_round"][rid] for s in ss]) -
                                    np.mean([s["q2_by_round"][rid] for s in ctrl])) for rid in rounds}
            r.update(effect=eff, effect_ci=bootstrap_diff_ci(q, cq), p_raw=raw[g], p_holm=adj[g],
                     effect_by_round=per_round,
                     cohens_d=eff / math.sqrt((np.var(q, ddof=1) + np.var(cq, ddof=1)) / 2)
                     if len(q) > 1 and (np.var(q, ddof=1) + np.var(cq, ddof=1)) > 0 else None,
                     ruinous_effect=r["ruinous_mean"] - float(np.mean([s["ruinous"] for s in ctrl])),
                     ruinous_effect_ci=bootstrap_diff_ci([s["ruinous"] for s in ss], [s["ruinous"] for s in ctrl]))
            r["G1"] = bool(adj[g] < ALPHA and eff >= MIN_EFFECT_Q2)
            r["G5"] = all(v > 0 for v in per_round.values())
            r["genuine_improvement"] = bool(r["G1"] and r["G2"] and r["G3"] and r["G4"] and r["G5"] and r["G6"]
                                            and r["E1"] and r["E2"] and r["E3"] and r["E4"] and r["K"] is not False)
        out[g] = r
    return out
