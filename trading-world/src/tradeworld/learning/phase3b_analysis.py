"""Phase 3 Experiment 1b: the pre-registered analysis (approved by the owner 2026-11-18).

Frozen before any Experiment 1b agent is trained: the manifest records this file's SHA-256 (and
those of every file it relies on); the report refuses to run if any of them changed.

Groups: C1m (size = e^z), C2-up and C2-tail (size = softplus(z)), with starting exploration noise
matched to the control in actual position-size terms (see the manifest). Six seed indices; the same
fresh training seed for a given index in every group (paired). Each agent is judged on its FINAL
checkpoint only, on two fresh evaluation rounds of 20 test runs (40 runs).

A seed is a SUCCESSFUL GROWTH AGENT only if all hold:
  W1  ruinous decisions below 1%
  W2  at most 1 death in the 40 test runs
  W3  median position size within x2 of the growth-optimal size (1.47x to 5.86x)
  W4  decision quality Q2.0 at least 0.50
  W5  mean log growth per run above zero with its 95% bootstrap interval above zero, and in each
      round above the 95th percentile of ten leverage- and turnover-matched random agents
  W6  previous-move tilt below 1.5 (as Experiment 1: mean size after a previous-bar move above
      +2 sd / below -2 sd, sd = 1.4826 x MAD, symmetric)
  W7  progressive learning in training: the median over training episodes of the episode's mean
      size is closer to the optimum (|ln(size / G*)|) in the last 20% of training episodes than in
      the first 20%, AND the ruinous share in the last 20% of training episodes is below 1%
  E   exploitation checks clear: own-training-path Q2 minus fresh Q2 at most 0.5; share of
      decisions at 1,000x or more below 0.1%; rejected-order share at most 0.1% (Experiment 1
      control: 0.000%); size output at its numerical safeguard bound in below 0.1% of decisions
A GROUP supports safe growth if at least 4 of its 6 seeds meet every criterion.

Interpretation (fixed in advance, checked in this order):
  "safe growth"                          the group supports safe growth (above)
  "earlier safety came from reduced exploration"   ruinous share >= 1% in at least 3 of 6 seeds
  "safe but timid"                       W1 met in at least 4 seeds and W3 in at most 2
  "mixed"                                anything else
Zero ruinous decisions alone is never counted as successful learning.

Statistics: per seed, bootstrap 95% intervals over the 40 runs; per group, mean with a bootstrap
interval across seeds. Planned comparisons, each new group against its Experiment 1 representation
(C1 or C2) and against the Experiment 1 control, on Q2, ruinous share and distance from the optimum
|ln(median / G*)|: exact two-sided permutation tests across seeds (924 relabellings), Holm-corrected
over all 18 comparisons, with bootstrap intervals of the difference. Between new groups on the same
seed index: exact two-sided sign-flip tests (64 assignments), descriptive only.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from . import phase3_analysis as p3a

RUIN_MAX, DEATHS_MAX = 0.01, 1
SIZE_FACTOR = 2.0
Q2_MIN = 0.50
TILT_MAX = 1.5
TRAIN_GAP_MAX = 0.5
BOUNDARY_MAX = 0.001
REJECT_MAX = 0.001
SAFEGUARD_MAX = 0.001
SUCCESS_SEEDS = 4
EXPLORATION_SEEDS = 3          # ruinous >= 1% in at least this many seeds
TIMID_SAFE_SEEDS, TIMID_SIZE_MAX = 4, 2
BOOT, BOOT_SEED = 10_000, 20261118
NEW_GROUPS = ("C1m", "C2-up", "C2-tail")
REFERENCE = {"C1m": "C1", "C2-up": "C2", "C2-tail": "C2"}
METRICS = ("q2", "ruinous", "size_distance")


# ---------------------------------------------------------------------------- statistics
def permutation_p_two_sided(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.r_[a, b]
    obs = abs(a.mean() - b.mean())
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        total += 1
        hits += abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-12
    return hits / total


def signflip_p_two_sided(d) -> float:
    d = np.asarray(d, float)
    obs = abs(d.mean())
    vals = [abs(np.mean(d * np.array(s))) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean(np.array(vals) >= obs - 1e-12))


def boot_diff(a, b) -> list[float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(BOOT_SEED)
    d = a[rng.integers(0, len(a), (BOOT, len(a)))].mean(1) - b[rng.integers(0, len(b), (BOOT, len(b)))].mean(1)
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def boot_mean(x) -> list[float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(BOOT_SEED)
    return [float(v) for v in np.percentile(x[rng.integers(0, len(x), (BOOT, len(x)))].mean(1), [2.5, 97.5])]


# ---------------------------------------------------------------------------- per seed
def training_progress(ep, g_star: float) -> dict:
    """W7 inputs from the complete training history (episodes in step order)."""
    if ep is None or not len(ep) or "mean_gross" not in ep:
        return dict(w7=False, note="no training history")
    k = max(1, len(ep) // 5)
    size = ep["mean_gross"].to_numpy()
    ruin = ep["share_ruinous"].fillna(0.0).to_numpy() if "share_ruinous" in ep else np.zeros(len(ep))
    s_first, s_last = float(np.median(size[:k])), float(np.median(size[-k:]))
    dist = lambda s: abs(math.log(max(s, 1e-12) / g_star))
    r_last = float(ruin[-k:].mean())
    return dict(train_size_first=s_first, train_size_last=s_last, train_dist_first=dist(s_first),
                train_dist_last=dist(s_last), train_ruin_last=r_last,
                w7=bool(dist(s_last) < dist(s_first) and r_last < RUIN_MAX))


def safeguard_share(logs, bounds) -> float:
    """Share of evaluated decisions whose raw size output is at (or beyond) its safeguard bound."""
    lo, hi = bounds
    raw = logs["action_exposure_raw"].to_numpy()
    tol = 1e-6 * max(1.0, hi - lo)
    return float(np.mean((raw <= lo + tol) | (raw >= hi - tol)))


def seed_flags(s: dict, g_star: float) -> dict:
    lo, hi = g_star / SIZE_FACTOR, g_star * SIZE_FACTOR
    f = dict(
        W1=s["ruinous"] < RUIN_MAX, W2=s["deaths"] <= DEATHS_MAX, W3=lo <= s["median_size"] <= hi,
        W4=s["q2"] >= Q2_MIN, W5=bool(s["growth_ok"]), W6=s["tilt"] < TILT_MAX, W7=bool(s.get("w7")),
        E=bool(s["train_gap"] is not None and s["train_gap"] <= TRAIN_GAP_MAX and s["boundary"] < BOUNDARY_MAX
               and s["rejected"] <= REJECT_MAX and s["safeguard_share"] < SAFEGUARD_MAX),
    )
    f["success"] = all(f.values())
    return f


# ---------------------------------------------------------------------------- per group
def interpretation(flags: list[dict], seeds: list[dict]) -> str:
    n_success = sum(f["success"] for f in flags)
    if n_success >= SUCCESS_SEEDS:
        return "safe growth"
    if sum(s["ruinous"] >= RUIN_MAX for s in seeds) >= EXPLORATION_SEEDS:
        return "earlier safety came from reduced exploration"
    if sum(f["W1"] for f in flags) >= TIMID_SAFE_SEEDS and sum(f["W3"] for f in flags) <= TIMID_SIZE_MAX:
        return "safe but timid"
    return "mixed"


def metric(s: dict, name: str, g_star: float) -> float:
    if name == "size_distance":
        return abs(math.log(max(s["median_size"], 1e-12) / g_star))
    return float(s[name])


def verdicts(seeds: dict, references: dict, g_star: float) -> dict:
    """seeds: {new group: [per-seed metric dicts]}; references: {Experiment 1 group: [per-seed dicts]}."""
    out = {}
    raw = {}
    for g in NEW_GROUPS:
        ss = sorted(seeds[g], key=lambda x: x["seed_index"])
        flags = [seed_flags(s, g_star) for s in ss]
        q = [s["q2"] for s in ss]
        r = dict(n=len(ss), flags=flags, successes=sum(f["success"] for f in flags),
                 counts={k: sum(f[k] for f in flags) for k in ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "E")},
                 supports_safe_growth=sum(f["success"] for f in flags) >= SUCCESS_SEEDS,
                 interpretation=interpretation(flags, ss), q2_mean=float(np.mean(q)), q2_ci=boot_mean(q),
                 ruinous_mean=float(np.mean([s["ruinous"] for s in ss])),
                 ruinous_ci=boot_mean([s["ruinous"] for s in ss]),
                 median_size_mean=float(np.mean([s["median_size"] for s in ss])),
                 median_size_ci=boot_mean([s["median_size"] for s in ss]),
                 seeds_ruinous_ge_1pct=sum(s["ruinous"] >= RUIN_MAX for s in ss), comparisons={})
        for ref in (REFERENCE[g], "control"):
            for mname in METRICS:
                a = [metric(s, mname, g_star) for s in ss]
                b = [metric(s, mname, g_star) for s in references[ref]]
                key = f"{g} vs Exp1 {ref}: {mname}"
                raw[key] = permutation_p_two_sided(a, b)
                r["comparisons"][key] = dict(difference=float(np.mean(a) - np.mean(b)), ci=boot_diff(a, b), p_raw=raw[key])
        out[g] = r
    adj = p3a.holm(raw)
    for g in NEW_GROUPS:
        for key, c in out[g]["comparisons"].items():
            c["p_holm"] = adj[key]
    paired = {}
    for a, b in itertools.combinations(NEW_GROUPS, 2):
        sa = {s["seed_index"]: s for s in seeds[a]}
        sb = {s["seed_index"]: s for s in seeds[b]}
        idx = sorted(set(sa) & set(sb))
        for mname in METRICS:
            d = [metric(sa[i], mname, g_star) - metric(sb[i], mname, g_star) for i in idx]
            paired[f"{a} - {b}: {mname}"] = dict(mean=float(np.mean(d)), p=signflip_p_two_sided(d), n=len(d))
    return dict(groups=out, paired=paired)


# ---------------------------------------------------------------------------- exploration record
EXPLORE_KEYS = ("size_ratio_sd", "size_up2_share", "size_down2_share", "size_sample_ge_ruin_share",
                "position_ratio_sd", "position_sample_ge_ruin_share", "size_mean_median", "size_mean_p99",
                "size_sample_p99", "size_clipped_share", "size_mean_near_bound_share",
                "band_lt1_ratio_sd", "band_1to3_ratio_sd", "band_3to9p5_ratio_sd", "band_ge9p5_ratio_sd",
                "band_lt1_share", "band_1to3_share", "band_3to9p5_share", "band_ge9p5_share")


def exploration_record(up) -> dict:
    """First, middle and last training update, plus the maximum of the safeguard measures."""
    import pandas as pd
    if up is None or not len(up):
        return {}
    cols = {k: pd.to_numeric(up[f"explore/{k}"], errors="coerce") for k in EXPLORE_KEYS if f"explore/{k}" in up}
    if not cols:
        return {}
    frame = pd.DataFrame(cols).dropna(how="all")
    pick = lambda i: {k: (None if not math.isfinite(v) else float(v)) for k, v in frame.iloc[i].items()}
    return dict(first=pick(0), middle=pick(len(frame) // 2), last=pick(-1), updates=int(len(frame)),
                max_clipped_share=float(frame["size_clipped_share"].max()) if "size_clipped_share" in frame else None,
                max_near_bound_share=float(frame["size_mean_near_bound_share"].max())
                if "size_mean_near_bound_share" in frame else None)
