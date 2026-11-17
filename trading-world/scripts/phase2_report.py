"""Aggregate runs/phase2/*.json into reports/phase2_synthetic.md using the pre-registered
criteria in docs/decisions/0002-phase2-gate-criteria.md."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "phase2"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase2_synthetic.md"))
    a = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(Path(a.runs).glob("*.json"))]
    by = defaultdict(dict)
    for r in runs:
        by[(r["task"], r["reward"])][r["seed"]] = r
    S = {k: {seed: summarise(r) for seed, r in v.items()} for k, v in by.items()}
    L = ["# Phase 2 gate: synthetic learning experiments\n",
         "Criteria were fixed before running (docs/decisions/0002-phase2-gate-criteria.md). "
         "Agents are trained on one synthetic market path and tested on a different, unseen path.\n",
         "| Task | Reward | Seed | Test growth (log) | t | Deaths | Mean gross | Mean net | Result |",
         "|---|---|---|---|---|---|---|---|---|"]
    task_pass = {}
    for (task, reward) in sorted(S):
        oks = []
        for seed, s in sorted(S[(task, reward)].items()):
            ref = S.get((task, "R1"), {}).get(seed) if reward == "R2" else None
            ok, why = verdict(task, reward, s, ref)
            oks.append(ok)
            L.append(f"| {task} | {reward} | {seed} | {s['lg']:+.4f} | {s['t']:.2f} | {100 * s['died']:.0f}% | "
                     f"{s['gross']:.3f}x | {s['net']:+.3f}x | {'✅' if ok else '❌'} {why} |")
        task_pass[(task, reward)] = (sum(oks), len(oks))
    L += ["", "## Gate summary (majority of seeds must pass)\n"]
    for (task, reward), (k, n) in sorted(task_pass.items()):
        L.append(f"- {'✅' if 2 * k > n else '❌'} {task} / {reward}: {k} of {n} seeds pass")
    any_run = next(iter(runs), None)
    if any_run:
        tr = any_run["tracking"]
        L += ["", "## Reproducibility\n",
              f"git {tr['git_commit'][:10]}, Python {tr['python']}, torch {tr['torch']}, SB3 {tr['sb3']}, "
              f"gymnasium {tr['gymnasium']}, numpy {tr['numpy']}. PPO {tr['ppo']}. Policy {tr['policy']}. "
              f"Env {tr['env']}. {tr['test_episodes']} test episodes x {tr['episode_decisions']} decisions. "
              f"Train steps per agent: {any_run['train_steps']:,}. Data hashes and seeds are in each run's JSON."]
    text = "\n".join(L) + "\n"
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
