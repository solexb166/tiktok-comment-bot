"""Aggregate runs/phase2/*.json into reports/phase2_synthetic.md using the pre-registered
criteria in docs/decisions/0002-phase2-gate-criteria.md."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradeworld.learning.gate import summarise, verdict  # noqa: E402


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
