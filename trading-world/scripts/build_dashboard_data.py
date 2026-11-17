"""Build the data files behind the agent dashboard (reports/dashboard/).

  agents.json     one row per planned Phase 2 agent: status, test results, pass/fail
  replay.csv      one recorded test episode per trained agent, decision by decision
  downloads.csv   historical data download status, one row per instrument-month

Usage: python scripts/build_dashboard_data.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tradeworld.learning.gate import summarise, verdict  # noqa: E402

OUT = ROOT / "reports" / "dashboard"
RUNS = ROOT / "runs" / "phase2"
SEEDS = (0, 1, 2)

TASK_NAMES = {
    "S1": "Hidden pattern, with trading costs",
    "S2": "Trend switches it is told about",
    "S3": "Pure noise, with trading costs",
    "S4": "Pure noise, free trading",
    "S5": "Leak trap",
    "S6": "Steady gains with sudden crashes",
    "S7": "Known best leverage (5x)",
}
REWARD_NAMES = {"R1": "Long-term growth", "R2": "Reckless (comparison)"}


def agents_rows():
    from phase2_synthetic import DEFAULT_JOBS, STEPS, DEFAULT_STEPS
    runs = {}
    for p in RUNS.glob("*.json"):
        r = json.loads(p.read_text())
        runs[(r["task"], r["reward"], r["seed"])] = r
    summ = {k: summarise(r) for k, r in runs.items()}
    rows = []
    for job in DEFAULT_JOBS:
        task, reward = job.split(":")
        for seed in SEEDS:
            key = (task, reward, seed)
            row = dict(agent=f"{task}-{reward}-{seed}", task=task, task_name=TASK_NAMES[task], reward=reward,
                       reward_name=REWARD_NAMES[reward], seed=seed, steps=STEPS.get(task, DEFAULT_STEPS))
            if key in summ:
                s = summ[key]
                ref = summ.get((task, "R1", seed)) if reward == "R2" else None
                ok, why = verdict(task, reward, s, ref)
                # rounded to what the dashboard shows, so its number checks match exactly
                row.update(status="trained", growth_pct=round(100 * (math.exp(s["lg"]) - 1), 1),
                           deaths_pct=round(100 * s["died"]), avg_leverage=round(s["gross"], 2),
                           avg_net=round(s["net"], 2), result="pass" if ok else "fail", why=why)
            else:
                row.update(status="training", growth_pct=None, deaths_pct=None, avg_leverage=None,
                           avg_net=None, result="pending", why="")
            rows.append(row)
    return rows, runs


def replay_rows(runs):
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from tradeworld.learning import synthetic_tasks as st
    from tradeworld.learning.train import model_policy, trace_episode

    torch.set_num_threads(1)
    cache = OUT / "replay_cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for (task, reward, seed), r in sorted(runs.items()):
        agent = f"{task}-{reward}-{seed}"
        zip_path = RUNS / f"{task}_{reward}_s{seed}.zip"
        c = cache / f"{agent}.json"
        if c.exists() and c.stat().st_mtime >= zip_path.stat().st_mtime:
            rows = json.loads(c.read_text())
        else:
            model = PPO.load(str(zip_path), device="cpu")
            test = st.build(task, 10_000 + seed, 60_000)
            rows = trace_episode(model_policy(model), test, 0, 1000, reward=reward)
            c.write_text(json.dumps(rows))
        p0 = rows[0]["price"]
        for x in rows:
            out.append(dict(agent=agent, step=x["step"], price=round(100 * x["price"] / p0, 4),
                            net_exposure=round(x["net"], 4), gross_exposure=round(x["gross"], 4),
                            equity=round(x["equity"], 4), regime="" if x["flag"] is None else int(x["flag"])))
    return out


def download_rows():
    import yaml
    plan = yaml.safe_load((ROOT / "configs/data/download_plan.yaml").read_text())
    root = ROOT / plan["archive_dir"]
    rows = []
    for sym in plan["instruments"]:
        for y in (2022, 2023):
            for m in range(1, 13):
                done = (root / sym / f"{y:04d}-{m:02d}.tar").exists()
                rows.append(dict(instrument=sym, month=f"{y:04d}-{m:02d}", downloaded="yes" if done else "no"))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    global RUNS, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    RUNS, OUT = Path(a.runs), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    agents, runs = agents_rows()
    (OUT / "agents.json").write_text("[\n" + ",\n".join(json.dumps(r) for r in agents) + "\n]\n")
    rep = replay_rows(runs)
    if rep:
        write_csv(OUT / "replay.csv", rep)
    else:
        (OUT / "replay.csv").write_text("agent,step,price,net_exposure,gross_exposure,equity,regime\n")
    write_csv(OUT / "downloads.csv", download_rows())
    print(f"agents: {len(agents)} ({sum(a['status'] == 'trained' for a in agents)} trained); "
          f"replay rows: {len(rep)}; written to {OUT}")


if __name__ == "__main__":
    main()
