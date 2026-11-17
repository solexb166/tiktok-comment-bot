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


# ---------------------------------------------------------------------------- agent registry
def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


def registry_rows(reg_root: Path):
    """Agent History data: identities, versions, checkpoints, evaluations, diagnosis, training
    curves and the worst test run of every evaluation for decision replay."""
    import io

    import numpy as np
    import pandas as pd

    from tradeworld.learning.compare import training_progress
    from tradeworld.learning.evaluate import decision_samples
    from tradeworld.registry.registry import Registry

    reg = Registry(reg_root)
    agents, versions, ckpts, evals, diag, train, decisions = [], [], [], [], [], [], []
    for a in reg.agents():
        i, st_ = reg.identity(a), reg.state(a)
        agents.append(dict(agent_id=a, name=i["name"], task=i["task"], task_name=TASK_NAMES.get(i["task"], i["task"]),
                           reward=i["reward"], reward_name=REWARD_NAMES.get(i["reward"], i["reward"]),
                           seed=i["original_seed"], created_at=i["created_at"], status=st_["status"].replace("_", " "),
                           current_version=st_["current_version"], training_attempts=st_["training_attempts"],
                           evaluations=st_["evaluations"], failed_evaluations=st_["failed_evaluations"],
                           passed_on_version=st_["passed_on_version"] or "",
                           passed_on_evaluation=st_["passed_on_evaluation"] or "",
                           champion_version=(st_["champion"] or {}).get("version", "")))
        for v in reg.versions(a):
            tr = v.get("training", {})
            ch = v.get("changes", {})
            versions.append(dict(agent_id=a, version=v["version"], parent=v.get("parent_version") or "",
                                 kind=v["kind"].replace("_", " "), reason=v.get("reason", ""),
                                 steps_added=tr.get("steps_added"), total_steps=tr.get("step_end"),
                                 changes="; ".join(f"{k} = {json.dumps(x)}" for k, x in ch.items()) or "none",
                                 checkpoints=len(v.get("checkpoints", [])),
                                 history="complete" if v.get("history", {}).get("complete") else "not recorded",
                                 recorded_at=v["recorded_at"], code=v.get("code_version", {}).get("git_commit", "")[:10]))
            for c in v.get("checkpoints", []):
                ckpts.append(dict(agent_id=a, version=v["version"], step=c.get("step"), kind=c.get("kind", ""),
                                  checksum=c["model_sha256"][:12],
                                  rng_state="yes" if c.get("state_sha256") else "no"))
        seen = set()
        for e in reg.evaluations(a):
            s, m = e["summary"], (e.get("diagnosis") or {}).get("metrics", {})
            original = e["version"] not in seen
            seen.add(e["version"])
            evals.append(dict(agent_id=a, version=e["version"], round_id=e["round_id"], eval_id=e["eval_id"],
                              original="first evaluation" if original else "later round",
                              result=e["result"], reason=e.get("reason", ""),
                              growth_pct=_r(100 * (math.exp(s["lg"]) - 1), 1), deaths_pct=round(100 * s["died"]),
                              avg_leverage=_r(s["gross"], 2), median_leverage=_r(m.get("median_leverage"), 2),
                              max_leverage=_r(m.get("max_leverage"), 1), share_over_fatal_pct=_r(100 * m["share_over_fatal"], 1)
                              if m.get("share_over_fatal") is not None else None,
                              fatal_leverage=_r(m.get("fatal_leverage"), 1),
                              trade_frequency_pct=_r(100 * m["trade_frequency"], 0) if m.get("trade_frequency") is not None else None,
                              aligned_pct=_r(100 * m["aligned_with_edge"], 0) if m.get("aligned_with_edge") is not None else None,
                              recorded_at=e["recorded_at"]))
            for kind, key in (("Observed", "observations"), ("Possible explanation", "possible_explanations")):
                for o in (e.get("diagnosis") or {}).get(key, []):
                    diag.append(dict(agent_id=a, version=e["version"], round_id=e["round_id"], kind=kind,
                                     code=o["code"], text=o["text"]))
            latest_round = max((x["recorded_at"], x["round_id"]) for x in reg.evaluations(a) if x["version"] == e["version"])[1]
            if e.get("decision_samples_sha256") and (original or e["round_id"] == latest_round):
                d = decision_samples(reg, e)
                worst = e["sample_episodes"][0]
                d = d[d["episode"] == worst]
                p0 = float(d["price"].iloc[0])
                for x in d.itertuples():
                    decisions.append(dict(
                        eval_id=e["eval_id"], step=int(x.step), price=_r(100 * x.price / p0, 3),
                        direction={1: "long", -1: "short", 0: "flat"}[int(x.direction)],
                        size_chosen=_r(x.gross_selected, 4), leverage=_r(x.leverage_after_fill, 3),
                        position_change=x.position_change.replace("_", " "), equity_before=_r(x.equity_before, 4),
                        equity_after=_r(x.equity_after, 4), reward=_r(x.reward, 6),
                        costs=_r(x.cost_spread + x.cost_slippage + x.cost_commission + x.cost_swap, 5),
                        market_move_pct=_r(100 * x.market_return, 3),
                        expected_move_pct=_r(100 * x.expected_return, 4) if np.isfinite(x.expected_return) else None,
                        rejected=x.rejected or "", died="yes" if x.died else "", death_reason=x.death_reason or ""))
        tp = training_progress(reg, a)
        for x in tp.itertuples() if len(tp) else []:
            train.append(dict(agent_id=a, version=int(x.version), step=int(x.step), growth=_r(x.growth, 4),
                              death_rate=_r(x.died, 3), avg_leverage=_r(x.mean_gross, 3), drawdown=_r(x.drawdown, 3)))
    return dict(agents=agents, versions=versions, checkpoints=ckpts, evals=evals, diagnosis=diag,
                training=train, decisions=decisions)


def write_registry(reg_root: Path, out: Path) -> dict:
    data = registry_rows(reg_root)
    n = {}
    dec = data.pop("decisions")
    data["decisions"] = [r for r in dec if r["eval_id"].endswith("-gate-1")]          # originals: load with the page
    data["decisions_p25"] = [r for r in dec if not r["eval_id"].endswith("-gate-1")]  # later rounds: on demand
    for k, rows in data.items():
        if k in ("training", "decisions", "decisions_p25"):
            path = out / f"registry_{k}.csv"
            if rows:
                write_csv(path, rows)
            else:
                path.write_text({"training": "agent_id,version,step,growth,death_rate,avg_leverage,drawdown",
                                 "decisions": "eval_id,agent_id,version,round_id,episode,step"}[k] + "\n")
        else:
            (out / f"registry_{k}.json").write_text("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")
        n[k] = len(rows)
    return n


def main():
    global RUNS, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--registry", default=str(ROOT / "registry"))
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
    if (Path(a.registry) / "agents").exists():
        print("registry:", write_registry(Path(a.registry), OUT))
    print(f"agents: {len(agents)} ({sum(a['status'] == 'trained' for a in agents)} trained); "
          f"replay rows: {len(rep)}; written to {OUT}")


if __name__ == "__main__":
    main()
