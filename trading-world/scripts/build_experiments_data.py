"""Build the agent monitor's "Learning experiments" datasets from each experiment's sealed verdict file.

Reads only the write-once verdict records (never recomputes anything) and writes:
  reports/dashboard/experiments.json       one row per experimental group: what was tried, the pre-registered verdict
  reports/dashboard/roadmap.json           the stages between here and a demo-account forward test, with status
  reports/dashboard/experiment_seeds.json  one row per agent (seed): size, ruin, deaths, growth, decision quality

Usage: python3 scripts/build_experiments_data.py [--worktrees DIR]   (DIR holds the phase3-* worktrees)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dashboard"

# experiment key, title, verdict path (relative to the worktrees dir), report path on its branch, branch
EXPERIMENTS = [
    ("exp1", "Experiment 1: learning changes for the trial-and-error (PPO) agent",
     "phase3-exp1/trading-world/registry-phase3/analyses/exp1-verdict/verdict.json", "claude/phase3-exp1"),
    ("exp1b", "Experiment 1b: exploration matched in position-size terms",
     "phase3-exp1b/trading-world/registry-phase3b/analyses/exp1b-verdict/verdict.json", "claude/phase3-exp1b"),
    ("rfl", "Experiment R-FL: full-liability reward",
     "phase3-rfl/trading-world/registry-rfl/analyses/rfl-verdict/verdict.json", "claude/phase3-rfl"),
    ("m1", "Experiment M1: model-based growth-optimal agent",
     "phase3-m1/trading-world/registry-m1/analyses/m1-verdict/verdict.json", "claude/phase3-m1"),
    ("m2", "Experiment M2: model-based agent v2 (exact mixture fit, model choice, edge shrinkage)",
     "phase3-m2/trading-world/registry-m2/analyses/m2-verdict/verdict.json", "claude/phase3-m2"),
    ("m3", "Experiment M3: model-based agent v3 with trading costs (calibration, persistence, cost-aware decisions)",
     "phase3-m3/trading-world/registry-m3/analyses/m3-verdict/verdict.json", "claude/phase3-m3"),
]

GROUPS = {
    "exp1": {"control": ("PPO, original settings", "S6", 2.93), "A": ("PPO, change A", "S6", 2.93),
             "C1": ("PPO, change C1", "S6", 2.93), "C2": ("PPO, change C2", "S6", 2.93), "D": ("PPO, change D", "S6", 2.93)},
    "exp1b": {"C1m": ("PPO, C1 matched exploration", "S6", 2.93), "C2-up": ("PPO, C2 upward exploration", "S6", 2.93),
              "C2-tail": ("PPO, C2 tail exploration", "S6", 2.93)},
    "rfl": {"R1": ("PPO, original reward (control)", "S6", 2.93), "R6": ("PPO, full-liability reward", "S6", 2.93),
            "R6-c0.05": ("PPO, full-liability reward (c = 0.05)", "S6", 2.93)},
    "m1": {"MK-S6": ("Model-based agent", "S6", 2.93), "MK-S7": ("Model-based agent", "S7", 5.0)},
    "m2": {"MK2-S6": ("Model-based agent v2", "S6", 2.93), "MK2-S7": ("Model-based agent v2", "S7", 5.0),
           "MK2-S4": ("Model-based agent v2", "S4", 0.0)},
    "m3": {"MK3-S1C": ("Model-based agent v3", "S1C", None), "MK3-S2C": ("Model-based agent v3", "S2C", None),
           "MK3-S3": ("Model-based agent v3", "S3", 0.0), "MK3-S6": ("Model-based agent v3", "S6", 2.93)},
}
MARKETS = {"S4": "No edge (random walk)", "S6": "Drift plus rare crashes", "S7": "Drift, no crashes",
           "S1C": "Bounce-back pattern, high trading costs", "S2C": "Announced trends, high trading costs",
           "S3": "Pure noise with trading costs"}


def num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, 6)


def group_rows(key: str, d: dict) -> list[dict]:
    out = []
    for g, (agent, market, gstar) in GROUPS[key].items():
        if key == "exp1":
            v = d["groups"][g]
            successes, verdict = None, ("genuine improvement" if v.get("genuine_improvement") else
                                        "no genuine improvement")
            q2, ruin, size = v.get("q2_mean"), v.get("ruinous_mean"), v.get("median_size_mean")
        else:
            v = d["verdicts"]["groups"][g] if "groups" in d["verdicts"] else d["verdicts"][g]
            successes = v.get("successes")
            if "interpretation" in v:
                verdict = v["interpretation"]
            elif "passes" in v:
                verdict = "passes" if v["passes"] else "does not pass"
            else:
                verdict = "safe growth" if v.get("supports_safe_growth") else "does not pass"
            q2, ruin, size = v.get("q2_mean"), v.get("ruinous_mean"), v.get("median_size_mean")
        passes = (successes is not None and successes >= 4)
        out.append(dict(experiment=key, group=g, agent=agent, market=market, market_name=MARKETS[market],
                        optimal_size=gstar, n=v.get("n", 6), successes=successes, passes=passes, verdict=verdict,
                        q2_mean=num(q2), ruinous_pct=None if ruin is None else num(100 * ruin), median_size=num(size),
                        deaths_total=v.get("deaths_total"), growth_mean=num(v.get("growth_mean")),
                        counts=v.get("counts")))
    return out


def seed_rows(key: str, d: dict) -> list[dict]:
    out = []
    for g, (agent, market, gstar) in GROUPS[key].items():
        seeds = sorted(d["seeds"][g], key=lambda s: s["seed_index"])
        if key == "exp1":
            flags = [None] * len(seeds)
        else:
            v = d["verdicts"]["groups"][g] if "groups" in d["verdicts"] else d["verdicts"][g]
            flags = v.get("flags") or [None] * len(seeds)
        for s, f in zip(seeds, flags):
            ci = s.get("growth_ci") or [None, None]
            out.append(dict(experiment=key, group=g, agent=agent, market=market, seed_index=s["seed_index"],
                            agent_id=s.get("agent_id"), success=None if f is None else bool(f["success"]),
                            failed=None if f is None else [k for k, ok in f.items() if k != "success" and not ok],
                            median_size=num(s.get("median_signed", s.get("median_size"))), optimal_size=gstar,
                            max_size=num(s.get("max_size")),
                            ruinous_pct=None if s.get("ruinous") is None else num(100 * s["ruinous"]),
                            deaths=s.get("deaths"), runs=s.get("runs"), growth=num(s.get("growth")),
                            growth_lo=num(ci[0]), growth_hi=num(ci[1]), q2=num(s.get("q2"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktrees", default=str(ROOT.parent))
    a = ap.parse_args()
    base = Path(a.worktrees)
    groups, seeds, status = [], [], []
    for key, title, rel, branch in EXPERIMENTS:
        p = base / rel
        if not p.exists():
            status.append(dict(experiment=key, title=title, branch=branch, complete=False, computed_at=None))
            continue
        d = json.loads(p.read_text())
        status.append(dict(experiment=key, title=title, branch=branch, complete=True, computed_at=d.get("computed_at")))
        groups += group_rows(key, d)
        seeds += seed_rows(key, d)
    for g in groups:
        g["title"] = next(s["title"] for s in status if s["experiment"] == g["experiment"])
    OUT.mkdir(parents=True, exist_ok=True)
    m2 = {g["group"]: g for g in groups if g["experiment"] == "m2"}
    st = lambda g: ("pending" if g not in m2 else "passed" if m2[g]["passes"] else "failed")
    dl = base / "trading-world" / "reports" / "dashboard" / "downloads.csv"
    dl_done = dl_total = 0
    if dl.exists():
        rows = dl.read_text().splitlines()[1:]
        dl_total, dl_done = len(rows), sum(r.rstrip().endswith(",yes") for r in rows)
    sd = {g: [x for x in seeds if x["group"] == g] for g in m2}
    ev2 = ev3 = ev4 = "Experiment M2 (not finished)"
    if m2:
        g6, g7 = m2["MK2-S6"], m2["MK2-S7"]
        m2v = json.loads((base / next(e[2] for e in EXPERIMENTS if e[0] == "m2")).read_text())
        bench7 = m2v["benchmarks"]["S7"]["deaths"]
        holders = [x for x in sd["MK2-S4"] if abs(x["median_size"] or 0) > 0.05]
        ev2 = (f"M2: {g6['successes']}/6 agents succeeded; typical size {g6['median_size']:.2f}x against the best "
               f"{g6['optimal_size']:.2f}x; {g6['deaths_total']} deaths in 720 test runs. Earlier trial-and-error agents "
               "(Experiments 1, 1b, R-FL) all failed this stage.")
        ev3 = (f"M2: {g7['successes']}/6 agents succeeded (4 needed); typical size {g7['median_size']:.2f}x against the best 5x. "
               f"{g7['deaths_total']} deaths in 720 test runs (holding the exact best size on the same runs: {bench7} in 120). "
               "The other two agents missed only the rule that size must end closer to 5x than it started.")
        ev4 = (f"M2: 6/6 claimed no real edge, as required. Caveat: {len(holders)} of 6 still held positions "
               f"({', '.join(f'{abs(x['median_size']):.2f}x' for x in holders)}) on chance drift. Harmless without trading "
               "costs, costly with them; the next stage tests exactly this.")
    m3 = {g["group"]: g for g in groups if g["experiment"] == "m3"}
    if not m3:
        st5, ev5 = ("running" if (base / "phase3-m3/trading-world/registry-m3/experiments/phase3-m3/manifest.json").exists()
                    else "not started"), "Experiment M3 (running): 24 cost-aware agents on S1C, S2C, S3 and an S6 regression check"
    else:
        ok = all(m3[g]["passes"] for g in ("MK3-S1C", "MK3-S2C", "MK3-S3"))
        st5 = "passed" if ok else "failed"
        v3 = json.loads((base / next(e[2] for e in EXPERIMENTS if e[0] == "m3")).read_text())["verdicts"]
        a1, a2 = v3["MK3-S1C"].get("ablation", {}), v3["MK3-S2C"].get("ablation", {})
        r2 = v3["MK3-S2C"]
        ev5 = ("M3: " + "; ".join(f"{m3[g]['market']} {m3[g]['successes']}/6 agents succeeded" for g in ("MK3-S1C", "MK3-S2C", "MK3-S3", "MK3-S6"))
               + f". Weighing costs is what made the difference: the same learned models trading as if costs were free earned "
               f"{a1.get('growth_mean', float('nan')):+.1f} per run in S1C ({a1.get('deaths_total', '?')} deaths in 720 runs) against "
               f"{v3['MK3-S1C']['growth_mean']:+.1f} with costs weighed. Caveat: in S2C the agents earned "
               f"{100 * r2['growth_mean'] / r2['reference']['growth']:.0f}% of the true-parameter reference but died more often "
               f"({r2['deaths_total']} in 720 runs, against the reference's {r2['reference']['deaths']} in 120).")
    roadmap = [
        dict(order=1, stage="Learning machinery and practice markets with known right answers", status="passed",
             evidence="Phase 2 gate, agent registry, decision replays (all below)"),
        dict(order=2, stage="Size positions safely in a market with rare crashes (S6)", status=st("MK2-S6"),
             evidence=ev2),
        dict(order=3, stage="Find the known best position size (S7)", status=st("MK2-S7"), evidence=ev3),
        dict(order=4, stage="Claim no edge where none exists (S4)", status=st("MK2-S4"), evidence=ev4),
        dict(order=5, stage="Profit from predictable patterns after trading costs (S1C, S2C, S3)", status=st5, evidence=ev5),
        dict(order=6, stage="Adapt when the market changes", status="not started", evidence=""),
        dict(order=7, stage="Real historical prices: train, then test once on sealed data", status="not started",
             evidence=f"Real data downloaded: {dl_done} of {dl_total} instrument-months"),
        dict(order=8, stage="Forward test on a demo account (simulated money, real prices)", status="not started",
             evidence="Only after stage 7 passes; the owner connects any account"),
        dict(order=9, stage="Readiness report", status="not started", evidence=""),
    ]
    (OUT / "roadmap.json").write_text(json.dumps(roadmap, indent=1))
    (OUT / "experiments.json").write_text(json.dumps(groups, indent=1))
    (OUT / "experiment_seeds.json").write_text(json.dumps(seeds, indent=1))
    print(f"{len(status)} experiments ({sum(s['complete'] for s in status)} complete), {len(groups)} groups, {len(seeds)} agents")


if __name__ == "__main__":
    main()
