"""Phase 2.5 S6 agent-evolution experiment (pre-registered; see the manifest).

  python scripts/phase25_s6.py plan          write the write-once manifest (once)
  python scripts/phase25_s6.py stage 1       train stage 1 (resumable)
  python scripts/phase25_s6.py round 1       fresh evaluation round 1 (resumable)
  python scripts/phase25_s6.py select        lock candidates (after rounds 1-3, before round 4)
  python scripts/phase25_s6.py report        write reports/phase25/s6_report.md
  python scripts/phase25_s6.py status
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradeworld.learning import phase25 as p25  # noqa: E402
from tradeworld.registry.registry import Registry  # noqa: E402

CONDITIONS = [
    "1 Six S6 agents, arms A and B, three stages of 1M steps; Version 1 preserved; new versions branch from their "
    "original checkpoints with complete parent-child history.",
    "2 Rounds 1-3 for learning curves and candidate selection; candidates recorded and locked before round 4; round 4 "
    "confirmation only; round 5 replication only; no re-selection from rounds 4-5; failures recorded honestly.",
    "3 Exact schedule: 33 versions, 138 evaluations (15, 24, 33, 33, 33).",
    "4 Decision-quality score documented and validated before training (docs/decisions/0004).",
    "5 Checkpoint every 250,000 steps; checkpoint, optimizer, exploration, RNG state and history preserved when "
    "branching.",
    "6 Frozen: S6 mechanics, rewards, scoring rules, simulator, costs, margin, death threshold, action space, pass/fail "
    "criteria; no leverage cap, new penalties, reward or learning-algorithm changes.",
    "7 Arms C and D not approved.",
    "8 Report every agent and version, including failures.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "stage", "round", "select", "report", "status"])
    ap.add_argument("n", nargs="?", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--registry", default=str(ROOT / "registry"))
    a = ap.parse_args()
    e = p25.Experiment(Registry(a.registry))
    if a.cmd == "plan":
        e.plan(CONDITIONS, compute=dict(training_steps=27_000_000, evaluations=138,
                                        estimate="about 2 h 15 min on 4 cores (37 min per stage, about 13 min of evaluations)",
                                        storage="about 80 MB in the repository (checkpoints about 60 MB)"))
        print("manifest written")
    elif a.cmd == "stage":
        e.run_stage(a.n, a.workers)
    elif a.cmd == "round":
        print(e.run_round(a.n, a.workers))
    elif a.cmd == "select":
        for k, c in e.select()["seeds"].items():
            print(k, "->", c["candidate"], "(no version met the size floor)" if c["none_eligible"] else "")
    elif a.cmd == "report":
        out = ROOT / "reports" / "phase25" / "s6_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(p25.report(e))
        print("written", out)
    elif a.cmd == "status":
        print("stages:", [k for k in (1, 2, 3) if e.stage_complete(k)],
              "rounds:", [n for n in range(1, 6) if e.round_complete(n)],
              "candidates:", bool(e.candidates()))


if __name__ == "__main__":
    main()
