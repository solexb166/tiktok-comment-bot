"""Phase 2.5 S6 agent-evolution experiment (pre-registered; see the manifest).

  python scripts/phase25_s6.py plan          write the write-once manifest (once)
  python scripts/phase25_s6.py stage 1       train stage 1 (resumable)
  python scripts/phase25_s6.py round 1       fresh evaluation round 1 (resumable)
  python scripts/phase25_s6.py select        lock candidates (after rounds 1-3, before round 4)
  python scripts/phase25_s6.py report        write reports/phase25/s6_report.md
  python scripts/phase25_s6.py status
"""
import argparse
import os
import subprocess
import sys
import time
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
    "9 Decision-quality score Q2.0 (clamped Kelly) approved by the owner on condition that the mathematics is correct; "
    "independent verification passed 36 of 36 checks (reports/phase25/q2_math_verification.md); frozen for S6.",
    "10 Selection: highest mean Q2.0 over rounds 1-3 among versions with median size >= 0.5x; growth seed 1 must also "
    "be closer to the growth-optimal size than version 1 (rounds 1-3). Locked before round 4.",
    "11 No trading style is prescribed (decision 0005); agents discover their own approach and position sizing.",
]


def commit(message: str) -> None:
    """Commit and push the registry and report after a step (retries around concurrent commits)."""
    trailer = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
               "Claude-Session: https://claude.ai/code/session_01GPvsEBXtZjzV2SPuWVYBnk")
    for attempt in range(6):
        g = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True)
        g("add", "registry", "reports/phase25")
        if g("diff", "--cached", "--quiet").returncode == 0:
            return
        if g("commit", "-q", "-m", message + trailer).returncode == 0:
            for _ in range(4):
                if g("push", "-q", "origin", "HEAD").returncode == 0:
                    return
                g("pull", "-q", "--rebase", "origin", "HEAD")
                time.sleep(5)
            return
        time.sleep(5 * (attempt + 1))                     # e.g. the downloader holds the index lock


def others_running() -> bool:
    me = str(os.getpid())
    out = subprocess.run(["pgrep", "-f", "phase25_s6.py (stage|round)"], capture_output=True, text=True).stdout.split()
    return any(pid != me for pid in out)


def run_all(e, workers: int) -> None:
    """The locked plan, in order. Every step is idempotent: re-running after an interruption
    resumes where it stopped (interrupted training continues from its last checkpoint)."""
    while others_running():
        print("waiting for a running stage or round to finish", flush=True)
        time.sleep(30)
    for k in (1, 2, 3):
        if not e.stage_complete(k):
            e.run_stage(k, workers)
            commit(f"Phase 2.5 S6: stage {k} trained")
        if not e.round_complete(k):
            rid = e.run_round(k, workers)
            commit(f"Phase 2.5 S6: round {k} ({rid}) evaluated")
    if e.candidates() is None:
        for key, c in e.select()["seeds"].items():
            print(key, "->", c["candidate"], flush=True)
        commit("Phase 2.5 S6: candidates selected from rounds 1-3 and locked before round 4")
    for n in (4, 5):
        if not e.round_complete(n):
            rid = e.run_round(n, workers)
            commit(f"Phase 2.5 S6: round {n} ({rid}, {'confirmation' if n == 4 else 'replication'}) evaluated")
    out = ROOT / "reports" / "phase25" / "s6_report.md"
    out.write_text(p25.report(e))
    commit("Phase 2.5 S6: final report")
    print("ALL DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "stage", "round", "select", "report", "status", "all"])
    ap.add_argument("n", nargs="?", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--registry", default=str(ROOT / "registry"))
    a = ap.parse_args()
    e = p25.Experiment(Registry(a.registry))
    if a.cmd == "plan":
        e.plan(CONDITIONS, compute=dict(training_steps=27_000_000, evaluations=138,
                                        estimate="about 2 h 10 min on 4 cores (measured: 12.4 min per 1M-step job with 4 in parallel, 21.5 s per evaluation)",
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
    elif a.cmd == "all":
        run_all(e, a.workers)
    elif a.cmd == "status":
        print("stages:", [k for k in (1, 2, 3) if e.stage_complete(k)],
              "rounds:", [n for n in range(1, 6) if e.round_complete(n)],
              "candidates:", bool(e.candidates()))


if __name__ == "__main__":
    main()
