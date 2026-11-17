"""Phase 3 Experiment 1 (S6 single-factor ablation; proposal 0006, approved 2026-11-17).

  python scripts/phase3_exp1.py setup      create the separate experiment registry (once)
  python scripts/phase3_exp1.py plan       allocate seeds and data, create the 30 agents, write the
                                           write-once manifest with the frozen criteria (once)
  python scripts/phase3_exp1.py train      train every agent to 3M steps (resumable)
  python scripts/phase3_exp1.py evaluate   evaluate every final checkpoint (+ curves, nulls; resumable)
  python scripts/phase3_exp1.py report     pre-registered analysis -> reports/phase3/exp1_report.md
  python scripts/phase3_exp1.py all        train, evaluate, report (resumable)
  python scripts/phase3_exp1.py status
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradeworld.learning import phase3 as p3  # noqa: E402
from tradeworld.registry.registry import Registry, blob_root, now_utc, seal, verify_records  # noqa: E402
from tradeworld.registry.store import write_once_json  # noqa: E402

REG = ROOT / "registry-phase3"
ORIGINAL = ROOT / "registry"
REPORT = ROOT / "reports" / "phase3" / "exp1_report.md"
STORAGE = dict(path="../../../phase3-exp1-artifacts/blobs", path_env="TRADEWORLD_PHASE3_BLOBS",
               branch="claude/phase3-exp1-artifacts",
               note="Permanent blobs of this registry (checkpoints, optimiser/RNG state, training histories, decision "
                    "samples) live on the separate branch above, checked out next to this repository. Git LFS was "
                    "refused by GitHub for this repository (public fork), so a dedicated blobs-only branch is used.")
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01GPvsEBXtZjzV2SPuWVYBnk")

APPROVAL = [
    "1 All existing agents, versions, checkpoints, reward rules and historical records untouched.",
    "2 Completely separate experimental branch (git branch claude/phase3-exp1; separate registry registry-phase3/) "
    "with independent agent identities and version histories.",
    "3 Five groups exactly as proposed: existing-settings control; size representation 1 (C1); size representation 2 "
    "(C2); steadier training updates (D); longer learning horizon (A).",
    "4 Six independent seeds per group.",
    "5 Evaluation criteria frozen before training begins.",
    "6 Every final checkpoint evaluated; no candidate selection from intermediate performance.",
    "7 All checkpoints and failed experiments preserved; separate storage preferred over lower checkpoint frequency.",
    "8 Report statistical uncertainty, effect sizes and individual seed results; no overstated conclusions.",
    "9 Reward function and all frozen rules unchanged.",
    "10 No exposure caps, additional penalties or manually imposed trading strategies.",
    "Experiment 2 (persistent exploration on S3) is separate and must not interfere with Experiment 1.",
    "No reward-function modification is investigated or implemented.",
]
PREDICTIONS = [
    "P-C (from the diagnosis, hypotheses H1/H2): C1 and C2 reduce the ruinous-decision share relative to control.",
    "P-D (H3): D lowers per-update KL and clip fraction and the previous-move tilt relative to control.",
    "P-A (falsification of the horizon explanation): A does not meet G1 and does not reduce the mean ruinous share "
    "by more than half relative to control.",
    "If none of C1, C2 or D meets the criteria, the evidence points at the reward basin (finding F4); any change to the "
    "reward would need a separate proposal and the owner's approval.",
]


def criteria_text() -> list[str]:
    doc = p3.an.__doc__
    return [ln.strip() for ln in doc.splitlines() if ln.strip()]


def git(cwd: Path, *a):
    return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)


def commit(message: str) -> None:
    """Blobs first (storage branch), then records (experiment branch); push both, with retries."""
    store = blob_root(REG).parent
    for cwd, paths in ((store, ["blobs"]), (ROOT, ["registry-phase3", "reports/phase3"])):
        for attempt in range(6):
            git(cwd, "add", *paths)
            if git(cwd, "diff", "--cached", "--quiet").returncode == 0:
                break
            if git(cwd, "commit", "-q", "-m", message + TRAILER).returncode == 0:
                break
            time.sleep(5 * (attempt + 1))
        for i in range(4):
            if git(cwd, "push", "-q", "-u", "origin", "HEAD").returncode == 0:
                break
            time.sleep(2 ** (i + 1))


def tree_hash(path: Path) -> str:
    """SHA-256 over every tracked-kind file under path (cache and lock files excluded)."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        rel = f.relative_to(path)
        if f.is_dir() or rel.parts[0] == "cache" or f.name.endswith(".lock"):
            continue
        h.update(str(rel).encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def setup() -> None:
    if (REG / "storage.json").exists():
        print("already set up")
        return
    REG.mkdir(parents=True)
    (REG / "data").mkdir()
    (REG / "metrics").mkdir()
    write_once_json(REG / "storage.json", STORAGE)
    write_once_json(REG / "data" / "inherits.json",
                    dict(ledgers=["../../registry/data/ledger.jsonl"],
                         note="The original registry's data ledger (read only): every seed it records counts as used here."))
    shutil.copyfile(ORIGINAL / "metrics" / "Q2.0.json", REG / "metrics" / "Q2.0.json")   # the frozen Q2.0 record, byte-identical
    blob_root(REG).mkdir(parents=True, exist_ok=True)
    (REG / "README.md").write_text(
        "# Phase 3 Experiment 1 registry\n\nSeparate registry for experiment `phase3-exp1` (proposal 0006). Its agents, "
        "versions, checkpoints, evaluations, rounds and seals are independent of the original registry (`registry/`), "
        "which this experiment never writes to. `metrics/Q2.0.json` is a byte-identical copy of the frozen Q2.0 record. "
        "`data/inherits.json` makes the original data ledger count as used, so all data here is fresh.\n\n"
        f"Blobs: branch `{STORAGE['branch']}`, checked out at `{STORAGE['path']}` relative to this directory "
        f"(or set `{STORAGE['path_env']}`).\n")
    print("experiment registry created at", REG)


def protection() -> dict:
    base = json.loads((REG / "protection_baseline.json").read_text())
    now = tree_hash(ORIGINAL)
    problems = verify_records(Registry(ORIGINAL))
    return {"Original registry content hash (before / now)": f"{base['registry_tree_sha256'][:16]}… / {now[:16]}… "
            f"({'identical' if now == base['registry_tree_sha256'] else 'CHANGED'})",
            "Original registry seal verification": "all sealed records intact" if not problems else f"{len(problems)} problems",
            "Experiment registry seal verification": ("all sealed records intact" if not verify_records(Registry(REG))
                                                     else "problems found"),
            "Original data ledger": "read only (inherited); no seed reused"}


def others_running() -> bool:
    me = str(os.getpid())
    out = subprocess.run(["pgrep", "-f", "phase3_exp1.py (train|evaluate|all)"], capture_output=True, text=True).stdout.split()
    return any(pid != me for pid in out)


def run_all(e, workers: int) -> None:
    if others_running():
        sys.exit("another Experiment 1 job is running")
    if not e.training_complete():
        e.run_training(workers, on_done=lambda key: commit(f"Phase 3 Exp 1: {key} trained"))
        seal(e.reg, note="Phase 3 Exp 1: training complete")
        commit("Phase 3 Exp 1: training complete (sealed)")
    if not e.evaluations_complete():
        e.run_evaluations(workers)
        seal(e.reg, note="Phase 3 Exp 1: evaluations complete")
        commit("Phase 3 Exp 1: all final checkpoints evaluated (sealed)")
    write_report(e)
    commit("Phase 3 Exp 1: pre-registered analysis and report")
    print("ALL DONE", flush=True)


def write_report(e) -> None:
    res = e.analyse()
    out = REG / "analyses" / "exp1-verdict" / "verdict.json"
    if not out.exists():
        write_once_json(out, dict(computed_at=now_utc(), groups=res["groups"], seeds=res["seeds"],
                                  analysis_sha256=res["analysis_sha256"], g_star=res["g_star"]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(p3.report(e, res, protection()))
    print("written", REPORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "plan", "train", "evaluate", "report", "all", "status"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
        return
    reg = Registry(REG)
    e = p3.Experiment(reg)
    if a.cmd == "plan":
        write_once_json(REG / "protection_baseline.json",
                        dict(at=now_utc(), registry_tree_sha256=tree_hash(ORIGINAL),
                             original_seal_problems=verify_records(Registry(ORIGINAL))))
        storage = dict(STORAGE, checkpoint_every=p3.CHECKPOINT_EVERY,
                       estimate="about 6-8 MB per agent, about 200-330 MB for the experiment")
        m = e.plan(APPROVAL, criteria_text(), PREDICTIONS, storage=storage,
                   compute=dict(training_steps=30 * p3.STEPS, evaluations=30 * 9, matched_null_sets=60,
                                estimate="about 5 h training on 4 cores (12.4 min per 1M steps per job, 4 jobs in "
                                         "parallel) plus about 1.5 h of evaluation"))
        seal(reg, note="Phase 3 Exp 1 planned: agents, data, rounds and criteria frozen before training")
        print(f"planned: {len(m['agents'])} agents; analysis sha256 {m['analysis']['sha256']}")
    elif a.cmd == "train":
        if others_running():
            sys.exit("another Experiment 1 job is running")
        e.run_training(a.workers, on_done=lambda key: commit(f"Phase 3 Exp 1: {key} trained"))
    elif a.cmd == "evaluate":
        if others_running():
            sys.exit("another Experiment 1 job is running")
        e.run_evaluations(a.workers)
    elif a.cmd == "report":
        write_report(e)
    elif a.cmd == "all":
        run_all(e, a.workers)
    elif a.cmd == "status":
        m = e.manifest()
        done = sum(bool(p3.an.lineage(reg, x["agent_id"], p3.STEPS)) for x in m["agents"].values())
        print(f"trained {done}/{len(m['agents'])}; evaluations pending {len(e.evaluation_jobs())}; "
              f"null sets pending {len(e.null_jobs())}")


if __name__ == "__main__":
    main()
