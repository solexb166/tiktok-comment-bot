"""Phase 3 Experiment 1b (C1 and C2 with exploration matched in position-size terms; approved 2026-11-18).

  python scripts/phase3_exp1b.py setup      create the separate experiment registry (once)
  python scripts/phase3_exp1b.py plan       fingerprint earlier work, allocate fresh paired seeds and data, create the
                                            18 agents, freeze configurations, code and criteria (write-once manifest)
  python scripts/phase3_exp1b.py all        train, evaluate, report (resumable; refuses if anything frozen changed)
  python scripts/phase3_exp1b.py report     pre-registered analysis -> reports/phase3/exp1b_report.md
  python scripts/phase3_exp1b.py status
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

from tradeworld.learning import phase3b as p3  # noqa: E402
from tradeworld.registry.registry import Registry, blob_root, now_utc, seal, verify_records  # noqa: E402
from tradeworld.registry.store import write_once_json  # noqa: E402

REG = ROOT / "registry-phase3b"
EARLIER = {"original registry": ROOT / "registry", "Phase 3 Experiment 1 registry": ROOT / "registry-phase3"}
REPORT = ROOT / "reports" / "phase3" / "exp1b_report.md"
STORAGE = dict(path="../../../phase3-exp1b-artifacts/blobs", path_env="TRADEWORLD_PHASE3B_BLOBS",
               branch="claude/phase3-exp1b-artifacts",
               note="Permanent blobs of this registry live on the separate blobs-only branch above, checked out next to "
                    "this repository (Git LFS is refused for this repository: public fork).")
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01GPvsEBXtZjzV2SPuWVYBnk")
APPROVAL = [
    "Three groups approved: C1m (exact match), C2-up (upward step matched at the optimum), C2-tail (ruin chance matched "
    "at the optimum); 18 agents, six independent paired seeds per group.",
    "Pre-registered criteria W1-W7 and E approved; a group supports safe growth if at least 4 of 6 seeds meet every one.",
    "1 Exact configurations preserved and their checksums frozen before training.",
    "2 Actual exploration distributions recorded throughout training, not only their initial settings.",
    "3 Zero ruinous bets alone is not evidence of successful learning.",
    "4 Every seed reported individually, including failures.",
    "5 Safe growth, reduced exploration and safe-but-timid behaviour kept distinct.",
    "6 Numerical safeguards must not act as exposure caps or clip usable size outputs; verification documented.",
    "7 Original agents verified to behave identically after the implementation.",
    "8 No change to frozen rewards, market rules, scoring rules or historical records.",
    "Experiment 2 not started; no combination of experimental changes.",
]


def git(cwd: Path, *a):
    return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)


def commit(message: str) -> None:
    store = blob_root(REG).parent
    for cwd, paths in ((store, ["blobs"]), (ROOT, ["registry-phase3b", "reports/phase3"])):
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
    (REG / "data").mkdir(parents=True)
    (REG / "metrics").mkdir()
    write_once_json(REG / "storage.json", STORAGE)
    write_once_json(REG / "data" / "inherits.json",
                    dict(ledgers=["../../registry/data/ledger.jsonl", "../../registry-phase3/data/ledger.jsonl"],
                         note="The original and Experiment 1 data ledgers (read only): every seed they record counts as used."))
    shutil.copyfile(ROOT / "registry" / "metrics" / "Q2.0.json", REG / "metrics" / "Q2.0.json")
    blob_root(REG).mkdir(parents=True, exist_ok=True)
    (REG / "README.md").write_text(
        "# Phase 3 Experiment 1b registry\n\nSeparate registry for experiment `phase3-exp1b`. Independent agents, versions, "
        "checkpoints, evaluations, rounds and seals. The original registry (`registry/`) and the Experiment 1 registry "
        "(`registry-phase3/`) are only read. `metrics/Q2.0.json` is a byte-identical copy of the frozen Q2.0 record.\n\n"
        f"Blobs: branch `{STORAGE['branch']}`, checked out at `{STORAGE['path']}` relative to this directory "
        f"(or set `{STORAGE['path_env']}`).\n")
    print("experiment registry created at", REG)


def protection() -> dict:
    base = json.loads((REG / "protection_baseline.json").read_text())
    out = {}
    for name, path in EARLIER.items():
        now = tree_hash(path)
        b = base["tree_sha256"][name]
        out[f"{name}: content hash before / now"] = f"{b[:16]}… / {now[:16]}… ({'identical' if now == b else 'CHANGED'})"
        out[f"{name}: seal verification"] = "all sealed records intact" if not verify_records(Registry(path)) else "PROBLEMS"
    out["Experiment 1b registry: seal verification"] = "all sealed records intact" if not verify_records(Registry(REG)) else "PROBLEMS"
    return out


def others_running() -> bool:
    me = str(os.getpid())
    out = subprocess.run(["pgrep", "-f", "phase3_exp1b.py (all|report)"], capture_output=True, text=True).stdout.split()
    return any(pid != me for pid in out)


def write_report(e) -> None:
    res = e.analyse()
    out = REG / "analyses" / "exp1b-verdict" / "verdict.json"
    if not out.exists():
        write_once_json(out, dict(computed_at=now_utc(), verdicts=res["verdicts"], seeds=res["seeds"], g_star=res["g_star"]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(p3.report(e, res, protection()))
    print("written", REPORT, flush=True)


def run_all(e, workers: int) -> None:
    if others_running():
        sys.exit("another Experiment 1b job is running")
    if not e.training_complete():
        e.run_training(workers, on_done=lambda key: commit(f"Phase 3 Exp 1b: {key} trained"))
        seal(e.reg, note="Phase 3 Exp 1b: training complete")
        commit("Phase 3 Exp 1b: training complete (sealed)")
    if not e.evaluations_complete():
        e.run_evaluations(workers)
        seal(e.reg, note="Phase 3 Exp 1b: evaluations complete")
        commit("Phase 3 Exp 1b: all final checkpoints evaluated (sealed)")
    write_report(e)
    commit("Phase 3 Exp 1b: pre-registered analysis and report")
    print("ALL DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "plan", "all", "report", "status"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
        return
    reg = Registry(REG)
    e = p3.Experiment(reg)
    if a.cmd == "plan":
        write_once_json(REG / "protection_baseline.json",
                        dict(at=now_utc(), tree_sha256={n: tree_hash(p) for n, p in EARLIER.items()},
                             seal_problems={n: verify_records(Registry(p)) for n, p in EARLIER.items()}))
        m = e.plan(APPROVAL, storage=dict(STORAGE, checkpoint_every=p3.CHECKPOINT_EVERY, estimate="about 190 MB"),
                   compute=dict(training_steps=18 * p3.STEPS, evaluations=18 * 9, matched_null_sets=36,
                                estimate="about 3.2 h training on 4 cores plus about 45 min of evaluation"))
        seal(reg, note="Phase 3 Exp 1b planned: configurations, code, data, rounds and criteria frozen before training")
        print("planned:", len(m["agents"]), "agents")
        for g, cfg in m["groups"].items():
            print(f"  {g}: config {cfg['config_sha256']}  initial {cfg['initial_settings']}")
    elif a.cmd == "all":
        run_all(e, a.workers)
    elif a.cmd == "report":
        write_report(e)
    elif a.cmd == "status":
        m = e.manifest()
        done = sum(bool(p3.p3a.lineage(reg, x["agent_id"], p3.STEPS)) for x in m["agents"].values())
        print(f"trained {done}/{len(m['agents'])}; evaluations pending {len(e.evaluation_jobs())}; null sets pending {len(e.null_jobs())}")


if __name__ == "__main__":
    main()
