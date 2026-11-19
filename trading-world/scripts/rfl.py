"""Experiment R-FL (proposal 0007: full-liability log reward vs the current reward).

  python scripts/rfl.py setup      create the separate experiment registry (once)
  python scripts/rfl.py plan       fingerprint earlier work, allocate fresh paired seeds and data, create the
                                            18 agents, freeze configurations, code and criteria (write-once manifest)
  python scripts/rfl.py all        train, evaluate, report (resumable; refuses if anything frozen changed)
  python scripts/rfl.py report     pre-registered analysis -> reports/phase3/rfl_report.md
  python scripts/rfl.py status
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

from tradeworld.learning import rfl as p3  # noqa: E402
from tradeworld.registry.registry import Registry, blob_root, now_utc, seal, verify_records  # noqa: E402
from tradeworld.registry.store import write_once_json  # noqa: E402

REG = ROOT / "registry-rfl"
EARLIER = {"original registry": ROOT / "registry", "Phase 3 Experiment 1 registry": ROOT / "registry-phase3",
           "Phase 3 Experiment 1b registry": ROOT / "registry-phase3b"}
REPORT = ROOT / "reports" / "phase3" / "rfl_report.md"
STORAGE = dict(path="../../../phase3-rfl-artifacts/blobs", path_env="TRADEWORLD_RFL_BLOBS",
               branch="claude/phase3-rfl-artifacts",
               note="Permanent blobs of this registry live on the separate blobs-only branch above, checked out next to "
                    "this repository (Git LFS is refused for this repository: public fork).")
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01GPvsEBXtZjzV2SPuWVYBnk")
APPROVAL = [
    "2026-11-19: the owner gave full control of the project to pursue its aim (autonomous agents that discover "
    "profitable opportunities, size positions intelligently and learn from experience) without further approvals.",
    "Design as recommended in proposal 0007 section 11: R1 control, R6 (c = 0.01, primary), R6-c0.05 (sensitivity); "
    "six paired seeds; everything except the reward identical to the Experiment 1 control.",
    "Kept from all earlier approvals: R1-R5, agents, versions, checkpoints, registries, evaluations and scoring rules "
    "untouched; R6 is a new, separately named reward for new agents only; no exposure caps, no added penalties, no "
    "trading style; criteria and code frozen by checksum before training; final checkpoints only; every seed reported.",
]


def git(cwd: Path, *a):
    return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)


def commit(message: str) -> None:
    store = blob_root(REG).parent
    for cwd, paths in ((store, ["blobs"]), (ROOT, ["registry-rfl", "reports/phase3"])):
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
                    dict(ledgers=["../../registry/data/ledger.jsonl", "../../registry-phase3/data/ledger.jsonl",
                                  "../../registry-phase3b/data/ledger.jsonl"],
                         note="The original, Experiment 1 and Experiment 1b data ledgers (read only): every seed they "
                              "record counts as used."))
    shutil.copyfile(ROOT / "registry" / "metrics" / "Q2.0.json", REG / "metrics" / "Q2.0.json")
    blob_root(REG).mkdir(parents=True, exist_ok=True)
    (REG / "README.md").write_text(
        "# Experiment R-FL registry\n\nSeparate registry for experiment `phase3-rfl`. Independent agents, versions, "
        "checkpoints, evaluations, rounds and seals. The original, Experiment 1 and Experiment 1b registries are only read. "
        "`metrics/Q2.0.json` is a byte-identical copy of the frozen Q2.0 record.\n\n"
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
    out["Experiment R-FL registry: seal verification"] = "all sealed records intact" if not verify_records(Registry(REG)) else "PROBLEMS"
    return out


def others_running() -> bool:
    me = str(os.getpid())
    out = subprocess.run(["pgrep", "-f", "scripts/rfl.py (all|report)"], capture_output=True, text=True).stdout.split()
    return any(pid != me for pid in out)


def write_report(e) -> None:
    res = e.analyse()
    out = REG / "analyses" / "rfl-verdict" / "verdict.json"
    if not out.exists():
        write_once_json(out, dict(computed_at=now_utc(), verdicts=res["verdicts"], seeds=res["seeds"], g_star=res["g_star"]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(p3.report(e, res, protection()))
    print("written", REPORT, flush=True)


def run_all(e, workers: int) -> None:
    if others_running():
        sys.exit("another R-FL job is running")
    if not e.training_complete():
        e.run_training(workers, on_done=lambda key: commit(f"R-FL: {key} trained"))
        seal(e.reg, note="R-FL: training complete")
        commit("R-FL: training complete (sealed)")
    if not e.evaluations_complete():
        e.run_evaluations(workers)
        seal(e.reg, note="R-FL: evaluations complete")
        commit("R-FL: all final checkpoints evaluated (sealed)")
    write_report(e)
    commit("R-FL: pre-registered analysis and report")
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
