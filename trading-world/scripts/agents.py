"""Agent registry command line.

  python scripts/agents.py import-gate                 import finished Phase 2 gate agents as v1
  python scripts/agents.py status                      every agent: status, version, attempts
  python scripts/agents.py round --tasks S6,S7         new evaluation round on fresh test data
  python scripts/agents.py evaluate --agent ag-.. --version 2 --round round-0001
  python scripts/agents.py train --agent ag-.. --parent 1 --steps 1000000 \\
          --set learning_rate=1e-4 --set log_std.exposure=-3 --fresh-data --reason "..."
  python scripts/agents.py recover --agent ag-.. --version 3 --steps 500000
  python scripts/agents.py compare --agent ag-.. --round round-0001
  python scripts/agents.py verify                      checksum every stored blob
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradeworld.registry.registry import Registry, RegistryError  # noqa: E402

GATE_RUNS = ROOT / "runs" / "phase2"
GATE_ROUND = "gate-1"
GATE_TEST_SEED_OFFSET = 10_000
GATE_ENV = dict(decision_interval=1, obs_resolution=1, window=30)
GATE_ACTIONS = dict(log10_min=-4.0, log10_max=4.0, cap=None)


def import_gate(reg: Registry, runs: Path, log=print) -> None:
    """Import the original 27-agent gate as version 1 of each agent, results untouched.
    Decision logs are reconstructed by replaying each checkpoint on its original test data and
    are attached only if the replay reproduces the stored results exactly."""
    import numpy as np
    from tradeworld.learning import synthetic_tasks as st
    from tradeworld.learning.evaluate import (pinned_criteria, load_model, model_policy, run_recorded,
                                              store_decision_log, test_starts)
    from tradeworld.learning.diagnosis import diagnose
    from tradeworld.learning.gate import summarise, verdict
    from tradeworld.registry.datasets import DataLedger
    import pandas as pd

    criteria = pinned_criteria(reg)
    ledger = DataLedger(reg.root)
    files = sorted(runs.glob("*.json"), key=lambda p: (json.loads(p.read_text())["reward"], p.name))   # R1 first
    for p in files:
        run = json.loads(p.read_text())
        task, reward, seed = run["task"], run["reward"], int(run["seed"])
        zip_path = p.with_suffix(".zip")
        if not zip_path.exists():
            log(f"skip {p.name}: no checkpoint")
            continue
        a = reg.find(task, reward, seed)
        if a and reg.versions(a):
            continue
        tr = run["tracking"]
        made = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if a is None:
            a = reg.create_agent(task=task, reward=reward, seed=seed, env_settings=tr["env"], action_space=GATE_ACTIONS,
                                 profile="A_synthetic", episode_decisions=tr["episode_decisions"], origin="phase2-gate-1",
                                 notes="Imported from the first Phase 2 gate run.")
        used = ledger.used(task)
        if used.get(seed) != "train":
            ledger.record(task, seed, "train", note="phase2 gate-1 training data")
        tseed = tr["test_seed"]
        if tseed not in ledger.used(task):
            ledger.record(task, tseed, "test", round_id=GATE_ROUND, note="phase2 gate-1 test data")
        sha = reg.blobs.put_file(zip_path, "zip")
        ckpt = dict(model_sha256=sha, state_sha256=None, step=int(run["train_steps"]), kind="final", legacy=True,
                    note="original gate checkpoint: weights, optimizer, step, hyperparameters; no RNG state")
        reg.start_training(a, 1, None, {})
        reg.record_version(a, dict(
            version=1, parent_version=None, kind="legacy_import", reason="Phase 2 gate, first attempt",
            changes={}, parent_checkpoint=None,
            training=dict(algorithm="PPO", library=dict(sb3=tr["sb3"], torch=tr["torch"], python=tr["python"],
                                                        numpy=tr["numpy"]),
                          steps_added=int(run["train_steps"]), step_start=0, step_end=int(run["train_steps"]),
                          wall_seconds=run.get("wall_seconds"), n_envs=8, seed=seed,
                          hyperparameters=dict(ppo=tr["ppo"], policy=tr["policy"],
                                               u_log_std_init=tr.get("u_log_std_init")),
                          settings_changes={},
                          train_data=[dict(task=task, seed=seed, data_hash=tr["train_data_hash"], steps=200_000)]),
            checkpoints=[ckpt], final_checkpoint=ckpt,
            history=dict(complete=False, episodes=0, updates=0,
                         note="the first gate did not record training history; only final results exist"),
            code_version=dict(git_commit=tr["git_commit"], dirty=None)))
            # reconstruct the decision log on the original test data and verify it
        ident = reg.identity(a)
        test = st.build(task, tseed, 60_000)
        starts = test_starts(test, ident)
        rows, stats = run_recorded(model_policy(load_model(reg, sha)), ident, test, starts)
        verified = all(math.isclose(x["log_growth"], y["log_growth"], rel_tol=0, abs_tol=1e-9)
                       and x["died"] == y["died"] and x["steps"] == y["steps"]
                       for x, y in zip(stats, run["episodes"]))
        s = summarise(run)
        ref = None
        if reward == "R2":
            r1 = reg.find(task, "R1", seed)
            ref = [e for e in reg.evaluations(r1) if e["round_id"] == GATE_ROUND][0]["summary"] if r1 else None
        ok, why = verdict(task, reward, s, ref)
        log_df = pd.DataFrame(rows)
        reg.record_evaluation(a, dict(
            version=1, round_id=GATE_ROUND, legacy=True, checkpoint_sha256=sha, criteria=criteria,
            test_data=dict(task=task, seed=tseed, data_hash=tr["test_data_hash"],
                           episode_starts=[int(x) for x in starts], episode_reset_seeds=list(range(len(starts)))),
            episodes=run["episodes"], null=run.get("null"), summary=s, result="pass" if ok else "fail", reason=why,
            reference_summary=ref, original_result_file=p.name,
            **(store_decision_log(reg, log_df, stats) if verified else {}), test_steps=60_000,
            decision_log_reconstructed=True, decision_log_verified=verified,
            diagnosis=diagnose(log_df, test, ident, s) if verified else None))
        log(f"imported {task} {reward} seed {seed} as {a} v1: {'pass' if ok else 'fail'}"
            f" (decision log {'verified' if verified else 'NOT reproduced; not attached'})")
    gate_data = []
    for a in reg.agents():
        i = reg.identity(a)
        for e in reg.evaluations(a):
            if e["round_id"] == GATE_ROUND:
                gate_data.append(dict(task=i["task"], agent_seed=i["original_seed"], seed=e["test_data"]["seed"],
                                      data_hash=e["test_data"]["data_hash"]))
    gate_path = reg.root / "rounds" / f"{GATE_ROUND}.json"
    if not gate_path.exists() and len(gate_data) == 27:   # write-once, recorded when the gate is complete
        reg.record_round(GATE_ROUND, dict(kind="legacy", note="Phase 2 gate, first attempt: one test seed per "
                                          "agent seed (10,000 + seed)", data=gate_data))


def status(reg: Registry) -> None:
    print(f"{'agent':15} {'name':16} {'status':20} {'ver':>3} {'tries':>5} {'evals':>5} passed-on")
    for a in reg.agents():
        i, s = reg.identity(a), reg.state(a)
        print(f"{a:15} {i['name']:16} {s['status']:20} {s['current_version']:>3} {s['training_attempts']:>5} "
              f"{s['evaluations']:>5} {('v' + str(s['passed_on_version'])) if s['passed_on_version'] else '-'}")


def parse_sets(items) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        if k.startswith("log_std."):
            out.setdefault("log_std", {})[k.split(".", 1)[1]] = float(v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "registry"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("import-gate")
    g.add_argument("--runs", default=str(GATE_RUNS))
    sub.add_parser("status")
    sub.add_parser("verify")
    r = sub.add_parser("round")
    r.add_argument("--tasks", required=True)
    r.add_argument("--note", default="")
    e = sub.add_parser("evaluate")
    e.add_argument("--agent", required=True)
    e.add_argument("--version", type=int, required=True)
    e.add_argument("--round", required=True)
    t = sub.add_parser("train")
    t.add_argument("--agent", required=True)
    t.add_argument("--parent", type=int)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--set", action="append")
    t.add_argument("--fresh-data", action="store_true")
    t.add_argument("--reason", default="")
    t.add_argument("--checkpoint-every", type=int, default=250_000)
    c = sub.add_parser("recover")
    c.add_argument("--agent", required=True)
    c.add_argument("--version", type=int, required=True)
    c.add_argument("--steps", type=int, required=True)
    m = sub.add_parser("compare")
    m.add_argument("--agent", required=True)
    m.add_argument("--round")
    a = ap.parse_args()
    reg = Registry(a.registry)
    if a.cmd == "import-gate":
        import torch
        torch.set_num_threads(1)
        import_gate(reg, Path(a.runs))
    elif a.cmd == "status":
        status(reg)
    elif a.cmd == "verify":
        bad = reg.blobs.verify_all()
        print("all blobs match their checksums" if not bad else f"CORRUPTED: {bad}")
        sys.exit(1 if bad else 0)
    elif a.cmd == "round":
        from tradeworld.learning.evaluate import create_round
        print(json.dumps(create_round(reg, a.tasks.split(","), a.note), indent=1))
    elif a.cmd == "evaluate":
        from tradeworld.learning.evaluate import evaluate
        rec = evaluate(reg, a.agent, a.version, a.round)
        print(rec["eval_id"], rec["result"], rec["reason"])
    elif a.cmd == "train":
        from tradeworld.learning.continual import train_version
        changes = parse_sets(a.set)
        if a.fresh_data:
            changes["train_data_seeds"] = ["fresh"]
        changes["train_steps"] = a.steps
        rec = train_version(reg, a.agent, steps=a.steps, parent=a.parent, changes=changes, reason=a.reason,
                            checkpoint_every=a.checkpoint_every)
        print(f"{a.agent} v{rec['version']} recorded")
    elif a.cmd == "recover":
        from tradeworld.learning.continual import recover
        rec = recover(reg, a.agent, a.version, steps=a.steps)
        print(f"{a.agent} v{rec['version']} recorded (continued from interrupted v{a.version})")
    elif a.cmd == "compare":
        import pandas as pd
        from tradeworld.learning.compare import compare_versions
        pd.set_option("display.width", 200)
        out = compare_versions(reg, a.agent, a.round)
        print("Original results per version:\n", out["original"].to_string(index=False))
        if a.round:
            print(f"\nAll versions on {a.round}:\n", out["same_round"].to_string(index=False))


if __name__ == "__main__":
    main()
