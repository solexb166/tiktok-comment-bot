"""Phase 2 gate: train PPO agents on synthetic markets S1-S7 and evaluate out of sample.

Usage:
    python scripts/phase2_synthetic.py --seeds 0,1,2 --workers 4        (resumable: finished jobs are skipped)
    python scripts/phase2_synthetic.py --jobs S7:R1,S7:R2 --steps 300000 --seeds 0
Each job writes runs/phase2/<task>_<reward>_s<seed>.json (+ model .zip); then run
scripts/phase2_report.py to produce reports/phase2_synthetic.md.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_JOBS = ["S1:R1", "S2:R1", "S3:R1", "S4:R1", "S5:R1", "S6:R1", "S6:R2", "S7:R1", "S7:R2"]
TRAIN_STEPS_DATA = 200_000
TEST_STEPS_DATA = 60_000
EPISODE = 1000
TEST_EPISODES = 20
NULL_AGENTS = 10
TEST_SEED_OFFSET = 10_000
# Training budget per task: the leverage tasks need longer (S7 leverage was still rising at 1M).
STEPS = {"S6": 3_000_000, "S7": 3_000_000}
DEFAULT_STEPS = 1_000_000


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "unknown"


def run_job(args):
    task_name, reward, seed, steps, out_dir, overrides = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import gymnasium
    import stable_baselines3

    from tradeworld.interface.actions import ActionMapper
    from tradeworld.learning import synthetic_tasks as st
    from tradeworld.learning.train import (POLICY_DEFAULTS, PPO_DEFAULTS, U_LOG_STD_INIT, matched_random_policy,
                                           model_policy, run_episodes, train_ppo)

    t0 = time.time()
    train = st.build(task_name, seed, TRAIN_STEPS_DATA)
    test = st.build(task_name, TEST_SEED_OFFSET + seed, TEST_STEPS_DATA)
    ppo_kw = dict(overrides.get("ppo", {}))
    pol_kw = dict(overrides.get("policy", {}))
    u_std = overrides.get("u_log_std_init", U_LOG_STD_INIT)
    model = train_ppo(train, reward, seed, steps, episode_decisions=EPISODE, ppo_kwargs=ppo_kw,
                      policy_kwargs=pol_kw, log=lambda m: print(m, flush=True), u_log_std_init=u_std)
    hi = len(test.md.times) - 31 - EPISODE - 5
    starts = np.linspace(0, hi, TEST_EPISODES).astype(int)
    stats = run_episodes(model_policy(model), test, starts, EPISODE, reward=reward)
    res = dict(
        task=task_name, reward=reward, seed=seed, train_steps=steps,
        episodes=[s.__dict__ for s in stats],
        params=train.params,
        tracking=dict(git_commit=git_commit(), train_data_hash=train.md.content_hash(),
                      test_data_hash=test.md.content_hash(), train_seed=seed,
                      test_seed=TEST_SEED_OFFSET + seed, python=platform.python_version(),
                      torch=torch.__version__, sb3=stable_baselines3.__version__,
                      gymnasium=gymnasium.__version__, numpy=np.__version__,
                      ppo={**PPO_DEFAULTS, **ppo_kw}, policy={**POLICY_DEFAULTS, **pol_kw}, u_log_std_init=u_std,
                      env=train.env_kwargs, episode_decisions=EPISODE, test_episodes=TEST_EPISODES),
    )
    if task_name == "S1":
        mapper = ActionMapper.primary(1)
        g = float(np.mean([s.mean_gross for s in stats]))
        turn = float(np.mean([s.turnover for s in stats]))
        switch = min(1.0, turn / max(2.0 * g, 1e-9))      # a direction flip moves exposure by 2G
        null = []
        for j in range(NULL_AGENTS):
            ns = run_episodes(matched_random_policy(mapper, g, switch, 50_000 + j), test, starts, EPISODE)
            null.append(float(np.mean([s.log_growth for s in ns])))
        res["null"] = dict(mean_log_growth_per_agent=null, matched_gross=g, matched_switch_prob=switch)
    res["wall_seconds"] = time.time() - t0
    out = Path(out_dir) / f"{task_name}_{reward}_s{seed}.json"
    out.write_text(json.dumps(res, indent=1))
    model.save(str(out.with_suffix(".zip")))
    print(f"done {out.name} in {res['wall_seconds']:.0f}s", flush=True)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=",".join(DEFAULT_JOBS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=None, help="override every job's step budget")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "runs" / "phase2"))
    ap.add_argument("--overrides", default="{}", help='JSON, e.g. {"policy": {"log_std_init": -1.0}}')
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    overrides = json.loads(a.overrides)
    jobs = []
    for j in a.jobs.split(","):
        task, reward = j.split(":")
        steps = a.steps or STEPS.get(task, DEFAULT_STEPS)
        for s in a.seeds.split(","):
            if (Path(a.out) / f"{task}_{reward}_s{int(s)}.json").exists():
                print(f"skip {task}:{reward} seed {s} (already done)", flush=True)
                continue
            jobs.append((task, reward, int(s), steps, a.out, overrides))
    jobs.sort(key=lambda x: -x[3])                    # longest first
    print(f"{len(jobs)} jobs, {a.workers} workers, {sum(x[3] for x in jobs):,} steps total", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for path in pool.imap_unordered(run_job, jobs):
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
