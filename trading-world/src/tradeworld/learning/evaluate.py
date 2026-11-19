"""Immutable evaluation of agent versions on a round's test data.

The pass/fail logic is learning/gate.py exactly as pre-registered (docs/decisions/0002); its
source checksum is pinned in the registry the first time it is used, and evaluation refuses to
run if it ever changes (a change needs a new, approved criteria version).

Episode definitions are identical to scripts/phase2_synthetic.py: 20 fixed-start test episodes
of 1,000 decisions, deterministic actions, episode i reset with seed i.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..interface.actions import ActionMapper
from ..registry.datasets import DataLedger
from ..registry.registry import ROOT, FrozenRuleError, Registry, RegistryError
from ..registry.store import read_json, write_once_json
from . import synthetic_tasks as st
from .env import TradeWorldEnv
from .gate import summarise, verdict
from .history import frame_bytes
from .recorder import DecisionRecorder

CRITERIA_ID = "0002-phase2-gate"
CRITERIA_FILES = ("src/tradeworld/learning/gate.py", "docs/decisions/0002-phase2-gate-criteria.md")
TEST_STEPS_DATA = 60_000          # as the original gate (scripts/phase2_synthetic.py)
TEST_EPISODES = 20
NULL_AGENTS = 10


def criteria_fingerprint() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CRITERIA_FILES}


def pinned_criteria(reg: Registry) -> dict:
    """Pin the criteria on first use; refuse to evaluate if they changed since."""
    path = reg.root / "criteria" / f"{CRITERIA_ID}.json"
    fp = criteria_fingerprint()
    if not path.exists():
        write_once_json(path, dict(criteria_id=CRITERIA_ID, files=fp))
    pinned = read_json(path)
    if pinned["files"] != fp:
        raise FrozenRuleError(f"evaluation criteria {CRITERIA_ID} changed since they were pinned; "
                              f"a criteria change needs a new approved version")
    return dict(criteria_id=CRITERIA_ID, files=fp)


def expected_returns(task) -> np.ndarray:
    """Known expected log return of the bar after each M1 row (synthetic tasks only).
    Used by diagnosis to judge decision quality; never shown to agents."""
    mid = task.md.mid_c()[:, 0]
    r = np.r_[0.0, np.log(mid[1:] / mid[:-1])]
    p = task.params
    if task.name == "S1":
        return p["phi"] * r
    if task.name == "S2":
        return task.extra[:, 0] * p["mu"]
    if task.name in ("S3", "S4", "S5"):
        return np.zeros_like(r)
    if task.name == "S6":
        e = (1 - p["crash_prob"]) * (p["mu"] - 0.5 * p["sigma"] ** 2) + p["crash_prob"] * math.log(1 - p["crash"])
        return np.full_like(r, e)
    if task.name == "S7":
        return np.full_like(r, p["mu"] - 0.5 * p["sigma"] ** 2)
    return np.full_like(r, np.nan)


def mapper_from(identity: dict, n: int = 1) -> ActionMapper:
    a = identity["frozen"]["action_space"]
    return ActionMapper(n, a.get("log10_min", -4.0), a.get("log10_max", 4.0), a.get("cap"),
                        a.get("size_repr", "log10"))


def make_env(identity: dict, task, recorder: bool = True, expected=None):
    f = identity["frozen"]
    env = TradeWorldEnv(task.md, task.profile, reward=f["reward"], episode_decisions=f["episode_decisions"],
                        extra=task.extra, mapper=mapper_from(identity), random_start=False, **f["env_settings"])
    return DecisionRecorder(env, expected) if recorder else env


def test_starts(task, identity) -> np.ndarray:
    w = identity["frozen"]["env_settings"]["window"]
    hi = len(task.md.times) - (w + 1) - identity["frozen"]["episode_decisions"] - 5
    return np.linspace(0, hi, TEST_EPISODES).astype(int)


def run_recorded(policy, identity, task, starts) -> tuple[list[dict], list[dict]]:
    env = make_env(identity, task, expected=expected_returns(task))
    stats = []
    for i, k0 in enumerate(starts):
        o, info = env.reset(seed=i, options={"start_decision": int(k0)})
        e0 = info["equity"]
        first = len(env.rows)
        done = False
        while not done:
            o, r, te, tr, info = env.step(policy(o))
            done = te or tr
        rows = env.rows[first:]
        alive = [x for x in rows if x["alive_after"]]
        net = [x["net_exposure_end"] for x in alive]
        turn, prev = 0.0, 0.0
        for n in net:
            turn += abs(n - prev)
            prev = n
        agree = None
        if task.extra is not None:
            a = [float(np.sign(x["net_exposure_end"]) == np.sign(task.extra[x["decision_row"], 0]))
                 for x in alive if abs(x["net_exposure_end"]) > 1e-6]
            agree = float(np.mean(a)) if a else 0.0
        e1 = info["equity"]
        stats.append(dict(log_growth=math.log(max(e1, 0.01 * e0) / e0), died=not info["alive"], steps=len(rows),
                          mean_gross=float(np.mean([x["gross_exposure_end"] for x in alive])) if alive else 0.0,
                          mean_net=float(np.mean(net)) if net else 0.0, turnover=turn / max(len(rows), 1),
                          costs=float(sum(env.unwrapped.account.costs_total.values())), flag_agreement=agree))
    return env.rows, stats


def model_policy(model):
    return lambda o: model.predict(o, deterministic=True)[0]


def null_distribution(identity, task, starts, stats) -> dict:
    """Leverage- and turnover-matched random agents, exactly as the original gate (S1 only)."""
    from .train import matched_random_policy, run_episodes
    mapper = mapper_from(identity)
    g = float(np.mean([s["mean_gross"] for s in stats]))
    turn = float(np.mean([s["turnover"] for s in stats]))
    switch = min(1.0, turn / max(2.0 * g, 1e-9))
    out = []
    for j in range(NULL_AGENTS):
        ns = run_episodes(matched_random_policy(mapper, g, switch, 50_000 + j), task, starts,
                          identity["frozen"]["episode_decisions"], mapper=mapper)
        out.append(float(np.mean([s.log_growth for s in ns])))
    return dict(mean_log_growth_per_agent=out, matched_gross=g, matched_switch_prob=switch)


# ---------------------------------------------------------------------------- rounds
def create_round(reg: Registry, tasks, note: str = "") -> dict:
    """A new evaluation round: one never-used test seed per market, shared by every agent
    evaluated in this round, so competing versions see identical fresh data."""
    ledger = DataLedger(reg.root)
    n = len([r for r in reg.rounds() if r["round_id"].startswith("round-")]) + 1
    round_id = f"round-{n:04d}"
    data = []
    for t in sorted(set(tasks)):
        seed = ledger.fresh_seed(t, "test")
        ledger.record(t, seed, "test", round_id=round_id)
        data.append(dict(task=t, seed=seed, data_hash=st.build(t, seed, TEST_STEPS_DATA).md.content_hash()))
    return reg.record_round(round_id, dict(kind="fresh", data=data, note=note, test_steps=TEST_STEPS_DATA))


def round_test_seed(rnd: dict, task: str, agent_seed: int | None) -> int:
    for d in rnd["data"]:
        if d["task"] == task and d.get("agent_seed", agent_seed) == agent_seed:
            return d["seed"]
    raise RegistryError(f"round {rnd['round_id']} has no test data for {task} (agent seed {agent_seed})")


def load_model(reg: Registry, sha: str):
    from stable_baselines3 import PPO
    return PPO.load(io.BytesIO(reg.blobs.get(sha)), device="cpu")


def policy_for(reg: Registry, identity: dict, sha: str):
    """The deterministic decision function of an evaluated checkpoint: a PPO policy (every agent without an
    `agent_type`, unchanged) or, for new model-based agents, the learned world model plus its sizing rule."""
    if identity.get("agent_type") == "model-kelly":
        from .model_kelly import load_policy
        return load_policy(reg.blobs, sha, identity)
    return model_policy(load_model(reg, sha))


def evaluate(reg: Registry, agent_id: str, version: int, round_id: str, *, checkpoint_sha: str | None = None,
             reference: dict | None = None, extra_fields: dict | None = None, policy=None) -> dict:
    """Evaluate one version on a round's test data and store the write-once record."""
    from .diagnosis import diagnose
    criteria = pinned_criteria(reg)
    from .metrics import check_q2
    q2_identity = check_q2(reg)
    ident = reg.identity(agent_id)
    ver = reg.version(agent_id, version)
    sha = checkpoint_sha or ver["final_checkpoint"]["model_sha256"]
    rnd = reg.round(round_id)
    seed = round_test_seed(rnd, ident["task"], ident["original_seed"])
    task = st.build(ident["task"], seed, rnd.get("test_steps", TEST_STEPS_DATA))
    starts = test_starts(task, ident)
    rows, stats = run_recorded(policy or policy_for(reg, ident, sha), ident, task, starts)
    run = dict(episodes=stats)
    if ident["task"] == "S1":
        run["null"] = null_distribution(ident, task, starts, stats)
    s = summarise(run)
    if ident["reward"] == "R2" and reference is None:
        ref_id = reg.find(ident["task"], "R1", ident["original_seed"])
        refs = [e for e in reg.evaluations(ref_id) if e["round_id"] == round_id] if ref_id else []
        reference = refs[-1]["summary"] if refs else None
    if ident["reward"] == "R2" and reference is None:
        result, why = "incomplete", "no R1 evaluation in this round to compare against"
    else:
        ok, why = verdict(ident["task"], ident["reward"], s, reference)
        result = "pass" if ok else "fail"
    log = pd.DataFrame(rows)
    stored = store_decision_log(reg, log, stats)
    rec = dict(
        version=version, round_id=round_id, checkpoint_sha256=sha, criteria=criteria, analysis_metrics=dict(Q2=q2_identity),
        test_steps=rnd.get("test_steps", TEST_STEPS_DATA),
        test_data=dict(task=ident["task"], seed=seed, data_hash=task.md.content_hash(),
                       episode_starts=[int(x) for x in starts], episode_reset_seeds=list(range(len(starts)))),
        episodes=stats, null=run.get("null"), summary=s, result=result, reason=why,
        reference_summary=reference, **stored, diagnosis=diagnose(log, task, ident, s),
        **(extra_fields or {}),
    )
    return reg.record_evaluation(agent_id, rec)


def log_fingerprint(log: pd.DataFrame) -> str:
    """Stable content fingerprint of a decision log (independent of the file format)."""
    cols = sorted(log.columns)
    return hashlib.sha256(log[cols].to_csv(index=False, float_format="%.9g").encode()).hexdigest()


def sample_episodes(log: pd.DataFrame, stats: list) -> list[int]:
    """The worst test run and a typical (median-growth) one, kept permanently for replays."""
    if not stats:
        return []
    g = np.array([s["log_growth"] for s in stats])
    order = np.argsort(g)
    return sorted({int(order[0]), int(order[len(order) // 2])})


def store_decision_log(reg: Registry, log: pd.DataFrame, stats: list) -> dict:
    """Full log -> regenerable cache; fingerprint + two sample episodes -> permanent store."""
    eps = sample_episodes(log, stats)
    samples = log[log["episode"].isin(eps)].copy()
    for c in samples.columns:
        if samples[c].dtype == np.float64:
            samples[c] = samples[c].astype(np.float32)
    return dict(decisions=len(log), decision_log_fingerprint=log_fingerprint(log),
                decision_log_cache_sha256=reg.cache.put(frame_bytes(log), "parquet"),
                sample_episodes=eps, decision_samples_sha256=reg.blobs.put(frame_bytes(samples), "parquet"))


def decision_samples(reg: Registry, evaluation: dict) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(reg.blobs.get(evaluation["decision_samples_sha256"])))


def decision_log(reg: Registry, evaluation: dict) -> pd.DataFrame:
    """The full decision log: from the cache, or regenerated from the evaluated checkpoint and
    the recorded test data, and accepted only if it matches the stored fingerprint."""
    sha = evaluation.get("decision_log_cache_sha256")
    if sha and reg.cache.exists(sha):
        return pd.read_parquet(io.BytesIO(reg.cache.get(sha)))
    ident = reg.identity(evaluation["agent_id"])
    td = evaluation["test_data"]
    task = st.build(td["task"], td["seed"], evaluation.get("test_steps", TEST_STEPS_DATA))
    if task.md.content_hash() != td["data_hash"]:
        raise RegistryError("test data no longer regenerates identically; cannot rebuild the decision log")
    rows, _ = run_recorded(policy_for(reg, ident, evaluation["checkpoint_sha256"]), ident, task, td["episode_starts"])
    log = pd.DataFrame(rows)
    if log_fingerprint(log) != evaluation["decision_log_fingerprint"]:
        raise RegistryError("regenerated decision log does not match its fingerprint")
    reg.cache.put(frame_bytes(log), "parquet")
    return log
