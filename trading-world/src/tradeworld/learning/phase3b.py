"""Phase 3 Experiment 1b (approved 2026-11-18): C1 and C2 with exploration matched to the control
in actual position-size terms.

Groups (six paired seeds each, 3M steps, S6, reward R1, all other settings as Experiment 1):
  C1m      size = e^z;            starting size-noise sigma = 0.756 (exact match: x2.13 per sd at every size)
  C2-up    size = softplus(z);    sigma chosen so that +1 sd at the optimum is x2.13 (as the control)
  C2-tail  size = softplus(z);    sigma chosen so that the chance of exploring to a ruinous size
                                  (>= 9.5x) from the optimum equals the control's (6.0%)
Only the STARTING noise is set; PPO keeps learning it. The exploration actually applied is recorded
at every training update (history.exploration_stats).

Separate registry (registry-phase3b/), separate blob store (storage.json), data ledger inheriting the
original and Experiment 1 ledgers (read only). The original registry and the Experiment 1 registry
are only read.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing as mp
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ..registry.datasets import DataLedger
from ..registry.registry import ROOT, Registry, RegistryError, now_utc
from ..registry.store import append_jsonl, canonical_json, read_json, read_jsonl, write_once_json
from . import phase3_analysis as p3a
from . import phase3b_analysis as an

EXPERIMENT_ID = "phase3-exp1b"
TASK, REWARD, PROFILE = "S6", "R1", "A_synthetic"
ENV_SETTINGS = dict(decision_interval=1, obs_resolution=1, window=30)
EPISODE_DECISIONS = 1000
STEPS = 3_000_000
CHECKPOINT_EVERY = 250_000
CURVE_EVERY = 500_000
N_ENVS = 8
SEEDS_PER_GROUP = 6
TRAIN_DATA_STEPS = 200_000
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None)
CONTROL_U_LOG_STD = -2.5
ROUND_A, ROUND_B, ROUND_TRAIN, ROUND_DEV = "exp1b-final-a", "exp1b-final-b", "exp1b-trainpaths", "exp1b-dev"
EXP1_VERDICT = "registry-phase3/analyses/exp1-verdict/verdict.json"
FROZEN_CODE = (
    "src/tradeworld/interface/actions.py", "src/tradeworld/learning/env.py", "src/tradeworld/learning/continual.py",
    "src/tradeworld/learning/history.py", "src/tradeworld/learning/train.py", "src/tradeworld/learning/evaluate.py",
    "src/tradeworld/learning/diagnosis.py", "src/tradeworld/learning/quality.py", "src/tradeworld/learning/rewards.py",
    "src/tradeworld/learning/synthetic_tasks.py", "src/tradeworld/learning/checkpoint.py",
    "src/tradeworld/learning/phase3_analysis.py", "src/tradeworld/learning/phase3b_analysis.py",
    "src/tradeworld/learning/phase3b.py", "src/tradeworld/registry/registry.py", "src/tradeworld/registry/datasets.py",
)


def _softplus(z):
    return np.logaddexp(0.0, z)


def _softplus_inv(G: float) -> float:
    return float(G + math.log(-math.expm1(-G)))


def calibration() -> dict:
    """Starting size-noise for each group, derived from the control's exploration (deterministic)."""
    from . import synthetic_tasks as st
    from .quality import KnownMarket
    mk = KnownMarket.from_task(st.build(TASK, 0, 1000))
    g_star, ruin = mk.optimal()[0], mk.ruin_long()
    L = 4 * math.log(10) * math.exp(CONTROL_U_LOG_STD)          # control: sd of ln(size) per sd of noise
    p_ruin = 1 - NormalDist().cdf(math.log(ruin / g_star) / L)  # control: chance of exploring to ruin from G*
    zs = _softplus_inv(g_star)
    sig_up = _softplus_inv(g_star * math.exp(L)) - zs
    sig_tail = (_softplus_inv(ruin) - zs) / NormalDist().inv_cdf(1 - p_ruin)
    return dict(g_star=g_star, ruin=ruin, control_lnsize_sd=L, control_multiplier_per_sd=math.exp(L),
                control_ruin_probability_from_optimum=p_ruin,
                sigma=dict(C1m=L, **{"C2-up": sig_up, "C2-tail": sig_tail}),
                size_log_std={"C1m": math.log(L), "C2-up": math.log(sig_up), "C2-tail": math.log(sig_tail)})


def groups() -> dict:
    c = calibration()
    return {
        "C1m": dict(action_space=dict(LOG10, size_repr="exp"), initial_settings=dict(size_log_std=c["size_log_std"]["C1m"]),
                    description="size = e^z, z in [-25, 25]; starting size-noise log std %.4f (sigma %.4f): x%.2f per sd at "
                                "every size, as the control" % (c["size_log_std"]["C1m"], c["sigma"]["C1m"], c["control_multiplier_per_sd"])),
        "C2-up": dict(action_space=dict(LOG10, size_repr="softplus"), initial_settings=dict(size_log_std=c["size_log_std"]["C2-up"]),
                      description="size = ln(1 + e^z), z in [-25, 10,000]; starting size-noise log std %.4f (sigma %.4f): "
                                  "+1 sd at the optimum = x%.2f, as the control" % (c["size_log_std"]["C2-up"], c["sigma"]["C2-up"], c["control_multiplier_per_sd"])),
        "C2-tail": dict(action_space=dict(LOG10, size_repr="softplus"), initial_settings=dict(size_log_std=c["size_log_std"]["C2-tail"]),
                        description="size = ln(1 + e^z), z in [-25, 10,000]; starting size-noise log std %.4f (sigma %.4f): "
                                    "chance of exploring to >= 9.5x from the optimum = %.1f%%, as the control"
                                    % (c["size_log_std"]["C2-tail"], c["sigma"]["C2-tail"], 100 * c["control_ruin_probability_from_optimum"])),
    }


def exploration_design(n: int = 1_000_000) -> dict:
    """Documented before training: starting exploration in actual size terms, by starting size."""
    c = calibration()
    eps = np.random.default_rng(0).standard_normal(n)
    reps = {"control": (np.exp, math.log, c["control_lnsize_sd"]), "C1m": (np.exp, math.log, c["sigma"]["C1m"]),
            "C2-up": (_softplus, _softplus_inv, c["sigma"]["C2-up"]), "C2-tail": (_softplus, _softplus_inv, c["sigma"]["C2-tail"]),
            "Exp1 C1": (np.exp, math.log, math.exp(CONTROL_U_LOG_STD)), "Exp1 C2": (_softplus, _softplus_inv, math.exp(CONTROL_U_LOG_STD))}
    table = {}
    for name, (f, inv, sig) in reps.items():
        rows = []
        for G0 in (0.1, 0.5, 0.6931, 1.0, 1.42, round(c["g_star"], 4), 5.0, c["ruin"], 20.0):
            G = f(inv(G0) + sig * eps)
            r = np.log(G / G0)
            q = lambda p: float(math.exp(np.percentile(r, p)))
            rows.append(dict(start=G0, lnsize_sd=float(np.std(r)), plus1sd=q(84.13), minus1sd=q(15.87), plus2sd=q(97.72),
                             minus2sd=q(2.28), median=float(np.median(G)), p1=float(np.percentile(G, 1)),
                             p99=float(np.percentile(G, 99)), p_ge_optimum=float(np.mean(G >= c["g_star"])),
                             p_ge_ruin=float(np.mean(G >= c["ruin"]))))
        table[name] = rows
    return dict(calibration=c, by_start_size=table, method="1,000,000 standard-normal draws (seed 0) on the size output "
                "around each starting size; size ratios are sampled / starting size. Control: u-noise exp(-2.5) on "
                "size = 10^(4u). Experiment 1 C1/C2 (unmatched) shown for reference.")


def code_checksums() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN_CODE}


def config_checksum(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg)).hexdigest()


def curve_round(step: int) -> str:
    return f"exp1b-curve-{step // 1000:04d}k"


class Experiment:
    def __init__(self, reg: Registry):
        self.reg = reg
        self.dir = reg.root / "experiments" / EXPERIMENT_ID
        self.progress_path = self.dir / "progress.jsonl"

    def manifest(self) -> dict:
        p = self.dir / "manifest.json"
        if not p.exists():
            raise RegistryError("experiment not planned yet")
        return read_json(p)

    def _log(self, kind: str, **data) -> None:
        append_jsonl(self.progress_path, dict(kind=kind, at=now_utc(), **data))

    def check_frozen(self) -> None:
        """Refuse to train, evaluate or analyse if any frozen configuration or code changed."""
        m = self.manifest()
        if code_checksums() != m["frozen_code_sha256"]:
            bad = [f for f, h in code_checksums().items() if m["frozen_code_sha256"].get(f) != h]
            raise RegistryError(f"frozen code changed after pre-registration: {bad}")
        for g, cfg in groups().items():
            if config_checksum(cfg) != m["groups"][g]["config_sha256"]:
                raise RegistryError(f"configuration of group {g} changed after pre-registration")

    # ------------------------------------------------------------------ plan
    def plan(self, approval: list[str], compute: dict, storage: dict) -> dict:
        from . import synthetic_tasks as st
        from .evaluate import TEST_STEPS_DATA
        from .metrics import Q2_DEFINITION, check_q2
        if (self.dir / "manifest.json").exists():
            raise RegistryError("already planned")
        q2 = check_q2(self.reg, require_frozen=True)
        ledger = DataLedger(self.reg.root)
        if len(ledger.inherited()) < 2:
            raise RegistryError("the ledger must inherit the original and the Experiment 1 ledgers")
        ref_path = ROOT / EXP1_VERDICT
        G = groups()
        agents = {}
        for k in range(SEEDS_PER_GROUP):
            seed = ledger.allocate(TASK, "train", note=f"{EXPERIMENT_ID} seed index {k + 1}: training data and model "
                                                       f"seed shared by C1m, C2-up and C2-tail (paired)")
            for g, cfg in G.items():
                aid = self.reg.create_agent(
                    task=TASK, reward=REWARD, seed=seed, env_settings=ENV_SETTINGS, action_space=cfg["action_space"],
                    profile=PROFILE, episode_decisions=EPISODE_DECISIONS, name=f"P3E1b-{g}-s{k + 1}", origin=EXPERIMENT_ID,
                    notes=f"Phase 3 Experiment 1b, group {g}: {cfg['description']}",
                    extra=dict(experiment=EXPERIMENT_ID, group=g, seed_index=k + 1, initial_settings=cfg["initial_settings"],
                               config_sha256=config_checksum(cfg)))
                agents[f"{g}-s{k + 1}"] = dict(agent_id=aid, group=g, seed_index=k + 1, seed=seed)
        rounds = {}
        for rid, purpose in ((ROUND_A, "final evaluation, round A"), (ROUND_B, "final evaluation, round B"),
                             (ROUND_DEV, "learning-curve development data (descriptive only)")):
            seed = ledger.allocate(TASK, "test", round_id=rid, note=f"{EXPERIMENT_ID} {purpose}")
            data = [dict(task=TASK, seed=seed, data_hash=st.build(TASK, seed, TEST_STEPS_DATA).md.content_hash())]
            if rid == ROUND_DEV:
                for step in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY):
                    self.reg.record_round(curve_round(step), dict(kind="fresh", data=data, test_steps=TEST_STEPS_DATA,
                                                                  note=f"{EXPERIMENT_ID} learning curve at {step:,} steps"))
                rounds["curve"] = {curve_round(s): seed for s in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY)}
            else:
                self.reg.record_round(rid, dict(kind="fresh", data=data, test_steps=TEST_STEPS_DATA, note=f"{EXPERIMENT_ID} {purpose}"))
                rounds[rid] = seed
        seeds = sorted({a["seed"] for a in agents.values()})
        train_data = [dict(task=TASK, seed=s, agent_seed=s, data_hash=st.build(TASK, s, TRAIN_DATA_STEPS).md.content_hash())
                      for s in seeds]
        self.reg.record_round(ROUND_TRAIN, dict(kind="training-data", data=train_data, test_steps=TRAIN_DATA_STEPS,
                                                note=f"{EXPERIMENT_ID}: each agent's own training paths (check E); not fresh"))
        rounds[ROUND_TRAIN] = "each agent's own training seed"
        m = dict(
            experiment_id=EXPERIMENT_ID, planned_at=now_utc(), market=TASK, reward=REWARD, profile=PROFILE,
            env_settings=ENV_SETTINGS, episode_decisions=EPISODE_DECISIONS, steps=STEPS, checkpoint_every=CHECKPOINT_EVERY,
            curve_every=CURVE_EVERY, n_envs=N_ENVS, seeds_per_group=SEEDS_PER_GROUP, paired_seeds=True,
            groups={g: dict(cfg, config_sha256=config_checksum(cfg)) for g, cfg in G.items()},
            exploration_design=exploration_design(), agents=agents, rounds=rounds,
            evaluated_checkpoint="each agent's final checkpoint (no selection)",
            decision_quality=dict(q2, definition=Q2_DEFINITION),
            analysis=dict(file="src/tradeworld/learning/phase3b_analysis.py", criteria=[ln.strip() for ln in an.__doc__.splitlines() if ln.strip()]),
            frozen_code_sha256=code_checksums(),
            references=dict(file=EXP1_VERDICT, sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                            note="Experiment 1 per-seed final results (control, C1, C2), read only; different evaluation "
                                 "data, Q2 depends only on the decisions"),
            safeguard_verification=(
                "Numerical size ranges: C1m z in [-25, 25] (sizes 1.4e-11x to 7.2e10x); C2 z in [-25, 10,000] (up to 10,000x, "
                "the same largest size as the existing representation). The margin rule (1:2000) is far below both. Verified "
                "by: (1) the share of sampled training actions outside the range (explore/size_clipped_share) and of mean "
                "actions within 5% of either end (explore/size_mean_near_bound_share), recorded at every update; (2) the share "
                "of evaluated decisions with the raw size output at its bound (criterion E, limit 0.1%); (3) tests that every "
                "size from 1e-4x to 10,000x is reachable and unclipped. Any non-zero training clipping is reported per agent."),
            approval_conditions=approval, compute=compute, storage=storage,
            frozen=("S6 market mechanics, reward R1, death at 5%, margin 1:2000, simulator, costs, Q2.0 score, open trading "
                    "principle; no exposure cap, no new penalty, no trading style; not combined with any other change."),
        )
        write_once_json(self.dir / "manifest.json", m)
        self._log("planned")
        return m

    # ------------------------------------------------------------------ training
    def training_job(self, key: str) -> dict | None:
        a = self.manifest()["agents"][key]
        aid = a["agent_id"]
        if p3a.lineage(self.reg, aid, STEPS):
            return None
        ev = self.reg.events(aid)
        closed = {e["version"] for e in ev if e["kind"] in ("training_finished", "training_abandoned")}
        open_ = [e["version"] for e in ev if e["kind"] == "training_started" and e["version"] not in closed]
        vs = self.reg.versions(aid)
        parents = {v.get("parent_version") for v in vs}
        stopped = [v["version"] for v in vs if v["kind"] == "interrupted" and v["version"] not in parents]
        return dict(key=key, agent_id=aid, group=a["group"], seed=a["seed"], recover=max(open_ + stopped, default=None),
                    initial_settings=self.manifest()["groups"][a["group"]]["initial_settings"])

    def run_training(self, workers: int = 4, on_done=None, log=print) -> None:
        self.check_frozen()
        keys = sorted(self.manifest()["agents"], key=lambda k: (int(k.rsplit("-s", 1)[1]), k))   # seed index first
        jobs = [j for j in (self.training_job(k) for k in keys) if j]
        log(f"training: {len(jobs)} agents to train")
        with mp.get_context("fork").Pool(workers) as pool:
            for job, rec in pool.imap_unordered(_train_job, [(str(self.reg.root), j) for j in jobs]):
                self._log("trained", agent_key=job["key"], version=rec["version"], step=rec["final_checkpoint"]["step"])
                log(f"  {job['key']} -> v{rec['version']} ({rec['training']['wall_seconds']:.0f}s)")
                if on_done:
                    on_done(job["key"])

    def training_complete(self) -> bool:
        return all(p3a.lineage(self.reg, a["agent_id"], STEPS) for a in self.manifest()["agents"].values())

    # ------------------------------------------------------------------ evaluation
    def evaluation_jobs(self) -> list[dict]:
        jobs = []
        for key, a in self.manifest()["agents"].items():
            chain = p3a.lineage(self.reg, a["agent_id"], STEPS)
            if not chain:
                continue
            final_v = chain[-1]["version"]
            done = {(e["round_id"], e["version"]) for e in self.reg.evaluations(a["agent_id"])}
            items = [(ROUND_A, final_v, None, "final"), (ROUND_B, final_v, None, "final"), (ROUND_TRAIN, final_v, None, "trainpaths")]
            for step in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY):
                v, c = p3a.checkpoint_at(chain, step)
                items.append((curve_round(step), v, c["model_sha256"], f"curve {step}"))
            jobs += [dict(key=key, agent_id=a["agent_id"], group=a["group"], version=v, round_id=rid, checkpoint=sha, purpose=pu)
                     for rid, v, sha, pu in items if (rid, v) not in done]
        return jobs

    def _null_path(self, e: dict) -> Path:
        return self.reg.root / "analyses" / "matched-null" / f"{e['eval_id']}.json"

    def null_jobs(self) -> list[dict]:
        return [dict(key=key, agent_id=a["agent_id"], eval_id=e["eval_id"])
                for key, a in self.manifest()["agents"].items() for e in self.reg.evaluations(a["agent_id"])
                if e["round_id"] in (ROUND_A, ROUND_B) and not self._null_path(e).exists()]

    def run_evaluations(self, workers: int = 4, log=print) -> None:
        from .metrics import check_q2
        self.check_frozen()
        if check_q2(self.reg, require_frozen=True)["checksum"] != self.manifest()["decision_quality"]["checksum"]:
            raise RegistryError("the decision-quality score differs from the one in the experiment plan")
        jobs = self.evaluation_jobs()
        log(f"evaluations: {len(jobs)}")
        with mp.get_context("fork").Pool(workers) as pool:
            for job, rec in pool.imap_unordered(_eval_job, [(str(self.reg.root), j) for j in jobs]):
                self._log("evaluated", agent_key=job["key"], round_id=job["round_id"], version=job["version"],
                          eval_id=rec["eval_id"], purpose=job["purpose"])
            jobs = self.null_jobs()
            log(f"matched random agents: {len(jobs)} evaluation sets")
            for job in pool.imap_unordered(_null_job, [(str(self.reg.root), j) for j in jobs]):
                self._log("null_computed", agent_key=job["key"], eval_id=job["eval_id"])

    def evaluations_complete(self) -> bool:
        return self.training_complete() and not self.evaluation_jobs() and not self.null_jobs()

    # ------------------------------------------------------------------ analysis
    def analyse(self) -> dict:
        import pandas as pd
        from . import synthetic_tasks as st
        from .evaluate import decision_log, mapper_from
        from .quality import KnownMarket
        self.check_frozen()
        m = self.manifest()
        ref_path = ROOT / m["references"]["file"]
        if hashlib.sha256(ref_path.read_bytes()).hexdigest() != m["references"]["sha256"]:
            raise RegistryError("the Experiment 1 reference record changed")
        market = KnownMarket.from_task(st.build(TASK, 0, 1000))
        g_star = market.optimal()[0]
        seeds, curves, explore = {g: [] for g in m["groups"]}, {}, {}
        for key, a in m["agents"].items():
            aid = a["agent_id"]
            chain = p3a.lineage(self.reg, aid, STEPS)
            final_v = chain[-1]["version"]
            evals = self.reg.evaluations(aid)
            finals = sorted([e for e in evals if e["version"] == final_v and e["round_id"] in (ROUND_A, ROUND_B)],
                            key=lambda e: e["round_id"])
            tp = next((e for e in evals if e["version"] == final_v and e["round_id"] == ROUND_TRAIN), None)
            logs = pd.concat([decision_log(self.reg, e).assign(round=e["round_id"]) for e in finals], ignore_index=True)
            nulls = [read_json(self._null_path(e)) for e in finals if self._null_path(e).exists()]
            s = p3a.seed_metrics(finals, logs, nulls, tp, market, g_star)
            ep, up = p3a.history_frames(self.reg, chain)
            s.update(p3a.history_metrics(ep, up))
            s.update(an.training_progress(ep, g_star))
            s["safeguard_share"] = an.safeguard_share(logs, mapper_from(self.reg.identity(aid)).size_bounds)
            s["time_in_market"] = float(np.mean(logs["leverage_after_fill"].to_numpy() > 1e-6))
            s["size_distance"] = abs(math.log(max(s["median_size"], 1e-12) / g_star))
            s["exploration"] = an.exploration_record(up)
            s.update(key=key, agent_id=aid, seed=a["seed"], seed_index=a["seed_index"], versions=[v["version"] for v in chain],
                     final_checkpoint=chain[-1]["final_checkpoint"]["model_sha256"])
            seeds[a["group"]].append(s)
            for e in evals:
                if e["round_id"].startswith("exp1b-curve-"):
                    mt = e["diagnosis"]["metrics"]
                    lg = decision_log(self.reg, e)
                    curves.setdefault(a["group"], {}).setdefault(e["round_id"], []).append(
                        dict(key=key, q2=mt["dq2"], ruinous=float(market.ruinous(lg["exposure_after_fill"].to_numpy()).mean()),
                             median=mt["median_leverage"]))
        ref = read_json(ref_path)["seeds"]
        v = an.verdicts(seeds, {g: ref[g] for g in ("control", "C1", "C2")}, g_star)
        return dict(seeds=seeds, verdicts=v, curves=curves, g_star=g_star, references={g: ref[g] for g in ("control", "C1", "C2")})


# ---------------------------------------------------------------------------- workers
def _train_job(args):
    root, job = args
    import torch
    torch.set_num_threads(1)
    from .continual import record_interrupted, recover, train_version
    reg = Registry(root)
    reason = f"{EXPERIMENT_ID} group {job['group']}: training from scratch, {STEPS:,} steps"
    if job.get("recover") is not None:
        v = record_interrupted(reg, job["agent_id"], job["recover"], "interrupted (container restart or error); resumed")
        if v is not None:
            return job, recover(reg, job["agent_id"], job["recover"], steps=max(STEPS - v["final_checkpoint"]["step"], 8192),
                                reason=reason + f" (resumed after interruption of v{job['recover']})", log=lambda m: None)
    rec = train_version(reg, job["agent_id"], steps=STEPS, parent=None, reason=reason, n_envs=N_ENVS,
                        checkpoint_every=CHECKPOINT_EVERY, seed=job["seed"], allow_parallel=True,
                        initial_settings=job["initial_settings"], log=lambda m: None)
    return job, rec


def _eval_job(args):
    root, job = args
    import torch
    torch.set_num_threads(1)
    from .evaluate import evaluate
    reg = Registry(root)
    rec = evaluate(reg, job["agent_id"], job["version"], job["round_id"], checkpoint_sha=job["checkpoint"],
                   extra_fields=dict(experiment=EXPERIMENT_ID, group=job["group"], purpose=job["purpose"]))
    return job, rec


def _null_job(args):
    root, job = args
    import torch
    torch.set_num_threads(1)
    from . import synthetic_tasks as st
    from .evaluate import null_distribution
    reg = Registry(root)
    e = next(x for x in reg.evaluations(job["agent_id"]) if x["eval_id"] == job["eval_id"])
    td = e["test_data"]
    task = st.build(td["task"], td["seed"], e["test_steps"])
    null = null_distribution(reg.identity(job["agent_id"]), task, td["episode_starts"], e["episodes"])
    write_once_json(reg.root / "analyses" / "matched-null" / f"{e['eval_id']}.json",
                    dict(eval_id=e["eval_id"], agent_id=job["agent_id"], computed_at=now_utc(), null=null,
                         method="leverage- and turnover-matched random-direction agents (evaluate.null_distribution), "
                                "10 agents, seeds 50000-50009, same test runs"))
    return job


# ---------------------------------------------------------------------------- report
def _f(x, fmt="{:+.2f}", none="—"):
    return none if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def _y(b):
    return "**yes**" if b else "no"


def report(exp: Experiment, res: dict, protection: dict | None = None) -> str:
    m = exp.manifest()
    V, S, g_star = res["verdicts"]["groups"], res["seeds"], res["g_star"]
    c = m["exploration_design"]["calibration"]
    L = [f"# Phase 3 Experiment 1b: C1 and C2 with exploration matched in position-size terms ({EXPERIMENT_ID})\n",
         f"Pre-registered plan: `registry-phase3b/experiments/{EXPERIMENT_ID}/manifest.json` (planned {m['planned_at']}); "
         f"configuration and code checksums verified identical before training, evaluation and this analysis.\n",
         f"Six paired seeds per group (the same fresh training seed in every group for a seed index), {STEPS:,} steps "
         f"each, final checkpoints only, two fresh evaluation rounds of 20 test runs. Growth-optimal size G* = {g_star:.2f}x; "
         "ruinous = 9.5x or more.\n",
         "| Group | Configuration | Config SHA-256 |", "|---|---|---|"]
    L += [f"| {g} | {cfg['description']} | `{cfg['config_sha256'][:12]}…` |" for g, cfg in m["groups"].items()]
    L += ["", "## Pre-registered verdicts\n",
          "A seed is a successful growth agent only if W1–W7 and E all hold. A group supports safe growth if at least 4 of 6 "
          "seeds do. Zero ruinous decisions alone is not counted as successful learning.\n",
          "| Group | W1 ruin<1% | W2 deaths≤1 | W3 size 1.47–5.86x | W4 Q2≥0.5 | W5 growth | W6 tilt<1.5 | W7 progress | E | "
          "**Successful seeds** | **Interpretation** |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, r in V.items():
        n = r["counts"]
        L.append(f"| {g} | {n['W1']}/6 | {n['W2']}/6 | {n['W3']}/6 | {n['W4']}/6 | {n['W5']}/6 | {n['W6']}/6 | {n['W7']}/6 | "
                 f"{n['E']}/6 | **{r['successes']}/6** | **{r['interpretation']}** |")
    L += ["", "Group means across seeds (95% bootstrap interval):\n",
          "| Group | Q2 | Ruinous share | Median size | Seeds with ruinous ≥ 1% |", "|---|---|---|---|---|"]
    for g, r in V.items():
        L.append(f"| {g} | {r['q2_mean']:+.2f} ({r['q2_ci'][0]:+.2f}, {r['q2_ci'][1]:+.2f}) | {100 * r['ruinous_mean']:.2f}% "
                 f"({100 * r['ruinous_ci'][0]:.2f}, {100 * r['ruinous_ci'][1]:.2f}) | {r['median_size_mean']:.2f}x "
                 f"({r['median_size_ci'][0]:.2f}, {r['median_size_ci'][1]:.2f}) | {r['seeds_ruinous_ge_1pct']}/6 |")
    L += ["", "## Comparison with Experiment 1 (planned; Holm-corrected over 18 comparisons)\n",
          "Difference = new group mean minus Experiment 1 group mean, per seed; distance = |ln(median size / G*)| (0 = optimal).\n",
          "| Comparison | Difference (95% CI) | p raw / Holm |", "|---|---|---|"]
    for g, r in V.items():
        for key, x in r["comparisons"].items():
            L.append(f"| {key} | {x['difference']:+.3f} ({x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}) | {x['p_raw']:.3f} / {x['p_holm']:.3f} |")
    L += ["", "Paired by seed index between the new groups (descriptive):\n", "| Difference | Mean | sign-flip p |", "|---|---|---|"]
    L += [f"| {k} | {x['mean']:+.3f} | {x['p']:.3f} |" for k, x in res["verdicts"]["paired"].items()]
    L += ["", "## Every agent (final checkpoint, 40 test runs)\n",
          "| Group | Seed | Agent | Q2 | Ruinous | Test deaths | Median size | Distance | 95th pct | Largest | In market | Long | "
          "Growth per run (95% CI) | Beats random A/B | Tilt | Train size first→last | Train ruin last 20% | Train deaths | "
          "Safeguard at bound | W1–W7,E | Success |", "|" + "---|" * 21]
    for g in m["groups"]:
        for s, f in ((s, V[g]["flags"][i]) for i, s in enumerate(sorted(S[g], key=lambda x: x["seed_index"]))):
            fl = "".join("✓" if f[k] else "✗" for k in ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "E"))
            L.append(f"| {g} | {s['seed_index']} ({s['seed']}) | {s['agent_id']} | {s['q2']:+.2f} | {100 * s['ruinous']:.2f}% | {s['deaths']} | "
                     f"{s['median_size']:.2f}x | {s['size_distance']:.2f} | {s['p95_size']:.2f}x | {s['max_size']:.2f}x | "
                     f"{100 * s['time_in_market']:.0f}% | {100 * s['long_share']:.0f}% | {s['growth']:+.4f} ({s['growth_ci'][0]:+.4f}, "
                     f"{s['growth_ci'][1]:+.4f}) | {'/'.join('y' if b['beats'] else 'n' for b in s['null'])} | {_f(s['tilt'], '{:.2f}')} | "
                     f"{_f(s.get('train_size_first'), '{:.2f}')}x→{_f(s.get('train_size_last'), '{:.2f}')}x | "
                     f"{_f(100 * s['train_ruin_last'] if s.get('train_ruin_last') is not None else None, '{:.2f}%')} | "
                     f"{s.get('training_deaths', '—')} | {100 * s['safeguard_share']:.3f}% | {fl} | {_y(f['success'])} |")
    L += ["", "## Exploration actually applied during training\n",
          "From the exploration monitor (every training update): spread of ln(sampled size / mean size) as a multiplier per "
          "standard deviation; share of sampled sizes at 9.5x or more; median mean size. First / middle / last update, "
          "averaged over the group's seeds; maximum share of sampled size outputs clipped by the numerical range.\n",
          "| Group | Size x per sd (first / mid / last) | Position x per sd (first / last) | Sampled ≥ 9.5x (first / last) | "
          "Median mean size (first / last) | Spread by mean size band, last update (<1x / 1–3x / 3–9.5x) | Max clipped share |",
          "|---|---|---|---|---|---|---|"]
    for g in m["groups"]:
        ex = [s["exploration"] for s in S[g] if s.get("exploration")]
        if not ex:
            continue
        avg = lambda part, k: np.nanmean([e[part].get(k) if e[part].get(k) is not None else np.nan for e in ex])
        mult = lambda part, k: math.exp(avg(part, k))
        L.append(f"| {g} | x{mult('first', 'size_ratio_sd'):.2f} / x{mult('middle', 'size_ratio_sd'):.2f} / x{mult('last', 'size_ratio_sd'):.2f} | "
                 f"x{mult('first', 'position_ratio_sd'):.2f} / x{mult('last', 'position_ratio_sd'):.2f} | "
                 f"{100 * avg('first', 'size_sample_ge_ruin_share'):.2f}% / {100 * avg('last', 'size_sample_ge_ruin_share'):.2f}% | "
                 f"{avg('first', 'size_mean_median'):.2f}x / {avg('last', 'size_mean_median'):.2f}x | "
                 f"{_f(math.exp(avg('last', 'band_lt1_ratio_sd')) if math.isfinite(avg('last', 'band_lt1_ratio_sd')) else None, 'x{:.2f}')} / "
                 f"{_f(math.exp(avg('last', 'band_1to3_ratio_sd')) if math.isfinite(avg('last', 'band_1to3_ratio_sd')) else None, 'x{:.2f}')} / "
                 f"{_f(math.exp(avg('last', 'band_3to9p5_ratio_sd')) if math.isfinite(avg('last', 'band_3to9p5_ratio_sd')) else None, 'x{:.2f}')} | "
                 f"{100 * max(e.get('max_clipped_share') or 0 for e in ex):.4f}% |")
    L += ["", "Designed starting exploration (pre-registered, by starting size): multiplier at +1 sd / -1 sd and chance of "
          "reaching 9.5x or more.\n", "| Start | " + " | ".join(m["exploration_design"]["by_start_size"]) + " |",
          "|---|" + "---|" * len(m["exploration_design"]["by_start_size"])]
    names = list(m["exploration_design"]["by_start_size"])
    for i, row in enumerate(m["exploration_design"]["by_start_size"][names[0]]):
        cells = []
        for nm in names:
            r = m["exploration_design"]["by_start_size"][nm][i]
            cells.append(f"x{r['plus1sd']:.2f} / ÷{1 / r['minus1sd']:.2f} / {100 * r['p_ge_ruin']:.2f}%")
        L.append(f"| {row['start']:.2f}x | " + " | ".join(cells) + " |")
    L += ["", "## Learning curves (development round; descriptive only)\n",
          "Group mean of Q2 / ruinous share / median size by training steps.\n",
          "| Group | " + " | ".join(f"{s // 1000}k" for s in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY)) + " |",
          "|---|" + "---|" * len(range(CURVE_EVERY, STEPS + 1, CURVE_EVERY))]
    for g in m["groups"]:
        cells = []
        for s in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY):
            xs = res["curves"].get(g, {}).get(curve_round(s), [])
            cells.append("—" if not xs else f"{np.mean([x['q2'] for x in xs]):+.2f} / {100 * np.mean([x['ruinous'] for x in xs]):.1f}% / "
                         f"{np.mean([x['median'] for x in xs]):.2f}x")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L += ["", "## Limits\n",
          "- Six seeds per group resolve only large effects; intervals show the range consistent with the data.",
          "- Experiment 1 references used different evaluation data; Q2 depends only on decisions, not on realised luck.",
          "- Verdicts are the pre-registered rules applied mechanically; nothing was changed after training began.",
          "- Every agent, version (including interrupted attempts), checkpoint and evaluation is preserved."]
    if protection:
        L += ["", "## Protection of earlier work\n"] + [f"- {k}: {v}" for k, v in protection.items()]
    return "\n".join(L) + "\n"
