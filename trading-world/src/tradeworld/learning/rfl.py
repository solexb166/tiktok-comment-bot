"""Experiment R-FL (proposal 0007, section 11): full-liability log reward against the current reward.

Groups (six paired seeds each, 3M steps, S6; everything else as the Experiment 1 control: original size
output u with size = 10^(4u), original exploration, PPO defaults):
  R1        current reward (concurrent control)
  R6        full-liability log return, c = 0.01 (identical to R1 on every bar that does not end the account)
  R6-c0.05  full-liability log return, c = 0.05 (sensitivity)
Separate registry (registry-rfl/), separate blob store, data ledger inheriting every earlier ledger (read only).
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from ..registry.datasets import DataLedger
from ..registry.registry import ROOT, Registry, RegistryError, now_utc
from ..registry.store import append_jsonl, canonical_json, read_json, write_once_json
from . import phase3_analysis as p3a
from . import phase3b_analysis as w
from . import rfl_analysis as an

EXPERIMENT_ID = "phase3-rfl"
TASK, PROFILE = "S6", "A_synthetic"
ENV_SETTINGS = dict(decision_interval=1, obs_resolution=1, window=30)
EPISODE_DECISIONS = 1000
STEPS = 3_000_000
CHECKPOINT_EVERY = 250_000
CURVE_EVERY = 500_000
N_ENVS = 8
SEEDS_PER_GROUP = 6
TRAIN_DATA_STEPS = 200_000
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
ROUND_A, ROUND_B, ROUND_TRAIN, ROUND_DEV = "rfl-final-a", "rfl-final-b", "rfl-trainpaths", "rfl-dev"
FROZEN_CODE = (
    "src/tradeworld/interface/actions.py", "src/tradeworld/learning/env.py", "src/tradeworld/learning/continual.py",
    "src/tradeworld/learning/history.py", "src/tradeworld/learning/train.py", "src/tradeworld/learning/evaluate.py",
    "src/tradeworld/learning/diagnosis.py", "src/tradeworld/learning/quality.py", "src/tradeworld/learning/rewards.py",
    "src/tradeworld/learning/synthetic_tasks.py", "src/tradeworld/learning/checkpoint.py", "src/tradeworld/learning/gate.py",
    "src/tradeworld/learning/phase3_analysis.py", "src/tradeworld/learning/phase3b_analysis.py",
    "src/tradeworld/learning/rfl_analysis.py", "src/tradeworld/learning/rfl.py",
    "src/tradeworld/registry/registry.py", "src/tradeworld/registry/datasets.py",
)


def groups() -> dict:
    return {
        "R1": dict(reward="R1", action_space=LOG10, initial_settings={},
                   description="current reward R1 (eps-floored log wealth); concurrent control"),
        "R6": dict(reward="R6", action_space=LOG10, initial_settings={},
                   description="full-liability log return, c = 0.01: ln M for M >= 0.01, tangent line below "
                               "(identical to R1 on every bar that does not end the account)"),
        "R6-c0.05": dict(reward="R6-c0.05", action_space=LOG10, initial_settings={},
                         description="full-liability log return, c = 0.05 (sensitivity)"),
    }


def code_checksums() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN_CODE}


def config_checksum(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg)).hexdigest()


def curve_round(step: int) -> str:
    return f"rfl-curve-{step // 1000:04d}k"


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
        if len(ledger.inherited()) < 3:
            raise RegistryError("the ledger must inherit the original, Experiment 1 and Experiment 1b ledgers")
        G = groups()
        agents = {}
        for k in range(SEEDS_PER_GROUP):
            seed = ledger.allocate(TASK, "train", note=f"{EXPERIMENT_ID} seed index {k + 1}: training data and model "
                                                       f"seed shared by R1, R6 and R6-c0.05 (paired)")
            for g, cfg in G.items():
                aid = self.reg.create_agent(
                    task=TASK, reward=cfg["reward"], seed=seed, env_settings=ENV_SETTINGS, action_space=cfg["action_space"],
                    profile=PROFILE, episode_decisions=EPISODE_DECISIONS, name=f"RFL-{g}-s{k + 1}", origin=EXPERIMENT_ID,
                    notes=f"Experiment R-FL, group {g}: {cfg['description']}",
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
            experiment_id=EXPERIMENT_ID, planned_at=now_utc(), market=TASK, rewards={g: c["reward"] for g, c in G.items()}, profile=PROFILE,
            env_settings=ENV_SETTINGS, episode_decisions=EPISODE_DECISIONS, steps=STEPS, checkpoint_every=CHECKPOINT_EVERY,
            curve_every=CURVE_EVERY, n_envs=N_ENVS, seeds_per_group=SEEDS_PER_GROUP, paired_seeds=True,
            groups={g: dict(cfg, config_sha256=config_checksum(cfg)) for g, cfg in G.items()},
            agents=agents, rounds=rounds,
            evaluated_checkpoint="each agent's final checkpoint (no selection)",
            decision_quality=dict(q2, definition=Q2_DEFINITION),
            analysis=dict(file="src/tradeworld/learning/rfl_analysis.py",
                          criteria=[ln.strip() for ln in an.__doc__.splitlines() if ln.strip()],
                          success_criteria=[ln.strip() for ln in w.__doc__.splitlines() if ln.strip()]),
            gate_note=("The frozen Phase 2 gate (gate.py) has no criterion for R6, so evaluation records of R6 agents carry "
                       "result 'fail: no criterion'. That verdict is not used; the experiment is judged by the "
                       "pre-registered criteria above."),
            frozen_code_sha256=code_checksums(),
            approval_conditions=approval, compute=compute, storage=storage,
            frozen=("S6 market mechanics, R1-R5 unchanged (R6 is a new, separately named reward for new agents), death at "
                    "5%, margin 1:2000, simulator, costs, Q2.0 score, open trading principle; no exposure cap, no new "
                    "penalty, no trading style; the only difference between groups is the reward."),
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
        from .evaluate import pinned_criteria
        pinned_criteria(self.reg)          # pin once here: parallel first use would race on the write-once record
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
        market = KnownMarket.from_task(st.build(TASK, 0, 1000))
        g_star = market.optimal()[0]
        seeds, curves = {g: [] for g in m["groups"]}, {}
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
            s.update(w.training_progress(ep, g_star))
            s["safeguard_share"] = w.safeguard_share(logs, mapper_from(self.reg.identity(aid)).size_bounds)
            s["time_in_market"] = float(np.mean(logs["leverage_after_fill"].to_numpy() > 1e-6))
            s["size_distance"] = abs(math.log(max(s["median_size"], 1e-12) / g_star))
            s["exploration"] = w.exploration_record(up)
            s["stability"] = an.stability(up)
            s["nonruinous_median"] = an.nonruinous_median(logs, market)
            s["recovery_after"], s["recovery_else"], s["recovery_n"] = an.recovery_sizes(logs)
            s.update(key=key, agent_id=aid, seed=a["seed"], seed_index=a["seed_index"], versions=[v["version"] for v in chain],
                     final_checkpoint=chain[-1]["final_checkpoint"]["model_sha256"])
            seeds[a["group"]].append(s)
            for e in evals:
                if e["round_id"].startswith("rfl-curve-"):
                    mt = e["diagnosis"]["metrics"]
                    lg = decision_log(self.reg, e)
                    curves.setdefault(a["group"], {}).setdefault(e["round_id"], []).append(
                        dict(key=key, q2=mt["dq2"], ruinous=float(market.ruinous(lg["exposure_after_fill"].to_numpy()).mean()),
                             median=mt["median_leverage"]))
        v = an.verdicts(seeds, g_star)
        return dict(seeds=seeds, verdicts=v, curves=curves, g_star=g_star)


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
    return "n/a" if b is None else ("**yes**" if b else "no")


def report(exp: Experiment, res: dict, protection: dict | None = None) -> str:
    m = exp.manifest()
    V, S, g_star = res["verdicts"]["groups"], res["seeds"], res["g_star"]
    L = [f"# Experiment R-FL: full-liability log reward vs the current reward ({EXPERIMENT_ID})\n",
         f"Proposal: `docs/proposals/0007-reward-investigation.md`. Pre-registered plan: `registry-rfl/experiments/"
         f"{EXPERIMENT_ID}/manifest.json` (planned {m['planned_at']}); configuration and code checksums verified identical "
         "before training, evaluation and this analysis.\n",
         f"Six paired seeds per group, {STEPS:,} steps each, final checkpoints only, two fresh rounds of 20 test runs. "
         f"Everything except the reward is identical to the Experiment 1 control. G* = {g_star:.2f}x; ruinous = 9.5x or more.\n",
         "| Group | Reward | Config SHA-256 |", "|---|---|---|"]
    L += [f"| {g} | {cfg['description']} | `{cfg['config_sha256'][:12]}…` |" for g, cfg in m["groups"].items()]
    L += ["", "## Pre-registered verdicts\n",
          "Per seed: W1–W7 and E (approved in Experiment 1b). A group supports safe growth if at least 4 of 6 seeds meet all.\n",
          "| Group | W1 ruin<1% | W2 deaths≤1 | W3 size 1.47–5.86x | W4 Q2≥0.5 | W5 growth | W6 tilt<1.5 | W7 progress | E | "
          "**Successful seeds** | **Interpretation** |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, r in V.items():
        n = r["counts"]
        L.append(f"| {g} | {n['W1']}/6 | {n['W2']}/6 | {n['W3']}/6 | {n['W4']}/6 | {n['W5']}/6 | {n['W6']}/6 | {n['W7']}/6 | "
                 f"{n['E']}/6 | **{r['successes']}/6** | **{r['interpretation']}** |")
    L += ["", f"Reward truncation confirmed as a cause (R6 'safe growth' and R1 not): "
              f"**{'yes' if res['verdicts']['truncation_confirmed_as_cause'] else 'no'}**.\n",
          "Group means across seeds (95% bootstrap interval):\n",
          "| Group | Q2 | Ruinous share | Median size | Seeds ruinous ≥1% | Training stable | Explained variance (last 20%) | Value loss (last 20%) |",
          "|---|---|---|---|---|---|---|---|"]
    for g, r in V.items():
        L.append(f"| {g} | {r['q2_mean']:+.2f} ({r['q2_ci'][0]:+.2f}, {r['q2_ci'][1]:+.2f}) | {100 * r['ruinous_mean']:.2f}% "
                 f"({100 * r['ruinous_ci'][0]:.2f}, {100 * r['ruinous_ci'][1]:.2f}) | {r['median_size_mean']:.2f}x "
                 f"({r['median_size_ci'][0]:.2f}, {r['median_size_ci'][1]:.2f}) | {r['seeds_ruinous_ge_1pct']}/6 | "
                 f"{_y(r['all_finite'])} | {_f(r['explained_variance_last20'])} | {_f(r['value_loss_last20'], '{:.3g}')} |")
    L += ["", "## Checks for unintended incentives\n",
          "| Group | Timidity: non-ruinous median size vs R1 (95% CI) | T ok (lower bound ≥ 0.75) | Gambling to recover: size after a >20% loss / elsewhere | GR ok (≤ 1.5) |",
          "|---|---|---|---|---|"]
    for g, r in V.items():
        t = r.get("T")
        tcell = "—" if t is None else f"x{t['ratio']:.2f} ({t['ci'][0]:.2f}, {t['ci'][1]:.2f})"
        gr = r["GR"]
        L.append(f"| {g} | {tcell} | {_y(None if t is None else t['ok'])} | "
                 f"{_f(gr['median_ratio'], '{:.2f}')} ({gr['decisions']} decisions) | {_y(gr['ok'])} |")
    L += ["", "## Comparison with the concurrent R1 control (Holm-corrected over 6 comparisons)\n",
          "| Comparison | Difference (95% CI) | p raw / Holm |", "|---|---|---|"]
    for g in an.NEW:
        for key, x in V[g]["comparisons"].items():
            L.append(f"| {key} | {x['difference']:+.3f} ({x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}) | {x['p_raw']:.3f} / {x['p_holm']:.3f} |")
    L += ["", "Paired by seed index (same training data and initial network):\n", "| Difference | Mean | sign-flip p |", "|---|---|---|"]
    L += [f"| {k} | {x['mean']:+.3f} | {x['p']:.3f} |" for k, x in res["verdicts"]["paired"].items()]
    L += ["", "## Every agent (final checkpoint, 40 test runs)\n",
          "| Group | Seed | Agent | Q2 | Ruinous | Test deaths | Median size | Non-ruinous median | 95th pct | Largest | Long | "
          "Growth per run (95% CI) | Beats random A/B | Tilt | Train size first→last | Train ruin last 20% | Train deaths | "
          "W1–W7,E | Success |", "|" + "---|" * 19]
    for g in m["groups"]:
        for i, s in enumerate(sorted(S[g], key=lambda x: x["seed_index"])):
            f = V[g]["flags"][i]
            fl = "".join("✓" if f[k] else "✗" for k in ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "E"))
            L.append(f"| {g} | {s['seed_index']} ({s['seed']}) | {s['agent_id']} | {s['q2']:+.2f} | {100 * s['ruinous']:.2f}% | "
                     f"{s['deaths']} | {s['median_size']:.2f}x | {s['nonruinous_median']:.2f}x | {s['p95_size']:.2f}x | "
                     f"{s['max_size']:.2f}x | {100 * s['long_share']:.0f}% | {s['growth']:+.4f} ({s['growth_ci'][0]:+.4f}, "
                     f"{s['growth_ci'][1]:+.4f}) | {'/'.join('y' if b['beats'] else 'n' for b in s['null'])} | "
                     f"{_f(s['tilt'], '{:.2f}')} | {_f(s.get('train_size_first'), '{:.2f}')}x→{_f(s.get('train_size_last'), '{:.2f}')}x | "
                     f"{_f(100 * s['train_ruin_last'] if s.get('train_ruin_last') is not None else None, '{:.2f}%')} | "
                     f"{s.get('training_deaths', '—')} | {fl} | {_y(f['success'])} |")
    L += ["", "## Learning curves (development round; descriptive only)\n",
          "Group mean of Q2 / ruinous share / median size by training steps.\n",
          "| Group | " + " | ".join(f"{st // 1000}k" for st in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY)) + " |",
          "|---|" + "---|" * len(range(CURVE_EVERY, STEPS + 1, CURVE_EVERY))]
    for g in m["groups"]:
        cells = []
        for st in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY):
            xs = res["curves"].get(g, {}).get(curve_round(st), [])
            cells.append("—" if not xs else f"{np.mean([x['q2'] for x in xs]):+.2f} / {100 * np.mean([x['ruinous'] for x in xs]):.1f}% / "
                         f"{np.mean([x['median'] for x in xs]):.2f}x")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L += ["", "## Limits\n",
          "- Six seeds per group resolve only large effects; intervals show the range consistent with the data.",
          "- Evaluation records of R6 agents carry the frozen Phase 2 gate's 'fail: no criterion' (the gate has no R6 rule); "
          "it is not used.",
          "- Verdicts are the pre-registered rules applied mechanically; nothing was changed after training began.",
          "- Every agent, version (including interrupted attempts), checkpoint and evaluation is preserved."]
    if protection:
        L += ["", "## Protection of earlier work\n"] + [f"- {k}: {v}" for k, v in protection.items()]
    return "\n".join(L) + "\n"
