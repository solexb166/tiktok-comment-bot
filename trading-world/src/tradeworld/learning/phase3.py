"""Phase 3 Experiment 1 (proposal 0006, approved 2026-11-17): S6 single-factor ablation.

Five groups of six NEW agents, each trained from scratch for 3M steps under the frozen S6 rules
(market, reward R1, death at 5%, margin, simulator); each group differs from the replicated
control in exactly one respect:
  control  existing settings (size output u, G = 10^(4u); PPO defaults)
  C1       size output z, G = e^z                  (proposal option C1)
  C2       size output z, G = softplus(z)          (proposal option C2)
  D        learning rate 1e-4, 4 epochs per update, KL limit 0.02 per update
  A        discount factor 0.999

The experiment lives in its own registry (registry-phase3/), with its own agents, versions,
checkpoints, evaluations, rounds and seals. Its data ledger inherits the original registry's
ledger read-only, so every training and test seed is one no earlier agent or round has used.
Its permanent blobs are written to a separate store (storage.json). The original registry is
never opened for writing.

Evaluation: every agent's FINAL checkpoint, on two fresh rounds (A and B) shared by all agents;
plus its own training paths (exploitation check E1) and learning-curve points every 500k steps on
one development round (descriptive only, never used for any decision).
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from ..registry.datasets import DataLedger
from ..registry.registry import ROOT, Registry, RegistryError, now_utc
from ..registry.store import append_jsonl, read_json, read_jsonl, write_once_json
from . import phase3_analysis as an

EXPERIMENT_ID = "phase3-exp1"
TASK, REWARD, PROFILE = "S6", "R1", "A_synthetic"
ENV_SETTINGS = dict(decision_interval=1, obs_resolution=1, window=30)
EPISODE_DECISIONS = 1000
STEPS = 3_000_000
CHECKPOINT_EVERY = 250_000
CURVE_EVERY = 500_000
N_ENVS = 8
SEEDS_PER_GROUP = 6
TRAIN_DATA_STEPS = 200_000
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
GROUPS = {
    "control": dict(action_space=LOG10, initial_settings={},
                    description="existing settings: size output u, G = 10^(4u); PPO lr 3e-4, 10 epochs, no KL limit, gamma 0.99"),
    "C1": dict(action_space=dict(LOG10, size_repr="exp"), initial_settings={},
               description="size output z with G = e^z (z in [-25, 25], numerical safeguard only)"),
    "C2": dict(action_space=dict(LOG10, size_repr="softplus"), initial_settings={},
               description="size output z with G = softplus(z) = ln(1 + e^z) (z in [-25, 10,000], numerical safeguard only)"),
    "D": dict(action_space=LOG10, initial_settings=dict(learning_rate=1e-4, n_epochs=4, target_kl=0.02),
              description="steadier updates: learning rate 1e-4, 4 epochs per update, KL limit 0.02 per update"),
    "A": dict(action_space=LOG10, initial_settings=dict(gamma=0.999),
              description="longer learning horizon: discount factor 0.999"),
}
ANALYSIS_FILE = "src/tradeworld/learning/phase3_analysis.py"
ROUND_A, ROUND_B, ROUND_TRAIN = "exp1-final-a", "exp1-final-b", "exp1-trainpaths"


def curve_round(step: int) -> str:
    return f"exp1-curve-{step // 1000:04d}k"


def analysis_checksum() -> str:
    return hashlib.sha256((ROOT / ANALYSIS_FILE).read_bytes()).hexdigest()


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

    def progress(self) -> list[dict]:
        return read_jsonl(self.progress_path)

    # ------------------------------------------------------------------ plan (write-once)
    def plan(self, approval: list[str], criteria_text: list[str], predictions: list[str], compute: dict,
             storage: dict) -> dict:
        from . import synthetic_tasks as st
        from .evaluate import TEST_STEPS_DATA
        from .metrics import Q2_DEFINITION, check_q2
        if (self.dir / "manifest.json").exists():
            raise RegistryError("already planned")
        q2 = check_q2(self.reg, require_frozen=True)
        ledger = DataLedger(self.reg.root)
        if not ledger.inherited():
            raise RegistryError("the experiment ledger must inherit the original registry's ledger")
        # agents: interleaved by seed index so that every group's seeds are drawn alike
        agents = {}
        for k in range(SEEDS_PER_GROUP):
            for g, cfg in GROUPS.items():
                seed = ledger.allocate(TASK, "train", note=f"{EXPERIMENT_ID} {g} seed {k + 1}: training data and model seed")
                aid = self.reg.create_agent(
                    task=TASK, reward=REWARD, seed=seed, env_settings=ENV_SETTINGS, action_space=cfg["action_space"],
                    profile=PROFILE, episode_decisions=EPISODE_DECISIONS, name=f"P3E1-{g}-s{k + 1}",
                    origin=EXPERIMENT_ID, notes=f"Phase 3 Experiment 1, group {g}: {cfg['description']}",
                    extra=dict(experiment=EXPERIMENT_ID, group=g, seed_index=k + 1,
                               initial_settings=cfg["initial_settings"]))
                agents[f"{g}-s{k + 1}"] = dict(agent_id=aid, group=g, seed_index=k + 1, seed=seed)
        # evaluation data, fixed before training
        rounds = {}
        for rid, purpose in ((ROUND_A, "final evaluation, round A"), (ROUND_B, "final evaluation, round B (replication)"),
                             ("exp1-dev", "learning-curve development data (descriptive only)")):
            seed = ledger.allocate(TASK, "test", round_id=rid, note=f"{EXPERIMENT_ID} {purpose}")
            data = [dict(task=TASK, seed=seed, data_hash=st.build(TASK, seed, TEST_STEPS_DATA).md.content_hash())]
            if rid == "exp1-dev":
                for step in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY):
                    self.reg.record_round(curve_round(step), dict(kind="fresh", data=data, test_steps=TEST_STEPS_DATA,
                                                                  note=f"{EXPERIMENT_ID} learning curve at {step:,} steps"))
                rounds["curve"] = {curve_round(s): seed for s in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY)}
            else:
                self.reg.record_round(rid, dict(kind="fresh", data=data, test_steps=TEST_STEPS_DATA, note=f"{EXPERIMENT_ID} {purpose}"))
                rounds[rid] = seed
        train_data = [dict(task=TASK, seed=a["seed"], agent_seed=a["seed"],
                           data_hash=st.build(TASK, a["seed"], TRAIN_DATA_STEPS).md.content_hash()) for a in agents.values()]
        self.reg.record_round(ROUND_TRAIN, dict(kind="training-data", data=train_data, test_steps=TRAIN_DATA_STEPS,
                                                note=f"{EXPERIMENT_ID}: each agent's own training paths (check E1); not fresh"))
        rounds[ROUND_TRAIN] = "each agent's own training seed"
        m = dict(
            experiment_id=EXPERIMENT_ID, planned_at=now_utc(), proposal="docs/proposals/0006-phase3-learning-interventions.md",
            market=TASK, reward=REWARD, profile=PROFILE, env_settings=ENV_SETTINGS, episode_decisions=EPISODE_DECISIONS,
            groups=GROUPS, seeds_per_group=SEEDS_PER_GROUP, steps=STEPS, checkpoint_every=CHECKPOINT_EVERY,
            curve_every=CURVE_EVERY, n_envs=N_ENVS, agents=agents, rounds=rounds,
            evaluated_checkpoint="each agent's final checkpoint (no selection)",
            decision_quality=dict(q2, definition=Q2_DEFINITION),
            analysis=dict(file=ANALYSIS_FILE, sha256=analysis_checksum(), criteria=criteria_text),
            predictions=predictions, approval_conditions=approval, compute=compute, storage=storage,
            frozen=("S6 market mechanics, reward R1, death at 5%, margin 1:2000, simulator, costs, Q2.0 score, phase 2 "
                    "gate criteria, open trading principle. No exposure cap, no new penalty, no trading style."),
        )
        write_once_json(self.dir / "manifest.json", m)
        self._log("planned")
        return m

    def check_analysis(self) -> None:
        if analysis_checksum() != self.manifest()["analysis"]["sha256"]:
            raise RegistryError("the pre-registered analysis code changed after it was frozen")

    # ------------------------------------------------------------------ training
    def training_job(self, key: str) -> dict | None:
        a = self.manifest()["agents"][key]
        aid = a["agent_id"]
        if an.lineage(self.reg, aid, STEPS):
            return None
        ev = self.reg.events(aid)
        closed = {e["version"] for e in ev if e["kind"] in ("training_finished", "training_abandoned")}
        open_ = [e["version"] for e in ev if e["kind"] == "training_started" and e["version"] not in closed]
        vs = self.reg.versions(aid)
        parents = {v.get("parent_version") for v in vs}
        stopped = [v["version"] for v in vs if v["kind"] == "interrupted" and v["version"] not in parents]
        stuck = max(open_ + stopped, default=None)
        return dict(key=key, agent_id=aid, group=a["group"], seed=a["seed"], recover=stuck,
                    initial_settings=GROUPS[a["group"]]["initial_settings"])

    def run_training(self, workers: int = 4, on_done=None, log=print) -> None:
        jobs = [j for j in (self.training_job(k) for k in self.manifest()["agents"]) if j]
        log(f"training: {len(jobs)} agents to train")
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for job, rec in pool.imap_unordered(_train_job, [(str(self.reg.root), j) for j in jobs]):
                self._log("trained", agent_key=job["key"], version=rec["version"], step=rec["final_checkpoint"]["step"])
                log(f"  {job['key']} -> v{rec['version']} ({rec['training']['wall_seconds']:.0f}s)")
                if on_done:
                    on_done(job["key"])

    def training_complete(self) -> bool:
        return all(an.lineage(self.reg, a["agent_id"], STEPS) for a in self.manifest()["agents"].values())

    # ------------------------------------------------------------------ evaluation
    def evaluation_jobs(self) -> list[dict]:
        m = self.manifest()
        jobs = []
        for key, a in m["agents"].items():
            chain = an.lineage(self.reg, a["agent_id"], STEPS)
            if not chain:
                continue
            final_v = chain[-1]["version"]
            done = {(e["round_id"], e["version"]) for e in self.reg.evaluations(a["agent_id"])}
            items = [(ROUND_A, final_v, None, "final"), (ROUND_B, final_v, None, "final"), (ROUND_TRAIN, final_v, None, "trainpaths")]
            for step in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY):
                v, c = an.checkpoint_at(chain, step)
                items.append((curve_round(step), v, c["model_sha256"], f"curve {step}"))
            for rid, v, sha, purpose in items:
                if (rid, v) not in done:
                    jobs.append(dict(key=key, agent_id=a["agent_id"], group=a["group"], version=v, round_id=rid,
                                     checkpoint=sha, purpose=purpose))
        return jobs

    def null_jobs(self) -> list[dict]:
        jobs = []
        for key, a in self.manifest()["agents"].items():
            for e in self.reg.evaluations(a["agent_id"]):
                if e["round_id"] in (ROUND_A, ROUND_B) and not self._null_path(e).exists():
                    jobs.append(dict(key=key, agent_id=a["agent_id"], eval_id=e["eval_id"]))
        return jobs

    def _null_path(self, e: dict) -> Path:
        return self.reg.root / "analyses" / "matched-null" / f"{e['eval_id']}.json"

    def run_evaluations(self, workers: int = 4, log=print) -> None:
        from .metrics import check_q2
        if check_q2(self.reg, require_frozen=True)["checksum"] != self.manifest()["decision_quality"]["checksum"]:
            raise RegistryError("the decision-quality score differs from the one in the experiment plan")
        jobs = self.evaluation_jobs()
        log(f"evaluations: {len(jobs)}")
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
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
        from . import synthetic_tasks as st
        from .evaluate import decision_log
        from .quality import KnownMarket
        self.check_analysis()
        m = self.manifest()
        market = KnownMarket.from_task(st.build(TASK, 0, 1000))
        g_star = market.optimal()[0]
        seeds, curves = {g: [] for g in GROUPS}, {}
        for key, a in m["agents"].items():
            aid = a["agent_id"]
            chain = an.lineage(self.reg, aid, STEPS)
            final_v = chain[-1]["version"]
            evals = self.reg.evaluations(aid)
            finals = [e for e in evals if e["version"] == final_v and e["round_id"] in (ROUND_A, ROUND_B)]
            finals.sort(key=lambda e: e["round_id"])
            tp = next((e for e in evals if e["version"] == final_v and e["round_id"] == ROUND_TRAIN), None)
            import pandas as pd
            logs = pd.concat([decision_log(self.reg, e).assign(round=e["round_id"]) for e in finals], ignore_index=True)
            nulls = [read_json(self._null_path(e)) for e in finals if self._null_path(e).exists()]
            s = an.seed_metrics(finals, logs, nulls, tp, market, g_star)
            ep, up = an.history_frames(self.reg, chain)
            s.update(an.history_metrics(ep, up))
            s.update(key=key, agent_id=aid, seed=a["seed"], seed_index=a["seed_index"], versions=[v["version"] for v in chain],
                     final_checkpoint=chain[-1]["final_checkpoint"]["model_sha256"],
                     wall_seconds=sum(v["training"].get("wall_seconds") or 0 for v in chain))
            seeds[a["group"]].append(s)
            for e in evals:
                if e["round_id"].startswith("exp1-curve-"):
                    mt = e["diagnosis"]["metrics"]
                    curves.setdefault(a["group"], {}).setdefault(e["round_id"], []).append(
                        dict(key=key, q2=mt["dq2"], ruinous=float(market.ruinous(np.asarray(
                            decision_log(self.reg, e)["exposure_after_fill"])).mean()), median=mt["median_leverage"]))
        return dict(seeds=seeds, groups=an.group_verdicts(seeds), curves=curves, g_star=g_star,
                    analysis_sha256=analysis_checksum())


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
            remaining = max(STEPS - v["final_checkpoint"]["step"], 8192)
            return job, recover(reg, job["agent_id"], job["recover"], steps=remaining,
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
    ident = reg.identity(job["agent_id"])
    td = e["test_data"]
    task = st.build(td["task"], td["seed"], e["test_steps"])
    null = null_distribution(ident, task, td["episode_starts"], e["episodes"])
    write_once_json(reg.root / "analyses" / "matched-null" / f"{e['eval_id']}.json",
                    dict(eval_id=e["eval_id"], agent_id=job["agent_id"], computed_at=now_utc(), null=null,
                         method="leverage- and turnover-matched random-direction agents (evaluate.null_distribution), "
                                "10 agents, seeds 50000-50009, same test runs"))
    return job


# ---------------------------------------------------------------------------- report
def _f(x, fmt="{:+.2f}", none="—"):
    return none if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def _yn(x):
    return "n/a" if x is None else ("**yes**" if x else "no")


def report(exp: Experiment, res: dict, protection: dict | None = None) -> str:
    m = exp.manifest()
    G, S, g_star = res["groups"], res["seeds"], res["g_star"]
    L = [f"# Phase 3 Experiment 1: S6 single-factor ablation ({EXPERIMENT_ID})\n",
         f"Pre-registered plan: `registry-phase3/experiments/{EXPERIMENT_ID}/manifest.json` (planned {m['planned_at']}). "
         f"Analysis code `{ANALYSIS_FILE}`, SHA-256 `{res['analysis_sha256'][:16]}…`, identical to the frozen version.\n",
         f"Five groups of {m['seeds_per_group']} new agents, each trained from scratch for {STEPS:,} steps on its own fresh S6 "
         f"market path. Every agent is judged on its final checkpoint only, on two fresh evaluation rounds (test runs "
         f"each, {m['rounds'][ROUND_A]} and {m['rounds'][ROUND_B]} market seeds). Growth-optimal size G* = {g_star:.2f}x; ruinous = 9.5x or more.\n",
         "| Group | Configuration |", "|---|---|"]
    L += [f"| {g} | {c['description']} |" for g, c in GROUPS.items()]
    L += ["", "## Pre-registered verdicts\n",
          "Q2 effect = group mean minus control mean of per-seed decision quality (Q2.0), with a 95% bootstrap interval "
          f"across seeds; p = exact one-sided permutation test ({math.comb(2 * m['seeds_per_group'], m['seeds_per_group'])} "
          "relabellings), Holm-corrected across the four groups.\n",
          "| Group | Q2 mean (sd) | Effect vs control (95% CI) | Cohen's d | p raw / Holm | G1 | G2 ruin<1% | G3 size | "
          "G4 growth | G5 both rounds | G6 history | E1–E4 clear | K catastrophe | **Genuine improvement** |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        r = G[g]
        n = r["n"]
        e_ok = all(r[k] for k in ("E1", "E2", "E3", "E4"))
        if g == "control":
            L.append(f"| control | {r['q2_mean']:+.2f} ({_f(r['q2_sd'], '{:.2f}')}) | — | — | — | — | {r['ruin_ok']}/{n} | "
                     f"{r['size_ok']}/{n} | {r['growth_ok']}/{n} | — | {r['history_ok']}/{n} | {_yn(e_ok)} | {r['K_status']} | — |")
            continue
        L.append(f"| {g} | {r['q2_mean']:+.2f} ({_f(r['q2_sd'], '{:.2f}')}) | {r['effect']:+.2f} ({r['effect_ci'][0]:+.2f}, "
                 f"{r['effect_ci'][1]:+.2f}) | {_f(r['cohens_d'])} | {r['p_raw']:.3f} / {r['p_holm']:.3f} | {_yn(r['G1'])} | "
                 f"{r['ruin_ok']}/{n} {_yn(r['G2'])} | {r['size_ok']}/{n} {_yn(r['G3'])} | {r['growth_ok']}/{n} {_yn(r['G4'])} | "
                 f"{_yn(r['G5'])} | {r['history_ok']}/{n} {_yn(r['G6'])} | {_yn(e_ok)} | {r['K_status']} | "
                 f"{'**YES**' if r['genuine_improvement'] else 'no'} |")
    L += ["", "Exploitation checks per group: E1 training-path advantage (mean Q2 gap, limit 0.5); E2 share of decisions at "
          "1,000x or more (limit 0.1%); E3 median previous-move tilt (limit 1.5); E4 rejected-order share (control + 0.1 pp).\n",
          "| Group | E1 gap | E2 boundary | E3 tilt | E4 rejected | Ruinous share effect vs control (95% CI) | K: change after deaths (control) |",
          "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        r = G[g]
        eff = "—" if g == "control" else f"{100 * r['ruinous_effect']:+.1f} pp ({100 * r['ruinous_effect_ci'][0]:+.1f}, {100 * r['ruinous_effect_ci'][1]:+.1f})"
        L.append(f"| {g} | {_f(r['train_gap_mean'])} {_yn(r['E1'])} | {100 * r['boundary_mean']:.3f}% {_yn(r['E2'])} | "
                 f"{r['tilt_median']:.2f} {_yn(r['E3'])} | {100 * r['rejected_mean']:.3f}% {_yn(r['E4'])} | {eff} | "
                 f"{_f(r.get('K_mean'), '{:+.4f}')} ({_f(r['K_control_mean'], '{:+.4f}')}) |")
    L += ["", "## Every agent (final checkpoint, rounds A and B pooled: 40 test runs)\n",
          "| Group | Seed | Agent | Q2 | Q2 A / B | Ruinous | Median size | 95th pct size | Largest | Long | Test deaths | "
          "Growth per run (95% CI) | Beats random A/B | Tilt | Q2 own paths | Train deaths | Ruin first→last 20% | "
          "Median KL | Clip frac | Gate A/B |",
          "|" + "---|" * 20]
    for g in GROUPS:
        for s in sorted(S[g], key=lambda x: x["seed_index"]):
            qa, qb = (s["q2_by_round"].get(ROUND_A), s["q2_by_round"].get(ROUND_B))
            L.append(f"| {g} | {s['seed_index']} ({s['seed']}) | {s['agent_id']} | {s['q2']:+.2f} | {_f(qa)} / {_f(qb)} | "
                     f"{100 * s['ruinous']:.2f}% | {s['median_size']:.2f}x | {s['p95_size']:.1f}x | {s['max_size']:.0f}x | "
                     f"{100 * s['long_share']:.0f}% | {s['deaths']} | {s['growth']:+.4f} ({s['growth_ci'][0]:+.4f}, {s['growth_ci'][1]:+.4f}) | "
                     f"{'/'.join('y' if b['beats'] else 'n' for b in s['null'])} | {_f(s['tilt'], '{:.2f}')} | {_f(s['q2_trainpaths'])} | "
                     f"{s.get('training_deaths', '—')} | {_f(100 * s['ruin_first'] if s.get('ruin_first') is not None else None, '{:.1f}%')}"
                     f"→{_f(100 * s['ruin_last'] if s.get('ruin_last') is not None else None, '{:.1f}%')} | "
                     f"{_f(s.get('median_kl'), '{:.3f}')} | {_f(s.get('median_clip_fraction'), '{:.2f}')} | "
                     f"{'/'.join(s['gate_results'].get(r, '—') for r in (ROUND_A, ROUND_B))} |")
    L += ["", "## Learning curves (development round; descriptive only, never used for any decision)\n",
          "Group mean across seeds of Q2 / ruinous share / median size, by training steps.\n",
          "| Group | " + " | ".join(f"{s // 1000}k" for s in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY)) + " |",
          "|---|" + "---|" * len(range(CURVE_EVERY, STEPS + 1, CURVE_EVERY))]
    for g in GROUPS:
        cells = []
        for s in range(CURVE_EVERY, STEPS + 1, CURVE_EVERY):
            xs = res["curves"].get(g, {}).get(curve_round(s), [])
            cells.append("—" if not xs else f"{np.mean([x['q2'] for x in xs]):+.2f} / {100 * np.mean([x['ruinous'] for x in xs]):.1f}% / "
                         f"{np.mean([x['median'] for x in xs]):.2f}x")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L += ["", "## Predictions recorded before training\n"] + [f"- {p}" for p in m["predictions"]]
    L += ["", "## How to read this (limits of six seeds)\n",
          "- Six seeds per group detect only large differences (about one between-seed standard deviation). A group that "
          "misses G1 has not been shown to be ineffective; its effect may be smaller than this experiment can resolve. "
          "The intervals above show the range of effects consistent with the data.",
          "- The verdict for each group is the pre-registered rule applied mechanically; no threshold was changed after "
          "training began.",
          "- Every agent, version, checkpoint and evaluation is preserved, including failures and interrupted attempts."]
    if protection:
        L += ["", "## Protection of the original system\n"] + [f"- {k}: {v}" for k, v in protection.items()]
    return "\n".join(L) + "\n"
