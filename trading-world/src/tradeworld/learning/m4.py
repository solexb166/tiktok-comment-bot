"""Experiment M4: model-based growth-optimal agent, version 4 (model_kelly4): markets that change.

Groups (six fresh seeds each; each agent learns from its own fresh 200,000-bar market experience):
  MK4-S1X  bounce-back pattern that flips to momentum and back at unannounced times (spread 0.4 sigma)
  MK4-S2X  announced trends whose meaning flips at unannounced times (spread 1.0 sigma)
  MK4-S6X  drift with crashes whose frequency switches between calm (1/300) and storm (1/50) spells
  MK4-S1C  S1C, which does not change (regression check against version 3)
  MK4-S6   S6, which does not change (regression check against version 2)
References (not agents): switching markets - the true-model filter (knows the rules, infers the current regime) and
the oracle (told the current regime; upper bound only); S1C - M3's true-parameter reference; S6 - the exact optimal
constant size. Ablation (not agents): each switching-market agent's own single-regime model on the same runs.
Separate registry (registry-m4/), blob store and ledger inheritance as before.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from ..registry.datasets import DataLedger
from ..registry.registry import ROOT, Registry, RegistryError, code_version, now_utc
from ..registry.store import append_jsonl, canonical_json, read_json, write_once_json
from . import m4_analysis as an
from . import model_kelly4 as mk
from . import m3_analysis as m3a
from . import model_kelly3 as mk3

EXPERIMENT_ID = "phase3-m4"
PROFILE = "A_synthetic"
ENV_SETTINGS = dict(decision_interval=1, obs_resolution=1, window=30)
EPISODE_DECISIONS = 1000
TRAIN_DATA_STEPS = 200_000
DATA_POINTS = (20_000, 200_000)
SEEDS_PER_GROUP = 6
FINAL_ROUNDS = 6
TASKS = ("S1X", "S2X", "S6X", "S1C", "S6")
SWITCHING = ("S1X", "S2X", "S6X")
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
FROZEN_CODE = (
    "src/tradeworld/learning/model_kelly4.py", "src/tradeworld/learning/model_kelly3.py", "src/tradeworld/learning/model_kelly2.py",
    "src/tradeworld/learning/model_kelly.py", "src/tradeworld/learning/m4.py", "src/tradeworld/learning/m4_analysis.py",
    "src/tradeworld/learning/m3_analysis.py",
    "src/tradeworld/learning/m2_analysis.py", "src/tradeworld/learning/evaluate.py", "src/tradeworld/learning/env.py",
    "src/tradeworld/learning/features.py", "src/tradeworld/learning/diagnosis.py", "src/tradeworld/learning/quality.py",
    "src/tradeworld/learning/synthetic_tasks.py", "src/tradeworld/learning/gate.py",
    "src/tradeworld/learning/phase3_analysis.py", "src/tradeworld/learning/phase3b_analysis.py",
    "src/tradeworld/interface/actions.py", "src/tradeworld/interface/observation.py",
    "src/tradeworld/registry/registry.py", "src/tradeworld/registry/datasets.py",
)


def groups() -> dict:
    return {f"MK4-{t}": dict(task=t, agent_type="model-kelly-v4", reward="R1", action_space=LOG10,
                             model_config=dict(mk.CONFIG), data_points=list(DATA_POINTS)) for t in TASKS}


def optimal_size(task: str) -> float:
    from . import synthetic_tasks as st
    from .quality import KnownMarket
    m = KnownMarket.from_task(st.build(task, 0, 1000))
    return 0.0 if m is None else float(m.optimal()[0])


def code_checksums() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN_CODE}


def config_checksum(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg)).hexdigest()


def rid(task: str, kind: str) -> str:
    return f"m4-{task.lower()}-{kind}"


def final_rids(task: str) -> list[str]:
    return [rid(task, f"final-{i}") for i in range(1, FINAL_ROUNDS + 1)]


def curve_rid(task: str, n: int) -> str:
    return rid(task, f"curve-{n // 1000:03d}k")


def curve_rid_for_step(task: str, step: int) -> str:
    """A checkpoint's learning-curve round is the planned data point nearest to its step (the last checkpoint learns
    from all 199,967 examples a 200,000-bar market yields after the 30-bar window and the one-bar-ahead target)."""
    return curve_rid(task, min(DATA_POINTS, key=lambda dp: abs(dp - step)))


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

    def amendments(self) -> list[dict]:
        d = self.dir / "amendments"
        return [read_json(p) for p in sorted(d.glob("[0-9][0-9][0-9][0-9].json"))] if d.exists() else []

    def frozen_code(self) -> dict:
        frozen = dict(self.manifest()["frozen_code_sha256"])
        for a in self.amendments():
            for f, ch in a["code_changes"].items():
                if frozen.get(f) != ch["old_sha256"]:
                    raise RegistryError(f"amendment {a['number']} does not follow the recorded checksum of {f}")
                frozen[f] = ch["new_sha256"]
        return frozen

    def amend(self, reason: str, changed: dict[str, str], state: dict, effect: str) -> dict:
        frozen, now = self.frozen_code(), code_checksums()
        for f, old in changed.items():
            if frozen.get(f) != old:
                raise RegistryError(f"{f}: the stated earlier checksum is not the recorded one")
        unexplained = [f for f, h in now.items() if frozen.get(f) != h and f not in changed]
        if unexplained:
            raise RegistryError(f"frozen code changed without being named in the amendment: {unexplained}")
        n = len(self.amendments()) + 1
        rec = dict(number=n, at=now_utc(), experiment=EXPERIMENT_ID, reason=reason, effect=effect, state_when_made=state,
                   code_changes={f: dict(old_sha256=old, new_sha256=now[f]) for f, old in changed.items()})
        write_once_json(self.dir / "amendments" / f"{n:04d}.json", rec)
        self._log("amendment", number=n, reason=reason)
        return rec

    def check_frozen(self) -> None:
        m = self.manifest()
        frozen = self.frozen_code()
        bad = [f for f, h in code_checksums().items() if frozen.get(f) != h]
        if bad:
            raise RegistryError(f"frozen code changed after pre-registration: {bad}")
        for g, cfg in groups().items():
            if config_checksum(cfg) != m["groups"][g]["config_sha256"]:
                raise RegistryError(f"configuration of group {g} changed after pre-registration")

    # ------------------------------------------------------------------ plan
    def _definition_version(self, aid: str, payload: dict, note: str) -> None:
        n = self.reg.start_training(aid, None, None, {}, info=dict(reference=True))
        sha = self.reg.blobs.put(json.dumps(payload, sort_keys=True).encode(), "json")
        ck = dict(step=0, model_sha256=sha, state_sha256=None, kind="final", note=note)
        self.reg.record_version(aid, dict(version=n, parent_version=None, kind="initial", reason=note, changes={},
                                          training=dict(algorithm="none (reference)"), checkpoints=[], final_checkpoint=ck,
                                          history={}, code_version=code_version()))

    def plan(self, approval: list[str], references: dict, compute: dict, storage: dict) -> dict:
        from . import synthetic_tasks as st
        from .evaluate import TEST_STEPS_DATA
        from .metrics import Q2_DEFINITION, check_q2
        if (self.dir / "manifest.json").exists():
            raise RegistryError("already planned")
        q2 = check_q2(self.reg, require_frozen=True)
        ledger = DataLedger(self.reg.root)
        G = groups()
        agents, rounds, refs_ids, ablations = {}, {}, {}, {}
        for g, cfg in G.items():
            t = cfg["task"]
            for k in range(SEEDS_PER_GROUP):
                seed = ledger.allocate(t, "train", note=f"{EXPERIMENT_ID} {g} seed {k + 1}: the agent's market experience")
                aid = self.reg.create_agent(
                    task=t, reward=cfg["reward"], seed=seed, env_settings=ENV_SETTINGS, action_space=cfg["action_space"],
                    profile=PROFILE, episode_decisions=EPISODE_DECISIONS, name=f"M4-{g}-s{k + 1}", origin=EXPERIMENT_ID,
                    notes=f"Experiment M4, group {g}: model-based growth-optimal agent v4 (regime tracking) on {t}",
                    extra=dict(agent_type="model-kelly-v4", experiment=EXPERIMENT_ID, group=g, seed_index=k + 1,
                               model_config=cfg["model_config"], config_sha256=config_checksum(cfg)))
                agents[f"{g}-s{k + 1}"] = dict(agent_id=aid, group=g, task=t, seed_index=k + 1, seed=seed)
            if t == "S6":
                size = optimal_size(t)
                rid_ = self.reg.create_agent(
                    task=t, reward="R1", seed=0, env_settings=ENV_SETTINGS, action_space=LOG10, profile=PROFILE,
                    episode_decisions=EPISODE_DECISIONS, name=f"M4-REF-{t}", origin=EXPERIMENT_ID,
                    notes=f"Reference, not an agent: always holds the exact growth-optimal size {size:.4f}x computed from "
                          f"{t}'s known parameters. Learns nothing.",
                    extra=dict(agent_type="benchmark-constant", experiment=EXPERIMENT_ID, benchmark_size=size))
                self._definition_version(rid_, dict(benchmark_size=size, task=t), "reference definition (no training)")
                refs_ids[t] = dict(agent_id=rid_, kind="exact optimal constant size", size=size)
            elif t == "S1C":
                rid_ = self.reg.create_agent(
                    task=t, reward="R1", seed=0, env_settings=ENV_SETTINGS, action_space=LOG10, profile=PROFILE,
                    episode_decisions=EPISODE_DECISIONS, name=f"M4-REF-{t}", origin=EXPERIMENT_ID,
                    notes="Reference, not an agent: Experiment M3's cost-aware decision with S1C's TRUE parameters "
                          "(model_kelly3.reference_policy). Learns nothing.",
                    extra=dict(agent_type="reference-cost-aware", experiment=EXPERIMENT_ID))
                self._definition_version(rid_, dict(reference="model_kelly3.reference_policy", task=t,
                                                    rho=mk3.true_rho(t, st.build(t, 0, 1000).params)),
                                         "reference definition (no training)")
                refs_ids[t] = dict(agent_id=rid_, kind="true-parameter cost-aware decision", size=None)
            else:
                for kind_, atype, note in (
                        ("true-filter", "reference-true-filter",
                         "version 4's decision with the market's TRUE regime model, inferring the current regime from "
                         "prices like the agent (model_kelly4.true_filter_policy)"),
                        ("oracle", "reference-oracle",
                         "version 4's decision told the TRUE current regime at every bar (model_kelly4.oracle_policy); "
                         "an upper bound no trader can reach")):
                    rid_ = self.reg.create_agent(
                        task=t, reward="R1", seed=0, env_settings=ENV_SETTINGS, action_space=LOG10, profile=PROFILE,
                        episode_decisions=EPISODE_DECISIONS, name=f"M4-{kind_.upper()}-{t}", origin=EXPERIMENT_ID,
                        notes=f"Reference, not an agent: {note}. Learns nothing.",
                        extra=dict(agent_type=atype, experiment=EXPERIMENT_ID))
                    self._definition_version(rid_, dict(reference=atype, task=t), "reference definition (no training)")
                    key = t if kind_ == "true-filter" else f"{t}-oracle"
                    refs_ids[key] = dict(agent_id=rid_, kind=kind_, size=None)
            for kind in [f"final-{i}" for i in range(1, FINAL_ROUNDS + 1)] + ["dev"]:
                r = rid(t, kind)
                seed = ledger.allocate(t, "test", round_id=r, note=f"{EXPERIMENT_ID} {t} {kind}")
                data = [dict(task=t, seed=seed, data_hash=st.build(t, seed, TEST_STEPS_DATA).md.content_hash())]
                if kind == "dev":
                    for npts in DATA_POINTS:
                        self.reg.record_round(curve_rid(t, npts), dict(kind="fresh", data=data, test_steps=TEST_STEPS_DATA,
                                                                       note=f"{EXPERIMENT_ID} {t}: after {npts:,} bars of experience"))
                    rounds[rid(t, "curve")] = {curve_rid(t, npts): seed for npts in DATA_POINTS}
                else:
                    self.reg.record_round(r, dict(kind="fresh", data=data, test_steps=TEST_STEPS_DATA, note=f"{EXPERIMENT_ID} {t} {kind}"))
                    rounds[r] = seed
            tdata = [dict(task=t, seed=a["seed"], agent_seed=a["seed"],
                          data_hash=st.build(t, a["seed"], TRAIN_DATA_STEPS).md.content_hash())
                     for a in agents.values() if a["task"] == t]
            self.reg.record_round(rid(t, "trainpaths"), dict(kind="training-data", data=tdata, test_steps=TRAIN_DATA_STEPS,
                                                             note=f"{EXPERIMENT_ID} {t}: each agent's own experience (check E)"))
            rounds[rid(t, "trainpaths")] = "each agent's own training seed"
        for key, a in agents.items():
            if a["task"] in SWITCHING:
                bid = self.reg.create_agent(
                    task=a["task"], reward="R1", seed=a["seed"], env_settings=ENV_SETTINGS, action_space=LOG10, profile=PROFILE,
                    episode_decisions=EPISODE_DECISIONS, name=f"M4-ABL-{key}", origin=EXPERIMENT_ID,
                    notes=f"Ablation, not an agent: agent {a['agent_id']}'s own single-regime model (K = 1, its static "
                          "fallback), deciding on the same runs, to measure what tracking regimes is worth.",
                    extra=dict(agent_type="model-kelly-v4-static", experiment=EXPERIMENT_ID, ablation_of=a["agent_id"]))
                ablations[key] = dict(agent_id=bid, ablation_of=a["agent_id"])
        refs = {k: dict(file=v, sha256=hashlib.sha256((ROOT / v).read_bytes()).hexdigest()) for k, v in references.items()}
        m = dict(
            experiment_id=EXPERIMENT_ID, planned_at=now_utc(), profile=PROFILE, env_settings=ENV_SETTINGS,
            episode_decisions=EPISODE_DECISIONS, train_data_bars=TRAIN_DATA_STEPS, data_points=list(DATA_POINTS),
            seeds_per_group=SEEDS_PER_GROUP, final_rounds=FINAL_ROUNDS,
            groups={g: dict(cfg, config_sha256=config_checksum(cfg)) for g, cfg in G.items()},
            agents=agents, references_policies=refs_ids, ablations=ablations, rounds=rounds,
            evaluated_checkpoint="each agent's final checkpoint (200,000 bars; no selection)",
            decision_quality=dict(q2, definition=Q2_DEFINITION, note="used for S6 only"),
            analysis=dict(file="src/tradeworld/learning/m4_analysis.py",
                          criteria=[ln.strip() for ln in an.__doc__.splitlines() if ln.strip()]),
            references=refs, frozen_code_sha256=code_checksums(), approval_conditions=approval, compute=compute,
            storage=storage,
            frozen=("market mechanics, R1-R6, scoring rules, simulator, margin, death rule, Q2.0; no exposure cap, no penalty, "
                    "no trading style; nothing about the market is given to the agents (only to the labelled references)."),
        )
        write_once_json(self.dir / "manifest.json", m)
        self._log("planned")
        return m

    # ------------------------------------------------------------------ training
    def trained(self, aid: str) -> dict | None:
        vs = [v for v in self.reg.versions(aid) if v["kind"] != "interrupted"]
        return vs[-1] if vs else None

    def run_training(self, workers: int = 4, on_done=None, log=print) -> None:
        self.check_frozen()
        m = self.manifest()
        jobs = [dict(key=k, agent_id=a["agent_id"], seed=a["seed"]) for k, a in m["agents"].items() if not self.trained(a["agent_id"])]
        log(f"training: {len(jobs)} agents")
        with mp.get_context("fork").Pool(workers) as pool:
            for job, rec in pool.imap_unordered(_train_job, [(str(self.reg.root), j) for j in jobs]):
                sel = rec["final_checkpoint"].get("selection", {})
                self._log("trained", agent_key=job["key"], version=rec["version"])
                log(f"  {job['key']} -> v{rec['version']} ({rec['training']['wall_seconds']:.0f}s, chose {sel.get('chosen')})")
                if on_done:
                    on_done(job["key"])
        # each ablation identity gets a definition version pointing at its agent's final model (no training)
        for key, ab in m["ablations"].items():
            if not self.trained(ab["agent_id"]):
                v = self.trained(ab["ablation_of"])
                n = self.reg.start_training(ab["agent_id"], None, None, {}, info=dict(ablation=True))
                ck = dict(v["final_checkpoint"], note=f"agent {ab['ablation_of']}'s final model, deciding with its single-regime model")
                self.reg.record_version(ab["agent_id"], dict(
                    version=n, parent_version=None, kind="initial", reason="ablation definition (no training)", changes={},
                    training=dict(algorithm="none (ablation of a trained agent)"), checkpoints=[], final_checkpoint=ck,
                    history={}, code_version=code_version()))

    def training_complete(self) -> bool:
        m = self.manifest()
        return all(self.trained(a["agent_id"]) for a in list(m["agents"].values()) + list(m["ablations"].values()))

    # ------------------------------------------------------------------ evaluation
    def evaluation_jobs(self) -> list[dict]:
        m = self.manifest()
        jobs = []
        for key, a in m["agents"].items():
            v = self.trained(a["agent_id"])
            if v is None:
                continue
            done = {(e["round_id"], e["version"]) for e in self.reg.evaluations(a["agent_id"])}
            t = a["task"]
            items = [(r, None, "final") for r in final_rids(t)] + [(rid(t, "trainpaths"), None, "trainpaths")]
            for c in v["checkpoints"] + [v["final_checkpoint"]]:
                items.append((curve_rid_for_step(t, c["step"]), c["model_sha256"], f"curve {c['step']}"))
            jobs += [dict(key=key, agent_id=a["agent_id"], group=a["group"], version=v["version"], round_id=r,
                          checkpoint=sha, purpose=pu) for r, sha, pu in items if (r, v["version"]) not in done]
        for key, b in m["references_policies"].items():
            t = key.split("-")[0]                          # "S1X-oracle" -> the S1X rounds
            done = {e["round_id"] for e in self.reg.evaluations(b["agent_id"])}
            jobs += [dict(key=f"REF-{key}", agent_id=b["agent_id"], group="reference", version=1, round_id=r, checkpoint=None,
                          purpose="oracle" if key.endswith("-oracle") else "reference") for r in final_rids(t) if r not in done]
        for key, ab in m["ablations"].items():
            v = self.trained(ab["agent_id"])
            if v is None:
                continue
            t = m["agents"][key]["task"]
            done = {e["round_id"] for e in self.reg.evaluations(ab["agent_id"])}
            jobs += [dict(key=f"ABL-{key}", agent_id=ab["agent_id"], group="ablation", version=v["version"], round_id=r,
                          checkpoint=None, purpose="single-regime ablation") for r in final_rids(t) if r not in done]
        return jobs

    def _null_path(self, e: dict) -> Path:
        return self.reg.root / "analyses" / "matched-null" / f"{e['eval_id']}.json"

    def null_jobs(self) -> list[dict]:
        out = []
        for key, a in self.manifest()["agents"].items():
            for e in self.reg.evaluations(a["agent_id"]):
                if e["round_id"] in final_rids(a["task"]) and not self._null_path(e).exists():
                    out.append(dict(key=key, agent_id=a["agent_id"], eval_id=e["eval_id"]))
        return out

    def run_evaluations(self, workers: int = 4, log=print) -> None:
        from .evaluate import pinned_criteria
        from .metrics import check_q2
        self.check_frozen()
        if check_q2(self.reg, require_frozen=True)["checksum"] != self.manifest()["decision_quality"]["checksum"]:
            raise RegistryError("the decision-quality score differs from the one in the experiment plan")
        pinned_criteria(self.reg)
        jobs = self.evaluation_jobs()
        log(f"evaluations: {len(jobs)}")
        with mp.get_context("fork").Pool(workers) as pool:
            for job, rec in pool.imap_unordered(_eval_job, [(str(self.reg.root), j) for j in jobs]):
                self._log("evaluated", agent_key=job["key"], round_id=job["round_id"], eval_id=rec["eval_id"], purpose=job["purpose"])
            jobs = self.null_jobs()
            log(f"matched random agents: {len(jobs)} evaluation sets")
            for job in pool.imap_unordered(_null_job, [(str(self.reg.root), j) for j in jobs]):
                self._log("null_computed", agent_key=job["key"], eval_id=job["eval_id"])

    def evaluations_complete(self) -> bool:
        return self.training_complete() and not self.evaluation_jobs() and not self.null_jobs()

    # ------------------------------------------------------------------ analysis
    def analyse(self) -> dict:
        import pandas as pd
        from . import m2_analysis as m2a
        from . import synthetic_tasks as st
        from .evaluate import decision_log, mapper_from
        from .quality import KnownMarket
        self.check_frozen()
        m = self.manifest()
        for k, ref in m["references"].items():
            if hashlib.sha256((ROOT / ref["file"]).read_bytes()).hexdigest() != ref["sha256"]:
                raise RegistryError(f"reference record {k} changed")

        def gstats(evals):
            g = m3a.growth_stats(evals)
            g["growth_runs"] = an.growth_runs(evals)
            return g

        refs, oracle = {}, {}
        for key, b in m["references_policies"].items():
            t = key.split("-")[0]
            ev = {e["round_id"]: e for e in self.reg.evaluations(b["agent_id"])}
            fin = [ev[r] for r in final_rids(t)]
            if key.endswith("-oracle"):
                oracle[t] = dict(gstats(fin), kind="oracle")
            elif t == "S6":
                refs[t] = dict(m2a.benchmark_metrics(fin, b["size"] or 0.0), died_runs=m3a.growth_stats(fin)["died_runs"])
            else:
                refs[t] = dict(gstats(fin), kind=b["kind"])
        ablation = {}
        for key, ab in m["ablations"].items():
            ev = [e for e in self.reg.evaluations(ab["agent_id"]) if e["round_id"] in final_rids(m["agents"][key]["task"])]
            if len(ev) == FINAL_ROUNDS:
                ablation[ab["ablation_of"]] = gstats(ev)
        seeds = {g: [] for g in m["groups"]}
        for key, a in m["agents"].items():
            aid, t = a["agent_id"], a["task"]
            v = self.trained(aid)
            evals = self.reg.evaluations(aid)
            finals = sorted([e for e in evals if e["version"] == v["version"] and e["round_id"] in final_rids(t)],
                            key=lambda e: e["round_id"])
            logs = pd.concat([decision_log(self.reg, e).assign(round=e["round_id"]) for e in finals], ignore_index=True)
            nulls = {e["eval_id"]: read_json(self._null_path(e)) for e in finals if self._null_path(e).exists()}
            bounds = mapper_from(self.reg.identity(aid)).size_bounds
            curve = {}
            for e in evals:
                if e["round_id"].startswith(rid(t, "curve-")):
                    npts = int(e["round_id"].rsplit("-", 1)[1].rstrip("k")) * 1000
                    g = [ep["log_growth"] for ep in e["episodes"]]
                    curve[npts] = dict(growth=float(np.mean(g)), deaths=int(sum(ep["died"] for ep in e["episodes"])))
                    if t == "S6":
                        curve[npts].update(m2a.curve_point(e, decision_log(self.reg, e), KnownMarket.from_task(st.build(t, 0, 1000))))
            if t == "S6":
                tp = next((e for e in evals if e["round_id"] == rid(t, "trainpaths")), None)
                s = m2a.seed_metrics(finals, logs, nulls, tp, KnownMarket.from_task(st.build(t, 0, 1000)), optimal_size(t),
                                     curve, bounds)
                s.update(costs_per_run=m3a.growth_stats(finals)["costs_per_run"])
            else:
                params = st.build(t, 0, 1000).params
                s = m3a.seed_metrics_cost(t, finals, logs, nulls, bounds, params,
                                          mk3.true_rho(t, params) if t == "S1C" else 0.0)
                s["curve"] = curve
                s["growth_runs"] = an.growth_runs(finals)
            fc = v["final_checkpoint"]
            s.update(key=key, agent_id=aid, seed=a["seed"], seed_index=a["seed_index"], task=t, selection=fc.get("selection", {}),
                     shrinkage=fc.get("shrinkage"), persistence=fc.get("persistence"), regimes=fc.get("regimes"))
            seeds[a["group"]].append(s)
        return dict(seeds=seeds, references=refs, oracle=oracle, ablation=ablation,
                    verdicts=an.verdicts(seeds, refs, oracle, ablation))


# ---------------------------------------------------------------------------- workers
def train_agent(reg: Registry, aid: str, seed: int) -> dict:
    import platform
    import pandas as pd
    import torch
    from . import synthetic_tasks as st
    from .history import frame_bytes
    torch.set_num_threads(1)
    ident = reg.identity(aid)
    t0 = time.time()
    n = reg.start_training(aid, None, None, {}, info=dict(seed=seed, data_points=list(DATA_POINTS)), allow_parallel=True)
    task = st.build(ident["task"], seed, TRAIN_DATA_STEPS)
    Xm, y, cols = mk.collect(task, ident)
    X = mk.Features(mk._encoder(ident)).from_market_columns(Xm, cols)
    cks, hists = [], []
    for i, npts in enumerate(DATA_POINTS):
        npts = min(npts, len(y))
        art, hist = mk.learn(X[:npts], y[:npts], seed)
        sha = reg.blobs.put(mk.to_bytes(art), "pkl")
        mo = art["model"]
        regimes = [dict(beta=[float(v) for v in mo["beta"][k]], vol_scale=float(mo["c"][k]),
                        weights=[float(v) for v in mo["pi"][k]], mean_length=float(1 / max(1 - mo["A"][k, k], 1e-12)))
                   for k in range(art["K"])]
        cks.append(dict(step=int(npts), model_sha256=sha, state_sha256=None, kind="final" if i == len(DATA_POINTS) - 1 else "periodic",
                        selection=art["selection"], shrinkage=art.get("shrinkage"), persistence=art.get("persistence"),
                        regimes=regimes, note="world model learned from the first {:,} bars of experience".format(npts)))
        hists += [dict({k: (v if not isinstance(v, list) else str(v)) for k, v in h.items()}, data_points=int(npts)) for h in hist]
    rec = dict(
        version=n, parent_version=None, kind="initial", reason=f"{EXPERIMENT_ID}: model-based agent v4 learning from experience",
        changes={}, recovered_from=None, parent_checkpoint=None,
        training=dict(algorithm="model-based v4: regime-switching world model (Baum-Welch; number of regimes by held-out "
                                "likelihood), tracked by forward filtering while trading; version 3's cost-aware decision",
                      library=dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version()),
                      steps_added=TRAIN_DATA_STEPS, step_start=0, step_end=TRAIN_DATA_STEPS, wall_seconds=round(time.time() - t0, 1),
                      seed=seed, model_config=dict(mk.CONFIG), market_columns=int(len(cols)),
                      train_data=[dict(task=task.name, seed=seed, data_hash=task.md.content_hash(), steps=TRAIN_DATA_STEPS)]),
        checkpoints=cks[:-1], final_checkpoint=cks[-1],
        history=dict(episodes_sha256=None, updates_sha256=reg.blobs.put(frame_bytes(pd.DataFrame(hists)), "parquet"),
                     episodes=0, updates=len(hists), complete=True, note="component and regime selection history per amount of experience"),
        code_version=code_version(),
    )
    return reg.record_version(aid, rec)


def _train_job(args):
    root, job = args
    reg = Registry(root)
    try:
        return job, train_agent(reg, job["agent_id"], job["seed"])
    except BaseException as e:
        started = [x for x in reg.events(job["agent_id"]) if x["kind"] == "training_started"]
        if started:
            reg.abandon_training(job["agent_id"], started[-1]["version"], f"{type(e).__name__}: {e}")
        raise


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


TITLES = {"MK4-S1X": "S1X: bounce-back pattern that flips to momentum and back at unannounced times",
          "MK4-S2X": "S2X: announced trends whose meaning flips at unannounced times",
          "MK4-S6X": "S6X: crash frequency switching between calm and storm spells",
          "MK4-S1C": "S1C: no change (regression check against version 3)",
          "MK4-S6": "S6: no change, no costs (regression check against version 2)"}


def report(exp: Experiment, res: dict, protection: dict | None = None) -> str:
    m = exp.manifest()
    V, S = res["verdicts"], res["seeds"]
    yn = lambda b: "**yes**" if b else "no"
    L = [f"# Experiment M4: model-based growth-optimal agent v4, markets that change ({EXPERIMENT_ID})\n",
         f"Pre-registered plan: `registry-m4/experiments/{EXPERIMENT_ID}/manifest.json` (planned {m['planned_at']}); "
         "configuration and code checksums verified identical before training, evaluation and this analysis.\n",
         "Each agent learns from its own fresh 200,000-bar market a regime-switching world model: how many hidden regimes "
         "the market has (one = it does not change), how each behaves (pattern, trend response, volatility, crash frequency) "
         "and how long each lasts. While trading it updates its belief about the current regime with every bar it sees and "
         "decides as version 3 (growth after costs). Nothing about the market is given to it. Final checkpoints only; six fresh "
         "rounds of 20 runs (120 runs per agent). The true-model filter knows the market's rules and infers the regime like the "
         "agent; the oracle is told the regime (an upper bound).\n",
         f"**M4 passes (all five markets): {'YES' if V['M4_passes'] else 'NO'}**\n"]
    for a in exp.amendments():
        L.append(f"**Protocol amendment {a['number']:04d}** ({a['at']}): {a['reason']} Effect: {a['effect']} State when made: "
                 + ", ".join(f"{k} {v}" for k, v in a["state_when_made"].items()) + ".\n")
    for g in ("MK4-S1X", "MK4-S2X", "MK4-S6X", "MK4-S1C", "MK4-S6"):
        r = V[g]
        ref = r["reference"]
        L += [f"## {TITLES[g]}\n",
              f"Successful seeds: **{r['successes']}/6** → market {'**passes**' if r['passes'] else '**does not pass**'}.\n",
              "Criterion counts: " + ", ".join(f"{k} {v}/6" for k, v in r["counts"].items()) + ".\n",
              f"Regimes found: " + ", ".join(f"{v} agents chose {k}" for k, v in r["regimes_chosen"].items() if v) + ". "
              f"Group growth per run {r['growth_mean']:+.3f} (95% CI {r['growth_ci'][0]:+.3f}, {r['growth_ci'][1]:+.3f}); "
              f"{r['deaths_total']} deaths in 720 runs.\n",
              f"Reference ({ref.get('kind', 'exact optimal constant size')}): growth {ref['growth']:+.3f} per run "
              f"(95% CI {ref['growth_ci'][0]:+.3f}, {ref['growth_ci'][1]:+.3f}), {ref['deaths']} deaths in {ref['runs']} runs.\n"]
        if r.get("oracle"):
            o = r["oracle"]
            L.append(f"Oracle (told the current regime; upper bound): growth {o['growth']:+.3f} per run, {o['deaths']} deaths in "
                     f"{o['runs']} runs. Own single-regime model (ablation): growth {r['ablation']['growth_mean']:+.3f} per run, "
                     f"{r['ablation']['deaths_total']} deaths in 720 runs.\n")
        if g == "MK4-S6":
            L += ["| Seed | Agent | Regimes | Median size | Ruinous | Deaths | Growth per run (95% CI) | Rounds beating random | Q2 | Success |",
                  "|" + "---|" * 10]
            for i, s in enumerate(sorted(S[g], key=lambda x: x["seed_index"])):
                f = r["flags"][i]
                L.append(f"| {s['seed_index']} ({s['seed']}) | {s['agent_id']} | {s['selection'].get('regimes')} | {s['median_signed']:+.2f}x | "
                         f"{100 * s['ruinous']:.2f}% | {s['deaths']} | {s['growth']:+.4f} ({s['growth_ci'][0]:+.4f}, {s['growth_ci'][1]:+.4f}) | "
                         f"{s['rounds_beating_null']}/6 | {_f(s['q2'])} | {yn(f['success'])} |")
        else:
            L += ["| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | "
                  "Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |",
                  "|" + "---|" * 11]
            for i, s in enumerate(sorted(S[g], key=lambda x: x["seed_index"])):
                f = r["flags"][i]
                dt = s.get("death_test")
                dts = f" ({dt['agent_only']}/{dt['reference_only']}, p={dt['p']:.2f})" if dt else ""
                ad = s.get("adaptation")
                ads = f"{ad['difference']:+.3f} ({ad['ci'][0]:+.3f}, {ad['ci'][1]:+.3f})" if ad else "—"
                reg = s.get("regimes") or []
                rgs = f"{len(reg)} ({', '.join(f'{x['mean_length']:.0f}' for x in reg)})" if reg else "—"
                failed = ", ".join(k for k, ok in f.items() if k != "success" and not ok) or "—"
                L.append(f"| {s['seed_index']} ({s['seed']}) | {s['agent_id']} | {rgs} | {s['median_size']:.1f}x / {s['max_size']:.1f}x | "
                         f"{100 * s['changes']:.0f}% | {s['deaths']}{dts} | {s['growth']:+.3f} ({s['growth_ci'][0]:+.3f}, {s['growth_ci'][1]:+.3f}) | "
                         f"{ads} | {s['rounds_beating_null']}/6 | {failed} | {yn(f['success'])} |")
        L.append("")
    L += ["## Limits\n",
          "- Six seeds per market; intervals show the range consistent with the data.",
          "- Synthetic markets with one instrument and known structure; real historical data is the next test.",
          "- The world model is linear in the last three returns and the visible side features within each regime; version 3's "
          "non-linear network is not used by version 4.",
          "- Full growth-optimal sizing of a weak signal that can silently reverse dies often even with the true rules (S2X); "
          "whether a live account should bet a fraction of it is the owner's risk decision.",
          "- References are given the true market rules and are references only, never agents.",
          "- Verdicts are the pre-registered rules applied mechanically; nothing was changed after training began.",
          "- Every agent, model, checkpoint and evaluation is preserved."]
    if protection:
        L += ["", "## Protection of earlier work\n"] + [f"- {k}: {v}" for k, v in protection.items()]
    return "\n".join(L) + "\n"
