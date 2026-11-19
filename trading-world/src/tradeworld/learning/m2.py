"""Experiment M2: model-based growth-optimal agent, version 2 (model_kelly2), on S6, S7 and S4.

Groups (six fresh seeds each; each agent learns from its own fresh 200,000-bar market experience):
  MK2-S6  drift plus rare crashes (growth-optimal size 2.93x)
  MK2-S7  drift, no crashes (growth-optimal size 5x)
  MK2-S4  no edge at all (the right behaviour is to claim none)
Test power was sized from Experiment M1's power analysis: six fresh final rounds of 20 runs (120 runs per agent).
Benchmarks (not agents): for each market, the exact growth-optimal constant size computed from the market's known
parameters (S4: flat) is run on the same test runs, so deaths and growth are judged against what optimal play
achieves on that very data. Separate registry (registry-m2/), blob store and ledger inheritance as before.
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
from . import m2_analysis as an
from . import model_kelly2 as mk

EXPERIMENT_ID = "phase3-m2"
PROFILE = "A_synthetic"
ENV_SETTINGS = dict(decision_interval=1, obs_resolution=1, window=30)
EPISODE_DECISIONS = 1000
TRAIN_DATA_STEPS = 200_000
DATA_POINTS = (20_000, 60_000, 200_000)
SEEDS_PER_GROUP = 6
FINAL_ROUNDS = 6
TASKS = ("S6", "S7", "S4")
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
FROZEN_CODE = (
    "src/tradeworld/learning/model_kelly2.py", "src/tradeworld/learning/model_kelly.py", "src/tradeworld/learning/m2.py",
    "src/tradeworld/learning/m2_analysis.py", "src/tradeworld/learning/evaluate.py", "src/tradeworld/learning/env.py",
    "src/tradeworld/learning/diagnosis.py", "src/tradeworld/learning/quality.py", "src/tradeworld/learning/synthetic_tasks.py",
    "src/tradeworld/learning/gate.py", "src/tradeworld/learning/phase3_analysis.py",
    "src/tradeworld/learning/phase3b_analysis.py", "src/tradeworld/interface/actions.py",
    "src/tradeworld/registry/registry.py", "src/tradeworld/registry/datasets.py",
)


def groups() -> dict:
    return {f"MK2-{t}": dict(task=t, agent_type="model-kelly-v2", reward="R1", action_space=LOG10,
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
    return f"m2-{task.lower()}-{kind}"


def final_rids(task: str) -> list[str]:
    return [rid(task, f"final-{i}") for i in range(1, FINAL_ROUNDS + 1)]


def curve_rid(task: str, n: int) -> str:
    return rid(task, f"curve-{n // 1000:03d}k")


def curve_rid_for_step(task: str, step: int) -> str:
    """Amendment 0001: a checkpoint's learning-curve round is the planned data point nearest to its step. (The last
    checkpoint learns from all 199,967 bars an agent's 200,000-bar market yields after the 30-bar window and the
    one-bar-ahead target; the plan names that round '200k'.)"""
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
        """Recorded protocol amendments (write-once, numbered, applied in order)."""
        d = self.dir / "amendments"
        return [read_json(p) for p in sorted(d.glob("[0-9][0-9][0-9][0-9].json"))] if d.exists() else []

    def frozen_code(self) -> dict:
        """The frozen code checksums: those of the manifest, as changed by each recorded amendment in turn. An
        amendment must name the checksums it replaces, so the chain cannot skip or rewrite a step."""
        frozen = dict(self.manifest()["frozen_code_sha256"])
        for a in self.amendments():
            for f, ch in a["code_changes"].items():
                if frozen.get(f) != ch["old_sha256"]:
                    raise RegistryError(f"amendment {a['number']} does not follow the recorded checksum of {f}")
                frozen[f] = ch["new_sha256"]
        return frozen

    def amend(self, reason: str, changed: dict[str, str], state: dict, effect: str) -> dict:
        """Record an amendment. `changed` maps each changed frozen file to its checksum before the change."""
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
    def plan(self, approval: list[str], references: dict, compute: dict, storage: dict) -> dict:
        from . import synthetic_tasks as st
        from .evaluate import TEST_STEPS_DATA
        from .metrics import Q2_DEFINITION, check_q2
        if (self.dir / "manifest.json").exists():
            raise RegistryError("already planned")
        q2 = check_q2(self.reg, require_frozen=True)
        ledger = DataLedger(self.reg.root)
        G = groups()
        agents, rounds, benchmarks = {}, {}, {}
        for g, cfg in G.items():
            t = cfg["task"]
            for k in range(SEEDS_PER_GROUP):
                seed = ledger.allocate(t, "train", note=f"{EXPERIMENT_ID} {g} seed {k + 1}: the agent's market experience")
                aid = self.reg.create_agent(
                    task=t, reward=cfg["reward"], seed=seed, env_settings=ENV_SETTINGS, action_space=cfg["action_space"],
                    profile=PROFILE, episode_decisions=EPISODE_DECISIONS, name=f"M2-{g}-s{k + 1}", origin=EXPERIMENT_ID,
                    notes=f"Experiment M2, group {g}: model-based growth-optimal agent v2 on {t}",
                    extra=dict(agent_type="model-kelly-v2", experiment=EXPERIMENT_ID, group=g, seed_index=k + 1,
                               model_config=cfg["model_config"], config_sha256=config_checksum(cfg)))
                agents[f"{g}-s{k + 1}"] = dict(agent_id=aid, group=g, task=t, seed_index=k + 1, seed=seed)
            size = optimal_size(t)
            bid = self.reg.create_agent(
                task=t, reward="R1", seed=0, env_settings=ENV_SETTINGS, action_space=LOG10, profile=PROFILE,
                episode_decisions=EPISODE_DECISIONS, name=f"M2-BENCH-{t}", origin=EXPERIMENT_ID,
                notes=f"Benchmark, not an agent: always holds the exact growth-optimal size {size:.4f}x computed from {t}'s "
                      "known parameters. Learns nothing; used only as a reference on the same test runs.",
                extra=dict(agent_type="benchmark-constant", experiment=EXPERIMENT_ID, benchmark_size=size))
            n = self.reg.start_training(bid, None, None, {}, info=dict(benchmark=True))
            sha = self.reg.blobs.put(json.dumps(dict(benchmark_size=size, task=t)).encode(), "json")
            ck = dict(step=0, model_sha256=sha, state_sha256=None, kind="final", note="benchmark definition (no training)")
            self.reg.record_version(bid, dict(version=n, parent_version=None, kind="initial", reason="benchmark definition",
                                              changes={}, training=dict(algorithm="none (benchmark)"), checkpoints=[],
                                              final_checkpoint=ck, history={}, code_version=code_version()))
            benchmarks[t] = dict(agent_id=bid, size=size)
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
        refs = {k: dict(file=v, sha256=hashlib.sha256((ROOT / v).read_bytes()).hexdigest()) for k, v in references.items()}
        m = dict(
            experiment_id=EXPERIMENT_ID, planned_at=now_utc(), profile=PROFILE, env_settings=ENV_SETTINGS,
            episode_decisions=EPISODE_DECISIONS, train_data_bars=TRAIN_DATA_STEPS, data_points=list(DATA_POINTS),
            seeds_per_group=SEEDS_PER_GROUP, final_rounds=FINAL_ROUNDS,
            groups={g: dict(cfg, config_sha256=config_checksum(cfg)) for g, cfg in G.items()},
            agents=agents, benchmarks=benchmarks, rounds=rounds,
            evaluated_checkpoint="each agent's final checkpoint (200,000 bars; no selection)",
            decision_quality=dict(q2, definition=Q2_DEFINITION, note="exact for S6; provisional for S7 (not used); undefined for S4"),
            analysis=dict(file="src/tradeworld/learning/m2_analysis.py",
                          criteria=[ln.strip() for ln in an.__doc__.splitlines() if ln.strip()]),
            references=refs, frozen_code_sha256=code_checksums(), approval_conditions=approval, compute=compute,
            storage=storage,
            frozen=("market mechanics, R1-R6, scoring rules, simulator, margin, death rule, Q2.0; no exposure cap, no penalty, "
                    "no trading style; nothing about the market is given to the agents (only to the labelled benchmarks)."),
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

    def training_complete(self) -> bool:
        return all(self.trained(a["agent_id"]) for a in self.manifest()["agents"].values())

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
        for t, b in m["benchmarks"].items():
            done = {e["round_id"] for e in self.reg.evaluations(b["agent_id"])}
            jobs += [dict(key=f"BENCH-{t}", agent_id=b["agent_id"], group="benchmark", version=1, round_id=r, checkpoint=None,
                          purpose="benchmark") for r in final_rids(t) if r not in done]
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
        from . import synthetic_tasks as st
        from .evaluate import decision_log, mapper_from
        from .quality import KnownMarket
        self.check_frozen()
        m = self.manifest()
        for k, ref in m["references"].items():
            if hashlib.sha256((ROOT / ref["file"]).read_bytes()).hexdigest() != ref["sha256"]:
                raise RegistryError(f"reference record {k} changed")
        bench = {}
        for t, b in m["benchmarks"].items():
            ev = {e["round_id"]: e for e in self.reg.evaluations(b["agent_id"])}
            bench[t] = an.benchmark_metrics([ev[r] for r in final_rids(t)], b["size"])
        seeds = {g: [] for g in m["groups"]}
        for key, a in m["agents"].items():
            aid, t = a["agent_id"], a["task"]
            market = KnownMarket.from_task(st.build(t, 0, 1000))
            v = self.trained(aid)
            evals = self.reg.evaluations(aid)
            finals = sorted([e for e in evals if e["version"] == v["version"] and e["round_id"] in final_rids(t)],
                            key=lambda e: e["round_id"])
            tp = next((e for e in evals if e["round_id"] == rid(t, "trainpaths")), None)
            logs = pd.concat([decision_log(self.reg, e).assign(round=e["round_id"]) for e in finals], ignore_index=True)
            nulls = {e["eval_id"]: read_json(self._null_path(e)) for e in finals if self._null_path(e).exists()}
            curve = {}
            for e in evals:
                if e["round_id"].startswith(rid(t, "curve-")):
                    npts = int(e["round_id"].rsplit("-", 1)[1].rstrip("k")) * 1000
                    lg = decision_log(self.reg, e)
                    curve[npts] = an.curve_point(e, lg, market)
            s = an.seed_metrics(finals, logs, nulls, tp, market, optimal_size(t), curve,
                                mapper_from(self.reg.identity(aid)).size_bounds)
            sel = v["final_checkpoint"].get("selection", {})
            s.update(key=key, agent_id=aid, seed=a["seed"], seed_index=a["seed_index"], task=t, selection=sel,
                     shrinkage=v["final_checkpoint"].get("shrinkage"))
            seeds[a["group"]].append(s)
        refs = {k: read_json(ROOT / ref["file"]) for k, ref in m["references"].items()}
        return dict(seeds=seeds, benchmarks=bench, verdicts=an.verdicts(seeds, bench, refs))


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
    X, y, cols = mk.collect(task, ident)
    cks, hists = [], []
    for i, npts in enumerate(DATA_POINTS):
        npts = min(npts, len(y))
        art, hist = mk.learn(X[:npts], y[:npts], seed)
        sha = reg.blobs.put(mk.to_bytes(art), "pt")
        cks.append(dict(step=int(npts), model_sha256=sha, state_sha256=None, kind="final" if i == len(DATA_POINTS) - 1 else "periodic",
                        selection=art["selection"], shrinkage=art.get("shrinkage"),
                        note="world model learned from the first {:,} bars of experience".format(npts)))
        hists += [dict(h, data_points=int(npts)) for h in hist]
    rec = dict(
        version=n, parent_version=None, kind="initial", reason=f"{EXPERIMENT_ID}: model-based agent v2 learning from experience",
        changes={}, recovered_from=None, parent_checkpoint=None,
        training=dict(algorithm="model-based v2: EM Gaussian mixture vs mixture-density network chosen by held-out likelihood; "
                                "James-Stein edge shrinkage; expected-log-growth sizing",
                      library=dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version()),
                      steps_added=TRAIN_DATA_STEPS, step_start=0, step_end=TRAIN_DATA_STEPS, wall_seconds=round(time.time() - t0, 1),
                      seed=seed, model_config=dict(mk.CONFIG), market_columns=int(len(cols)),
                      train_data=[dict(task=task.name, seed=seed, data_hash=task.md.content_hash(), steps=TRAIN_DATA_STEPS)]),
        checkpoints=cks[:-1], final_checkpoint=cks[-1],
        history=dict(episodes_sha256=None, updates_sha256=reg.blobs.put(frame_bytes(pd.DataFrame(hists)), "parquet"),
                     episodes=0, updates=len(hists), complete=True, note="model selection and fitting history per amount of experience"),
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


def report(exp: Experiment, res: dict, protection: dict | None = None) -> str:
    m = exp.manifest()
    V, S = res["verdicts"], res["seeds"]
    yn = lambda b: "**yes**" if b else "no"
    L = [f"# Experiment M2: model-based growth-optimal agent v2 ({EXPERIMENT_ID})\n",
         f"Pre-registered plan: `registry-m2/experiments/{EXPERIMENT_ID}/manifest.json` (planned {m['planned_at']}); "
         "configuration and code checksums verified identical before training, evaluation and this analysis.\n",
         "Each agent learns from its own fresh 200,000-bar market: it fits an unconditional return mixture (exactly, by EM) and "
         "a conditional mixture-density network, keeps whichever predicts held-out experience better, corrects its edge for "
         "estimation error (James-Stein), and sizes every decision for maximum expected log growth. Nothing about the market "
         "is given to it. Final checkpoints only; six fresh rounds of 20 runs (120 runs per agent). Benchmarks hold the exact "
         "growth-optimal size on the same runs.\n",
         f"**M2 passes (all three markets): {'YES' if V['M2_passes'] else 'NO'}**\n"]
    for a in exp.amendments():
        L.append(f"**Protocol amendment {a['number']:04d}** ({a['at']}; recorded before any result was analysed): {a['reason']} "
                 f"Effect: {a['effect']} State when made: " + ", ".join(f"{k} {v}" for k, v in a["state_when_made"].items()) + ".\n")
    for g, title in (("MK2-S6", "S6: drift plus rare crashes (optimal 2.93x)"), ("MK2-S7", "S7: drift, no crashes (optimal 5x)"),
                     ("MK2-S4", "S4: no edge (right behaviour: claim none)")):
        r = V[g]
        b = r["benchmark"]
        L += [f"## {title}\n",
              f"Successful seeds: **{r['successes']}/6** → market {'**passes**' if r['passes'] else '**does not pass**'}"
              + (f"; interpretation: **{r['interpretation']}**" if "interpretation" in r else "") + ".\n",
              "Criterion counts: " + ", ".join(f"{k} {v}/6" for k, v in r["counts"].items()) + ".\n",
              f"Models chosen by the agents: {r['chosen_models']['unconditional']} unconditional, "
              f"{r['chosen_models']['conditional']} conditional. Benchmark (exact optimal {b['size']:.2f}x): growth {b['growth']:+.4f} per run "
              f"(95% CI {b['growth_ci'][0]:+.4f}, {b['growth_ci'][1]:+.4f}), {b['deaths']} deaths in {b['runs']} runs.\n"]
        if g == "MK2-S6":
            L += [f"Group: Q2 {r['q2_mean']:+.2f} ({r['q2_ci'][0]:+.2f}, {r['q2_ci'][1]:+.2f}); ruinous {100 * r['ruinous_mean']:.2f}%; "
                  f"median size {r['median_size_mean']:.2f}x ({r['median_size_ci'][0]:.2f}, {r['median_size_ci'][1]:.2f}); "
                  f"{r['deaths_total']} deaths in {6 * 120} runs.\n",
                  "| Comparison (Holm over 6) | MK2-S6 minus reference (95% CI) | Reference mean | p raw / Holm |", "|---|---|---|---|"]
            L += [f"| {k} | {c['difference']:+.3f} ({c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}) | {c['reference_mean']:+.3f} | "
                  f"{c['p_raw']:.3f} / {c['p_holm']:.3f} |" for k, c in r["comparisons"].items()]
            L.append("")
        L += ["| Seed | Agent | Model | Edge t / shrink | Median size | Largest | Ruinous | Deaths | Growth per run (95% CI) | t | "
              "Rounds beating random | Q2 | Tilt | Learning curve: median size (Q2) at 20k → 60k → 200k bars | Success |",
              "|" + "---|" * 15]
        for i, s in enumerate(sorted(S[g], key=lambda x: x["seed_index"])):
            f = r["flags"][i]
            sh = s.get("shrinkage") or {}
            cv = s["curve"]
            cc = " → ".join(f"{cv[k]['median']:.2f}x ({_f(cv[k]['q2'])})" for k in sorted(cv))
            L.append(f"| {s['seed_index']} ({s['seed']}) | {s['agent_id']} | {s['selection'].get('chosen')} K={s['selection'].get('components')} | "
                     f"{_f(sh.get('t'), '{:.1f}')} / {_f(sh.get('shrink'), '{:.2f}')} | {s['median_signed']:+.2f}x | {s['max_size']:.2f}x | "
                     f"{100 * s['ruinous']:.2f}% | {s['deaths']} | {s['growth']:+.4f} ({s['growth_ci'][0]:+.4f}, {s['growth_ci'][1]:+.4f}) | "
                     f"{s['growth_t']:+.1f} | {s['rounds_beating_null']}/6 | {_f(s['q2'])} | {_f(s['tilt'], '{:.2f}')} | {cc} | {yn(f['success'])} |")
        L.append("")
    L += ["## Limits\n",
          "- Six seeds per market; intervals show the range consistent with the data.",
          "- S4, S6 and S7 have i.i.d. returns and no trading costs; predictable markets with costs are the next test.",
          "- Benchmarks are given the true market parameters and are references only, never agents.",
          "- Verdicts are the pre-registered rules applied mechanically; nothing was changed after training began.",
          "- Every agent, model, checkpoint and evaluation is preserved."]
    if protection:
        L += ["", "## Protection of earlier work\n"] + [f"- {k}: {v}" for k, v in protection.items()]
    return "\n".join(L) + "\n"
