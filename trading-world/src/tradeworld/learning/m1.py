"""Experiment M1: the model-based growth-optimal agent (model_kelly) on S6 and S7.

Groups (six fresh seeds each; each agent experiences its own fresh 200,000-bar market, the same amount of
market data each PPO agent trained on):
  MK-S6   S6 (drift plus rare crashes; growth-optimal size 2.93x)
  MK-S7   S7 (drift, no crashes; growth-optimal size 5x)
Each agent is trained on the first 20,000, 60,000 and 200,000 bars of its experience (checkpoints at those
amounts of data), so learning from experience can be measured. The final checkpoint (200,000 bars) is the
agent that is evaluated; the earlier ones only form the learning curve. Separate registry (registry-m1/),
separate blob store, data ledger inheriting every earlier ledger (read only).
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from ..registry.datasets import DataLedger
from ..registry.registry import ROOT, Registry, RegistryError, code_version, now_utc
from ..registry.store import append_jsonl, canonical_json, read_json, write_once_json
from . import m1_analysis as an
from . import model_kelly as mk
from . import phase3_analysis as p3a

EXPERIMENT_ID = "phase3-m1"
PROFILE = "A_synthetic"
ENV_SETTINGS = dict(decision_interval=1, obs_resolution=1, window=30)
EPISODE_DECISIONS = 1000
TRAIN_DATA_STEPS = 200_000
DATA_POINTS = (20_000, 60_000, 200_000)
SEEDS_PER_GROUP = 6
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None, size_repr="log10")
FROZEN_CODE = (
    "src/tradeworld/learning/model_kelly.py", "src/tradeworld/learning/m1.py", "src/tradeworld/learning/m1_analysis.py",
    "src/tradeworld/learning/evaluate.py", "src/tradeworld/learning/env.py", "src/tradeworld/learning/diagnosis.py",
    "src/tradeworld/learning/quality.py", "src/tradeworld/learning/synthetic_tasks.py", "src/tradeworld/learning/gate.py",
    "src/tradeworld/learning/phase3_analysis.py", "src/tradeworld/learning/phase3b_analysis.py",
    "src/tradeworld/interface/actions.py", "src/tradeworld/registry/registry.py", "src/tradeworld/registry/datasets.py",
)


def groups() -> dict:
    return {g: dict(task=t, agent_type="model-kelly", reward="R1", action_space=LOG10, model_config=dict(mk.CONFIG),
                    data_points=list(DATA_POINTS), description=d)
            for g, t, d in (("MK-S6", "S6", "model-based growth-optimal agent on S6 (crashes)"),
                            ("MK-S7", "S7", "model-based growth-optimal agent on S7 (no crashes)"))}


def code_checksums() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN_CODE}


def config_checksum(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg)).hexdigest()


def rid(task: str, kind: str) -> str:
    return f"m1-{task.lower()}-{kind}"


def curve_rid(task: str, n: int) -> str:
    return rid(task, f"curve-{n // 1000:03d}k")


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
        m = self.manifest()
        bad = [f for f, h in code_checksums().items() if m["frozen_code_sha256"].get(f) != h]
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
        agents, rounds = {}, {}
        for g, cfg in G.items():
            t = cfg["task"]
            for k in range(SEEDS_PER_GROUP):
                seed = ledger.allocate(t, "train", note=f"{EXPERIMENT_ID} {g} seed {k + 1}: the agent's market experience")
                aid = self.reg.create_agent(
                    task=t, reward=cfg["reward"], seed=seed, env_settings=ENV_SETTINGS, action_space=cfg["action_space"],
                    profile=PROFILE, episode_decisions=EPISODE_DECISIONS, name=f"M1-{g}-s{k + 1}", origin=EXPERIMENT_ID,
                    notes=f"Experiment M1, group {g}: {cfg['description']}",
                    extra=dict(agent_type="model-kelly", experiment=EXPERIMENT_ID, group=g, seed_index=k + 1,
                               model_config=cfg["model_config"], config_sha256=config_checksum(cfg)))
                agents[f"{g}-s{k + 1}"] = dict(agent_id=aid, group=g, task=t, seed_index=k + 1, seed=seed)
        for t in ("S6", "S7"):
            for kind, purpose in (("final-a", "final evaluation, round A"), ("final-b", "final evaluation, round B"),
                                  ("dev", "learning curve over amount of experience (descriptive and W7')")):
                r = rid(t, kind)
                seed = ledger.allocate(t, "test", round_id=r, note=f"{EXPERIMENT_ID} {t} {purpose}")
                data = [dict(task=t, seed=seed, data_hash=st.build(t, seed, TEST_STEPS_DATA).md.content_hash())]
                if kind == "dev":
                    for n in DATA_POINTS:
                        self.reg.record_round(curve_rid(t, n), dict(kind="fresh", data=data, test_steps=TEST_STEPS_DATA,
                                                                    note=f"{EXPERIMENT_ID} {t}: after {n:,} bars of experience"))
                    rounds[rid(t, "curve")] = {curve_rid(t, n): seed for n in DATA_POINTS}
                else:
                    self.reg.record_round(r, dict(kind="fresh", data=data, test_steps=TEST_STEPS_DATA, note=f"{EXPERIMENT_ID} {t} {purpose}"))
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
            seeds_per_group=SEEDS_PER_GROUP, groups={g: dict(cfg, config_sha256=config_checksum(cfg)) for g, cfg in G.items()},
            agents=agents, rounds=rounds, evaluated_checkpoint="each agent's final checkpoint (200,000 bars; no selection)",
            decision_quality=dict(q2, definition=Q2_DEFINITION, note="exact for S6; provisional for S7 (reported only)"),
            analysis=dict(file="src/tradeworld/learning/m1_analysis.py",
                          criteria=[ln.strip() for ln in an.__doc__.splitlines() if ln.strip()]),
            references=refs, frozen_code_sha256=code_checksums(), approval_conditions=approval, compute=compute,
            storage=storage,
            frozen=("market mechanics, R1-R6, scoring rules, simulator, margin, death rule, Q2.0; no exposure cap, no penalty, "
                    "no trading style; nothing about the market is given to the agent."),
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
                self._log("trained", agent_key=job["key"], version=rec["version"])
                log(f"  {job['key']} -> v{rec['version']} ({rec['training']['wall_seconds']:.0f}s, "
                    f"final model {rec['final_checkpoint']['epochs']} epochs)")
                if on_done:
                    on_done(job["key"])

    def training_complete(self) -> bool:
        return all(self.trained(a["agent_id"]) for a in self.manifest()["agents"].values())

    # ------------------------------------------------------------------ evaluation
    def evaluation_jobs(self) -> list[dict]:
        jobs = []
        for key, a in self.manifest()["agents"].items():
            v = self.trained(a["agent_id"])
            if v is None:
                continue
            done = {(e["round_id"], e["version"]) for e in self.reg.evaluations(a["agent_id"])}
            t = a["task"]
            items = [(rid(t, "final-a"), None, "final"), (rid(t, "final-b"), None, "final"), (rid(t, "trainpaths"), None, "trainpaths")]
            for c in v["checkpoints"] + [v["final_checkpoint"]]:
                items.append((curve_rid(t, c["step"]), c["model_sha256"], f"curve {c['step']}"))
            jobs += [dict(key=key, agent_id=a["agent_id"], group=a["group"], version=v["version"], round_id=r,
                          checkpoint=sha, purpose=pu) for r, sha, pu in items if (r, v["version"]) not in done]
        return jobs

    def _null_path(self, e: dict) -> Path:
        return self.reg.root / "analyses" / "matched-null" / f"{e['eval_id']}.json"

    def null_jobs(self) -> list[dict]:
        out = []
        for key, a in self.manifest()["agents"].items():
            for e in self.reg.evaluations(a["agent_id"]):
                if e["round_id"] in (rid(a["task"], "final-a"), rid(a["task"], "final-b")) and not self._null_path(e).exists():
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
        from . import phase3b_analysis as w
        from . import synthetic_tasks as st
        from .evaluate import decision_log, mapper_from
        from .quality import KnownMarket
        self.check_frozen()
        m = self.manifest()
        for k, ref in m["references"].items():
            if hashlib.sha256((ROOT / ref["file"]).read_bytes()).hexdigest() != ref["sha256"]:
                raise RegistryError(f"reference record {k} changed")
        seeds, curves = {g: [] for g in m["groups"]}, {}
        for key, a in m["agents"].items():
            aid, t = a["agent_id"], a["task"]
            market = KnownMarket.from_task(st.build(t, 0, 1000))
            g_star = market.optimal()[0]
            v = self.trained(aid)
            evals = self.reg.evaluations(aid)
            finals = sorted([e for e in evals if e["version"] == v["version"] and e["round_id"] in (rid(t, "final-a"), rid(t, "final-b"))],
                            key=lambda e: e["round_id"])
            tp = next((e for e in evals if e["round_id"] == rid(t, "trainpaths")), None)
            logs = pd.concat([decision_log(self.reg, e).assign(round=e["round_id"]) for e in finals], ignore_index=True)
            nulls = [read_json(self._null_path(e)) for e in finals if self._null_path(e).exists()]
            s = p3a.seed_metrics(finals, logs, nulls, tp, market, g_star)
            s["safeguard_share"] = w.safeguard_share(logs, mapper_from(self.reg.identity(aid)).size_bounds)
            s["time_in_market"] = float(np.mean(logs["leverage_after_fill"].to_numpy() > 1e-6))
            s["size_distance"] = abs(math.log(max(s["median_size"], 1e-12) / g_star))
            s["g_star"] = g_star
            curve = {}
            for e in evals:
                if e["round_id"].startswith(rid(t, "curve-")):
                    lg = decision_log(self.reg, e)
                    n = int(e["round_id"].rsplit("-", 1)[1].rstrip("k")) * 1000
                    curve[n] = dict(q2=e["diagnosis"]["metrics"]["dq2"], median=e["diagnosis"]["metrics"]["median_leverage"],
                                    ruinous=float(market.ruinous(lg["exposure_after_fill"].to_numpy()).mean()),
                                    deaths=int(sum(ep["died"] for ep in e["episodes"])))
            s["curve"] = curve
            s.update(an.learning_curve_flag(curve, s))
            s["model"] = dict(epochs=v["final_checkpoint"].get("epochs"), best_val_nll=v["final_checkpoint"].get("best_val_nll"))
            s.update(key=key, agent_id=aid, seed=a["seed"], seed_index=a["seed_index"], versions=[v["version"]])
            seeds[a["group"]].append(s)
            curves.setdefault(a["group"], []).append(curve)
        refs = {k: read_json(ROOT / ref["file"]) for k, ref in m["references"].items()}
        return dict(seeds=seeds, curves=curves, verdicts=an.verdicts(seeds, refs))


# ---------------------------------------------------------------------------- workers
def train_agent(reg: Registry, aid: str, seed: int) -> dict:
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
        art, hist = mk.train(X[:npts], y[:npts], seed)
        sha = reg.blobs.put(mk.to_bytes(art), "pt")
        cks.append(dict(step=int(npts), model_sha256=sha, state_sha256=None, epochs=art["epochs"],
                        best_val_nll=art["best_val_nll"], kind="final" if i == len(DATA_POINTS) - 1 else "periodic",
                        note="world model trained on the first {:,} bars of experience".format(npts)))
        hists += [dict(h, data_points=int(npts)) for h in hist]
    up = pd.DataFrame(hists)
    import platform
    rec = dict(
        version=n, parent_version=None, kind="initial", reason=f"{EXPERIMENT_ID}: model-based agent learning from experience",
        changes={}, recovered_from=None, parent_checkpoint=None,
        training=dict(algorithm="model-based: mixture-density world model (maximum likelihood, early stopping) + expected-log-growth sizing",
                      library=dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version()),
                      steps_added=TRAIN_DATA_STEPS, step_start=0, step_end=TRAIN_DATA_STEPS, wall_seconds=round(time.time() - t0, 1),
                      seed=seed, model_config=dict(mk.CONFIG), market_columns=int(len(cols)),
                      train_data=[dict(task=task.name, seed=seed, data_hash=task.md.content_hash(), steps=TRAIN_DATA_STEPS)]),
        checkpoints=cks[:-1], final_checkpoint=cks[-1],
        history=dict(episodes_sha256=None, updates_sha256=reg.blobs.put(frame_bytes(up), "parquet"), episodes=0,
                     updates=len(up), complete=True, note="model fitting history per amount of experience"),
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
    L = [f"# Experiment M1: model-based growth-optimal agent ({EXPERIMENT_ID})\n",
         f"Pre-registered plan: `registry-m1/experiments/{EXPERIMENT_ID}/manifest.json` (planned {m['planned_at']}); "
         "configuration and code checksums verified identical before training, evaluation and this analysis.\n",
         "Each agent watches its own fresh 200,000-bar market (the same amount of market data each PPO agent trained on), "
         "learns the distribution of the next bar's return from what it saw (mixture-density network, maximum likelihood, "
         "early stopping), and sizes every decision to maximise expected log growth under what it learned. Nothing about the "
         "market is given to it. Final checkpoints only; two fresh rounds of 20 test runs.\n",
         "## S6 (crashes): pre-registered verdict\n",
         "| W1 ruin<1% | W2 deaths≤1 | W3 size 1.47–5.86x | W4 Q2≥0.5 | W5 growth | W6 tilt<1.5 | W7' learns from experience | E | "
         "**Successful seeds** | **Interpretation** |", "|---|---|---|---|---|---|---|---|---|---|"]
    r = V["MK-S6"]
    n = r["counts"]
    L.append(f"| {n['W1']}/6 | {n['W2']}/6 | {n['W3']}/6 | {n['W4']}/6 | {n['W5']}/6 | {n['W6']}/6 | {n['W7']}/6 | {n['E']}/6 | "
             f"**{r['successes']}/6** | **{r['interpretation']}** |")
    L += ["", f"Group means (95% bootstrap interval): Q2 {r['q2_mean']:+.2f} ({r['q2_ci'][0]:+.2f}, {r['q2_ci'][1]:+.2f}); "
              f"ruinous {100 * r['ruinous_mean']:.2f}% ({100 * r['ruinous_ci'][0]:.2f}, {100 * r['ruinous_ci'][1]:.2f}); "
              f"median size {r['median_size_mean']:.2f}x ({r['median_size_ci'][0]:.2f}, {r['median_size_ci'][1]:.2f}); "
              f"test deaths {r['deaths_total']} in {6 * 40} runs.\n",
          "Against earlier PPO agents (historical references; Holm over 6 comparisons):\n",
          "| Comparison | MK-S6 minus reference (95% CI) | Reference mean | p raw / Holm |", "|---|---|---|---|"]
    for k, c in r["comparisons"].items():
        L.append(f"| {k} | {c['difference']:+.3f} ({c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}) | {c['reference_mean']:+.3f} | "
                 f"{c['p_raw']:.3f} / {c['p_holm']:.3f} |")
    for g, title in (("MK-S6", "S6"), ("MK-S7", "S7 (generality check; growth-optimal size 5x; Q2 provisional, not used)")):
        if g == "MK-S7":
            r7 = V["MK-S7"]
            n7 = r7["counts"]
            L += ["", f"## {title}\n",
                  "| W1 | W2 | W3 size 2.5–10x | W5 growth | W6 | W7'' | E | **Successful seeds** | **Passes (≥4/6)** |",
                  "|---|---|---|---|---|---|---|---|---|",
                  f"| {n7['W1']}/6 | {n7['W2']}/6 | {n7['W3']}/6 | {n7['W5']}/6 | {n7['W6']}/6 | {n7['W7']}/6 | {n7['E']}/6 | "
                  f"**{r7['successes']}/6** | **{'yes' if r7['passes'] else 'no'}** |"]
        L += ["", f"### Every {g} agent\n",
              "| Seed | Agent | Q2 | Ruinous | Deaths | Median size | 95th pct | Largest | Long | Growth per run (95% CI) | "
              "Beats random A/B | Tilt | Train-path gap | Learning curve: median size / Q2 at 20k → 60k → 200k bars | Success |",
              "|" + "---|" * 15]
        for i, s in enumerate(sorted(S[g], key=lambda x: x["seed_index"])):
            f = V[g]["flags"][i]
            cv = s["curve"]
            cc = " → ".join(f"{cv[k]['median']:.2f}x / {cv[k]['q2']:+.2f}" for k in sorted(cv))
            L.append(f"| {s['seed_index']} ({s['seed']}) | {s['agent_id']} | {s['q2']:+.2f} | {100 * s['ruinous']:.2f}% | {s['deaths']} | "
                     f"{s['median_size']:.2f}x | {s['p95_size']:.2f}x | {s['max_size']:.2f}x | {100 * s['long_share']:.0f}% | "
                     f"{s['growth']:+.4f} ({s['growth_ci'][0]:+.4f}, {s['growth_ci'][1]:+.4f}) | "
                     f"{'/'.join('y' if b['beats'] else 'n' for b in s['null'])} | {_f(s['tilt'], '{:.2f}')} | {_f(s['train_gap'])} | "
                     f"{cc} | {'**yes**' if f['success'] else 'no'} |")
    L += ["", "## Limits\n",
          "- Six seeds per market; intervals show the range consistent with the data.",
          "- The PPO references used different test data; Q2 depends only on the decisions.",
          "- S6 and S7 have i.i.d. returns and no trading costs; markets with predictable structure and costs are the next test.",
          "- Verdicts are the pre-registered rules applied mechanically; nothing was changed after training began.",
          "- Every agent, model, checkpoint and evaluation is preserved."]
    if protection:
        L += ["", "## Protection of earlier work\n"] + [f"- {k}: {v}" for k, v in protection.items()]
    return "\n".join(L) + "\n"
