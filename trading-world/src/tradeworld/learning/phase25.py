"""Phase 2.5 (S6 agent evolution): pre-registered continued-training experiment.

Design (approved 2026-11-17, conditions in the manifest):
  * six S6 agents: growth-scored (R1) seeds 0-2 are the subjects; reckless (R2) seeds 0-2 are
    controls. Version 1 of each is the original gate agent and is never modified.
  * arms: A = more training on the agent's original practice market; B = more training on new,
    never-used S6 market paths. Controls run arm B only.
  * three stages of 1M steps; each stage continues from the previous stage of the same arm
    (A1 <- v1, A2 <- A1, A3 <- A2; B likewise), with all learned state carried over.
  * rounds 1-3: learning curves and candidate selection. Before round 4, one candidate per growth
    seed is locked (write-once). Round 4 = confirmation only, round 5 = replication only; their
    results can never change the candidate.

Symbolic labels: v1, A1..A3, B1..B3. The progress log maps labels to registry version numbers.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from ..registry.registry import Registry, RegistryError, now_utc
from ..registry.store import append_jsonl, read_json, read_jsonl, write_once_json

EXPERIMENT_ID = "phase2.5-s6"
STAGES = 3
STEPS_PER_STAGE = 1_000_000
CHECKPOINT_EVERY = 250_000
GROWTH, CONTROL = "R1", "R2"
ARMS = {"A": "more training on the original practice market", "B": "more training on new S6 market paths"}
CONTROL_ARMS = ("B",)
MIN_MEDIAN_SIZE = 0.5
BOOTSTRAP = 10_000


def labels_for(reward: str, upto_stage: int) -> list[str]:
    arms = ARMS if reward == GROWTH else CONTROL_ARMS
    return ["v1"] + [f"{a}{k}" for a in arms for k in range(1, upto_stage + 1)]


def schedule() -> dict:
    """Exact evaluation schedule: which labels each round evaluates."""
    rounds = {}
    for r in range(1, 6):
        stage = min(r, STAGES)
        rounds[r] = {GROWTH: labels_for(GROWTH, stage), CONTROL: labels_for(CONTROL, stage)}
    return rounds


def evaluation_count(n_seeds: int = 3) -> dict:
    s = schedule()
    per = {r: n_seeds * (len(x[GROWTH]) + len(x[CONTROL])) for r, x in s.items()}
    return dict(per_round=per, total=sum(per.values()),
                versions=n_seeds * (len(labels_for(GROWTH, STAGES)) + len(labels_for(CONTROL, STAGES))))


def parent_label(label: str) -> str:
    arm, k = label[0], int(label[1:])
    return "v1" if k == 1 else f"{arm}{k - 1}"


def r2_reference_label(label: str) -> str:
    """A control's original verdict compares it with the growth agent of the same seed and the
    same label (v1 with v1, Bk with Bk)."""
    return label


def matched_control_label(candidate: str) -> str:
    """The control version matched to a growth candidate: same stage, arm B (controls have no A)."""
    return "v1" if candidate == "v1" else f"B{candidate[1:]}"


class Experiment:
    def __init__(self, reg: Registry):
        self.reg = reg
        self.dir = reg.root / "experiments" / EXPERIMENT_ID
        self.progress_path = self.dir / "progress.jsonl"

    # ------------------------------------------------------------------ manifest
    def manifest(self) -> dict:
        p = self.dir / "manifest.json"
        if not p.exists():
            raise RegistryError("experiment not planned yet")
        return read_json(p)

    def plan(self, conditions: list[str], steps: int = STEPS_PER_STAGE, checkpoint_every: int = CHECKPOINT_EVERY,
             compute: dict | None = None) -> dict:
        from .metrics import Q2_DEFINITION, check_q2
        q2_identity = check_q2(self.reg, require_frozen=True)
        agents = {}
        for reward in (GROWTH, CONTROL):
            for seed in (0, 1, 2):
                a = self.reg.find("S6", reward, seed)
                if a is None or not self.reg.versions(a):
                    raise RegistryError(f"S6 {reward} seed {seed} is not in the registry with a version 1")
                v1 = self.reg.version(a, 1)
                agents[f"{reward}-{seed}"] = dict(agent_id=a, reward=reward, seed=seed,
                                                  v1_checkpoint=v1["final_checkpoint"]["model_sha256"])
        m = dict(
            experiment_id=EXPERIMENT_ID, planned_at=now_utc(), market="S6", agents=agents,
            arms=ARMS, control_arms=list(CONTROL_ARMS), stages=STAGES, steps_per_stage=steps,
            checkpoint_every=checkpoint_every, schedule={str(k): v for k, v in schedule().items()},
            evaluations=evaluation_count(), approval_conditions=conditions,
            decision_quality=dict(q2_identity, definition=Q2_DEFINITION),
            selection_rule=("Rounds 1-3 only. For each growth seed, among its six continued versions (A1-A3, B1-B3), "
                            "choose the one with the highest mean Q2.0 over every test run it played in rounds 1-3, "
                            f"among eligible versions. Eligible: mean median position size over those rounds of at "
                            f"least {MIN_MEDIAN_SIZE}x; for growth seed 1 also a mean median position size closer "
                            "(in ratio) to the growth-optimal size than version 1's, both measured over rounds 1-3. "
                            "If no version is eligible, choose the highest Q2 and record that none was eligible. "
                            "Ties: lower ruinous share. Recorded and locked (write-once) before round 4; rounds 4 and 5 "
                            "can never change it."),
            success_criteria=[
                "C1 ruinous decisions fall to at most half of version 1's share, or below 1% (round 4)",
                "C2 decision quality Q2.0 rises: 95% paired-bootstrap interval of (candidate - version 1) over "
                "the 20 identical test runs lies above zero (round 4)",
                f"C3 median position size at least {MIN_MEDIAN_SIZE}x; if version 1's median was below "
                f"{MIN_MEDIAN_SIZE}x, the candidate's median must also be closer (in ratio) to the growth-optimal "
                "size (round 4)",
                "C4 replication: C1 and C2 hold again in round 5",
                "Improvement is claimed only if at least 2 of 3 growth seeds meet C1-C4. If the matched controls "
                "meet C1 and C2 as often, the improvement is reported as not clearly due to learning from "
                "consequences.",
            ],
            frozen=("S6 market mechanics, reward functions, scoring rules, simulator, trading costs, margin, death "
                    "threshold, action space, pass/fail criteria, learning algorithm; no leverage cap, no new "
                    "penalties. Arms C and D not approved."),
            compute=compute or {},
        )
        write_once_json(self.dir / "manifest.json", m)
        self._log("planned")
        return m

    # ------------------------------------------------------------------ progress
    def _log(self, kind: str, **data) -> None:
        append_jsonl(self.progress_path, dict(kind=kind, at=now_utc(), **data))

    def progress(self) -> list[dict]:
        return read_jsonl(self.progress_path)

    def version_of(self, key: str, label: str) -> int | None:
        if label == "v1":
            return 1
        hits = [e for e in self.progress() if e["kind"] == "trained" and e["agent_key"] == key and e["label"] == label]
        return hits[-1]["version"] if hits else None

    def round_id(self, n: int) -> str | None:
        hits = [e for e in self.progress() if e["kind"] == "round_created" and e["round"] == n]
        return hits[-1]["round_id"] if hits else None

    def stage_complete(self, k: int) -> bool:
        m = self.manifest()
        return all(self.version_of(key, lab) is not None
                   for key, a in m["agents"].items() for lab in labels_for(a["reward"], k) if lab != "v1")

    def round_complete(self, n: int) -> bool:
        rid = self.round_id(n)
        return rid is not None and all(self._eval(key, lab, rid) is not None for key, lab in self._round_items(n))

    def _round_items(self, n: int) -> list[tuple[str, str]]:
        m = self.manifest()
        sched = m["schedule"][str(n)]
        return [(key, lab) for key, a in m["agents"].items() for lab in sched[a["reward"]]]

    def _eval(self, key: str, label: str, round_id: str) -> dict | None:
        a = self.manifest()["agents"][key]["agent_id"]
        v = self.version_of(key, label)
        if v is None:
            return None
        hits = [e for e in self.reg.evaluations(a) if e["version"] == v and e["round_id"] == round_id]
        return hits[0] if hits else None

    # ------------------------------------------------------------------ training
    def _interrupted(self, agent_id: str, parent: int, arm: str) -> int | None:
        """An attempt for this arm and stage that started but never finished (e.g. container restart)."""
        versions = {v["version"]: v for v in self.reg.versions(agent_id)}
        used_as_parent = {v.get("parent_version") for v in versions.values()}
        closed = {e["version"] for e in self.reg.events(agent_id) if e["kind"] == "training_abandoned"}
        for e in self.reg.events(agent_id):
            if e["kind"] != "training_started" or e["version"] in closed or e["version"] in used_as_parent:
                continue
            rec = versions.get(e["version"])
            if rec is not None and rec["kind"] != "interrupted":           # finished normally
                continue
            if e.get("parent") == parent and (("train_data_seeds" in e.get("changes", {})) == (arm == "B")):
                return e["version"]                                        # killed, or stopped with an error
        return None

    def stage_jobs(self, k: int) -> list[dict]:
        if k > 1 and not self.stage_complete(k - 1):
            raise RegistryError(f"stage {k - 1} is not complete")
        m = self.manifest()
        jobs = []
        for key, a in m["agents"].items():
            for arm in (ARMS if a["reward"] == GROWTH else CONTROL_ARMS):
                label = f"{arm}{k}"
                if self.version_of(key, label) is not None:
                    continue
                parent = self.version_of(key, parent_label(label))
                stuck = self._interrupted(a["agent_id"], parent, arm)
                if stuck is not None:
                    target = self.reg.version(a["agent_id"], parent)["final_checkpoint"]["step"] + m["steps_per_stage"]
                    jobs.append(dict(key=key, agent_id=a["agent_id"], label=label, parent=parent, recover=stuck,
                                     target_step=target, reason=f"{EXPERIMENT_ID} arm {arm}, stage {k}: resumed after "
                                     f"interruption of v{stuck}"))
                    continue
                changes = {"train_steps": m["steps_per_stage"], "checkpoint_every": m["checkpoint_every"]}
                if arm == "B":
                    changes["train_data_seeds"] = ["fresh"]
                jobs.append(dict(key=key, agent_id=a["agent_id"], label=label, parent=parent, changes=changes,
                                 steps=m["steps_per_stage"], checkpoint_every=m["checkpoint_every"],
                                 reason=f"{EXPERIMENT_ID} arm {arm} ({ARMS[arm]}), stage {k} of {STAGES}"))
        return jobs

    def run_stage(self, k: int, workers: int = 4, log=print) -> None:
        jobs = self.stage_jobs(k)
        log(f"stage {k}: {len(jobs)} training jobs")
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, maxtasksperchild=1) as pool:
            for job, rec in pool.imap_unordered(_train_job, [(str(self.reg.root), j) for j in jobs]):
                self._log("trained", agent_key=job["key"], label=job["label"], version=rec["version"],
                          parent_version=job["parent"], stage=k)
                log(f"  {job['key']} {job['label']} -> v{rec['version']} "
                    f"({rec['training']['steps_added']:,} steps, {rec['training']['wall_seconds']:.0f}s)")

    # ------------------------------------------------------------------ rounds
    def candidates(self) -> dict | None:
        p = self.dir / "candidates.json"
        return read_json(p) if p.exists() else None

    def run_round(self, n: int, workers: int = 4, log=print) -> str:
        from .evaluate import create_round
        if n > 1 and not self.round_complete(n - 1):
            raise RegistryError(f"round {n - 1} is not complete")
        if not self.stage_complete(min(n, STAGES)):
            raise RegistryError(f"stage {min(n, STAGES)} must be trained before round {n}")
        if n >= 4 and self.candidates() is None:
            raise RegistryError("candidates must be selected and locked before the confirmation round")
        from .metrics import check_q2
        if check_q2(self.reg, require_frozen=True)["checksum"] != self.manifest()["decision_quality"]["checksum"]:
            raise RegistryError("the decision-quality score differs from the one in the experiment plan")
        rid = self.round_id(n)
        if rid is None:
            rid = create_round(self.reg, ["S6"], note=f"{EXPERIMENT_ID} round {n} "
                               f"({'learning curve / selection' if n <= 3 else 'confirmation' if n == 4 else 'replication'})")["round_id"]
            self._log("round_created", round=n, round_id=rid)
        items = self._round_items(n)
        m = self.manifest()
        for reward in (GROWTH, CONTROL):                       # controls need the growth agents' results
            todo = []
            for key, lab in items:
                a = m["agents"][key]
                if a["reward"] != reward or self._eval(key, lab, rid) is not None:
                    continue
                ref = None
                if reward == CONTROL:
                    ref_e = self._eval(f"{GROWTH}-{a['seed']}", r2_reference_label(lab), rid)
                    ref = ref_e["summary"] if ref_e else None
                todo.append(dict(key=key, agent_id=a["agent_id"], label=lab, version=self.version_of(key, lab),
                                 round_id=rid, reference=ref, round=n))
            log(f"round {n} ({rid}): {len(todo)} {reward} evaluations")
            ctx = mp.get_context("fork")
            with ctx.Pool(workers, maxtasksperchild=4) as pool:
                for job, rec in pool.imap_unordered(_eval_job, [(str(self.reg.root), j) for j in todo]):
                    self._log("evaluated", agent_key=job["key"], label=job["label"], version=job["version"],
                              round=n, round_id=rid, eval_id=rec["eval_id"], result=rec["result"])
        return rid

    # ------------------------------------------------------------------ selection
    def select(self) -> dict:
        if self.candidates() is not None:
            raise RegistryError("candidates are already locked")
        if self.round_id(4) is not None:
            raise RegistryError("the confirmation round has started; candidates can no longer be selected")
        for n in (1, 2, 3):
            if not self.round_complete(n):
                raise RegistryError(f"round {n} is not complete")
        m = self.manifest()
        out = dict(locked_at=now_utc(), rule=m["selection_rule"], seeds={})
        for key, a in m["agents"].items():
            if a["reward"] != GROWTH:
                continue
            g_star = self._g_star()

            def pooled(lab):
                Q, med, ruin, rounds = [], [], [], []
                for n in (1, 2, 3):                                      # selection uses rounds 1-3 only
                    e = self._eval(key, lab, self.round_id(n))
                    if e is None:
                        continue
                    mt = e["diagnosis"]["metrics"]
                    Q += mt["dq2_by_episode"]
                    med.append(mt["median_leverage"])
                    ruin.append(mt["dq_ruinous_share"])
                    rounds.append(n)
                return dict(label=lab, version=self.version_of(key, lab), rounds=rounds, q2=float(np.mean(Q)),
                            median_size=float(np.mean(med)), ruinous_share=float(np.mean(ruin)))
            base = pooled("v1")
            dist = lambda x: abs(math.log(max(x, 1e-9) / g_star))
            rows = []
            for lab in labels_for(GROWTH, STAGES)[1:]:
                r = pooled(lab)
                r["eligible"] = r["median_size"] >= MIN_MEDIAN_SIZE
                if a["seed"] == 1:                                       # seed 1: must move toward the optimum
                    r["closer_to_optimum_than_v1"] = dist(r["median_size"]) < dist(base["median_size"])
                    r["eligible"] = r["eligible"] and r["closer_to_optimum_than_v1"]
                rows.append(r)
            pool = [r for r in rows if r["eligible"]] or rows
            best = sorted(pool, key=lambda r: (-r["q2"], r["ruinous_share"]))[0]
            out["seeds"][key] = dict(candidate=best["label"], version=best["version"],
                                     none_eligible=not any(r["eligible"] for r in rows), version1_rounds_1_3=base,
                                     matched_control=matched_control_label(best["label"]), table=rows)
        write_once_json(self.dir / "candidates.json", out)
        self._log("candidates_locked", candidates={k: v["candidate"] for k, v in out["seeds"].items()})
        return out

    # ------------------------------------------------------------------ verdicts
    def _g_star(self) -> float:
        from .quality import KnownMarket
        from . import synthetic_tasks as st
        return KnownMarket.from_task(st.build("S6", 0, 1000)).optimal()[0]

    def criteria(self, key: str, label: str, baseline: str = "v1") -> dict:
        """C1-C4 for one version against version 1 (rounds 4 and 5)."""
        g_star = self._g_star()
        res = {}
        for n in (4, 5):
            rid = self.round_id(n)
            c, b = (self._eval(key, label, rid), self._eval(key, baseline, rid)) if rid else (None, None)
            if c is None or b is None:
                res[n] = None
                continue
            mc, mb = c["diagnosis"]["metrics"], b["diagnosis"]["metrics"]
            d = np.array(mc["dq2_by_episode"]) - np.array(mb["dq2_by_episode"])
            rng = np.random.default_rng(12345)
            boot = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            c1 = mc["dq_ruinous_share"] <= 0.5 * mb["dq_ruinous_share"] or mc["dq_ruinous_share"] < 0.01
            c2 = lo > 0
            c3 = mc["median_leverage"] >= MIN_MEDIAN_SIZE
            if mb["median_leverage"] < MIN_MEDIAN_SIZE or key == f"{GROWTH}-1":
                dist = lambda x: abs(math.log(max(x, 1e-9) / g_star))
                c3 = c3 and dist(mc["median_leverage"]) < dist(mb["median_leverage"])
            res[n] = dict(c1=bool(c1), c2=bool(c2), c3=bool(c3), q_diff=float(d.mean()), q_ci=[float(lo), float(hi)],
                          ruin=[mb["dq_ruinous_share"], mc["dq_ruinous_share"]],
                          median=[mb["median_leverage"], mc["median_leverage"]])
        r4, r5 = res.get(4), res.get(5)
        passed = None
        if r4 and r5:
            passed = r4["c1"] and r4["c2"] and r4["c3"] and r5["c1"] and r5["c2"]
        return dict(round4=r4, round5=r5, c4=(r5["c1"] and r5["c2"]) if r5 else None, passed=passed)


def _train_job(args):
    root, job = args
    import torch
    torch.set_num_threads(1)
    from .continual import record_interrupted, recover, train_version
    reg = Registry(root)
    if job.get("recover") is not None:
        v = record_interrupted(reg, job["agent_id"], job["recover"], "interrupted (container restart); resumed")
        if v is not None:                                   # it saved checkpoints: continue from the last one
            remaining = max(job["target_step"] - v["final_checkpoint"]["step"], 8192)
            rec = recover(reg, job["agent_id"], job["recover"], steps=remaining, reason=job["reason"], log=lambda m: None)
            return job, rec
        job = dict(job, changes={"train_steps": job["target_step"]}, recover=None)   # nothing saved: start again
        m = Experiment(reg).manifest()
        job.update(steps=m["steps_per_stage"], checkpoint_every=m["checkpoint_every"],
                   changes={"train_steps": m["steps_per_stage"], "checkpoint_every": m["checkpoint_every"],
                            **({"train_data_seeds": ["fresh"]} if job["label"].startswith("B") else {})})
    rec = train_version(reg, job["agent_id"], steps=job["steps"], parent=job["parent"], changes=job["changes"],
                        reason=job["reason"], checkpoint_every=job["checkpoint_every"], allow_parallel=True,
                        log=lambda m: None)
    return job, rec


def _eval_job(args):
    root, job = args
    import torch
    torch.set_num_threads(1)
    from .evaluate import evaluate
    reg = Registry(root)
    rec = evaluate(reg, job["agent_id"], job["version"], job["round_id"], reference=job["reference"],
                   extra_fields=dict(experiment=EXPERIMENT_ID, experiment_label=job["label"],
                                     experiment_round=job["round"]))
    return job, rec


# ---------------------------------------------------------------------------- report
def _fmt(x, f="{:.2f}", none="—"):
    return none if x is None or (isinstance(x, float) and not math.isfinite(x)) else f.format(x)


def report(exp: Experiment) -> str:
    reg, m = exp.reg, exp.manifest()
    L = [f"# Phase 2.5: S6 agent evolution ({EXPERIMENT_ID})\n",
         "Pre-registered design, schedule, selection rule and success criteria: "
         f"`registry/experiments/{EXPERIMENT_ID}/manifest.json`. Decision-quality score: docs/decisions/0004.\n"]
    stages = [k for k in range(1, STAGES + 1) if exp.stage_complete(k)]
    rounds = [n for n in range(1, 6) if exp.round_complete(n)]
    L.append(f"Stages complete: {stages or 'none'}. Rounds complete: {rounds or 'none'}. "
             f"Candidates locked: {'yes' if exp.candidates() else 'no'}.\n")
    L += ["## Every version in every round\n",
          "Growth = average change per 20-run test round. Q2 = decision quality Q2.0 (1 = optimal sizing, 0 = staying "
          "out, -21.35 = ruinous). q = per-decision quality on non-ruinous decisions. Ruinous = share of decisions at 9.5x or more. "
          "Jumps = share of consecutive decisions whose size changed more than tenfold. Crash exposure = average "
          "size before the largest moves / elsewhere (about 1 is right: crashes are unpredictable). After loss = "
          "size after a >2% loss / size before.\n",
          "| Agent | Label | Version | Round | Verdict | Growth | Deaths | Ruinous | Q2 | q | Median size | Largest | "
          "Jumps | Typical size change | Crash exposure | Long | After loss |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, a in m["agents"].items():
        for lab in labels_for(a["reward"], STAGES):
            for n in range(1, 6):
                rid = exp.round_id(n)
                e = exp._eval(key, lab, rid) if rid else None
                if e is None:
                    continue
                s, mt = e["summary"], e["diagnosis"]["metrics"]
                ce = (mt["leverage_before_big_moves"] / mt["leverage_elsewhere"]
                      if mt.get("leverage_before_big_moves") is not None and mt.get("leverage_elsewhere") else None)
                L.append(f"| {key} | {lab} | v{e['version']} | {n} | {e['result']} | {100 * (math.exp(s['lg']) - 1):+.1f}% | "
                         f"{100 * s['died']:.0f}% | {_fmt(100 * mt['dq_ruinous_share'], '{:.1f}%')} | "
                         f"{_fmt(mt.get('dq2'))} | {_fmt(mt['dq_quality'])} | {_fmt(mt['median_leverage'])}x | "
                         f"{_fmt(mt['max_leverage'], '{:.0f}')}x | {_fmt(100 * mt['share_size_jumps_10x'] if mt.get('share_size_jumps_10x') is not None else None, '{:.1f}%')} | "
                         f"{_fmt(mt.get('median_size_change'), '{:.2f}x')} | {_fmt(ce)} | {_fmt(100 * mt['long_share'], '{:.0f}%')} | "
                         f"{_fmt(mt.get('size_after_loss_ratio'))} |")
    L += ["", "## Training history (per version)\n",
          "| Agent | Label | Version | Steps | Training episodes | Death rate first / last 20% | Ruinous share first / last 20% | Quality first / last 20% |",
          "|---|---|---|---|---|---|---|---|"]
    import io
    import pandas as pd
    for key, a in m["agents"].items():
        for lab in labels_for(a["reward"], STAGES)[1:]:
            v = exp.version_of(key, lab)
            if v is None:
                continue
            rec = reg.version(a["agent_id"], v)
            sha = rec["history"].get("episodes_sha256")
            if not sha:
                continue
            ep = pd.read_parquet(io.BytesIO(reg.blobs.get(sha))).sort_values("step")
            k = max(1, len(ep) // 5)
            first, last = ep.iloc[:k], ep.iloc[-k:]
            col = lambda d, c: d[c].mean() if c in d else float("nan")
            L.append(f"| {key} | {lab} | v{v} | {rec['training']['steps_added']:,} | {len(ep)} | "
                     f"{100 * col(first, 'died'):.0f}% / {100 * col(last, 'died'):.0f}% | "
                     f"{100 * col(first, 'share_ruinous'):.1f}% / {100 * col(last, 'share_ruinous'):.1f}% | "
                     f"{col(first, 'decision_quality'):.2f} / {col(last, 'decision_quality'):.2f} |")
    cand = exp.candidates()
    if cand:
        L += ["", "## Candidate selection (locked before round 4)\n", f"Locked at {cand['locked_at']}. Rule: {cand['rule']}\n",
              "| Growth seed | Label | Rounds | Mean Q2 | Mean median size | Mean ruinous share | Eligible | Selected |",
              "|---|---|---|---|---|---|---|---|"]
        for key, c in cand["seeds"].items():
            for r in c["table"]:
                L.append(f"| {key} | {r['label']} | {r['rounds']} | {r['q2']:.2f} | {r['median_size']:.2f}x | "
                         f"{100 * r['ruinous_share']:.1f}% | {'yes' if r['eligible'] else 'no'} | "
                         f"{'**selected**' if r['label'] == c['candidate'] else ''} |")
        L += ["", "## Confirmation (round 4) and replication (round 5)\n",
              "| Agent | Candidate | Round | Ruinous v1 -> cand | C1 | Q2 difference (95% CI) | C2 | Median v1 -> cand | C3 |",
              "|---|---|---|---|---|---|---|---|---|"]
        passes, ctrl = 0, 0
        for key, c in cand["seeds"].items():
            seed = key.split("-")[1]
            for who, lab in ((key, c["candidate"]), (f"{CONTROL}-{seed}", c["matched_control"])):
                r = exp.criteria(who, lab)
                for n in (4, 5):
                    x = r[f"round{n}"]
                    if x is None:
                        continue
                    L.append(f"| {who} | {lab} | {n} | {100 * x['ruin'][0]:.1f}% -> {100 * x['ruin'][1]:.1f}% | "
                             f"{'yes' if x['c1'] else 'no'} | {x['q_diff']:+.2f} ({x['q_ci'][0]:+.2f}, {x['q_ci'][1]:+.2f}) | "
                             f"{'yes' if x['c2'] else 'no'} | {x['median'][0]:.2f}x -> {x['median'][1]:.2f}x | "
                             f"{'yes' if x['c3'] else 'no'} |")
                if who == key and r["passed"]:
                    passes += 1
                if who != key and r["round4"] and r["round4"]["c1"] and r["round4"]["c2"]:
                    ctrl += 1
        if exp.round_complete(5):
            L += ["", "## Pre-registered conclusion\n",
                  f"Growth seeds meeting C1-C4: **{passes} of 3** "
                  f"({'genuine behavioural improvement demonstrated' if passes >= 2 else 'improvement NOT demonstrated'}). "
                  f"Matched controls meeting C1 and C2 in round 4: {ctrl} of 3"
                  + (" (as often as the growth agents: the change is not clearly due to learning from consequences)."
                     if passes >= 2 and ctrl >= passes else ".")]
    return "\n".join(L) + "\n"
