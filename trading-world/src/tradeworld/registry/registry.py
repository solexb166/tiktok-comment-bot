"""Agent registry: permanent identity, immutable versions, write-once evaluations, lifecycle.

Layout (under the registry root, git-tracked for now):

  agents/<agent_id>/identity.json          write-once: who the agent is (never changes)
  agents/<agent_id>/events.jsonl           append-only lifecycle log (status is derived from it)
  agents/<agent_id>/versions/v0001.json    write-once: one per training attempt
  agents/<agent_id>/evaluations/<id>.json  write-once: one per evaluation
  rounds/<round_id>.json                   write-once: the test data of one evaluation round
  blobs/..                                 content-addressed payloads (store.LocalBlobStore)

Frozen experimental rules: the practice market, scoring rule (reward), environment settings,
action space and simulator profile are fixed in the identity. A new version may change only
the training settings in ALLOWED_CHANGES; anything else raises FrozenRuleError and must go
through a separate, approved proposal.
"""
from __future__ import annotations

import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .store import (LocalBlobStore, RegistryError, WriteOnceError, append_jsonl, read_json,
                    read_jsonl, write_once_json)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "registry"

# Training settings a continued-training version may change (Part 8: controlled changes).
ALLOWED_CHANGES = {
    "train_steps", "train_data_seeds", "learning_rate", "log_std", "ent_coef", "clip_range",
    "n_epochs", "batch_size", "gamma", "gae_lambda", "max_grad_norm", "vf_coef", "n_envs",
    "checkpoint_every", "resume_from_checkpoint",
}
# Everything that defines the experiment itself. Never changed by a new version.
FROZEN_FIELDS = ("task", "reward", "env_settings", "action_space", "profile", "episode_decisions")

STATUSES = ("created", "training", "awaiting_evaluation", "passed", "failed", "continued_training",
            "improved", "retired")


class FrozenRuleError(RegistryError):
    """A requested change would alter the experimental rules (needs a separate approval)."""


class LifecycleError(RegistryError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_version() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""
    status = git("status", "--porcelain", "--", "src", "scripts", "configs")
    return dict(git_commit=git("rev-parse", "HEAD") or "unknown", dirty=bool(status))


def check_changes(changes: dict) -> None:
    bad = sorted(set(changes) - ALLOWED_CHANGES)
    if bad:
        raise FrozenRuleError(
            f"not an allowed training-setting change: {bad}. The market, scoring rule, simulator, "
            f"environment settings and evaluation criteria are frozen; propose such a change "
            f"separately for approval.")


class Registry:
    def __init__(self, root: Path | str = DEFAULT_REGISTRY):
        self.root = Path(root)
        self.blobs = LocalBlobStore(self.root / "blobs")          # permanent (git-tracked for now)
        self.cache = LocalBlobStore(self.root / "cache")          # regenerable, not tracked

    # ------------------------------------------------------------------ paths
    def _agent_dir(self, agent_id: str) -> Path:
        d = self.root / "agents" / agent_id
        if not (d / "identity.json").exists():
            raise RegistryError(f"unknown agent {agent_id}")
        return d

    # ------------------------------------------------------------------ identity
    def create_agent(self, *, task: str, reward: str, seed: int, env_settings: dict, action_space: dict,
                     profile: str, episode_decisions: int, name: str | None = None,
                     origin: str = "new", created_at: str | None = None, notes: str = "") -> str:
        agent_id = "ag-" + uuid.uuid4().hex[:10]
        identity = dict(
            agent_id=agent_id, name=name or f"{task}-{reward}-seed{seed}", task=task, reward=reward,
            original_seed=int(seed), created_at=created_at or now_utc(), origin=origin, notes=notes,
            frozen=dict(task=task, reward=reward, env_settings=env_settings, action_space=action_space,
                        profile=profile, episode_decisions=int(episode_decisions)),
        )
        d = self.root / "agents" / agent_id
        write_once_json(d / "identity.json", identity)
        self.event(agent_id, "created", at=identity["created_at"])
        return agent_id

    def identity(self, agent_id: str) -> dict:
        return read_json(self._agent_dir(agent_id) / "identity.json")

    def agents(self) -> list[str]:
        base = self.root / "agents"
        return sorted(p.name for p in base.iterdir() if (p / "identity.json").exists()) if base.exists() else []

    def find(self, task: str, reward: str, seed: int) -> str | None:
        for a in self.agents():
            i = self.identity(a)
            if i["task"] == task and i["reward"] == reward and i["original_seed"] == seed:
                return a
        return None

    # ------------------------------------------------------------------ lifecycle
    def event(self, agent_id: str, kind: str, at: str | None = None, **data) -> None:
        d = self.root / "agents" / agent_id
        if kind != "created":
            self._agent_dir(agent_id)
        append_jsonl(d / "events.jsonl", dict(kind=kind, at=at or now_utc(), **data))

    def events(self, agent_id: str) -> list[dict]:
        return read_jsonl(self._agent_dir(agent_id) / "events.jsonl")

    def state(self, agent_id: str) -> dict:
        """Current status, version and attempt counts, derived from the immutable records."""
        ev = self.events(agent_id)
        versions = self.versions(agent_id)
        evals = self.evaluations(agent_id)
        latest = versions[-1]["version"] if versions else 0
        training = {e["version"] for e in ev if e["kind"] == "training_started"} - \
                   {e["version"] for e in ev if e["kind"] in ("training_finished", "training_abandoned")}
        retired = any(e["kind"] == "retired" for e in ev)
        latest_evals = [e for e in evals if e["version"] == latest and e["result"] in ("pass", "fail")]
        ever_failed = any(e["result"] == "fail" for e in evals)
        if retired:
            status = "retired"
        elif training:
            status = "continued_training" if max(training) > 1 else "training"
        elif not versions:
            status = "created"
        elif not latest_evals:
            status = "awaiting_evaluation"
        elif latest_evals[-1]["result"] == "pass":
            status = "improved" if ever_failed else "passed"
        else:
            status = "failed"
        first_pass = next((e for e in evals if e["result"] == "pass"), None)
        champions = [e for e in ev if e["kind"] == "champion"]
        return dict(
            agent_id=agent_id, status=status, current_version=latest,
            training_attempts=len(versions), evaluations=len(evals),
            evaluation_rounds=len({e["round_id"] for e in evals}),
            failed_evaluations=sum(e["result"] == "fail" for e in evals),
            passed_on_version=first_pass["version"] if first_pass else None,
            passed_on_evaluation=(evals.index(first_pass) + 1) if first_pass else None,
            champion=champions[-1] if champions else None,
        )

    # ------------------------------------------------------------------ versions
    def versions(self, agent_id: str) -> list[dict]:
        d = self._agent_dir(agent_id) / "versions"
        return [read_json(p) for p in sorted(d.glob("v*.json"))] if d.exists() else []

    def version(self, agent_id: str, n: int) -> dict:
        p = self._agent_dir(agent_id) / "versions" / f"v{n:04d}.json"
        if not p.exists():
            raise RegistryError(f"{agent_id} has no version {n}")
        return read_json(p)

    def next_version(self, agent_id: str) -> int:
        v = self.versions(agent_id)
        return (v[-1]["version"] + 1) if v else 1

    def start_training(self, agent_id: str, version: int, parent: int | None, changes: dict,
                       info: dict | None = None) -> None:
        check_changes(changes)
        if version != self.next_version(agent_id):
            raise LifecycleError(f"{agent_id}: next version is {self.next_version(agent_id)}, not {version}")
        if parent is not None:
            self.version(agent_id, parent)                         # must exist
        if self.state(agent_id)["status"] in ("training", "continued_training"):
            raise LifecycleError(f"{agent_id} is already training")
        if self.state(agent_id)["status"] == "retired":
            raise LifecycleError(f"{agent_id} is retired")
        self.event(agent_id, "training_started", version=version, parent=parent, changes=changes, info=info or {})

    def record_version(self, agent_id: str, record: dict) -> dict:
        """Write version `record['version']` once. Verifies the frozen rules did not change."""
        ident = self.identity(agent_id)
        n = int(record["version"])
        if n != self.next_version(agent_id):
            raise WriteOnceError(f"{agent_id}: version {n} exists or skips a number")
        if record.get("parent_version") is not None:
            self.version(agent_id, record["parent_version"])
        check_changes(record.get("changes", {}))
        frozen = record.get("frozen", ident["frozen"])
        for k in FROZEN_FIELDS:
            if frozen.get(k) != ident["frozen"].get(k):
                raise FrozenRuleError(f"{agent_id} v{n}: frozen field {k!r} differs from the agent's identity")
        for c in record.get("checkpoints", []):
            for key in ("model_sha256", "state_sha256"):
                if c.get(key) and not self.blobs.exists(c[key]):
                    raise RegistryError(f"checkpoint blob {c[key]} is not in the store")
        rec = dict(record, agent_id=agent_id, frozen=ident["frozen"], recorded_at=now_utc())
        write_once_json(self._agent_dir(agent_id) / "versions" / f"v{n:04d}.json", rec)
        self.event(agent_id, "training_finished", version=n)
        return rec

    def abandon_training(self, agent_id: str, version: int, reason: str) -> None:
        self.event(agent_id, "training_abandoned", version=version, reason=reason)

    # ------------------------------------------------------------------ evaluations
    def evaluations(self, agent_id: str) -> list[dict]:
        d = self._agent_dir(agent_id) / "evaluations"
        recs = [read_json(p) for p in d.glob("*.json")] if d.exists() else []
        return sorted(recs, key=lambda r: (r["recorded_at"], r["eval_id"]))

    def record_evaluation(self, agent_id: str, record: dict) -> dict:
        v = self.version(agent_id, int(record["version"]))
        eval_id = f"{agent_id}-v{int(record['version']):04d}-{record['round_id']}"
        if record.get("checkpoint_sha256") and record["checkpoint_sha256"] != v["final_checkpoint"]["model_sha256"] \
                and record["checkpoint_sha256"] not in {c["model_sha256"] for c in v.get("checkpoints", [])}:
            raise RegistryError("evaluated checkpoint does not belong to this version")
        rec = dict(record, eval_id=eval_id, agent_id=agent_id, recorded_at=now_utc())
        path = self._agent_dir(agent_id) / "evaluations" / f"{eval_id}.json"
        if path.exists():
            raise WriteOnceError(f"evaluation {eval_id} already exists and cannot be overwritten")
        write_once_json(path, rec)
        self.event(agent_id, "evaluated", version=rec["version"], eval_id=eval_id, round_id=rec["round_id"],
                   result=rec["result"])
        if rec["result"] == "pass":
            self.event(agent_id, "champion", version=rec["version"], eval_id=eval_id,
                       checkpoint_sha256=rec.get("checkpoint_sha256"))
        return rec

    def retire(self, agent_id: str, reason: str) -> None:
        self.event(agent_id, "retired", reason=reason)

    # ------------------------------------------------------------------ rounds
    def record_round(self, round_id: str, record: dict) -> dict:
        rec = dict(record, round_id=round_id, recorded_at=now_utc())
        write_once_json(self.root / "rounds" / f"{round_id}.json", rec)
        return rec

    def round(self, round_id: str) -> dict:
        p = self.root / "rounds" / f"{round_id}.json"
        if not p.exists():
            raise RegistryError(f"unknown evaluation round {round_id}")
        return read_json(p)

    def rounds(self) -> list[dict]:
        d = self.root / "rounds"
        return sorted((read_json(p) for p in d.glob("*.json")), key=lambda r: r["recorded_at"]) if d.exists() else []
