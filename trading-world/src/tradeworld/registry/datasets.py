"""Test-data management: every evaluation round uses data no agent or round has used before.

Synthetic markets are generated from a seed, so "fresh data" means a seed never used before
for that market, for any purpose. The ledger (append-only) records every seed's use:

  train   - training data          (seeds below 10,000; continued training draws from 1,000+)
  test    - evaluation rounds      (the original gate used 10,000-10,002; new rounds 20,000+)
  sealed  - final assessment only  (900,000+): never trained, validated or tuned on. Reading a
            sealed seed requires an explicit unseal with a reason, which is logged.

Real market data uses fixed calendar splits (configs/data/splits.yaml); the final holdout is
sealed the same way and `check_period` refuses any overlap with it for other purposes.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from .registry import ROOT, FrozenRuleError, now_utc
from .store import RegistryError, append_jsonl, locked, read_jsonl

TRAIN_FRESH_START = 1_000
TEST_ROUND_START = 20_000
SEALED_START = 900_000
PURPOSES = ("train", "test", "sealed")


class SealedDataError(RegistryError):
    pass


class DataLedger:
    def __init__(self, registry_root: Path):
        self.path = Path(registry_root) / "data" / "ledger.jsonl"

    def entries(self) -> list[dict]:
        return read_jsonl(self.path)

    def used(self, task: str) -> dict[int, str]:
        return {e["seed"]: e["purpose"] for e in self.entries() if e["task"] == task}

    def record(self, task: str, seed: int, purpose: str, round_id: str | None = None, note: str = "") -> None:
        if purpose not in PURPOSES:
            raise ValueError(purpose)
        used = self.used(task)
        if seed in used and not (purpose == "train" and used[seed] == "train"):
            raise RegistryError(f"{task} seed {seed} was already used for {used[seed]}; data must be fresh")
        if purpose != "sealed" and seed >= SEALED_START:
            raise SealedDataError(f"{task} seed {seed} is in the sealed range")
        append_jsonl(self.path, dict(task=task, seed=int(seed), purpose=purpose, round_id=round_id,
                                     note=note, at=now_utc()))

    def fresh_seed(self, task: str, purpose: str) -> int:
        start = {"train": TRAIN_FRESH_START, "test": TEST_ROUND_START}[purpose]
        used = self.used(task)
        s = start
        while s in used:
            s += 1
        limit = TEST_ROUND_START if purpose == "train" else SEALED_START
        if s >= limit:
            raise RegistryError(f"no fresh {purpose} seeds left for {task}")
        return s

    def allocate(self, task: str, purpose: str, round_id: str | None = None, note: str = "") -> int:
        """Pick and record a never-used seed in one atomic step (safe for parallel jobs)."""
        with locked(self.path.parent / "allocate"):        # separate from the append lock (no self-deadlock)
            seed = self.fresh_seed(task, purpose)
            self.record(task, seed, purpose, round_id=round_id, note=note)
        return seed

    def sealed_seed(self, task: str, reason: str, approved_by: str) -> int:
        """The sealed final-assessment seed. Every access is logged with its reason."""
        if not reason or not approved_by:
            raise SealedDataError("unsealing the final test data needs a reason and who approved it")
        seed = SEALED_START
        append_jsonl(self.path.parent / "sealed_access.jsonl",
                     dict(task=task, seed=seed, reason=reason, approved_by=approved_by, at=now_utc()))
        return seed

    def sealed_accesses(self) -> list[dict]:
        return read_jsonl(self.path.parent / "sealed_access.jsonl")


def load_splits(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "configs" / "data" / "splits.yaml").read_text())


def check_period(start: date, end: date, purpose: str, splits: dict | None = None) -> None:
    """Refuse real-data use that overlaps the sealed holdout (or a split not meant for it)."""
    s = splits or load_splits()
    hits = []
    for name, sp in s["splits"].items():
        a, b = date.fromisoformat(str(sp["start"])), date.fromisoformat(str(sp["end"]))
        if start <= b and end >= a:
            hits.append((name, sp))
    for name, sp in hits:                                 # the sealed holdout is checked first
        if sp.get("sealed") and purpose != "final_assessment":
            raise SealedDataError(f"{start}..{end} overlaps the sealed split {name!r}")
    for name, sp in hits:
        if purpose not in sp["allowed"]:
            raise FrozenRuleError(f"{start}..{end} overlaps split {name!r}, which does not allow {purpose!r}")
