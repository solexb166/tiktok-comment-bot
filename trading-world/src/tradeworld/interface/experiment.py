"""Experiment-level settings that shape the agent interface (leverage benchmark caps etc.)."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..brokerspec.profile import CONFIG_ROOT
from .actions import ActionMapper


def load_experiment(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_ROOT / path
    return yaml.safe_load(path.read_text())


def mapper_for(cfg: dict, n: int) -> ActionMapper:
    cap = cfg.get("exposure_cap")
    return ActionMapper.primary(n) if cap is None else ActionMapper.benchmark(n, float(cap))
