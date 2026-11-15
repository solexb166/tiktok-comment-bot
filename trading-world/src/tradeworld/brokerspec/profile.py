"""Environment profiles (A / B / C) loaded from YAML, with per-field verification status.

Any YAML value may be written plainly or as {value, status, source}. Status values:
  verified   - confirmed from an official source (recorded in docs/exness-spec-verification.md)
  model      - a simulator modelling parameter, not a broker fact (slippage, death rule, thresholds)
  assumption - an explicit modelling assumption (allowed in environments A and C only)
  conflicting / unverified / unknown - NOT allowed in a strict environment-B load

Environment B is guarded: loading it strictly fails with UnverifiedSpecError listing every
non-verified field. `sensitivity=True` allows a run with explicit candidate values; such
profiles carry the tag 'unverified-spec'.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..accounts.margin import HMRWindow, MarginModel, UnlimitedEligibility
from ..errors import ConfigError, UnverifiedSpecError
from ..market.instruments import InstrumentSpec, Universe

STATUSES = {"verified", "model", "assumption", "conflicting", "unverified", "unknown"}
# Broker-specification fields: in a strict (environment B) load, a plain value without an explicit
# status is treated as 'unverified' -- nothing becomes verified by accident.
BROKER_FIELDS = ("instruments", "instrument_overrides", "margin", "stop_out_level", "margin_call_level",
                 "nbp", "account_currency", "currency_per_usd", "stop_out_order")
IDENTITY = {"symbol", "base", "quote", "asset_class", "account_type"}   # naming, not broker claims
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


@dataclass(frozen=True)
class SlippageModel:
    """Adverse slippage as a fraction of price: base + range_coef * bar_range/price + |N(0,1)| * noise."""
    base_bps: float = 0.2
    noise_bps: float = 0.2
    range_coef: float = 0.05

    def draw(self, rng: np.random.Generator, range_frac: float) -> float:
        noise = abs(rng.standard_normal()) * self.noise_bps if self.noise_bps else 0.0
        return max(0.0, (self.base_bps + noise) / 1e4 + self.range_coef * range_frac)


@dataclass(frozen=True)
class EnvProfile:
    name: str
    environment: str                       # "A" | "B" | "C"
    universe: Universe
    margin: MarginModel
    initial_equity_usd: float = 10.0
    account_currency: str = "USD"
    currency_per_usd: float = 1.0          # 100 for USC (cent) accounts
    death_fraction: float = 0.05
    stop_out_level: float = 0.0            # margin level (equity / used margin) that triggers stop-out
    margin_call_level: float | None = 0.6
    nbp: bool = False                      # negative balance protection
    stop_out_order: str = "most_unprofitable"
    min_trade_notional_usd: float = 0.001  # numerical threshold for fractional sizing
    slippage: SlippageModel = field(default_factory=SlippageModel)
    spread_multiplier: float = 1.0
    rollover_hour_ny: int = 17
    tags: tuple = ()
    field_status: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.environment not in ("A", "B", "C"):
            raise ConfigError("environment must be A, B or C")
        if not 0 < self.death_fraction < 1:
            raise ConfigError("death_fraction must be in (0, 1)")
        if self.stop_out_order not in ("most_unprofitable",):
            raise ConfigError(f"unsupported stop_out_order {self.stop_out_order!r}")
        if self.initial_equity_usd <= 0 or self.spread_multiplier <= 0:
            raise ConfigError("initial equity and spread multiplier must be > 0")

    @property
    def death_equity(self) -> float:
        return self.death_fraction * self.initial_equity_usd

    def with_(self, **changes) -> "EnvProfile":
        from dataclasses import replace
        return replace(self, **changes)


# ---------------------------------------------------------------------------- YAML loading
def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and "value" not in v:
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path, root: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for inc in data.pop("include", []) or []:
        merged = _deep_merge(merged, _read(root / inc, root))
    return _deep_merge(merged, data)


def _unwrap(obj, path: str, statuses: dict, strict: bool = False):
    """Strip {value, status, source} wrappers, recording each field's status by dotted path.
    Plain leaves are 'model' parameters, except broker-spec fields in strict mode ('unverified')."""
    if isinstance(obj, dict) and "value" in obj and set(obj) <= {"value", "status", "source", "note"}:
        status = obj.get("status", "unverified")
        if status not in STATUSES:
            raise ConfigError(f"{path}: unknown status {status!r}")
        if obj["value"] is None:
            status = "unknown"
        statuses[path] = status
        return _unwrap(obj["value"], path, {}, strict)
    if isinstance(obj, dict):
        return {k: _unwrap(v, f"{path}.{k}" if path else k, statuses, strict) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap(v, f"{path}[{i}]", statuses, strict) for i, v in enumerate(obj)]
    if strict and path.split(".")[0].split("[")[0] in BROKER_FIELDS and path.rsplit(".", 1)[-1] not in IDENTITY:
        statuses[path] = "unverified"
    return obj


def _num(x):
    if isinstance(x, str) and x.lower() in ("inf", "unlimited"):
        return math.inf
    return x


def load_profile(path: str | Path, *, strict: bool | None = None, sensitivity: bool = False,
                 overrides: dict | None = None, root: Path | None = None) -> EnvProfile:
    """Load an environment profile. Environment B is strict by default (verified-spec guard)."""
    root = Path(root or CONFIG_ROOT)
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    raw = _read(path, root)
    if overrides:
        raw = _deep_merge(raw, overrides)
    env = raw.get("environment")
    strict = (env == "B") if strict is None else strict
    statuses: dict[str, str] = {}
    cfg = _unwrap(raw, "", statuses, strict)
    allowed = {"verified", "model"} if strict else {"verified", "model", "assumption"}
    bad = {k: v for k, v in statuses.items() if v not in allowed}
    unknown = {k: v for k, v in statuses.items() if v == "unknown"}
    tags = tuple(cfg.get("tags", ()))
    if bad:
        if strict and not sensitivity:
            raise UnverifiedSpecError(cfg.get("name", str(path)), bad)
        if unknown:
            raise UnverifiedSpecError(cfg.get("name", str(path)) + " (no candidate values)", unknown)
        if not strict and not sensitivity:
            raise ConfigError(f"{cfg.get('name')}: fields with status other than verified/assumption: {bad}")
        tags = tags + ("unverified-spec",)
    return _build(cfg, statuses, tags)


def _build(cfg: dict, statuses: dict, tags: tuple) -> EnvProfile:
    specs = []
    overrides = cfg.get("instrument_overrides", {}) or {}
    for item in cfg["instruments"]:
        item = {**item, **overrides.get(item["symbol"], {})}
        item = {k: _num(v) for k, v in item.items()}
        specs.append(InstrumentSpec(**item))
    m = cfg.get("margin", {})
    unl = m.get("unlimited")
    margin = MarginModel(
        account_leverage=float(_num(m.get("account_leverage", 2000))),
        equity_tiers=tuple((float(_num(u)), float(_num(l))) for u, l in m.get("equity_tiers", [])),
        unlimited=UnlimitedEligibility(**{k: float(_num(v)) if k != "min_closed_trades" else int(v)
                                          for k, v in unl.items()}) if unl else None,
        hmr_windows=tuple(HMRWindow(**w) for w in m.get("hmr_windows", [])),
    )
    sl = cfg.get("slippage", {})
    keys = ("initial_equity_usd", "account_currency", "currency_per_usd", "death_fraction",
            "stop_out_level", "margin_call_level", "nbp", "stop_out_order",
            "min_trade_notional_usd", "spread_multiplier", "rollover_hour_ny")
    return EnvProfile(
        name=cfg["name"], environment=cfg["environment"], universe=Universe(tuple(specs)),
        margin=margin, slippage=SlippageModel(**sl), tags=tags, field_status=statuses,
        **{k: cfg[k] for k in keys if k in cfg},
    )
