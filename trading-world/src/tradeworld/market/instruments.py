"""Instrument specifications and the instrument universe (including currency conversion paths)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..errors import ConfigError

ASSET_CLASSES = ("fx", "metal", "crypto", "index")


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    base: str
    quote: str
    asset_class: str
    contract_size: float = 1.0          # base units per lot (fractional env: 1 unit = 1 base unit)
    lot_step: float = 0.0               # 0 => fractional sizing
    min_lot: float = 0.0
    max_lots: float = math.inf          # broker maximum position size, in lots
    margin_mode: str = "leverage"       # "leverage" (depends on account leverage) | "fixed"
    fixed_margin_rate: float = 0.0      # fraction of notional, for margin_mode == "fixed"
    max_leverage: float = math.inf      # instrument-level cap for leverage-based margin
    swap_long: float = 0.0              # annual rate on notional; negative = charge
    swap_short: float = 0.0
    swap_free: bool = False
    triple_swap_weekday: int = 2        # 0 = Monday ... 4 = Friday (NY-local rollover day)
    commission_per_lot: float = 0.0     # USD per lot per side
    commission_bps: float = 0.0         # bps of notional per side
    account_type: str = "generic"       # profile integrity: one account type per account

    def __post_init__(self):
        if self.asset_class not in ASSET_CLASSES:
            raise ConfigError(f"{self.symbol}: unknown asset_class {self.asset_class!r}")
        if self.margin_mode not in ("leverage", "fixed"):
            raise ConfigError(f"{self.symbol}: margin_mode must be 'leverage' or 'fixed'")
        if self.contract_size <= 0:
            raise ConfigError(f"{self.symbol}: contract_size must be > 0")
        if self.lot_step < 0 or self.min_lot < 0:
            raise ConfigError(f"{self.symbol}: lot_step/min_lot must be >= 0")
        if not 0 <= self.triple_swap_weekday <= 6:
            raise ConfigError(f"{self.symbol}: triple_swap_weekday must be 0..6")

    @property
    def fractional(self) -> bool:
        return self.lot_step == 0.0

    def with_(self, **changes) -> "InstrumentSpec":
        return replace(self, **changes)


@dataclass(frozen=True)
class Universe:
    """Ordered set of instruments. Account currency for all P&L is USD."""

    specs: tuple[InstrumentSpec, ...]
    _conv: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        symbols = [s.symbol for s in self.specs]
        if len(set(symbols)) != len(symbols):
            raise ConfigError(f"duplicate symbols in universe: {symbols}")
        types = {s.account_type for s in self.specs}
        if len(types) > 1:
            raise ConfigError(f"one account cannot mix account types: {sorted(types)}")
        object.__setattr__(self, "_conv", tuple(self._conversion_path(s) for s in self.specs))

    def _conversion_path(self, spec: InstrumentSpec):
        """How to convert one unit of the instrument's quote currency into USD."""
        q = spec.quote
        if q == "USD":
            return ("one", -1)
        for j, other in enumerate(self.specs):
            if other.base == "USD" and other.quote == q:
                return ("inv", j)       # e.g. JPY -> USD via 1 / USDJPY
            if other.base == q and other.quote == "USD":
                return ("dir", j)       # e.g. EUR -> USD via EURUSD
        raise ConfigError(f"{spec.symbol}: no instrument to convert {q} to USD")

    @property
    def n(self) -> int:
        return len(self.specs)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(s.symbol for s in self.specs)

    def index(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def quote_to_usd(self, mid: np.ndarray) -> np.ndarray:
        """mid: current mid prices, shape [n]. Returns USD per unit of each instrument's quote currency."""
        out = np.ones(self.n)
        for i, (kind, j) in enumerate(self._conv):
            if kind == "inv":
                out[i] = 1.0 / mid[j]
            elif kind == "dir":
                out[i] = mid[j]
        return out

    def quote_to_usd_series(self, mid: np.ndarray) -> np.ndarray:
        """Vectorized version for mid of shape [T, n]."""
        out = np.ones_like(mid)
        for i, (kind, j) in enumerate(self._conv):
            if kind == "inv":
                out[:, i] = 1.0 / mid[:, j]
            elif kind == "dir":
                out[:, i] = mid[:, j]
        return out


def default_specs() -> tuple[InstrumentSpec, ...]:
    """The six project instruments with neutral fractional specs (environment profiles override
    margin, swaps and costs)."""
    return (
        InstrumentSpec("XAUUSD", "XAU", "USD", "metal"),
        InstrumentSpec("EURUSD", "EUR", "USD", "fx"),
        InstrumentSpec("GBPUSD", "GBP", "USD", "fx"),
        InstrumentSpec("USDJPY", "USD", "JPY", "fx"),
        InstrumentSpec("BTCUSD", "BTC", "USD", "crypto", triple_swap_weekday=4),
        InstrumentSpec("NAS100", "NAS", "USD", "index", triple_swap_weekday=4),
    )
