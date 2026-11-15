from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Action(str, Enum):
    HOLD = "hold"
    ENTER_LONG = "enter_long"     # if short, the executor closes it first (reversal)
    ENTER_SHORT = "enter_short"   # if long, the executor closes it first (reversal)
    EXIT = "exit"                 # flatten whatever is open


@dataclass
class Decision:
    action: Action
    stop_distance: float | None = None   # price distance for the protective stop
    reason: str = ""


class Strategy(ABC):
    """A strategy adds indicator columns in `prepare` and makes one decision per closed bar
    in `decide`. Backtests and live trading call exactly the same two methods."""

    name: str = "base"
    defaults: dict = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"Unknown params for {self.name}: {sorted(unknown)}")
        self.params = {**self.defaults, **params}

    def __getattr__(self, item):
        params = self.__dict__.get("params", {})
        if item in params:
            return params[item]
        raise AttributeError(item)

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars needed before indicators are valid."""

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with indicator columns added."""

    @abstractmethod
    def decide(self, row: pd.Series, position: int) -> Decision:
        """position: 1 long, -1 short, 0 flat. `row` is the just-closed bar."""

    def describe(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
