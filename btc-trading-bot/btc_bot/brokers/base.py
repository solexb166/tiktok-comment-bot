from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class BrokerError(RuntimeError):
    pass


class RealMoneyBlocked(BrokerError):
    """Raised when a live broker points at a real-money account without explicit opt-in."""


@dataclass
class Position:
    side: int            # 1 long, -1 short
    qty: float           # in BTC
    entry_price: float
    stop_price: float | None = None
    id: str | None = None


@dataclass
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class Broker(ABC):
    """Executes the bot's orders. Quantities are always in BTC; brokers convert to lots etc."""

    #: True if the stop-loss lives on the broker's server (survives the bot crashing).
    server_side_stops: bool = False

    @abstractmethod
    def connect(self) -> str:
        """Validate credentials/account; return a one-line description. Must enforce the
        demo/sandbox guard."""

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def quote(self) -> Quote: ...

    @abstractmethod
    def position(self) -> Position | None: ...

    @abstractmethod
    def open(self, side: int, qty: float, stop_price: float, comment: str = "") -> Position: ...

    @abstractmethod
    def close(self, reason: str = "") -> float | None:
        """Close the bot's position; return the fill price if known."""
