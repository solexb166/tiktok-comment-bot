"""Simulated broker: real prices from the feed, fake fills. State persists to a JSON file."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from .base import Broker, Position, Quote

log = logging.getLogger(__name__)


class PaperBroker(Broker):
    server_side_stops = False

    def __init__(self, price_fn: Callable[[], float], state_file: str | Path,
                 initial_equity: float = 10_000.0, fee_bps: float = 6.0, slippage_bps: float = 2.0,
                 spread_bps: float = 1.0):
        self.price_fn = price_fn
        self.state_file = Path(state_file)
        self.fee, self.slip, self.spread = fee_bps / 1e4, slippage_bps / 1e4, spread_bps / 1e4
        self.balance = initial_equity
        self.pos: Position | None = None
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.balance = data["balance"]
            self.pos = Position(**data["position"]) if data.get("position") else None

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(
            {"balance": self.balance, "position": self.pos.__dict__ if self.pos else None}, indent=2))

    def connect(self) -> str:
        return f"paper account, balance {self.balance:,.2f} USD (state: {self.state_file})"

    def quote(self) -> Quote:
        mid = self.price_fn()
        return Quote(mid * (1 - self.spread / 2), mid * (1 + self.spread / 2))

    def equity(self) -> float:
        if not self.pos:
            return self.balance
        q = self.quote()
        mark = q.bid if self.pos.side > 0 else q.ask
        return self.balance + self.pos.side * self.pos.qty * (mark - self.pos.entry_price)

    def position(self) -> Position | None:
        return self.pos

    def open(self, side: int, qty: float, stop_price: float, comment: str = "") -> Position:
        q = self.quote()
        px = q.ask * (1 + self.slip) if side > 0 else q.bid * (1 - self.slip)
        self.balance -= qty * px * self.fee
        self.pos = Position(side, qty, px, stop_price, id="paper")
        self._save()
        log.info("[paper] opened %s %.6f BTC @ %.2f stop %.2f", "LONG" if side > 0 else "SHORT", qty, px, stop_price)
        return self.pos

    def close(self, reason: str = "") -> float | None:
        if not self.pos:
            return None
        q = self.quote()
        px = q.bid * (1 - self.slip) if self.pos.side > 0 else q.ask * (1 + self.slip)
        return self._close_at(px, reason)

    def _close_at(self, px: float, reason: str) -> float:
        pnl = self.pos.side * self.pos.qty * (px - self.pos.entry_price)
        self.balance += pnl - self.pos.qty * px * self.fee
        log.info("[paper] closed @ %.2f pnl %.2f (%s)", px, pnl, reason)
        self.pos = None
        self._save()
        return px
