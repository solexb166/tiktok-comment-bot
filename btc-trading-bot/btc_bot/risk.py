"""Position sizing and safety limits, shared by the backtester and the live runner."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01      # fraction of equity lost if the stop is hit (1%)
    max_leverage: float = 1.0         # max position notional / equity
    max_position_btc: float = 1.0     # hard cap on size in BTC
    min_position_btc: float = 0.0001  # below this, skip the trade
    daily_loss_limit: float = 0.03    # stop opening trades after losing 3% in a UTC day
    flatten_on_daily_limit: bool = True
    max_trades_per_day: int = 10


@dataclass
class RiskState:
    day: str = ""
    day_start_equity: float = 0.0
    trades_today: int = 0
    halted_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RiskManager:
    def __init__(self, config: RiskConfig, state: RiskState | None = None):
        self.cfg = config
        self.state = state or RiskState()

    def on_bar(self, day: date | str, equity: float) -> None:
        """Call once per bar with the bar's UTC date and current equity."""
        day = str(day)
        if day != self.state.day:
            self.state = RiskState(day=day, day_start_equity=equity)

    def daily_loss(self, equity: float) -> float:
        if self.state.day_start_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self.state.day_start_equity)

    def daily_limit_hit(self, equity: float) -> bool:
        return self.daily_loss(equity) >= self.cfg.daily_loss_limit

    def can_open(self, equity: float) -> tuple[bool, str]:
        if equity <= 0:
            return False, "no equity"
        if self.daily_limit_hit(equity):
            return False, f"daily loss limit hit ({self.daily_loss(equity):.2%})"
        if self.state.trades_today >= self.cfg.max_trades_per_day:
            return False, f"max trades per day reached ({self.cfg.max_trades_per_day})"
        return True, ""

    def position_size(self, equity: float, price: float, stop_distance: float) -> float:
        """BTC quantity so that hitting the stop loses `risk_per_trade` of equity,
        capped by leverage and the absolute BTC cap."""
        if stop_distance is None or stop_distance <= 0 or price <= 0 or equity <= 0:
            return 0.0
        qty = equity * self.cfg.risk_per_trade / stop_distance
        qty = min(qty, equity * self.cfg.max_leverage / price, self.cfg.max_position_btc)
        qty = math.floor(qty * 1e6) / 1e6
        return qty if qty >= self.cfg.min_position_btc else 0.0

    def record_open(self) -> None:
        self.state.trades_today += 1
