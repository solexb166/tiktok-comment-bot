"""Account: netting positions, event-sourced ledger, valuation, swaps, stop-out, NBP and death.

Money is float64 USD internally (a Cent account shows USC = USD x 100). Positions are NET per
instrument (one signed position per symbol). Longs are valued at bid, shorts at ask.

Balance changes ONLY through: realized P&L, commissions, swaps and NBP resets. Therefore
    equity = balance + sum(unrealized)        (invariant, test group 3)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..brokerspec.profile import EnvProfile

ALIVE, DEAD = "alive", "dead"


@dataclass
class Event:
    t: int                 # UTC minute of the bar in which it happened
    kind: str              # fill | swap | stop_out | death | nbp_reset | trade_closed | reject | ...
    data: dict

    def canonical(self) -> str:
        def fmt(v):
            if isinstance(v, float):
                return float(v).hex()
            if isinstance(v, (np.floating,)):
                return float(v).hex()
            if isinstance(v, (np.integer,)):
                return str(int(v))
            return repr(v)
        body = ",".join(f"{k}={fmt(v)}" for k, v in sorted(self.data.items()))
        return f"{self.t}|{self.kind}|{body}"


@dataclass
class Prices:
    """Valuation inputs for one instant: per-instrument bid/ask and quote->USD conversion."""
    bid: np.ndarray
    ask: np.ndarray
    conv: np.ndarray

    @property
    def mid(self) -> np.ndarray:
        return (self.bid + self.ask) / 2.0


class Account:
    def __init__(self, agent_id: str, profile: EnvProfile, rng: np.random.Generator):
        self.agent_id = agent_id
        self.profile = profile
        self.u = profile.universe
        self.rng = rng
        n = self.u.n
        self.units = np.zeros(n)
        self.avg = np.zeros(n)
        self.open_t = np.full(n, -1, dtype=np.int64)
        self.trade_pnl = np.zeros(n)          # realized + commissions + swaps of the open trade
        self.initial = float(profile.initial_equity_usd)
        self.balance = self.initial
        self.status = ALIVE
        self.death_reason: str | None = None
        self.death_t: int | None = None
        self.events: list[Event] = []
        self.closed_trades = 0
        self.volume_lots = 0.0
        self.trades: list[dict] = []
        self.costs = dict(spread=0.0, slippage=0.0, commission=0.0, swap=0.0)  # since last decision
        self.costs_total = dict(spread=0.0, slippage=0.0, commission=0.0, swap=0.0)
        self.max_gross_interval = 0.0
        self.min_equity_interval = self.initial

    # ------------------------------------------------------------------ valuation
    @property
    def alive(self) -> bool:
        return self.status == ALIVE

    @property
    def has_positions(self) -> bool:
        return bool(np.any(self.units != 0.0))

    def unrealized(self, p: Prices) -> np.ndarray:
        mark = np.where(self.units > 0, p.bid, p.ask)
        return self.units * (mark - self.avg) * p.conv

    def equity(self, p: Prices) -> float:
        return self.balance + float(np.sum(self.unrealized(p)))

    def notional(self, p: Prices, units: np.ndarray | None = None) -> np.ndarray:
        units = self.units if units is None else units
        return np.abs(units) * p.mid * p.conv

    def margin_used(self, p: Prices, t: int, equity: float, units: np.ndarray | None = None) -> float:
        rates = self.profile.margin.rates(self.u, equity, t, self.closed_trades, self.volume_lots)
        return float(np.sum(self.notional(p, units) * rates))

    def margin_level(self, p: Prices, t: int) -> float:
        eq = self.equity(p)
        used = self.margin_used(p, t, eq)
        return math.inf if used <= 0 else eq / used

    def exposure(self, p: Prices) -> np.ndarray:
        """Signed notional / equity per instrument."""
        eq = self.equity(p)
        signed = self.units * p.mid * p.conv
        return signed / eq if eq > 0 else np.where(signed != 0, np.sign(signed) * np.inf, 0.0)

    # ------------------------------------------------------------------ ledger primitives
    def _event(self, t: int, kind: str, **data) -> None:
        self.events.append(Event(int(t), kind, data))

    def _add_cost(self, kind: str, amount: float) -> None:
        self.costs[kind] += amount
        self.costs_total[kind] += amount

    def fill(self, t: int, i: int, du: float, price: float, conv: float, ref_mid: float,
             ref_quote: float, reason: str) -> None:
        """Apply a fill of signed size `du` at `price` (quote currency). `ref_quote` is the
        bid/ask before slippage, `ref_mid` the mid, used only for cost attribution."""
        if du == 0.0:
            return
        spec = self.u.specs[i]
        cur = self.units[i]
        commission = (abs(du) / spec.contract_size * spec.commission_per_lot
                      + abs(du) * price * conv * spec.commission_bps / 1e4)
        realized = 0.0
        if cur == 0.0 or np.sign(cur) == np.sign(du):          # open or add
            new = cur + du
            self.avg[i] = (abs(cur) * self.avg[i] + abs(du) * price) / abs(new)
            if cur == 0.0:
                self.open_t[i] = t
                self.trade_pnl[i] = 0.0
            self.units[i] = new
            self.trade_pnl[i] -= commission
        else:                                                   # reduce, close or reverse
            closing = min(abs(du), abs(cur))
            remainder = abs(du) - closing
            c_close = commission * closing / abs(du)
            realized = closing * (price - self.avg[i]) * np.sign(cur) * conv
            self.balance += realized
            self.trade_pnl[i] += realized - c_close
            new = cur + du
            if remainder > 0 or new == 0.0:
                self._close_trade(t, i, price, closing, self.trade_pnl[i])
            if remainder > 0:
                self.units[i] = np.sign(du) * remainder
                self.avg[i] = price
                self.open_t[i] = t
                self.trade_pnl[i] = -(commission - c_close)
            elif new == 0.0:
                self.units[i] = 0.0
                self.avg[i] = 0.0
                self.open_t[i] = -1
                self.trade_pnl[i] = 0.0
            else:
                self.units[i] = new
        self.balance -= commission
        self.volume_lots += abs(du) / spec.contract_size
        spread_cost = abs(du) * abs(ref_quote - ref_mid) * conv
        slip_cost = abs(du) * abs(price - ref_quote) * conv
        self._add_cost("spread", spread_cost)
        self._add_cost("slippage", slip_cost)
        self._add_cost("commission", commission)
        self._event(t, "fill", i=i, du=float(du), price=float(price), conv=float(conv),
                    realized=float(realized), commission=float(commission), reason=reason,
                    balance=float(self.balance))

    def _close_trade(self, t, i, price, closing, pnl):
        self.closed_trades += 1
        self.trades.append(dict(i=i, symbol=self.u.specs[i].symbol, side=int(np.sign(self.units[i])),
                                open_t=int(self.open_t[i]), close_t=int(t), entry=float(self.avg[i]),
                                exit=float(price), units=float(closing), pnl=float(pnl)))
        self._event(t, "trade_closed", i=i, pnl=float(pnl))

    def apply_swaps(self, t: int, p: Prices, weekday: int) -> None:
        for i, spec in enumerate(self.u.specs):
            if self.units[i] == 0.0 or spec.swap_free:
                continue
            rate = spec.swap_long if self.units[i] > 0 else spec.swap_short
            if rate == 0.0:
                continue
            mult = 3 if weekday == spec.triple_swap_weekday else 1
            amount = abs(self.units[i]) * p.mid[i] * p.conv[i] * rate / 360.0 * mult
            self.balance += amount
            self.trade_pnl[i] += amount
            self._add_cost("swap", -amount)
            self._event(t, "swap", i=i, amount=float(amount), mult=mult, balance=float(self.balance))

    # ------------------------------------------------------------------ forced closes
    def _close_at(self, t: int, i: int, mark: float, conv: float, mid: float, range_frac: float,
                  reason: str) -> None:
        """Close position i at `mark` (bid for longs / ask for shorts) with adverse slippage."""
        du = -self.units[i]
        s = self.profile.slippage.draw(self.rng, range_frac)
        price = mark * (1 - s) if du < 0 else mark * (1 + s)
        self.fill(t, i, du, price, conv, mid, mark, reason)

    def die(self, t: int, reason: str, marks: np.ndarray, conv: np.ndarray, mids: np.ndarray,
            range_frac: np.ndarray) -> None:
        for i in np.flatnonzero(self.units != 0.0):
            self._close_at(t, int(i), float(marks[i]), float(conv[i]), float(mids[i]),
                           float(range_frac[i]), reason)
        self.status = DEAD
        self.death_reason = reason
        self.death_t = int(t)
        self._event(t, "death", reason=reason, balance=float(self.balance))
        self._apply_nbp(t)

    def _apply_nbp(self, t: int) -> None:
        if self.balance < 0 and not self.has_positions:
            if self.profile.nbp:
                debt = self.balance
                self.balance = 0.0
                self._event(t, "nbp_reset", debt=float(debt))
            else:
                self._event(t, "negative_balance", debt=float(self.balance))
