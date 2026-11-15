"""Bar-by-bar backtester.

Rules (chosen to avoid look-ahead bias):
  * Decisions are made on a bar's close and filled at the NEXT bar's open (+ slippage).
  * Protective stops are checked against each bar's high/low; a gap through the stop
    fills at the open (worse price), otherwise at the stop price (+ slippage).
  * Fees are charged on the notional of every fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import bars_per_year
from .risk import RiskConfig, RiskManager
from .strategies import Action, Strategy


@dataclass
class Trade:
    side: int
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    fees: float = 0.0

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.side * self.qty * (self.exit_price - self.entry_price) - self.fees


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    stats: dict = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{**t.__dict__, "pnl": t.pnl} for t in self.trades])


def run_backtest(df: pd.DataFrame, strategy: Strategy, risk: RiskConfig | None = None,
                 initial_equity: float = 10_000.0, fee_bps: float = 6.0,
                 slippage_bps: float = 2.0, timeframe: str = "1h") -> BacktestResult:
    data = strategy.prepare(df)
    rm = RiskManager(risk or RiskConfig())
    fee, slip = fee_bps / 1e4, slippage_bps / 1e4

    balance = initial_equity
    pos: Trade | None = None
    pending: tuple | None = None           # ("open", side, stop_distance, reason) | ("close", reason)
    trades: list[Trade] = []
    equity_curve = np.empty(len(data))

    opens, highs, lows, closes = (data[c].to_numpy() for c in ("open", "high", "low", "close"))
    times = data.index

    def close_pos(i: int, price: float, reason: str) -> None:
        nonlocal balance, pos
        exit_fee = pos.qty * price * fee
        pos.exit_time, pos.exit_price, pos.exit_reason = times[i], price, reason
        pos.fees += exit_fee
        balance += pos.side * pos.qty * (price - pos.entry_price) - exit_fee  # entry fee paid at open
        trades.append(pos)
        pos = None

    for i in range(len(data)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # 1) Fill orders decided at the previous close, at this bar's open.
        if pending is not None:
            kind = pending[0]
            if pos is not None and (kind == "close" or pending[1] != pos.side):
                exit_px = o * (1 - slip) if pos.side > 0 else o * (1 + slip)
                close_pos(i, exit_px, pending[-1])
            if kind == "open" and pos is None:
                side, stop_dist = pending[1], pending[2]
                entry_px = o * (1 + slip) if side > 0 else o * (1 - slip)
                qty = rm.position_size(balance, entry_px, stop_dist)
                if qty > 0:
                    stop_px = entry_px - side * stop_dist
                    entry_fee = qty * entry_px * fee
                    pos = Trade(side, qty, times[i], entry_px, stop_px, fees=entry_fee)
                    balance -= entry_fee
                    rm.record_open()
            pending = None

        # 2) Protective stop, checked against this bar's range.
        if pos is not None:
            if pos.side > 0 and l <= pos.stop_price:
                close_pos(i, min(o, pos.stop_price) * (1 - slip), "stop")
            elif pos.side < 0 and h >= pos.stop_price:
                close_pos(i, max(o, pos.stop_price) * (1 + slip), "stop")

        # 3) Mark to market, apply risk limits, ask the strategy for a decision.
        equity = balance + (pos.side * pos.qty * (c - pos.entry_price) if pos else 0.0)
        rm.on_bar(times[i].date(), equity)
        equity_curve[i] = equity

        if i < strategy.warmup or i == len(data) - 1:
            continue
        side_now = pos.side if pos else 0
        if pos is not None and rm.cfg.flatten_on_daily_limit and rm.daily_limit_hit(equity):
            pending = ("close", "daily loss limit")
            continue
        decision = strategy.decide(data.iloc[i], side_now)
        if decision.action == Action.EXIT and pos is not None:
            pending = ("close", decision.reason or "exit")
        elif decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
            side = 1 if decision.action == Action.ENTER_LONG else -1
            if side != side_now:
                ok, _ = rm.can_open(equity)
                if ok:
                    pending = ("open", side, decision.stop_distance, decision.reason or "reversal")
                elif pos is not None:
                    pending = ("close", "opposite signal (new entry blocked by risk)")

    # Close anything still open at the final close so stats are realised.
    if pos is not None:
        close_pos(len(data) - 1, closes[-1], "end of data")
        equity_curve[-1] = balance

    equity = pd.Series(equity_curve, index=times, name="equity")
    result = BacktestResult(equity=equity, trades=trades)
    result.stats = compute_stats(equity, trades, data["close"], initial_equity, timeframe)
    return result


def compute_stats(equity: pd.Series, trades: list[Trade], close: pd.Series,
                  initial_equity: float, timeframe: str) -> dict:
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).total_seconds() / (365 * 24 * 3600), 1e-9)
    total_return = equity.iloc[-1] / initial_equity - 1
    drawdown = equity / equity.cummax() - 1
    pnls = np.array([t.pnl for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    sharpe = (rets.mean() / rets.std() * np.sqrt(bars_per_year(timeframe))) if rets.std() > 0 else 0.0
    in_market = sum(((t.exit_time - t.entry_time).total_seconds() for t in trades), 0.0)
    span = (equity.index[-1] - equity.index[0]).total_seconds() or 1.0
    return {
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(((1 + total_return) ** (1 / years) - 1) * 100, 2) if total_return > -1 else -100.0,
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        "avg_trade_pnl": round(float(pnls.mean()), 2) if trades else 0.0,
        "fees_paid": round(float(sum(t.fees for t in trades)), 2),
        "exposure_pct": round(in_market / span * 100, 1),
        "buy_and_hold_return_pct": round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
    }
