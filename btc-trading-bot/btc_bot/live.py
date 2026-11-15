"""Live / paper trading loop.

Each time a candle closes:
  1. pull recent closed candles from the feed, compute indicators
  2. read the bot's position + equity from the broker (the broker is the source of truth)
  3. apply risk limits, ask the strategy for a decision, send orders
Between candles it polls every `poll_seconds` to enforce client-side stops for brokers
that don't hold stops server-side.

Safety: create a file named STOP in the state dir to block new entries (open positions
are still managed). Live brokers refuse real-money accounts unless allow_real_money: true.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .brokers import Broker, BrokerError
from .data import CoinbaseFeed, timeframe_seconds
from .risk import RiskManager, RiskState
from .strategies import Action, Strategy

log = logging.getLogger(__name__)


class LiveRunner:
    def __init__(self, strategy: Strategy, broker: Broker, feed: CoinbaseFeed, risk: RiskManager,
                 timeframe: str, state_dir: str | Path, bot_name: str,
                 poll_seconds: int = 30, bar_close_delay: int = 10):
        self.strategy, self.broker, self.feed, self.risk = strategy, broker, feed, risk
        self.timeframe, self.bot_name = timeframe, bot_name
        self.tf_seconds = timeframe_seconds(timeframe)
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / f"{bot_name}.runner.json"
        self.kill_file = self.state_dir / "STOP"
        self.poll_seconds, self.bar_close_delay = poll_seconds, bar_close_delay
        self.last_bar: str = ""
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.last_bar = data.get("last_bar", "")
            self.risk.state = RiskState(**data.get("risk", {}))

    def _save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(
            {"last_bar": self.last_bar, "risk": self.risk.state.to_dict()}, indent=2))

    # -- one candle --------------------------------------------------------
    def on_bar_close(self) -> str:
        """Process the latest closed candle once. Returns a short status string."""
        bars = self.feed.recent(self.timeframe, self.strategy.warmup + 100)
        if bars.empty:
            return "no data"
        bar_time = str(bars.index[-1])
        if bar_time == self.last_bar:
            return f"bar {bar_time} already processed"
        data = self.strategy.prepare(bars)
        row = data.iloc[-1]

        pos = self.broker.position()
        side = pos.side if pos else 0
        equity = self.broker.equity()
        self.risk.on_bar(datetime.now(timezone.utc).date(), equity)

        status = self._act(row, side, equity)
        self.last_bar = bar_time
        self._save()
        log.info("bar %s close=%.2f equity=%.2f pos=%s -> %s", bar_time, row["close"], equity,
                 {1: "LONG", -1: "SHORT", 0: "FLAT"}[side], status)
        return status

    def _act(self, row, side: int, equity: float) -> str:
        if side and self.risk.cfg.flatten_on_daily_limit and self.risk.daily_limit_hit(equity):
            self.broker.close("daily loss limit")
            return "flattened: daily loss limit"

        decision = self.strategy.decide(row, side)
        if decision.action == Action.HOLD:
            return "hold"
        if decision.action == Action.EXIT:
            if side:
                self.broker.close(decision.reason)
                return f"exit ({decision.reason})"
            return "hold"

        want = 1 if decision.action == Action.ENTER_LONG else -1
        if want == side:
            return "hold (already in position)"
        if side:
            self.broker.close(f"reversal: {decision.reason}")
            side = 0
            equity = self.broker.equity()

        if self.kill_file.exists():
            return f"entry blocked: kill switch {self.kill_file} present"
        ok, why = self.risk.can_open(equity)
        if not ok:
            return f"entry blocked: {why}"

        q = self.broker.quote()
        ref = q.ask if want > 0 else q.bid
        qty = self.risk.position_size(equity, ref, decision.stop_distance)
        if qty <= 0:
            return "entry skipped: size below minimum"
        stop = ref - want * decision.stop_distance
        try:
            self.broker.open(want, qty, stop, comment=f"{self.bot_name}")
        except BrokerError as exc:
            log.error("Order failed: %s", exc)
            return f"entry failed: {exc}"
        self.risk.record_open()
        return f"{'LONG' if want > 0 else 'SHORT'} {qty:.6f} BTC @~{ref:.2f} stop {stop:.2f} ({decision.reason})"

    # -- client-side stops -------------------------------------------------
    def check_stop(self) -> bool:
        if self.broker.server_side_stops:
            return False
        pos = self.broker.position()
        if not pos or not pos.stop_price:
            return False
        q = self.broker.quote()
        hit = (pos.side > 0 and q.bid <= pos.stop_price) or (pos.side < 0 and q.ask >= pos.stop_price)
        if hit:
            log.warning("Stop hit at bid %.2f / ask %.2f (stop %.2f), closing", q.bid, q.ask, pos.stop_price)
            self.broker.close("stop")
        return hit

    # -- main loop ---------------------------------------------------------
    def next_bar_due(self, now: float) -> float:
        return (now // self.tf_seconds + 1) * self.tf_seconds + self.bar_close_delay

    def run(self, once: bool = False) -> None:
        log.info("Starting %s: %s on %s via %s", self.bot_name, self.strategy.describe(),
                 self.timeframe, type(self.broker).__name__)
        log.info("Broker: %s", self.broker.connect())
        self.on_bar_close()
        if once:
            return
        due = self.next_bar_due(time.time())
        while True:
            try:
                now = time.time()
                if now >= due:
                    self.on_bar_close()
                    due = self.next_bar_due(now)
                else:
                    self.check_stop()
            except KeyboardInterrupt:
                raise
            except Exception:  # keep the bot alive through transient API errors
                log.exception("Loop error; retrying next poll")
            time.sleep(max(1.0, min(self.poll_seconds, due - time.time())))
