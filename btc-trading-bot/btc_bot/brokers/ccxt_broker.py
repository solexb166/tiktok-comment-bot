"""Crypto-exchange broker via CCXT (Binance, Bybit, OKX, Kraken, ...).

Defaults to the exchange's sandbox/testnet. Spot markets are long-only.
The bot keeps a client-side stop (checked every poll) and also tries to place an
exchange-side stop order where the exchange supports CCXT's unified `stopLossPrice`.

API keys: env CCXT_API_KEY / CCXT_SECRET (and CCXT_PASSWORD for exchanges that need one).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .base import Broker, BrokerError, Position, Quote, RealMoneyBlocked

log = logging.getLogger(__name__)


class CcxtBroker(Broker):
    server_side_stops = False   # we can't rely on every exchange supporting stop orders

    def __init__(self, exchange: str = "binance", symbol: str = "BTC/USDT", sandbox: bool = True,
                 allow_real_money: bool = False, state_file: str | Path = "state/ccxt.json",
                 options: dict | None = None):
        try:
            import ccxt  # noqa: WPS433 (optional dependency)
        except ImportError as exc:
            raise BrokerError("pip install ccxt to use broker.type=ccxt") from exc
        if not sandbox and not allow_real_money:
            raise RealMoneyBlocked("ccxt.sandbox is false but allow_real_money is not set. Refusing.")
        cls = getattr(ccxt, exchange)
        self.ex = cls({
            "apiKey": os.environ.get("CCXT_API_KEY"),
            "secret": os.environ.get("CCXT_SECRET"),
            "password": os.environ.get("CCXT_PASSWORD"),
            "enableRateLimit": True,
            "options": options or {},
        })
        if sandbox:
            self.ex.set_sandbox_mode(True)
        self.symbol, self.sandbox = symbol, sandbox
        self.state_file = Path(state_file)   # remembers entry price / stop for our position
        self.market: dict = {}

    def _state(self) -> dict:
        return json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def _save(self, data: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(data, indent=2))

    def connect(self) -> str:
        self.ex.load_markets()
        self.market = self.ex.market(self.symbol)
        mode = "SANDBOX" if self.sandbox else "LIVE (real money)"
        return f"ccxt {self.ex.id} {mode} {self.symbol} ({self.market.get('type')}), equity {self.equity():,.2f}"

    def quote(self) -> Quote:
        t = self.ex.fetch_ticker(self.symbol)
        return Quote(float(t["bid"] or t["last"]), float(t["ask"] or t["last"]))

    def equity(self) -> float:
        bal = self.ex.fetch_balance()
        quote_ccy, base_ccy = self.market["quote"], self.market["base"]
        cash = float(bal.get("total", {}).get(quote_ccy) or 0.0)
        btc = float(bal.get("total", {}).get(base_ccy) or 0.0)
        return cash + btc * self.quote().bid

    def position(self) -> Position | None:
        st = self._state()
        if not st.get("qty"):
            return None
        held = float(self.ex.fetch_balance().get("total", {}).get(self.market["base"]) or 0.0)
        if held < 0.5 * st["qty"]:   # exchange-side stop (or a manual sell) already closed it
            log.info("[ccxt] position no longer held (balance %.6f); clearing state", held)
            self._save({})
            return None
        return Position(1, st["qty"], st["entry_price"], st.get("stop_price"), id=st.get("stop_order_id"))

    def open(self, side: int, qty: float, stop_price: float, comment: str = "") -> Position:
        if side < 0 and self.market.get("spot"):
            raise BrokerError("Shorting is not supported on spot markets; set allow_short: false")
        amount = float(self.ex.amount_to_precision(self.symbol, qty))
        order = self.ex.create_order(self.symbol, "market", "buy" if side > 0 else "sell", amount)
        price = float(order.get("average") or order.get("price") or self.quote().ask)
        stop_id = None
        try:
            stop = self.ex.create_order(self.symbol, "market", "sell", amount, None,
                                        {"stopLossPrice": float(self.ex.price_to_precision(self.symbol, stop_price))})
            stop_id = stop.get("id")
        except Exception as exc:  # exchange-specific; fall back to client-side stop
            log.warning("Exchange-side stop not placed (%s); relying on client-side stop", exc)
        self._save({"qty": amount, "entry_price": price, "stop_price": stop_price, "stop_order_id": stop_id})
        log.info("[ccxt] bought %.6f %s @ %.2f stop %.2f", amount, self.symbol, price, stop_price)
        return self.position()

    def close(self, reason: str = "") -> float | None:
        st = self._state()
        if not st.get("qty"):
            return None
        if st.get("stop_order_id"):
            try:
                self.ex.cancel_order(st["stop_order_id"], self.symbol)
            except Exception as exc:
                log.warning("Could not cancel stop order (%s)", exc)
        base_free = float(self.ex.fetch_balance().get("free", {}).get(self.market["base"]) or 0.0)
        amount = float(self.ex.amount_to_precision(self.symbol, min(st["qty"], base_free)))
        price = None
        if amount > 0:
            order = self.ex.create_order(self.symbol, "market", "sell", amount)
            price = order.get("average") or order.get("price")
        self._save({})
        log.info("[ccxt] sold %.6f %s (%s)", amount, self.symbol, reason)
        return price
