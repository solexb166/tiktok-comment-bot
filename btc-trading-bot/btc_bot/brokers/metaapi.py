"""MetaApi (metaapi.cloud) REST broker for MetaTrader 4/5 accounts.

Auth: the `auth-token` header carries your MetaApi token (env METAAPI_TOKEN).
The account must already be added and deployed in the MetaApi dashboard.

Positions are tagged with a per-bot `magic` number so several bots can share one account.
Stop-losses are sent with the order, so they live on the broker's server.
"""
from __future__ import annotations

import logging
import math
import os
import uuid

import requests

from .base import Broker, BrokerError, Position, Quote, RealMoneyBlocked

log = logging.getLogger(__name__)

OK_CODES = {"TRADE_RETCODE_DONE", "TRADE_RETCODE_PLACED", "TRADE_RETCODE_DONE_PARTIAL", "ERR_NO_ERROR"}
DEMO_TYPES = {"ACCOUNT_TRADE_MODE_DEMO", "ACCOUNT_TRADE_MODE_CONTEST"}


class MetaApiBroker(Broker):
    server_side_stops = True

    def __init__(self, account_id: str, symbol: str = "BTCUSD", magic: int = 20261115,
                 base_url: str = "https://mt-client-api-v1.new-york.agiliumtrade.ai",
                 token: str | None = None, allow_real_money: bool = False, timeout: float = 30.0,
                 session: requests.Session | None = None):
        if not account_id:
            raise BrokerError("MetaApi account_id is required (config broker.metaapi.account_id "
                              "or env METAAPI_ACCOUNT_ID)")
        self.account_id, self.symbol, self.magic = account_id, symbol, int(magic)
        self.base = f"{base_url.rstrip('/')}/users/current/accounts/{account_id}"
        self.allow_real_money = allow_real_money
        self.timeout = timeout
        self.session = session or requests.Session()
        # When running behind a credential-injecting proxy a placeholder token is fine.
        self.session.headers.update({"auth-token": token or os.environ.get("METAAPI_TOKEN", "placeholder"),
                                     "Content-Type": "application/json"})
        self.spec: dict = {}

    # -- HTTP helpers ------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs):
        resp = self.session.request(method, self.base + path, timeout=self.timeout, **kwargs)
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        if resp.status_code >= 400:
            msg = body.get("message") if isinstance(body, dict) else body
            raise BrokerError(f"MetaApi {method} {path} -> HTTP {resp.status_code}: {msg}")
        return body

    def _trade(self, payload: dict) -> dict:
        payload.setdefault("clientId", f"btcbot-{uuid.uuid4().hex[:12]}")
        result = self._request("POST", "/trade", json=payload)
        code = result.get("stringCode", "")
        if code not in OK_CODES:
            raise BrokerError(f"Trade rejected: {code} {result.get('message', '')}")
        return result

    # -- Broker interface --------------------------------------------------
    def account_information(self) -> dict:
        return self._request("GET", "/account-information")

    def connect(self) -> str:
        info = self.account_information()
        acct_type = info.get("type", "UNKNOWN")
        if acct_type not in DEMO_TYPES and not self.allow_real_money:
            raise RealMoneyBlocked(
                f"Account {info.get('login')} on {info.get('server')} is type {acct_type}, not demo. "
                "Refusing to trade. Set broker.allow_real_money: true only if you really mean it.")
        self.spec = self._request("GET", f"/symbols/{self.symbol}/specification")
        return (f"MetaApi {info.get('platform', '').upper()} {info.get('broker')} login {info.get('login')} "
                f"({acct_type}), balance {info.get('balance')} {info.get('currency')}, "
                f"symbol {self.symbol} contractSize={self.spec.get('contractSize')} "
                f"minVolume={self.spec.get('minVolume')} step={self.spec.get('volumeStep')}")

    def equity(self) -> float:
        return float(self.account_information()["equity"])

    def quote(self) -> Quote:
        p = self._request("GET", f"/symbols/{self.symbol}/current-price")
        return Quote(float(p["bid"]), float(p["ask"]))

    def _bot_positions(self) -> list[dict]:
        positions = self._request("GET", "/positions")
        return [p for p in positions if p.get("symbol") == self.symbol and int(p.get("magic", 0)) == self.magic]

    def position(self) -> Position | None:
        mine = self._bot_positions()
        if not mine:
            return None
        if len(mine) > 1:
            log.warning("Found %d bot positions on %s; using the first", len(mine), self.symbol)
        p = mine[0]
        side = 1 if p["type"] == "POSITION_TYPE_BUY" else -1
        return Position(side, self.lots_to_btc(float(p["volume"])), float(p["openPrice"]),
                        p.get("stopLoss"), id=str(p["id"]))

    def btc_to_lots(self, qty: float) -> float:
        contract = float(self.spec.get("contractSize") or 1.0)
        step = float(self.spec.get("volumeStep") or 0.01)
        lots = math.floor(qty / contract / step + 1e-9) * step
        lots = min(lots, float(self.spec.get("maxVolume") or lots))
        return round(lots, 8) if lots >= float(self.spec.get("minVolume") or step) else 0.0

    def lots_to_btc(self, lots: float) -> float:
        return lots * float(self.spec.get("contractSize") or 1.0)

    def _round_price(self, price: float) -> float:
        digits = self.spec.get("digits")
        return round(price, int(digits)) if digits is not None else price

    def open(self, side: int, qty: float, stop_price: float, comment: str = "") -> Position:
        lots = self.btc_to_lots(qty)
        if lots <= 0:
            raise BrokerError(f"{qty:.6f} BTC is below the broker minimum volume "
                              f"({self.spec.get('minVolume')} lots x {self.spec.get('contractSize')})")
        result = self._trade({
            "actionType": "ORDER_TYPE_BUY" if side > 0 else "ORDER_TYPE_SELL",
            "symbol": self.symbol,
            "volume": lots,
            "stopLoss": self._round_price(stop_price),
            "magic": self.magic,
            "comment": comment[:26],   # MT comment field is short
        })
        log.info("[metaapi] %s %s lots (%.6f BTC) stop %.2f -> %s", "BUY" if side > 0 else "SELL",
                 lots, self.lots_to_btc(lots), stop_price, result.get("stringCode"))
        return self.position() or Position(side, self.lots_to_btc(lots), 0.0, stop_price,
                                           id=str(result.get("positionId")))

    def close(self, reason: str = "") -> float | None:
        mine = self._bot_positions()
        for p in mine:
            result = self._trade({"actionType": "POSITION_CLOSE_ID", "positionId": str(p["id"])})
            log.info("[metaapi] closed position %s (%s) -> %s", p["id"], reason, result.get("stringCode"))
        return None
