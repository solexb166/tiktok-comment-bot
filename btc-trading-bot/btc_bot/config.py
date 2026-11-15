"""YAML config loading. Secrets come from environment variables, never from the YAML."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

from .risk import RiskConfig

DEFAULTS: dict = {
    "bot_name": "btc-bot",
    "feed": {"product": "BTC-USD"},
    "timeframe": "1h",
    "strategy": {"name": "mean_reversion", "params": {}},
    "risk": {},
    "backtest": {"days": 365, "initial_equity": 10_000, "fee_bps": 6, "slippage_bps": 2},
    "broker": {
        "type": "paper",              # paper | metaapi | ccxt
        "allow_real_money": False,    # live brokers refuse non-demo accounts unless true
        "paper": {"initial_equity": 10_000, "fee_bps": 6, "slippage_bps": 2},
        "metaapi": {
            "account_id": "",         # or env METAAPI_ACCOUNT_ID
            "symbol": "BTCUSD",
            "magic": 20261115,
            "base_url": "https://mt-client-api-v1.new-york.agiliumtrade.ai",
        },
        "ccxt": {"exchange": "binance", "symbol": "BTC/USDT", "sandbox": True, "options": {}},
    },
    "live": {"poll_seconds": 30, "state_dir": "state", "log_dir": "logs", "bar_close_delay_seconds": 10},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None) -> dict:
    user = yaml.safe_load(Path(path).read_text()) if path else {}
    cfg = _merge(DEFAULTS, user or {})
    meta = cfg["broker"]["metaapi"]
    meta["account_id"] = meta.get("account_id") or os.environ.get("METAAPI_ACCOUNT_ID", "")
    return cfg


def risk_config(cfg: dict) -> RiskConfig:
    return RiskConfig(**cfg.get("risk", {}))
