"""Command line entry point.

  python -m btc_bot backtest -c configs/mean_reversion.yaml
  python -m btc_bot compare --days 730
  python -m btc_bot check-broker -c configs/metaapi_demo_breakout.yaml
  python -m btc_bot live -c configs/metaapi_demo_breakout.yaml [--once]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .backtest import run_backtest
from .brokers import BrokerError
from .config import load_config, risk_config
from .data import CoinbaseFeed, load_or_download, read_csv
from .live import LiveRunner
from .risk import RiskManager
from .strategies import STRATEGIES, make_strategy


def setup_logging(log_dir: str | None = None, name: str = "btc-bot") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / f"{name}.log"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_broker(cfg: dict, feed: CoinbaseFeed):
    b = cfg["broker"]
    state_dir = Path(cfg["live"]["state_dir"])
    if b["type"] == "paper":
        from .brokers.paper import PaperBroker
        return PaperBroker(lambda: float(feed.recent("1m", 1)["close"].iloc[-1]),
                           state_dir / f"{cfg['bot_name']}.paper.json", **b["paper"])
    if b["type"] == "metaapi":
        from .brokers.metaapi import MetaApiBroker
        return MetaApiBroker(allow_real_money=b["allow_real_money"], **b["metaapi"])
    if b["type"] == "ccxt":
        from .brokers.ccxt_broker import CcxtBroker
        return CcxtBroker(allow_real_money=b["allow_real_money"],
                          state_file=state_dir / f"{cfg['bot_name']}.ccxt.json", **b["ccxt"])
    raise ValueError(f"Unknown broker type {b['type']!r}")


def _load_bars(cfg: dict, args) -> "pd.DataFrame":
    if getattr(args, "csv", None):
        return read_csv(args.csv)
    days = args.days or cfg["backtest"]["days"]
    return load_or_download(CoinbaseFeed(cfg["feed"]["product"]), cfg["timeframe"], days)


def _bt_kwargs(cfg: dict) -> dict:
    bt = cfg["backtest"]
    return dict(risk=risk_config(cfg), initial_equity=bt["initial_equity"], fee_bps=bt["fee_bps"],
                slippage_bps=bt["slippage_bps"], timeframe=cfg["timeframe"])


def cmd_backtest(cfg: dict, args) -> None:
    bars = _load_bars(cfg, args)
    strat = make_strategy(cfg["strategy"]["name"], cfg["strategy"]["params"])
    res = run_backtest(bars, strat, **_bt_kwargs(cfg))
    print(f"\n{strat.describe()} on {cfg['timeframe']} candles ({len(bars)} bars)")
    for k, v in res.stats.items():
        print(f"  {k:<26}{v}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        res.trades_frame().to_csv(out / f"{cfg['bot_name']}_trades.csv", index=False)
        res.equity.to_csv(out / f"{cfg['bot_name']}_equity.csv")
        (out / f"{cfg['bot_name']}_stats.json").write_text(json.dumps(res.stats, indent=2))
        print(f"  saved trades/equity/stats to {out}/")


def cmd_compare(cfg: dict, args) -> None:
    bars = _load_bars(cfg, args)
    rows = []
    for name in STRATEGIES:
        params = cfg["strategy"]["params"] if name == cfg["strategy"]["name"] else {}
        res = run_backtest(bars, make_strategy(name, params), **_bt_kwargs(cfg))
        rows.append((name, res.stats))
    keys = ["total_return_pct", "max_drawdown_pct", "sharpe", "trades", "win_rate_pct",
            "profit_factor", "buy_and_hold_return_pct"]
    print(f"\n{cfg['timeframe']} candles, {bars.index[0]:%Y-%m-%d} -> {bars.index[-1]:%Y-%m-%d}")
    print(f"{'strategy':<16}" + "".join(f"{k:>24}" for k in keys))
    for name, s in rows:
        print(f"{name:<16}" + "".join(f"{s[k]:>24}" for k in keys))


def cmd_check_broker(cfg: dict, args) -> None:
    feed = CoinbaseFeed(cfg["feed"]["product"])
    broker = build_broker(cfg, feed)
    print(broker.connect())
    q = broker.quote()
    print(f"quote: bid {q.bid} ask {q.ask}")
    print(f"equity: {broker.equity()}")
    print(f"bot position: {broker.position()}")


def cmd_live(cfg: dict, args) -> None:
    feed = CoinbaseFeed(cfg["feed"]["product"])
    live = cfg["live"]
    runner = LiveRunner(
        strategy=make_strategy(cfg["strategy"]["name"], cfg["strategy"]["params"]),
        broker=build_broker(cfg, feed), feed=feed, risk=RiskManager(risk_config(cfg)),
        timeframe=cfg["timeframe"], state_dir=live["state_dir"], bot_name=cfg["bot_name"],
        poll_seconds=live["poll_seconds"], bar_close_delay=live["bar_close_delay_seconds"])
    runner.run(once=args.once)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="btc_bot", description="BTC trading bot")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("backtest", "compare", "check-broker", "live"):
        sp = sub.add_parser(name)
        sp.add_argument("-c", "--config", help="YAML config file")
        if name in ("backtest", "compare"):
            sp.add_argument("--days", type=int, help="days of history (default from config)")
            sp.add_argument("--csv", help="use a local OHLCV CSV instead of downloading")
            sp.add_argument("--timeframe", help="override config timeframe")
        if name == "backtest":
            sp.add_argument("--strategy", help="override strategy name")
            sp.add_argument("--out", help="directory to save trades/equity CSVs")
        if name == "live":
            sp.add_argument("--once", action="store_true", help="process one candle and exit")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "timeframe", None):
        cfg["timeframe"] = args.timeframe
    if getattr(args, "strategy", None) and args.strategy != cfg["strategy"]["name"]:
        cfg["strategy"] = {"name": args.strategy, "params": {}}

    setup_logging(cfg["live"]["log_dir"] if args.cmd == "live" else None, cfg["bot_name"])
    try:
        {"backtest": cmd_backtest, "compare": cmd_compare,
         "check-broker": cmd_check_broker, "live": cmd_live}[args.cmd](cfg, args)
    except BrokerError as exc:
        logging.getLogger("btc_bot").error("%s", exc)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
