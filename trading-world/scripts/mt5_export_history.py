"""Export LONG price history (hourly, 4-hour and daily bars, as far back as the broker has it) from the MetaTrader 5
terminal on the owner's own PC (READ ONLY), for learning slower patterns.

Same rules as scripts/mt5_export.py: connects to the MT5 terminal already open and logged in, never reads or stores a
password or account data, never places, modifies or cancels orders.

Writes one file per instrument and bar size:  datasets/mt5_history/<INSTRUMENT>/<H1|H4|D1>.parquet
  columns: time (UTC minute of the bar's open), bid_o, bid_h, bid_l, bid_c, ask_o, ask_h, ask_l, ask_c (bid + spread x
  point, an approximation), spread_points, tick_volume

Usage (on the PC with MT5 open and logged in, from the trading-world folder):
    python scripts/mt5_export_history.py
Then commit and push the new files under datasets/mt5_history (a few MB in total).
If MT5 returns little history: in MT5, Tools > Options > Charts > "Max bars in chart" = Unlimited, restart MT5, and
open each symbol's chart on that bar size once (scroll left) so the terminal downloads the history.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_export import INSTRUMENTS, ROOT, find_symbol  # noqa: E402

FRAMES = ("H1", "H4", "D1")


def main():
    try:
        import MetaTrader5 as mt5
        import pandas as pd
    except ImportError:
        sys.exit("Install first:  pip install MetaTrader5 pandas pyarrow")
    if not mt5.initialize():
        sys.exit(f"Could not connect to MT5 (is it open and logged in?): {mt5.last_error()}")
    out = ROOT / "datasets" / "mt5_history"
    t0, t1 = datetime(1990, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc)
    try:
        for ours, cands in INSTRUMENTS.items():
            sym = find_symbol(mt5, cands)
            if sym is None:
                print(f"{ours}: no matching symbol at this broker, skipped")
                continue
            mt5.symbol_select(sym, True)
            point = float(mt5.symbol_info(sym).point)
            d = out / ours
            d.mkdir(parents=True, exist_ok=True)
            for fr in FRAMES:
                rates = mt5.copy_rates_range(sym, getattr(mt5, f"TIMEFRAME_{fr}"), t0, t1)
                if rates is None or len(rates) == 0:
                    print(f"{ours} {fr}: no bars ({mt5.last_error()})")
                    continue
                df = pd.DataFrame(rates)
                sp = df["spread"].astype(float) * point
                outdf = pd.DataFrame({
                    "time": (df["time"] // 60).astype("int64"),
                    "bid_o": df["open"], "bid_h": df["high"], "bid_l": df["low"], "bid_c": df["close"],
                    "ask_o": df["open"] + sp, "ask_h": df["high"] + sp, "ask_l": df["low"] + sp, "ask_c": df["close"] + sp,
                    "spread_points": df["spread"].astype("int32"), "tick_volume": df["tick_volume"].astype("int64")})
                outdf.to_parquet(d / f"{fr}.parquet", index=False)
                first = datetime.fromtimestamp(int(df["time"].iloc[0]), timezone.utc).date()
                print(f"{ours} {fr}: {len(outdf):,} bars from {first}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
