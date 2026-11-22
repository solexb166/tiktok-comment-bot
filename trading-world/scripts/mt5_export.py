"""Export 1-minute price history from a MetaTrader 5 terminal on the owner's own PC (READ ONLY).

What it does
  * Connects to the MT5 terminal that is already open and logged in (it never asks for, reads or stores a password).
  * Finds the broker's symbol for each instrument (e.g. EURUSDm, USTEC) and downloads M1 bars month by month.
  * Writes one file per instrument-month:  datasets/mt5/<INSTRUMENT>/<YYYY-MM>.parquet
      columns: time (UTC minute), bid_o, bid_h, bid_l, bid_c, ask_o, ask_h, ask_l, ask_c, spread_points, tick_volume
      (MT5 bars are bid prices with one spread per minute; ask = bid + spread x point, an approximation that is
       recorded in each file's metadata)
  * Writes datasets/mt5/<INSTRUMENT>/symbol_info.json: the broker's contract specification (digits, point, contract
    size, swap rates, margin settings) - so the simulator can match the broker. No account details are saved.

What it never does: place, modify or cancel orders; read or save account login, balance or personal data.

Usage (on the PC with MT5 open and logged in, from the trading-world folder):
    pip install MetaTrader5 pandas pyarrow
    python scripts/mt5_export.py --start 2022-01 --end 2026-10
Then commit and push the new files under datasets/mt5.
If MT5 returns no bars for old months: in MT5, Tools > Options > Charts > "Max bars in chart" = Unlimited, restart MT5.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS = {   # our name -> broker symbol names to try (first match wins; suffixes such as "m" or "." are handled)
    "EURUSD": ["EURUSD"], "GBPUSD": ["GBPUSD"], "USDJPY": ["USDJPY"], "XAUUSD": ["XAUUSD", "GOLD"],
    "BTCUSD": ["BTCUSD"], "NAS100": ["USTEC", "NAS100", "US100", "NDX100", "USTECH", "NQ100"],
}
SPEC_FIELDS = ("name", "description", "digits", "point", "trade_contract_size", "currency_base", "currency_profit",
               "currency_margin", "volume_min", "volume_max", "volume_step", "swap_long", "swap_short", "swap_mode",
               "swap_rollover3days", "margin_initial", "margin_maintenance", "margin_hedged", "trade_calc_mode",
               "trade_stops_level", "spread", "spread_float", "trade_tick_size", "trade_tick_value")


def find_symbol(mt5, candidates):
    names = [s.name for s in (mt5.symbols_get() or [])]
    for c in candidates:
        for n in names:                                     # exact first, then with a broker suffix/prefix
            if n.upper() == c:
                return n
        for n in names:
            u = n.upper().replace(".", "").replace("#", "")
            if u.startswith(c) and len(u) <= len(c) + 4:
                return n
    return None


def month_iter(start: str, end: str):
    y, m = (int(x) for x in start.split("-"))
    ye, me = (int(x) for x in end.split("-"))
    while (y, m) <= (ye, me):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "mt5"))
    ap.add_argument("--only", nargs="*", help="subset of instruments, e.g. EURUSD XAUUSD")
    a = ap.parse_args()
    try:
        import MetaTrader5 as mt5
        import pandas as pd
    except ImportError:
        sys.exit("Install first:  pip install MetaTrader5 pandas pyarrow")
    if not mt5.initialize():
        sys.exit(f"Could not connect to MT5 (is it open and logged in?): {mt5.last_error()}")
    out = Path(a.out)
    try:
        for ours, cands in INSTRUMENTS.items():
            if a.only and ours not in a.only:
                continue
            sym = find_symbol(mt5, cands)
            if sym is None:
                print(f"{ours}: no matching symbol at this broker, skipped")
                continue
            mt5.symbol_select(sym, True)
            info = mt5.symbol_info(sym)
            d = out / ours
            d.mkdir(parents=True, exist_ok=True)
            spec = {f: getattr(info, f) for f in SPEC_FIELDS if hasattr(info, f)}
            (d / "symbol_info.json").write_text(json.dumps(dict(instrument=ours, broker_symbol=sym, spec=spec,
                                                                exported_at=datetime.now(timezone.utc).isoformat()), indent=1, default=str))
            point = float(info.point)
            for y, m in month_iter(a.start, a.end):
                f = d / f"{y:04d}-{m:02d}.parquet"
                if f.exists():
                    continue
                t0 = datetime(y, m, 1, tzinfo=timezone.utc)
                t1 = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=timezone.utc)
                rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, t0, t1)
                if rates is None or len(rates) == 0:
                    print(f"{ours} {y}-{m:02d}: no bars ({mt5.last_error()})")
                    continue
                df = pd.DataFrame(rates)
                df = df[(df["time"] >= int(t0.timestamp())) & (df["time"] < int(t1.timestamp()))]
                sp = df["spread"].astype(float) * point
                outdf = pd.DataFrame({
                    "time": (df["time"] // 60).astype("int64"),
                    "bid_o": df["open"], "bid_h": df["high"], "bid_l": df["low"], "bid_c": df["close"],
                    "ask_o": df["open"] + sp, "ask_h": df["high"] + sp, "ask_l": df["low"] + sp, "ask_c": df["close"] + sp,
                    "spread_points": df["spread"].astype("int32"), "tick_volume": df["tick_volume"].astype("int64")})
                outdf.attrs["note"] = "MT5 M1 bars are bid prices; ask = bid + bar spread x point (approximation)"
                outdf.to_parquet(f, index=False)
                print(f"{ours} {y}-{m:02d}: {len(outdf):,} bars -> {f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
