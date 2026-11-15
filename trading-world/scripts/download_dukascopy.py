"""Download Dukascopy M1 bid/ask candles and build an aligned, hashed Parquet store.

Run on your own machine (Dukascopy rate-limits shared cloud IPs):
    python scripts/download_dukascopy.py --start 2018-01-01 --end 2018-01-31 --out data/dukascopy
"""
import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from tradeworld.data import align, dukascopy, store  # noqa: E402
from tradeworld.data.quality import validate  # noqa: E402


def val(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, type=date.fromisoformat)
    ap.add_argument("--end", required=True, type=date.fromisoformat)
    ap.add_argument("--out", default=str(ROOT / "data" / "dukascopy"))
    ap.add_argument("--symbols", nargs="*", help="subset of instruments (default: all six)")
    ap.add_argument("--skip-download", action="store_true", help="only decode/align existing raw files")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "data" / "dukascopy.yaml").read_text())
    symbols = args.symbols or list(cfg["instruments"])
    frames = {}
    for sym in symbols:
        ic = cfg["instruments"][sym]
        if not args.skip_download:
            print(f"downloading {sym} ({ic['source']}) {args.start}..{args.end}")
            dukascopy.download_range(sym, ic["source"], args.start, args.end, args.out, cfg["base_url"],
                                     pause=cfg["request_pause_seconds"])
        frames[sym] = dukascopy.load_raw_days(args.out, sym, args.start, args.end,
                                              float(val(ic["price_divisor"])), tuple(ic["range"]))
    start_min = int(datetime(args.start.year, args.start.month, args.start.day, tzinfo=timezone.utc).timestamp() // 60)
    end_min = int(datetime(args.end.year, args.end.month, args.end.day, tzinfo=timezone.utc).timestamp() // 60) + 1440
    md = align.align(frames, symbols, start_min, end_min, bool(val(cfg["zero_volume_is_unquoted"])))
    for issue in validate(md):
        print(issue)
    dest = Path(args.out) / "aligned" / f"{args.start}_{args.end}"
    digest = store.save(md, dest, meta={"source": "dukascopy", "symbols_source": {s: cfg["instruments"][s]["source"] for s in symbols}})
    print(f"saved {md.T} minutes x {md.N} instruments to {dest} (sha256 {digest[:12]})")


if __name__ == "__main__":
    main()
