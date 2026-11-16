"""Resumable, polite Dukascopy downloader for throttled environments.

Downloads daily BID/ASK M1 files in plan order (configs/data/download_plan.yaml). Finished
instrument-months are packed into datasets/dukascopy/<SYMBOL>/<YYYY-MM>.tar and, with --commit,
committed and pushed immediately, so progress survives container resets. Partial months live in
a scratch work dir. Stops cleanly after --max-minutes.

    python scripts/trickle_download.py --max-minutes 110 --commit
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
import yaml  # noqa: E402

from tradeworld.data.dukascopy import day_url  # noqa: E402
from tradeworld.data.dukascopy_archive import archive_path, month_days, pack_month  # noqa: E402

TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01GPvsEBXtZjzV2SPuWVYBnk\n")
ASSET = {"BTCUSD": "crypto"}


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def commit_and_push(paths, message):
    git("add", *[str(p) for p in paths])
    if git("diff", "--cached", "--quiet").returncode == 0:
        return
    git("commit", "-q", "-m", message + TRAILER)
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    for attempt in range(4):
        r = git("push", "-q", "-u", "origin", branch)
        if r.returncode == 0:
            return
        git("pull", "-q", "--rebase", "origin", branch)
        time.sleep(2 ** (attempt + 1))
    log(f"WARNING: push failed: {r.stderr.strip()[:200]}")


def nap(seconds, deadline):
    """Sleep, but never past the run's time budget."""
    time.sleep(max(0.0, min(seconds, deadline - time.time())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=110)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--work", default="/tmp/tradeworld-dukascopy-work")
    args = ap.parse_args()
    plan = yaml.safe_load((ROOT / "configs/data/download_plan.yaml").read_text())
    src = yaml.safe_load((ROOT / "configs/data/dukascopy.yaml").read_text())
    base, root = src["base_url"], ROOT / plan["archive_dir"]
    work = Path(args.work)
    deadline = time.time() + args.max_minutes * 60
    yesterday = date.today() - timedelta(days=1)
    sess = requests.Session()
    sess.headers["User-Agent"] = "tradeworld-data/0.1"
    stats = {"ok": 0, "empty": 0, "throttled": 0, "months_packed": 0}
    backoff = plan["pause_seconds"]

    for ph in plan["phases"]:
        start = date.fromisoformat(str(ph["start"]))
        end = yesterday if ph["end"] == "yesterday" else date.fromisoformat(str(ph["end"]))
        for y, m in months(start, end):
            if date(y, m, 1) > yesterday:
                continue
            last_day = month_days(y, m)[-1]
            if last_day > yesterday:
                continue                      # only complete months are archived
            for sym in plan["instruments"]:
                if archive_path(root, sym, y, m).exists():
                    continue
                source = src["instruments"][sym]["source"]
                closed = set() if ASSET.get(sym) == "crypto" else {5}
                files, statuses = {}, {}
                for day in month_days(y, m):
                    for side in ("BID", "ASK"):
                        name = f"{day.isoformat()}_{side}.bi5"
                        wpath = work / sym / name
                        if day.weekday() in closed:
                            files[name], statuses[name] = b"", "closed"
                            continue
                        if wpath.exists():
                            meta = json.loads(wpath.with_suffix(".json").read_text())
                            files[name], statuses[name] = wpath.read_bytes(), meta["status"]
                            continue
                        while True:
                            if time.time() > deadline:
                                log(f"time budget reached; stats {stats}")
                                return
                            try:
                                r = sess.get(day_url(base, source, day, side), timeout=30)
                                code = r.status_code
                            except requests.RequestException as e:
                                code = f"conn:{type(e).__name__}"
                            if code == 200 or code == 404:
                                payload = r.content if code == 200 else b""
                                status = "ok" if payload else "empty"
                                wpath.parent.mkdir(parents=True, exist_ok=True)
                                wpath.write_bytes(payload)
                                wpath.with_suffix(".json").write_text(json.dumps({"status": status}))
                                files[name], statuses[name] = payload, status
                                stats[status] += 1
                                backoff = plan["pause_seconds"]
                                nap(plan["pause_seconds"], deadline)
                                break
                            stats["throttled"] += 1
                            backoff = min(backoff * 2, plan["max_backoff_seconds"])
                            log(f"throttled ({code}) on {sym} {name}; retrying in {backoff:.0f}s")
                            nap(backoff, deadline)
                path = pack_month(root, sym, y, m, files, statuses)
                stats["months_packed"] += 1
                log(f"packed {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB); stats {stats}")
                if args.commit:
                    commit_and_push([path], f"Data: Dukascopy {sym} {y:04d}-{m:02d} (raw M1 bid/ask archive)")
    log(f"plan complete; stats {stats}")


if __name__ == "__main__":
    main()
