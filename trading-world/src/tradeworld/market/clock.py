"""Time utilities. Simulator time is int64 minutes since 1970-01-01 00:00 UTC."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

NY = ZoneInfo("America/New_York")
MIN_PER_DAY = 1440


def to_minute(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() // 60)


def from_minute(m: int) -> datetime:
    return datetime.fromtimestamp(int(m) * 60, tz=timezone.utc)


def weekday_utc(minutes: np.ndarray) -> np.ndarray:
    """0 = Monday ... 6 = Sunday (1970-01-01 was a Thursday)."""
    return ((np.asarray(minutes) // MIN_PER_DAY) + 3) % 7


def minute_of_day_utc(minutes: np.ndarray) -> np.ndarray:
    return np.asarray(minutes) % MIN_PER_DAY


@lru_cache(maxsize=4096)
def rollover_minute(d: date, hour_local: int = 17) -> int:
    """UTC minute of the daily rollover on NY-local date d (17:00 New York, i.e. 21:00 UTC in
    summer / 22:00 UTC in winter, matching Exness's documented swap time)."""
    local = datetime(d.year, d.month, d.day, hour_local, 0, tzinfo=NY)
    return to_minute(local.astimezone(timezone.utc))


def rollover_schedule(times: np.ndarray, hour_local: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """For each M1 bar: is it the rollover bar, and the NY-local weekday of that rollover.
    Only Monday-Friday rollovers are charged (Exness: swaps applied Monday to Friday)."""
    times = np.asarray(times)
    is_roll = np.zeros(times.shape, dtype=bool)
    weekday = np.full(times.shape, -1, dtype=np.int8)
    if times.size == 0:
        return is_roll, weekday
    first = from_minute(int(times[0])).date() - timedelta(days=1)
    last = from_minute(int(times[-1])).date() + timedelta(days=1)
    d = first
    t0 = int(times[0])
    while d <= last:
        if d.weekday() < 5:
            m = rollover_minute(d, hour_local)
            k = m - t0
            if 0 <= k < times.size and times[k] == m:
                is_roll[k] = True
                weekday[k] = d.weekday()
        d += timedelta(days=1)
    return is_roll, weekday
