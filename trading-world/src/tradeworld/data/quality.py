"""Data-quality checks (architecture §10.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..market.clock import MIN_PER_DAY, weekday_utc


@dataclass(frozen=True)
class Issue:
    severity: str        # "error" | "warning" | "info"
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.detail}"


def validate(md, spread_outlier_mult: float = 20.0) -> list[Issue]:
    issues: list[Issue] = []
    t = md.times
    if t.ndim != 1 or t.size == 0:
        return [Issue("error", "times", "empty or not 1-D")]
    d = np.diff(t)
    if np.any(d <= 0):
        issues.append(Issue("error", "monotonic", f"{int(np.sum(d <= 0))} non-increasing timestamps"))
    if np.any(d != 1):
        issues.append(Issue("error", "regular_grid", "times must be a regular 1-minute grid"))
    q = md.quoted
    for name, arr in (("bid", (md.bid_o, md.bid_h, md.bid_l, md.bid_c)),
                      ("ask", (md.ask_o, md.ask_h, md.ask_l, md.ask_c))):
        o, h, l, c = arr
        if not np.all(np.isfinite(np.stack(arr))):
            issues.append(Issue("error", f"{name}_finite", "non-finite prices"))
            continue
        if np.any(np.stack(arr) <= 0):
            issues.append(Issue("error", f"{name}_positive", "non-positive prices"))
        bad = q & ((l > np.minimum(o, c) + 1e-12) | (h < np.maximum(o, c) - 1e-12) | (l > h))
        if bad.any():
            issues.append(Issue("error", f"{name}_ohlc", f"{int(bad.sum())} bars violate low<=open,close<=high"))
    crossed = q & ((md.bid_o > md.ask_o) | (md.bid_c > md.ask_c) | (md.bid_l > md.ask_l) | (md.bid_h > md.ask_h))
    if crossed.any():
        issues.append(Issue("error", "bid_le_ask", f"{int(crossed.sum())} bars with bid > ask"))
    spread = md.ask_c - md.bid_c
    for j, sym in enumerate(md.symbols):
        s = spread[q[:, j], j]
        if s.size:
            med = np.median(s)
            if med > 0:
                n_out = int(np.sum(s > spread_outlier_mult * med))
                if n_out:
                    issues.append(Issue("warning", "spread_outlier",
                                        f"{sym}: {n_out} bars with spread > {spread_outlier_mult}x median"))
    issues.extend(classify_gaps(md))
    return issues


def classify_gaps(md, min_gap: int = 2) -> list[Issue]:
    """Report runs of unquoted minutes per instrument, classified as weekend / daily-break / outage."""
    out = []
    for j, sym in enumerate(md.symbols):
        unq = ~md.quoted[:, j]
        if not unq.any():
            continue
        edges = np.diff(np.r_[0, unq.astype(np.int8), 0])
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        counts = {"weekend": 0, "daily_break": 0, "outage": 0}
        for s, e in zip(starts, ends):
            n = e - s
            if n < min_gap:
                continue
            wd = int(weekday_utc(md.times[s]))
            if n >= 24 * 60 or wd in (4, 5, 6) and n >= 6 * 60:
                counts["weekend"] += 1
            elif n <= 3 * 60:
                counts["daily_break"] += 1
            else:
                counts["outage"] += 1
        out.append(Issue("info", "gaps", f"{sym}: {counts}"))
        if counts["outage"]:
            out.append(Issue("warning", "outage", f"{sym}: {counts['outage']} unexplained multi-hour gaps"))
    return out
