"""MarketData: aligned M1 bid/ask bars for several instruments on one regular UTC minute grid.

Conventions
-----------
* times[t] is the OPEN minute of M1 bar t; the bar closes at times[t] + 1.
* `quoted[t, i]` is True only where the source had a real quote. Unquoted minutes (weekends,
  session breaks, outages) carry the last known close forward for VALUATION only; the execution
  engine never fills on an unquoted bar.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..errors import DataQualityError

FIELDS = ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c", "volume")


@dataclass
class MarketData:
    times: np.ndarray            # int64 [T]
    symbols: tuple[str, ...]
    bid_o: np.ndarray            # float64 [T, N]
    bid_h: np.ndarray
    bid_l: np.ndarray
    bid_c: np.ndarray
    ask_o: np.ndarray
    ask_h: np.ndarray
    ask_l: np.ndarray
    ask_c: np.ndarray
    volume: np.ndarray
    quoted: np.ndarray           # bool [T, N]

    @property
    def T(self) -> int:
        return int(self.times.shape[0])

    @property
    def N(self) -> int:
        return len(self.symbols)

    def mid_c(self) -> np.ndarray:
        return (self.bid_c + self.ask_c) / 2.0

    def mid_o(self) -> np.ndarray:
        return (self.bid_o + self.ask_o) / 2.0

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_raw(cls, times, symbols, arrays: dict[str, np.ndarray], quoted: np.ndarray,
                 validate: bool = True) -> "MarketData":
        """Build from raw arrays where unquoted cells may hold NaN. Prices are forward-filled
        (then back-filled before the first quote) for valuation; `quoted` marks real quotes."""
        times = np.asarray(times, dtype=np.int64)
        quoted = np.asarray(quoted, dtype=bool).copy()
        out = {}
        for f in FIELDS:
            a = np.array(arrays[f], dtype=np.float64, copy=True)
            if f == "volume":
                a[~quoted] = 0.0
                out[f] = np.nan_to_num(a)
                continue
            a[~quoted] = np.nan
            out[f] = a
        # Forward-fill: an unquoted bar is flat at the previous close (per side).
        for side in ("bid", "ask"):
            close = _ffill_bfill(out[f"{side}_c"])
            prev_close = np.vstack([close[:1], close[:-1]])
            for f in ("o", "h", "l", "c"):
                a = out[f"{side}_{f}"]
                fill = close if f == "c" else prev_close
                # unquoted bar: o=h=l=c = last known close (prev bar's filled close)
                a[~quoted] = np.where(f == "c", close, prev_close)[~quoted]
                out[f"{side}_{f}"] = a
            # bars before first quote: back-filled values above may still be NaN in o/h/l
            for f in ("o", "h", "l"):
                a = out[f"{side}_{f}"]
                nan = np.isnan(a)
                a[nan] = out[f"{side}_c"][nan]
        md = cls(times=times, symbols=tuple(symbols), quoted=quoted, **out)
        if validate:
            from .quality import validate as _validate
            issues = _validate(md)
            errors = [i for i in issues if i.severity == "error"]
            if errors:
                raise DataQualityError("; ".join(str(e) for e in errors[:10]))
        return md

    def slice(self, start: int, stop: int) -> "MarketData":
        kw = {f: getattr(self, f)[start:stop] for f in FIELDS}
        return MarketData(times=self.times[start:stop], symbols=self.symbols,
                          quoted=self.quoted[start:stop], **kw)

    def copy(self) -> "MarketData":
        kw = {f: getattr(self, f).copy() for f in FIELDS}
        return MarketData(times=self.times.copy(), symbols=self.symbols, quoted=self.quoted.copy(), **kw)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.times).tobytes())
        h.update("|".join(self.symbols).encode())
        for f in FIELDS:
            h.update(np.ascontiguousarray(getattr(self, f)).tobytes())
        h.update(np.ascontiguousarray(self.quoted).tobytes())
        return h.hexdigest()


def _ffill_bfill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along axis 0, then back-fill leading NaNs."""
    a = a.copy()
    T = a.shape[0]
    idx = np.where(~np.isnan(a), np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    a = a[idx, np.arange(a.shape[1])[None, :]]
    # back-fill leading NaNs
    for j in range(a.shape[1]):
        col = a[:, j]
        valid = np.flatnonzero(~np.isnan(col))
        if valid.size:
            col[: valid[0]] = col[valid[0]]
    return a
