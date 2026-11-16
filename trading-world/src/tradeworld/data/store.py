"""Immutable Parquet store for aligned MarketData, with a content hash in a manifest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .market_data import FIELDS, MarketData


def save(md: MarketData, path: str | Path, meta: dict | None = None) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cols = {"minute": np.repeat(md.times, md.N), "symbol": np.tile(np.array(md.symbols), md.T)}
    for f in FIELDS:
        cols[f] = getattr(md, f).reshape(-1)
    cols["quoted"] = md.quoted.reshape(-1)
    pd.DataFrame(cols).to_parquet(path / "bars.parquet", index=False)
    digest = md.content_hash()
    (path / "manifest.json").write_text(json.dumps(
        {"symbols": list(md.symbols), "start_minute": int(md.times[0]), "minutes": md.T,
         "content_sha256": digest, **(meta or {})}, indent=2))
    return digest


def load(path: str | Path, verify: bool = True) -> MarketData:
    path = Path(path)
    man = json.loads((path / "manifest.json").read_text())
    df = pd.read_parquet(path / "bars.parquet")
    symbols = tuple(man["symbols"])
    T, N = man["minutes"], len(symbols)
    if len(df) != T * N or tuple(df["symbol"].iloc[:N]) != symbols:
        raise ValueError("store layout does not match manifest")
    kw = {f: df[f].to_numpy().reshape(T, N) for f in FIELDS}
    md = MarketData(times=df["minute"].to_numpy()[::N].astype(np.int64), symbols=symbols,
                    quoted=df["quoted"].to_numpy().reshape(T, N), **kw)
    if verify and md.content_hash() != man["content_sha256"]:
        raise ValueError("content hash mismatch: store was modified")
    return md
