"""Historical data infrastructure: Dukascopy decoding, alignment, hashed store, replay on stored data."""
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tradeworld.agents.baselines import RandomPolicy
from tradeworld.brokerspec import load_profile
from tradeworld.data import align, dukascopy, store
from tradeworld.data.quality import validate
from conftest import run

DAY = date(2024, 1, 8)
DAY_MIN = int(datetime(2024, 1, 8, tzinfo=timezone.utc).timestamp() // 60)


def candles(n, start_price, step, minutes=None):
    minutes = DAY_MIN + np.arange(n) if minutes is None else minutes
    close = start_price + step * np.arange(1, len(minutes) + 1)
    open_ = close - step
    return pd.DataFrame({"open": open_, "close": close, "low": np.minimum(open_, close) - abs(step),
                         "high": np.maximum(open_, close) + abs(step), "volume": 1.5},
                        index=pd.Index(minutes, name="minute"))


def test_url_uses_zero_based_month():
    assert dukascopy.day_url("https://x", "EURUSD", date(2024, 1, 8), "bid") == \
        "https://x/EURUSD/2024/00/08/BID_candles_min_1.bi5"


def test_bi5_round_trip():
    df = candles(120, 1.1, 0.00001)
    raw = dukascopy.encode_bi5(df, DAY, 100000)
    back = dukascopy.decode_bi5(raw, DAY, 100000)
    assert back.index.equals(df.index)
    assert np.allclose(back[["open", "close", "low", "high"]], df[["open", "close", "low", "high"]], atol=1e-9)
    assert dukascopy.decode_bi5(b"", DAY, 100000).empty


def test_load_raw_days_checks_price_divisor(tmp_path):
    d = tmp_path / "raw" / "EURUSD"
    d.mkdir(parents=True)
    for side, off in (("BID", 0.0), ("ASK", 0.0001)):
        (d / f"{DAY.isoformat()}_{side}.bi5").write_bytes(dukascopy.encode_bi5(candles(60, 1.1 + off, 1e-5), DAY, 100000))
    ok = dukascopy.load_raw_days(tmp_path, "EURUSD", DAY, DAY, 100000, (0.5, 2.0))
    assert len(ok["bid"]) == 60 and (ok["ask"]["close"] > ok["bid"]["close"]).all()
    with pytest.raises(ValueError, match="price_divisor"):
        dukascopy.load_raw_days(tmp_path, "EURUSD", DAY, DAY, 1000, (0.5, 2.0))


def test_align_store_and_replay(tmp_path):
    prof = load_profile("environments/A_fractional.yaml")
    frames = {}
    for j, sym in enumerate(prof.universe.symbols):
        base = [2000.0, 1.1, 1.27, 150.0, 40000.0, 16000.0][j]
        minutes = DAY_MIN + np.arange(1440)
        if sym == "XAUUSD":
            minutes = minutes[(minutes - DAY_MIN < 21 * 60) | (minutes - DAY_MIN >= 22 * 60)]  # daily break
        bid = candles(0, base, base * 1e-5, minutes)
        ask = bid.copy()
        for c in ("open", "close", "low", "high"):
            ask[c] = bid[c] * (1 + 1e-4)
        frames[sym] = {"bid": bid, "ask": ask}
    md = align.align(frames, prof.universe.symbols, DAY_MIN, DAY_MIN + 1440)
    gold = prof.universe.index("XAUUSD")
    assert not md.quoted[21 * 60: 22 * 60, gold].any() and md.quoted[:21 * 60, gold].all()
    assert not [i for i in validate(md) if i.severity == "error"]
    digest = store.save(md, tmp_path / "s")
    md2 = store.load(tmp_path / "s")
    assert md2.content_hash() == digest
    res = run(md2, prof, {"r": RandomPolicy(6, 1)}, interval=5, resolution=5, window=12)
    assert len(res.records) > 100
    # tampering is detected
    df = pd.read_parquet(tmp_path / "s" / "bars.parquet")
    df.loc[5, "bid_c"] *= 1.01
    df.to_parquet(tmp_path / "s" / "bars.parquet", index=False)
    with pytest.raises(ValueError, match="hash"):
        store.load(tmp_path / "s")
