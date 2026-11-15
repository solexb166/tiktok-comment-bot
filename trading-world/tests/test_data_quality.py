"""§10.1 data-quality checks, sessions and the rollover clock."""
from datetime import date

import numpy as np
import pytest

from tradeworld.data.market_data import MarketData
from tradeworld.data.quality import validate
from tradeworld.data.synthetic import generate, session_mask
from tradeworld.errors import DataQualityError
from tradeworld.market.clock import from_minute, rollover_minute, rollover_schedule, weekday_utc
from conftest import MON_2024_01_08


def test_synthetic_is_valid_and_deterministic(universe6):
    a = generate(universe6, MON_2024_01_08, 3 * 1440, seed=7)
    b = generate(universe6, MON_2024_01_08, 3 * 1440, seed=7)
    c = generate(universe6, MON_2024_01_08, 3 * 1440, seed=8)
    assert a.content_hash() == b.content_hash() != c.content_hash()
    assert not [i for i in validate(a) if i.severity == "error"]
    assert np.all(a.bid_c <= a.ask_c)


def _corrupt(md, field, t, j, value):
    arrays = {f: getattr(md, f).copy() for f in ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o",
                                                  "ask_h", "ask_l", "ask_c", "volume")}
    arrays[field][t, j] = value
    return arrays


@pytest.mark.parametrize("field,value,check", [
    ("bid_c", 9.9, "bid_le_ask"),          # bid above ask
    ("ask_h", 0.5, "ask_ohlc"),            # high below open/close
    ("bid_l", -1.0, "bid_positive"),
])
def test_quality_catches_corruption(universe6, field, value, check):
    md = generate(universe6, MON_2024_01_08, 600, seed=1)
    arrays = _corrupt(md, field, 100, 1, value)
    with pytest.raises(DataQualityError, match=check):
        MarketData.from_raw(md.times, md.symbols, arrays, md.quoted)


def test_irregular_grid_rejected(universe6):
    md = generate(universe6, MON_2024_01_08, 100, seed=1)
    times = md.times.copy()
    times[50:] += 1
    arrays = {f: getattr(md, f) for f in ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h",
                                          "ask_l", "ask_c", "volume")}
    with pytest.raises(DataQualityError, match="regular_grid"):
        MarketData.from_raw(times, md.symbols, arrays, md.quoted)


def test_unquoted_bars_are_flat_at_last_close(universe6):
    md = generate(universe6, MON_2024_01_08, 7 * 1440, seed=3)
    j = md.symbols.index("EURUSD")
    unq = np.flatnonzero(~md.quoted[:, j])
    assert unq.size > 0
    t = unq[10]
    assert md.bid_o[t, j] == md.bid_h[t, j] == md.bid_l[t, j] == md.bid_c[t, j] == md.bid_c[t - 1, j]


def test_sessions():
    sat_noon = MON_2024_01_08 - 2 * 1440 + 12 * 60
    assert not session_mask(np.array([sat_noon]), "fx")[0]
    assert session_mask(np.array([sat_noon]), "crypto")[0]
    tue_2130 = MON_2024_01_08 + 1440 + 21 * 60 + 30
    assert session_mask(np.array([tue_2130]), "fx")[0]
    assert not session_mask(np.array([tue_2130]), "metal")[0]       # daily break
    assert weekday_utc(np.array([MON_2024_01_08]))[0] == 0


def test_rollover_is_17_new_york_dst_aware():
    assert from_minute(rollover_minute(date(2024, 1, 10))).hour == 22   # winter: 22:00 UTC
    assert from_minute(rollover_minute(date(2024, 7, 10))).hour == 21   # summer: 21:00 UTC
    times = MON_2024_01_08 + np.arange(7 * 1440)
    is_roll, wd = rollover_schedule(times)
    assert is_roll.sum() == 5                                         # Mon-Fri only
    assert sorted(wd[is_roll].tolist()) == [0, 1, 2, 3, 4]
