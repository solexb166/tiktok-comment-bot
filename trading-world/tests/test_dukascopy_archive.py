from datetime import date

import numpy as np
import pytest

from tradeworld.data import dukascopy
from tradeworld.data.dukascopy_archive import archive_path, load_archived, month_days, pack_month, read_month
from test_historical_data import candles, DAY_MIN


def test_pack_read_and_load(tmp_path):
    days = month_days(2024, 1)
    files, statuses = {}, {}
    for day in days:
        for side, off in (("BID", 0.0), ("ASK", 1e-4)):
            name = f"{day.isoformat()}_{side}.bi5"
            if day.weekday() == 5:
                files[name], statuses[name] = b"", "closed"
                continue
            m0 = DAY_MIN + (day - date(2024, 1, 8)).days * 1440
            df = candles(0, 1.1 + off, 1e-5, m0 + np.arange(10))
            files[name], statuses[name] = dukascopy.encode_bi5(df, day, 100000), "ok"
    p1 = pack_month(tmp_path, "EURUSD", 2024, 1, files, statuses)
    b1 = p1.read_bytes()
    p2 = pack_month(tmp_path, "EURUSD", 2024, 1, files, statuses)
    assert p2.read_bytes() == b1                                  # deterministic archive
    manifest, back = read_month(p1)
    assert back == files and manifest[f"{days[5].isoformat()}_BID.bi5"]["status"] == "closed"
    frames = load_archived(tmp_path, "EURUSD", date(2024, 1, 8), date(2024, 1, 9), 100000)
    assert len(frames["bid"]) == 20 and (frames["ask"]["close"] > frames["bid"]["close"]).all()
    with pytest.raises(FileNotFoundError):
        load_archived(tmp_path, "EURUSD", date(2024, 1, 1), date(2024, 2, 2), 100000)


def test_corrupt_archive_detected(tmp_path):
    p = pack_month(tmp_path, "X", 2024, 1, {"2024-01-01_BID.bi5": b"abc"}, {"2024-01-01_BID.bi5": "ok"})
    raw = bytearray(p.read_bytes())
    i = raw.find(b"abc")
    raw[i] = ord("x")
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum"):
        read_month(p)
