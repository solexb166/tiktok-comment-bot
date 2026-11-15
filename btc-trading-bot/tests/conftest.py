import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_bars(closes, spread=0.0, start="2025-01-01"):
    closes = np.asarray(closes, dtype=float)
    opens = np.r_[closes[0], closes[:-1]]
    idx = pd.date_range(start, periods=len(closes), freq="1h", tz="UTC", name="time")
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + spread,
                         "low": np.minimum(opens, closes) - spread, "close": closes,
                         "volume": 1.0}, index=idx)


@pytest.fixture
def bars_factory():
    return make_bars
