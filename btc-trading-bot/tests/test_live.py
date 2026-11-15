import numpy as np
import pandas as pd

from btc_bot.brokers.paper import PaperBroker
from btc_bot.live import LiveRunner
from btc_bot.risk import RiskConfig, RiskManager
from btc_bot.strategies.base import Action, Decision, Strategy
from conftest import make_bars


class FakeFeed:
    def __init__(self, bars):
        self.bars = bars

    def recent(self, timeframe, n):
        return self.bars.iloc[-n:]


class OneShot(Strategy):
    name = "oneshot"
    defaults = {"action": Action.ENTER_LONG}
    warmup = 0

    def prepare(self, df):
        return df

    def decide(self, row, position):
        return Decision(self.action, 5.0, "test")


def make_runner(tmp_path, action, price=100.0, risk=None):
    bars = make_bars([100.0] * 10)
    broker = PaperBroker(lambda: price, tmp_path / "p.json", initial_equity=1000,
                         fee_bps=0, slippage_bps=0, spread_bps=0)
    rm = RiskManager(risk or RiskConfig(max_leverage=10, max_position_btc=10))
    runner = LiveRunner(OneShot(action=action), broker, FakeFeed(bars), rm, "1h", tmp_path, "t")
    return runner, broker


def test_entry_sizes_and_sets_stop(tmp_path):
    runner, broker = make_runner(tmp_path, Action.ENTER_LONG)
    status = runner.on_bar_close()
    pos = broker.position()
    assert "LONG" in status and pos.qty == 2.0 and pos.stop_price == 95.0   # 1% of 1000 / 5


def test_same_bar_not_processed_twice(tmp_path):
    runner, broker = make_runner(tmp_path, Action.ENTER_LONG)
    runner.on_bar_close()
    assert "already processed" in runner.on_bar_close()


def test_kill_switch_blocks_entries(tmp_path):
    runner, broker = make_runner(tmp_path, Action.ENTER_LONG)
    (tmp_path / "STOP").touch()
    assert "kill switch" in runner.on_bar_close()
    assert broker.position() is None


def test_reversal_closes_then_opens(tmp_path):
    runner, broker = make_runner(tmp_path, Action.ENTER_LONG)
    runner.on_bar_close()
    runner.strategy.params["action"] = Action.ENTER_SHORT
    runner.last_bar = ""
    runner.on_bar_close()
    assert broker.position().side == -1


def test_client_side_stop(tmp_path):
    price = {"p": 100.0}
    runner, broker = make_runner(tmp_path, Action.ENTER_LONG)
    broker.price_fn = lambda: price["p"]
    runner.on_bar_close()
    price["p"] = 94.0
    assert runner.check_stop() is True and broker.position() is None
