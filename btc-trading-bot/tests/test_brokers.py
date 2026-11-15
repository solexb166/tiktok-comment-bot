import json

import pytest

from btc_bot.brokers import BrokerError, RealMoneyBlocked
from btc_bot.brokers.metaapi import MetaApiBroker
from btc_bot.brokers.paper import PaperBroker


class FakeResp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    """Minimal stand-in for requests.Session that routes by (method, path suffix)."""
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def request(self, method, url, timeout=None, json=None, **kw):
        self.calls.append((method, url, json))
        for (m, suffix), handler in self.routes.items():
            if m == method and url.endswith(suffix):
                body = handler(json) if callable(handler) else handler
                return FakeResp(*body) if isinstance(body, tuple) else FakeResp(body)
        return FakeResp({"message": "not found"}, 404)


SPEC = {"symbol": "BTCUSD", "contractSize": 1, "minVolume": 0.01, "maxVolume": 10,
        "volumeStep": 0.01, "digits": 2}


def make_meta(acct_type="ACCOUNT_TRADE_MODE_DEMO", positions=None, trade_code="TRADE_RETCODE_DONE", **kw):
    positions = positions if positions is not None else []
    routes = {
        ("GET", "/account-information"): {"type": acct_type, "login": "123", "server": "Demo",
                                          "balance": 10000, "equity": 10050, "currency": "USD",
                                          "platform": "mt5", "broker": "TestBroker"},
        ("GET", "/symbols/BTCUSD/specification"): SPEC,
        ("GET", "/symbols/BTCUSD/current-price"): {"bid": 64000.0, "ask": 64020.0},
        ("GET", "/positions"): lambda _: positions,
        ("POST", "/trade"): lambda payload: {"stringCode": trade_code, "message": "ok", "positionId": "77"},
    }
    sess = FakeSession(routes)
    return MetaApiBroker("acc-1", session=sess, **kw), sess


def test_metaapi_demo_guard():
    broker, _ = make_meta(acct_type="ACCOUNT_TRADE_MODE_REAL")
    with pytest.raises(RealMoneyBlocked):
        broker.connect()
    ok, _ = make_meta(acct_type="ACCOUNT_TRADE_MODE_REAL", allow_real_money=True)
    assert "REAL" in ok.connect()
    demo, _ = make_meta()
    assert "DEMO" in demo.connect()


def test_metaapi_requires_account_id():
    with pytest.raises(BrokerError):
        MetaApiBroker("")


def test_metaapi_lot_conversion():
    broker, _ = make_meta()
    broker.connect()
    assert broker.btc_to_lots(0.1234) == 0.12
    assert broker.btc_to_lots(0.005) == 0.0          # below minVolume
    assert broker.btc_to_lots(50) == 10              # capped at maxVolume
    broker.spec = {**SPEC, "contractSize": 0.1}      # some brokers: 1 lot = 0.1 BTC
    assert broker.btc_to_lots(0.25) == 2.5


def test_metaapi_open_payload_and_position_filtering():
    positions = [
        {"id": "1", "symbol": "BTCUSD", "magic": 999, "type": "POSITION_TYPE_BUY", "volume": 1, "openPrice": 1},
        {"id": "2", "symbol": "BTCUSD", "magic": 20261115, "type": "POSITION_TYPE_SELL",
         "volume": 0.2, "openPrice": 64000, "stopLoss": 65000},
    ]
    broker, sess = make_meta(positions=positions)
    broker.connect()
    pos = broker.position()
    assert pos.id == "2" and pos.side == -1 and pos.qty == pytest.approx(0.2)   # ignores other magic
    broker.open(-1, 0.2, 65000.123, comment="a-very-long-comment-that-gets-truncated")
    payload = [c for c in sess.calls if c[0] == "POST"][-1][2]
    assert payload["actionType"] == "ORDER_TYPE_SELL" and payload["volume"] == 0.2
    assert payload["stopLoss"] == 65000.12 and payload["magic"] == 20261115
    assert len(payload["comment"]) <= 26 and payload["clientId"].startswith("btcbot-")
    broker.close("test")
    close_payload = [c for c in sess.calls if c[0] == "POST"][-1][2]
    assert close_payload["actionType"] == "POSITION_CLOSE_ID" and close_payload["positionId"] == "2"


def test_metaapi_rejected_trade_raises():
    broker, _ = make_meta(trade_code="TRADE_RETCODE_NO_MONEY")
    broker.connect()
    with pytest.raises(BrokerError, match="NO_MONEY"):
        broker.open(1, 0.1, 60000)


def test_paper_broker_roundtrip_and_persistence(tmp_path):
    price = {"p": 100.0}
    f = tmp_path / "paper.json"
    b = PaperBroker(lambda: price["p"], f, initial_equity=1000, fee_bps=0, slippage_bps=0, spread_bps=0)
    b.open(1, 2.0, 90.0)
    price["p"] = 110.0
    assert b.equity() == pytest.approx(1020.0)
    b2 = PaperBroker(lambda: price["p"], f, initial_equity=1000, fee_bps=0, slippage_bps=0, spread_bps=0)
    assert b2.position().qty == 2.0                  # state reloaded from disk
    b2.close()
    assert b2.position() is None and b2.equity() == pytest.approx(1020.0)
