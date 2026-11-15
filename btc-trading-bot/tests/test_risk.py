from btc_bot.risk import RiskConfig, RiskManager


def test_size_by_risk():
    rm = RiskManager(RiskConfig(risk_per_trade=0.01, max_leverage=10, max_position_btc=10))
    # $10k equity, $1000 stop -> risk $100 -> 0.1 BTC
    assert rm.position_size(10_000, 50_000, 1_000) == 0.1


def test_size_capped_by_leverage_and_btc_cap():
    rm = RiskManager(RiskConfig(risk_per_trade=0.05, max_leverage=1.0, max_position_btc=10))
    assert rm.position_size(10_000, 50_000, 100) == 0.2          # 1x leverage cap
    rm2 = RiskManager(RiskConfig(risk_per_trade=0.05, max_leverage=100, max_position_btc=0.3))
    assert rm2.position_size(10_000, 50_000, 100) == 0.3         # BTC cap


def test_size_zero_for_bad_stop_or_tiny():
    rm = RiskManager(RiskConfig(min_position_btc=0.001))
    assert rm.position_size(10_000, 50_000, 0) == 0.0
    assert rm.position_size(1, 50_000, 1_000) == 0.0


def test_daily_limit_and_reset():
    rm = RiskManager(RiskConfig(daily_loss_limit=0.03, max_trades_per_day=2))
    rm.on_bar("2025-01-01", 10_000)
    assert rm.can_open(9_800)[0]
    assert not rm.can_open(9_700)[0]          # -3%
    rm.record_open(); rm.record_open()
    assert "max trades" in rm.can_open(10_000)[1]
    rm.on_bar("2025-01-02", 9_700)            # new day resets baseline and count
    assert rm.can_open(9_700)[0]
