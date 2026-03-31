"""日内风控测试。"""

import pytest

from strategies.intraday.risk import IntradayRiskManager
from strategies.intraday.position import IntradayPositionManager
from strategies.intraday.signal import IntradaySignal


def _make_signal(
    symbol: str = "IF",
    direction: str = "LONG",
    entry_price: float = 4000.0,
    stop_loss: float = 3960.0,
    dt: str = "2026-03-18 02:00:00",
) -> IntradaySignal:
    return IntradaySignal(
        symbol=symbol,
        datetime=dt,
        direction=direction,
        score=70,
        entry_price=entry_price,
        stop_loss=stop_loss,
        signal_type="INTRADAY",
        components={},
        reason="test",
    )


class TestDailyLossLimit:
    """当日亏损限额测试。"""

    def test_reject_after_loss_limit(self):
        """累计亏损超限后拒绝开仓。"""
        risk = IntradayRiskManager({"max_daily_loss": 10000})
        pos_mgr = IntradayPositionManager()

        # 模拟亏损
        risk.on_trade_complete(-5000, "IF")
        risk.on_trade_complete(-6000, "IF")  # 累计 -11000

        sig = _make_signal()
        allowed, reason = risk.check_pre_trade(sig, pos_mgr)
        assert not allowed
        assert "限额" in reason


class TestFridayCloseAll:
    """周五关仓与风控时间窗口测试。"""

    def test_no_trade_opening_5min(self):
        """开盘5分钟内不交易。"""
        risk = IntradayRiskManager()
        pos_mgr = IntradayPositionManager()

        sig = _make_signal(dt="2026-03-18 01:32:00")  # 09:32 北京，开盘2分钟
        allowed, reason = risk.check_pre_trade(sig, pos_mgr)
        assert not allowed
        assert "5分钟" in reason

    def test_no_trade_after_1450(self):
        """14:50后不开新仓。"""
        risk = IntradayRiskManager()
        pos_mgr = IntradayPositionManager()

        sig = _make_signal(dt="2026-03-18 06:55:00")  # 14:55 北京
        allowed, reason = risk.check_pre_trade(sig, pos_mgr)
        assert not allowed
        assert "尾盘" in reason


class TestConsecutiveLosses:
    """连续亏损冷静期测试。"""

    def test_cooldown_after_3_losses(self):
        """连续3笔亏损后暂停。"""
        risk = IntradayRiskManager({"max_consecutive_losses": 3})
        pos_mgr = IntradayPositionManager()

        risk.on_trade_complete(-1000, "IF")
        risk.on_trade_complete(-1500, "IF")
        risk.on_trade_complete(-800, "IF")

        sig = _make_signal()
        allowed, reason = risk.check_pre_trade(sig, pos_mgr)
        assert not allowed
        assert "连续亏损" in reason

    def test_win_resets_counter(self):
        """盈利重置连续亏损计数。"""
        risk = IntradayRiskManager({"max_consecutive_losses": 3})
        pos_mgr = IntradayPositionManager()

        risk.on_trade_complete(-1000, "IF")
        risk.on_trade_complete(-1500, "IF")
        risk.on_trade_complete(2000, "IF")  # 盈利，重置

        sig = _make_signal()
        allowed, _ = risk.check_pre_trade(sig, pos_mgr)
        assert allowed


class TestStopLossLimit:
    """止损距离限制测试。"""

    def test_reject_wide_stop(self):
        """止损超过100bps被拒绝。"""
        risk = IntradayRiskManager({"stop_loss_bps": 100})
        pos_mgr = IntradayPositionManager()

        # 150bps止损
        sig = _make_signal(entry_price=4000, stop_loss=3940)
        allowed, reason = risk.check_pre_trade(sig, pos_mgr)
        assert not allowed
        assert "bps" in reason

    def test_accept_tight_stop(self):
        """止损在100bps以内通过。"""
        risk = IntradayRiskManager({"stop_loss_bps": 100})
        pos_mgr = IntradayPositionManager()

        sig = _make_signal(entry_price=4000, stop_loss=3965)  # 87.5 bps
        allowed, _ = risk.check_pre_trade(sig, pos_mgr)
        assert allowed


class TestDailyReset:
    """每日重置测试。"""

    def test_reset_clears_state(self):
        """每日重置清除所有日内统计。"""
        risk = IntradayRiskManager()

        risk.on_trade_complete(-5000, "IF")
        risk.on_trade_complete(-3000, "IF")

        risk.reset_daily()
        pos_mgr = IntradayPositionManager()
        sig = _make_signal()
        allowed, _ = risk.check_pre_trade(sig, pos_mgr)
        assert allowed
