"""仓位管理 + 锁仓测试。"""

import pytest

from strategies.intraday.position import (
    IntradayPositionManager, FuturesPosition, LockPair,
)
from strategies.intraday.signal import IntradaySignal


def _make_signal(
    symbol: str = "IF",
    direction: str = "LONG",
    score: int = 70,
    entry_price: float = 4000.0,
    stop_loss: float = 3960.0,
    signal_type: str = "INTRADAY",
) -> IntradaySignal:
    return IntradaySignal(
        symbol=symbol,
        datetime="2026-03-18 02:00:00",
        direction=direction,
        score=score,
        entry_price=entry_price,
        stop_loss=stop_loss,
        signal_type=signal_type,
        components={},
        reason="test",
    )


class TestOpenAndLock:
    """开仓后锁仓测试。"""

    def test_open_and_lock_net_zero(self):
        """开仓后锁仓，净持仓为0。"""
        mgr = IntradayPositionManager()
        sig = _make_signal()

        pos = mgr.open_position(sig, 4000.0)
        assert pos is not None
        assert mgr.get_net_positions() == {"IF": 1}

        # 锁仓
        result = mgr.close_position(
            pos.position_id, 4010.0, "STOP_LOSS",
            use_lock=True, current_time="2026-03-18 03:00:00",
            next_trade_date="20260319",
        )
        assert result is not None
        assert result["action"] == "LOCK"
        assert result["pnl"] == (4010 - 4000) * 1 * 300  # +3000

        # 净持仓应为0（多+空）
        assert mgr.get_net_positions() == {}
        assert len(mgr.lock_pairs) == 1

    def test_direct_close(self):
        """直接平仓（use_lock=False）。"""
        mgr = IntradayPositionManager()
        sig = _make_signal()
        pos = mgr.open_position(sig, 4000.0)

        result = mgr.close_position(
            pos.position_id, 3990.0, "FRIDAY_CLOSE",
            use_lock=False,
        )
        assert result["action"] == "CLOSE"
        assert result["pnl"] == (3990 - 4000) * 1 * 300  # -3000
        assert pos.position_id not in mgr.positions


class TestUnlockNextDay:
    """锁仓对第二天双平测试。"""

    def test_unlock_removes_both(self):
        """解锁后两个持仓都移除。"""
        mgr = IntradayPositionManager()
        sig = _make_signal()
        pos = mgr.open_position(sig, 4000.0)

        mgr.close_position(
            pos.position_id, 4010.0, "EOD_CLOSE",
            use_lock=True, current_time="2026-03-18 06:50:00",
            next_trade_date="20260319",
        )
        assert len(mgr.positions) == 2  # 原始 + 锁

        # 第二天开盘解锁
        actions = mgr.process_daily_open("20260319", {"IF": 4012.0})
        assert len(actions) == 1
        assert actions[0]["action"] == "UNLOCK"
        assert len(mgr.positions) == 0
        assert len(mgr.lock_pairs) == 0


class TestMaxLots:
    """持仓上限测试。"""

    def test_reject_when_total_full(self):
        """总持仓达上限时拒绝开仓。"""
        mgr = IntradayPositionManager({"max_total_lots": 2})

        sig1 = _make_signal(symbol="IF", direction="LONG")
        sig2 = _make_signal(symbol="IH", direction="SHORT",
                            entry_price=2700, stop_loss=2727)

        pos1 = mgr.open_position(sig1, 4000.0)
        pos2 = mgr.open_position(sig2, 2700.0)
        assert pos1 is not None
        assert pos2 is not None
        assert mgr._total_net_lots() == 2

        # 第三个品种应被拒绝
        sig3 = _make_signal(symbol="IM", direction="LONG",
                            entry_price=8000, stop_loss=7920)
        pos3 = mgr.open_position(sig3, 8000.0)
        assert pos3 is None

    def test_no_same_direction_add(self):
        """已有同方向持仓不加仓。"""
        mgr = IntradayPositionManager()
        sig1 = _make_signal(symbol="IF", direction="LONG")
        mgr.open_position(sig1, 4000.0)

        # 同品种同方向应被拒绝
        sig2 = _make_signal(symbol="IF", direction="LONG")
        assert not mgr.can_open("IF", "LONG")


class TestStopLoss:
    """止损检查测试。"""

    def test_long_stop_triggered(self):
        """多头止损触发。"""
        mgr = IntradayPositionManager()
        sig = _make_signal(direction="LONG", entry_price=4000, stop_loss=3960)
        pos = mgr.open_position(sig, 4000.0)

        # 价格跌到止损位
        triggers = mgr.check_stop_loss({"IF": 3955.0})
        assert len(triggers) == 1
        assert triggers[0][0] == pos.position_id

    def test_short_stop_triggered(self):
        """空头止损触发。"""
        mgr = IntradayPositionManager()
        sig = _make_signal(direction="SHORT", entry_price=4000, stop_loss=4040)
        pos = mgr.open_position(sig, 4000.0)

        triggers = mgr.check_stop_loss({"IF": 4045.0})
        assert len(triggers) == 1

    def test_no_trigger_in_range(self):
        """价格在安全范围内不触发。"""
        mgr = IntradayPositionManager()
        sig = _make_signal(direction="LONG", entry_price=4000, stop_loss=3960)
        mgr.open_position(sig, 4000.0)

        triggers = mgr.check_stop_loss({"IF": 3980.0})
        assert len(triggers) == 0


class TestTrailingStop:
    """移动止盈测试。"""

    def test_trailing_stop_moves_up(self):
        """浮盈50bps → 止损移到盈亏平衡。"""
        mgr = IntradayPositionManager()
        sig = _make_signal(direction="LONG", entry_price=4000, stop_loss=3960)
        pos = mgr.open_position(sig, 4000.0)

        # 价格涨50bps = 20点
        mgr.update_trailing_stop({"IF": 4020.0})
        assert pos.stop_loss >= 4000.0  # 至少移到盈亏平衡

    def test_trailing_stop_only_moves_favorably(self):
        """止损只向有利方向移动。"""
        mgr = IntradayPositionManager()
        sig = _make_signal(direction="LONG", entry_price=4000, stop_loss=3960)
        pos = mgr.open_position(sig, 4000.0)

        # 先涨到50bps
        mgr.update_trailing_stop({"IF": 4020.0})
        stop_after_up = pos.stop_loss

        # 然后回落 — 止损不应该降低
        mgr.update_trailing_stop({"IF": 4005.0})
        assert pos.stop_loss >= stop_after_up


class TestEODClose:
    """收盘平仓测试。"""

    def test_intraday_closed_at_eod(self):
        """日内仓14:50后平仓。"""
        mgr = IntradayPositionManager()
        sig = _make_signal(signal_type="INTRADAY")
        pos = mgr.open_position(sig, 4000.0)

        to_close = mgr.check_eod_close(
            "06:50", weekday=2, prices={"IF": 4010.0}
        )
        assert len(to_close) == 1
        assert to_close[0][1] == "EOD_CLOSE"
        assert to_close[0][2] is True  # use_lock

    def test_friday_close_all(self):
        """周五全部直接平仓。"""
        mgr = IntradayPositionManager()
        sig = _make_signal(signal_type="OVERNIGHT")
        pos = mgr.open_position(sig, 4000.0)

        to_close = mgr.check_eod_close(
            "06:50", weekday=4, prices={"IF": 4010.0}
        )
        assert len(to_close) == 1
        assert to_close[0][1] == "FRIDAY_CLOSE"
        assert to_close[0][2] is False  # 直接平仓，不锁仓
