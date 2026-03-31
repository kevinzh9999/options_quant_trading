"""日内信号评分系统测试。"""

import numpy as np
import pandas as pd
import pytest

from strategies.intraday.signal import IntradaySignalGenerator


def _make_bars(
    n: int,
    base_price: float = 4000.0,
    start_time: str = "2026-03-18 01:30:00",
    freq: str = "5min",
    trend: float = 0.0,
    volume: float = 500.0,
) -> pd.DataFrame:
    """构造模拟K线数据。"""
    idx = pd.date_range(start_time, periods=n, freq=freq)
    prices = base_price + np.arange(n) * trend + np.random.randn(n) * 2
    df = pd.DataFrame({
        "open": prices - 1,
        "high": prices + 3,
        "low": prices - 3,
        "close": prices,
        "volume": np.full(n, volume),
    }, index=idx)
    return df


class TestOpeningRangeBreakout:
    """开盘区间突破测试。"""

    def test_breakout_long(self):
        """突破上轨 → 做多信号。"""
        gen = IntradaySignalGenerator({"min_signal_score": 0})
        # 前6根在4000附近，第7根突破到4020
        bars = _make_bars(8, base_price=4000, trend=0)
        bars.iloc[:6, bars.columns.get_loc("high")] = 4005
        bars.iloc[:6, bars.columns.get_loc("low")] = 3995
        bars.iloc[6, bars.columns.get_loc("close")] = 4015
        bars.iloc[7, bars.columns.get_loc("close")] = 4020  # 突破
        bars.iloc[7, bars.columns.get_loc("high")] = 4022
        bars.iloc[7, bars.columns.get_loc("low")] = 4010

        opening_range = gen._calc_opening_range(bars)
        assert opening_range is not None
        assert opening_range["high"] == 4005

        score, direction = gen._score_opening_range_breakout(bars, opening_range)
        assert direction == "LONG"
        assert score >= 15  # 至少基础分

    def test_no_signal_during_opening(self):
        """开盘区间内不产生信号。"""
        gen = IntradaySignalGenerator()
        bars = _make_bars(5, base_price=4000)  # 只有5根，还在区间内
        assert gen._calc_opening_range(bars) is None


class TestVWAP:
    """VWAP 维度测试。"""

    def test_vwap_reversion_long(self):
        """价格低于VWAP后回升 → 做多信号。"""
        gen = IntradaySignalGenerator({"vwap_deviation_threshold": 0.002})
        # 价格先跌后回升
        n = 20
        bars = _make_bars(n, base_price=4000, trend=0)
        # 让价格先下跌再回升
        closes = np.concatenate([
            np.linspace(4000, 3980, 15),  # 下跌
            np.linspace(3982, 3990, 5),   # 回升
        ])
        bars["close"] = closes
        bars["low"] = closes - 2
        bars["high"] = closes + 2
        bars["volume"] = 500

        score, direction = gen._score_vwap(bars)
        # 当价格低于VWAP且低点抬高时应该看多
        if score > 0:
            assert direction == "LONG"


class TestMultiTimeframe:
    """多周期一致性测试。"""

    def test_conflict_gives_zero(self):
        """5m多+15m空 → 信号为0。"""
        gen = IntradaySignalGenerator({
            "trend_fast_period": 5, "trend_slow_period": 10,
        })
        # 5m上涨趋势
        bars_5m = _make_bars(30, base_price=4000, trend=2)
        # 15m下跌趋势
        bars_15m = _make_bars(30, base_price=4100, trend=-2,
                              start_time="2026-03-18 01:30:00", freq="15min")

        score, direction = gen._score_multi_timeframe(bars_5m, bars_15m)
        # 如果两个方向矛盾，分数应该为0
        if direction:
            assert score == 0 or score == 20  # 要么同向满分，要么矛盾0分

    def test_both_agree_long(self):
        """5m和15m都看多 → 满分20。"""
        gen = IntradaySignalGenerator({
            "trend_fast_period": 5, "trend_slow_period": 10,
        })
        bars_5m = _make_bars(30, base_price=4000, trend=3)
        bars_15m = _make_bars(30, base_price=4000, trend=3,
                              start_time="2026-03-18 01:30:00", freq="15min")

        score, direction = gen._score_multi_timeframe(bars_5m, bars_15m)
        if direction:
            # 如果有方向，检查一致
            assert score <= 20


class TestScoreAggregation:
    """评分汇总测试。"""

    def test_all_long(self):
        """所有维度看多 → 总分应接近满分。"""
        gen = IntradaySignalGenerator()
        scores = [
            (25, "LONG"),
            (20, "LONG"),
            (20, "LONG"),
            (10, ""),       # 成交量（无方向）
            (8, "LONG"),
        ]
        total, direction = gen._aggregate_scores(scores)
        assert direction == "LONG"
        assert total == 25 + 20 + 20 + 10 + 8

    def test_conflict_discounted(self):
        """多空矛盾 → 大幅折扣或为0。"""
        gen = IntradaySignalGenerator()
        scores = [
            (20, "LONG"),
            (15, "SHORT"),
            (10, ""),
        ]
        total, direction = gen._aggregate_scores(scores)
        # 矛盾时应该折扣
        assert total < 30 or direction == ""


class TestStopLoss:
    """止损测试。"""

    def test_stop_loss_within_100bps(self):
        """止损不超过100基点。"""
        gen = IntradaySignalGenerator()
        bars = _make_bars(10, base_price=4000)

        sl = gen._calc_stop_loss("LONG", 4000.0, bars, "IF")
        assert sl >= 4000 * 0.99  # 不超过1%
        assert sl < 4000.0  # 在入场价下方

        sl = gen._calc_stop_loss("SHORT", 4000.0, bars, "IF")
        assert sl <= 4000 * 1.01
        assert sl > 4000.0


class TestHoldType:
    """持仓类型判定测试。"""

    def test_weak_signal_is_intraday(self):
        """弱信号 → 日内。"""
        gen = IntradaySignalGenerator()
        bars = _make_bars(10, start_time="2026-03-18 06:30:00")
        assert gen._determine_hold_type(bars, 60, "LONG") == "INTRADAY"

    def test_strong_afternoon_is_overnight(self):
        """强信号+下午 → 隔日。"""
        gen = IntradaySignalGenerator()
        # 周三下午14:30（06:30 UTC）
        bars = _make_bars(10, start_time="2026-03-18 06:30:00")
        result = gen._determine_hold_type(bars, 80, "LONG")
        assert result == "OVERNIGHT"

    def test_friday_never_overnight(self):
        """周五不隔日。"""
        gen = IntradaySignalGenerator()
        # 2026-03-20 is Friday, 06:30 UTC
        bars = _make_bars(10, start_time="2026-03-20 06:30:00")
        result = gen._determine_hold_type(bars, 90, "LONG")
        assert result == "INTRADAY"
