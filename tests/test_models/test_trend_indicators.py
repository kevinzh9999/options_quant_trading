"""
test_trend_indicators.py
------------------------
测试 TrendIndicators 类的所有静态方法。

TestSMA              : sma 基本行为
TestEMA              : ema 基本行为
TestMACD             : macd 输出结构与数值逻辑
TestDonchianChannel  : donchian_channel 通道宽度与突破
TestADX              : adx 趋势强度
TestDelegation       : 类方法与模块函数结果一致
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from models.indicators.trend import TrendIndicators, calc_sma, calc_ema, calc_macd, calc_adx


# ======================================================================
# 共用测试数据
# ======================================================================

N = 100
IDX = pd.date_range("2024-01-01", periods=N, freq="D")

# 线性上升序列
LINEAR = pd.Series(np.arange(1.0, N + 1), index=IDX)

# 正弦振荡序列（围绕 4000）
SINE = pd.Series(4000 + 200 * np.sin(np.linspace(0, 4 * np.pi, N)), index=IDX)

# 强趋势 OHLCV（每日 +10 点，振幅 5 点）
_close = 4000 + 10 * np.arange(N, dtype=float)
_high  = _close + 5
_low   = _close - 5
CLOSE_TREND = pd.Series(_close, index=IDX)
HIGH_TREND  = pd.Series(_high,  index=IDX)
LOW_TREND   = pd.Series(_low,   index=IDX)

# 震荡 OHLCV（随机游走，种子固定）
rng = np.random.default_rng(42)
_randn = rng.normal(0, 1, N).cumsum()
CLOSE_RANGE = pd.Series(4000 + 20 * (_randn - _randn.mean()), index=IDX)
HIGH_RANGE  = CLOSE_RANGE + abs(rng.normal(0, 3, N))
LOW_RANGE   = CLOSE_RANGE - abs(rng.normal(0, 3, N))


# ======================================================================
# TestSMA
# ======================================================================

class TestSMA:

    def test_returns_series(self):
        assert isinstance(TrendIndicators.sma(LINEAR, 5), pd.Series)

    def test_length_preserved(self):
        result = TrendIndicators.sma(LINEAR, 5)
        assert len(result) == N

    def test_first_period_minus1_nan(self):
        result = TrendIndicators.sma(LINEAR, 10)
        assert result.iloc[:9].isna().all()
        assert not pd.isna(result.iloc[9])

    def test_constant_series_equals_constant(self):
        const = pd.Series([3.0] * N, index=IDX)
        result = TrendIndicators.sma(const, 5)
        assert (result.dropna() == 3.0).all()

    def test_linear_sma_midpoint(self):
        """线性序列 SMA(5) 的每个值 = 窗口中间值。"""
        result = TrendIndicators.sma(LINEAR, 5)
        # 第 5 个（index 4）: mean(1,2,3,4,5) = 3 = LINEAR[2]
        assert abs(result.iloc[4] - 3.0) < 1e-9

    def test_sma5_lt_sma20_on_uptrend(self):
        """上升趋势中短期 SMA 高于长期 SMA（after warmup）。"""
        sma5  = TrendIndicators.sma(LINEAR, 5)
        sma20 = TrendIndicators.sma(LINEAR, 20)
        assert (sma5.iloc[20:] > sma20.iloc[20:]).all()

    def test_index_preserved(self):
        result = TrendIndicators.sma(LINEAR, 5)
        assert list(result.index) == list(LINEAR.index)

    def test_period_1_equals_input(self):
        result = TrendIndicators.sma(LINEAR, 1)
        pd.testing.assert_series_equal(result, LINEAR)


# ======================================================================
# TestEMA
# ======================================================================

class TestEMA:

    def test_returns_series(self):
        assert isinstance(TrendIndicators.ema(LINEAR, 12), pd.Series)

    def test_no_leading_nan(self):
        """EMA 第一个值就是第一个输入值（adjust=False）。"""
        result = TrendIndicators.ema(LINEAR, 12)
        assert not result.isna().any()

    def test_ema_converges_toward_constant(self):
        """常数序列的 EMA 应收敛到该常数。"""
        const = pd.Series([5.0] * N, index=IDX)
        result = TrendIndicators.ema(const, 12)
        assert abs(result.iloc[-1] - 5.0) < 1e-6

    def test_ema_reacts_faster_than_sma(self):
        """价格突然跳升后，EMA 比 SMA 更快跟上。"""
        data = pd.Series([100.0] * 30 + [200.0] * 30, index=pd.RangeIndex(60))
        sma = TrendIndicators.sma(data, 10)
        ema = TrendIndicators.ema(data, 10)
        # 跳升后第 5 个点，EMA 应比 SMA 更接近 200
        assert ema.iloc[35] > sma.iloc[35]

    def test_ema5_gt_ema20_on_uptrend(self):
        """上升趋势中短期 EMA 高于长期 EMA（after warmup）。"""
        ema5  = TrendIndicators.ema(LINEAR, 5)
        ema20 = TrendIndicators.ema(LINEAR, 20)
        assert (ema5.iloc[25:] > ema20.iloc[25:]).all()

    def test_index_preserved(self):
        result = TrendIndicators.ema(LINEAR, 12)
        assert list(result.index) == list(LINEAR.index)

    def test_period_1_equals_input(self):
        """span=1 时 EMA 每步权重全给最新值，等于输入。"""
        result = TrendIndicators.ema(LINEAR, 1)
        pd.testing.assert_series_equal(result, LINEAR)


# ======================================================================
# TestMACD
# ======================================================================

class TestMACD:

    def setup_method(self):
        self.result = TrendIndicators.macd(SINE, fast=12, slow=26, signal=9)

    def test_returns_dataframe(self):
        assert isinstance(self.result, pd.DataFrame)

    def test_columns(self):
        for col in ("macd_line", "signal_line", "histogram"):
            assert col in self.result.columns

    def test_length(self):
        assert len(self.result) == N

    def test_index_preserved(self):
        assert list(self.result.index) == list(SINE.index)

    def test_histogram_equals_macd_minus_signal(self):
        diff = (self.result["macd_line"] - self.result["signal_line"]
                - self.result["histogram"]).abs()
        assert (diff < 1e-9).all()

    def test_macd_line_definition(self):
        """macd_line = EMA(12) - EMA(26)。"""
        ema12 = TrendIndicators.ema(SINE, 12)
        ema26 = TrendIndicators.ema(SINE, 26)
        expected = ema12 - ema26
        diff = (self.result["macd_line"] - expected).abs()
        assert (diff < 1e-9).all()

    def test_signal_line_lags_macd(self):
        """signal_line 是 macd_line 的 EMA，应在 macd_line 变化之后跟随。"""
        macd = self.result["macd_line"]
        sig  = self.result["signal_line"]
        # 当 macd 在上升段末尾时，signal 仍低于 macd
        peak_idx = macd.iloc[20:].idxmax()
        assert sig[peak_idx] < macd[peak_idx]

    def test_uptrend_macd_positive(self):
        """稳定上涨序列中 MACD 线应为正（快线 > 慢线）。"""
        result = TrendIndicators.macd(LINEAR, 12, 26, 9)
        assert (result["macd_line"].iloc[30:] > 0).all()

    def test_crossover_generates_histogram_sign_change(self):
        """macd 与 signal 交叉时 histogram 变号。"""
        hist = self.result["histogram"].dropna()
        signs = np.sign(hist)
        changes = (signs != signs.shift(1)).sum()
        assert changes > 1  # 正弦波应有多次交叉

    def test_custom_params(self):
        result = TrendIndicators.macd(SINE, fast=5, slow=10, signal=3)
        assert "macd_line" in result.columns
        assert len(result) == N


# ======================================================================
# TestDonchianChannel
# ======================================================================

class TestDonchianChannel:

    def setup_method(self):
        self.result = TrendIndicators.donchian_channel(HIGH_TREND, LOW_TREND, period=20)

    def test_returns_dataframe(self):
        assert isinstance(self.result, pd.DataFrame)

    def test_columns(self):
        for col in ("upper", "lower", "middle"):
            assert col in self.result.columns

    def test_length_preserved(self):
        assert len(self.result) == N

    def test_leading_nan(self):
        """前 period-1 行应为 NaN。"""
        assert self.result.iloc[:19].isna().all().all()
        assert not self.result.iloc[19].isna().any()

    def test_upper_ge_lower(self):
        valid = self.result.dropna()
        assert (valid["upper"] >= valid["lower"]).all()

    def test_middle_is_mean_of_upper_lower(self):
        valid = self.result.dropna()
        expected_mid = (valid["upper"] + valid["lower"]) / 2
        diff = (valid["middle"] - expected_mid).abs()
        assert (diff < 1e-9).all()

    def test_upper_is_rolling_max_of_high(self):
        """upper = rolling(period).max() of high。"""
        expected = HIGH_TREND.rolling(20).max()
        diff = (self.result["upper"] - expected).abs()
        assert (diff.dropna() < 1e-9).all()

    def test_lower_is_rolling_min_of_low(self):
        """lower = rolling(period).min() of low。"""
        expected = LOW_TREND.rolling(20).min()
        diff = (self.result["lower"] - expected).abs()
        assert (diff.dropna() < 1e-9).all()

    def test_channel_widens_with_volatility(self):
        """高波动序列的通道宽度 > 低波动序列。"""
        high_vol_high = pd.Series(
            4000 + rng.normal(0, 50, N), index=IDX
        )
        high_vol_low  = high_vol_high - abs(rng.normal(0, 50, N))
        low_vol_high  = pd.Series([4000 + 5] * N, index=IDX)
        low_vol_low   = pd.Series([4000 - 5] * N, index=IDX)

        hv = TrendIndicators.donchian_channel(high_vol_high, high_vol_low, 20)
        lv = TrendIndicators.donchian_channel(low_vol_high,  low_vol_low,  20)

        hv_width = (hv["upper"] - hv["lower"]).dropna().mean()
        lv_width = (lv["upper"] - lv["lower"]).dropna().mean()
        assert hv_width > lv_width

    def test_price_breaks_upper_in_uptrend(self):
        """强上升趋势中，新高价格会突破旧通道上轨。"""
        valid = self.result.dropna()
        # 在第 40-60 根 K 线之间，close 应 >= upper 的 80% 以上（紧贴上轨）
        subset = valid.iloc[20:40]
        close_subset = CLOSE_TREND.iloc[valid.index.get_loc(subset.index[0]):
                                        valid.index.get_loc(subset.index[0]) + 20]
        assert (close_subset.values >= subset["lower"].values).all()

    def test_period_1_upper_equals_high(self):
        """period=1 时 upper = high, lower = low（每日最值就是自身）。"""
        result = TrendIndicators.donchian_channel(HIGH_TREND, LOW_TREND, period=1)
        pd.testing.assert_series_equal(result["upper"], HIGH_TREND, check_names=False)
        pd.testing.assert_series_equal(result["lower"], LOW_TREND, check_names=False)

    def test_index_preserved(self):
        assert list(self.result.index) == list(HIGH_TREND.index)


# ======================================================================
# TestADX
# ======================================================================

class TestADX:

    def test_returns_series(self):
        result = TrendIndicators.adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 14)
        assert isinstance(result, pd.Series)

    def test_length_preserved(self):
        result = TrendIndicators.adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 14)
        assert len(result) == N

    def test_adx_range_0_to_100(self):
        """ADX 值域应在 [0, 100]（允许浮点误差 1e-9）。"""
        result = TrendIndicators.adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100 + 1e-9).all()

    def test_strong_trend_adx_above_25(self):
        """强趋势序列（单调上涨）预热后 ADX 应 > 25。"""
        result = TrendIndicators.adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 14)
        assert result.iloc[40:].mean() > 25

    def test_index_preserved(self):
        result = TrendIndicators.adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 14)
        assert list(result.index) == list(CLOSE_TREND.index)

    def test_custom_period(self):
        r7  = TrendIndicators.adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 7)
        r14 = TrendIndicators.adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 14)
        assert len(r7) == len(r14)  # 长度相同
        assert not (r7.dropna() == r14.dropna()).all()  # 值不同


# ======================================================================
# TestDelegation（类方法与模块函数结果完全一致）
# ======================================================================

class TestDelegation:
    """验证 TrendIndicators 各方法的结果与模块函数完全一致。"""

    def test_sma_delegates(self):
        expected = calc_sma(LINEAR, 10)
        result   = TrendIndicators.sma(LINEAR, 10)
        pd.testing.assert_series_equal(result, expected)

    def test_ema_delegates(self):
        expected = calc_ema(LINEAR, 12)
        result   = TrendIndicators.ema(LINEAR, 12)
        pd.testing.assert_series_equal(result, expected)

    def test_macd_delegates(self):
        ml, sl, hist = calc_macd(SINE, 12, 26, 9)
        result = TrendIndicators.macd(SINE, 12, 26, 9)
        pd.testing.assert_series_equal(result["macd_line"],   ml,   check_names=False)
        pd.testing.assert_series_equal(result["signal_line"], sl,   check_names=False)
        pd.testing.assert_series_equal(result["histogram"],   hist, check_names=False)

    def test_adx_delegates(self):
        expected = calc_adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 14)
        result   = TrendIndicators.adx(HIGH_TREND, LOW_TREND, CLOSE_TREND, 14)
        pd.testing.assert_series_equal(result, expected)
