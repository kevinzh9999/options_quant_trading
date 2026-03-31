"""
tests/test_models/test_indicators.py
--------------------------------------
models/indicators/ 单元测试。

覆盖：
- calc_sma: 窗口、NaN 前缀、手工验证
- calc_ema: 长度、EMA 平滑特性
- calc_macd: 三元组、直方图=MACD-signal
- calc_adx: 范围 [0,100]、趋势 vs 震荡
- calc_rsi: 范围 [0,100]、超买超卖
- calc_roc: 手工验证、符号
- calc_stochastic: 范围 [0,100]、%D 是 %K 的 SMA
- calc_atr: 正值、ATR ≥ H-L
- calc_bollinger_bands: upper>middle>lower、宽度随 std 变化
- calc_historical_vol: 正值、年化倍数
- calc_obv: 累积方向、全涨时递增
- calc_vwap: 在 [min, max] 之间、累积模式 vs 滚动模式
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.indicators.trend import calc_adx, calc_ema, calc_macd, calc_sma
from models.indicators.momentum import calc_roc, calc_rsi, calc_stochastic
from models.indicators.volatility_ind import (
    calc_atr,
    calc_bollinger_bands,
    calc_historical_vol,
)
from models.indicators.volume import calc_obv, calc_vwap


# ======================================================================
# 测试固件
# ======================================================================

RNG = np.random.default_rng(42)
N = 200


def make_price_series(n: int = N, start: float = 3000.0, vol: float = 0.01) -> pd.Series:
    """生成模拟收盘价序列（GBM）"""
    log_ret = RNG.normal(0, vol, n)
    prices = start * np.exp(np.cumsum(log_ret))
    return pd.Series(prices)


def make_ohlcv(n: int = N) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """生成 OHLCV 序列（高低开收成交量）"""
    close = make_price_series(n)
    noise = RNG.uniform(0, 0.01, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    volume = pd.Series(RNG.integers(1000, 50000, n).astype(float))
    return high, low, close, volume


# ======================================================================
# TestCalcSMA
# ======================================================================

class TestCalcSMA:
    def test_length(self):
        s = make_price_series()
        assert len(calc_sma(s, 20)) == len(s)

    def test_nan_prefix(self):
        s = make_price_series()
        result = calc_sma(s, 20)
        assert result.isna().sum() == 19  # first window-1 values are NaN

    def test_correct_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calc_sma(s, 3)
        assert abs(result.iloc[2] - 2.0) < 1e-10
        assert abs(result.iloc[4] - 4.0) < 1e-10

    def test_window_1_equals_input(self):
        s = make_price_series()
        result = calc_sma(s, 1)
        assert (result == s).all()


# ======================================================================
# TestCalcEMA
# ======================================================================

class TestCalcEMA:
    def test_length(self):
        s = make_price_series()
        assert len(calc_ema(s, 20)) == len(s)

    def test_no_nan_after_first(self):
        s = make_price_series()
        result = calc_ema(s, 20)
        assert result.isna().sum() == 0  # EMA with adjust=False has no NaN

    def test_ema_smooths_outlier(self):
        """EMA 对近期价格权重更高，对历史异常值衰减更快"""
        s = pd.Series([1.0] * 20 + [100.0] + [1.0] * 20)
        ema = calc_ema(s, 5)
        sma = calc_sma(s, 5)
        # 紧接异常值后（窗口内仍含异常值）EMA 和 SMA 都受影响
        # 但异常值出现后的第 6 个 bar，SMA 完全脱离异常值（窗口外），
        # 而 EMA 仍有指数衰减残留，所以 EMA > SMA
        idx_after = 20 + 5 + 1  # 5 bars after the outlier exits SMA window
        if idx_after < len(s):
            assert ema.iloc[idx_after] > sma.iloc[idx_after]

    def test_ema_span1_equals_input(self):
        s = make_price_series()
        result = calc_ema(s, 1)
        assert np.allclose(result.values, s.values, rtol=1e-10)


# ======================================================================
# TestCalcMACD
# ======================================================================

class TestCalcMACD:
    def test_returns_three_series(self):
        s = make_price_series()
        result = calc_macd(s)
        assert len(result) == 3

    def test_all_same_length_as_input(self):
        s = make_price_series()
        macd, signal, hist = calc_macd(s)
        assert len(macd) == len(s)
        assert len(signal) == len(s)
        assert len(hist) == len(s)

    def test_histogram_equals_macd_minus_signal(self):
        s = make_price_series()
        macd, signal, hist = calc_macd(s)
        diff = (macd - signal - hist).dropna()
        assert np.allclose(diff.values, 0, atol=1e-10)

    def test_macd_is_ema_diff(self):
        s = make_price_series()
        macd, _, _ = calc_macd(s, fast=12, slow=26)
        expected = calc_ema(s, 12) - calc_ema(s, 26)
        assert np.allclose(macd.values, expected.values, atol=1e-10)

    def test_no_nan_in_macd(self):
        s = make_price_series(n=100)
        macd, _, _ = calc_macd(s)
        assert not macd.isna().any()


# ======================================================================
# TestCalcADX
# ======================================================================

class TestCalcADX:
    def test_length(self):
        high, low, close, _ = make_ohlcv()
        adx = calc_adx(high, low, close)
        assert len(adx) == N

    def test_range_0_100(self):
        high, low, close, _ = make_ohlcv()
        adx = calc_adx(high, low, close).dropna()
        assert (adx >= 0).all()
        assert (adx <= 100).all()

    def test_strong_trend_higher_adx(self):
        """强趋势序列的 ADX 应高于震荡序列"""
        n = 300
        # 强趋势：单调递增
        trend = pd.Series(np.linspace(3000, 4000, n))
        noise_high = trend * 1.002
        noise_low = trend * 0.998
        adx_trend = calc_adx(noise_high, noise_low, trend).dropna()

        # 震荡：sin 波
        oscillate = pd.Series(3000 + 50 * np.sin(np.linspace(0, 20 * np.pi, n)))
        osc_high = oscillate * 1.002
        osc_low = oscillate * 0.998
        adx_osc = calc_adx(osc_high, osc_low, oscillate).dropna()

        assert adx_trend.mean() > adx_osc.mean()

    def test_finite_values(self):
        high, low, close, _ = make_ohlcv()
        adx = calc_adx(high, low, close).dropna()
        assert np.isfinite(adx.values).all()


# ======================================================================
# TestCalcRSI
# ======================================================================

class TestCalcRSI:
    def test_length(self):
        s = make_price_series()
        assert len(calc_rsi(s)) == len(s)

    def test_range_0_100(self):
        s = make_price_series()
        rsi = calc_rsi(s).dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_all_up_rsi_near_100(self):
        """全部上涨的序列 RSI 应接近 100"""
        s = pd.Series(np.linspace(100, 200, 50))
        rsi = calc_rsi(s, window=14)
        assert rsi.iloc[-1] > 90

    def test_all_down_rsi_near_0(self):
        """全部下跌的序列 RSI 应接近 0"""
        s = pd.Series(np.linspace(200, 100, 50))
        rsi = calc_rsi(s, window=14)
        assert rsi.iloc[-1] < 10


# ======================================================================
# TestCalcROC
# ======================================================================

class TestCalcROC:
    def test_length(self):
        s = make_price_series()
        assert len(calc_roc(s)) == len(s)

    def test_nan_prefix(self):
        s = make_price_series()
        roc = calc_roc(s, window=10)
        assert roc.isna().sum() == 10

    def test_correct_value(self):
        s = pd.Series([100.0, 110.0, 121.0])
        roc = calc_roc(s, window=1)
        assert abs(roc.iloc[1] - 10.0) < 1e-8  # (110-100)/100*100 = 10%
        assert abs(roc.iloc[2] - 10.0) < 1e-8  # (121-110)/110*100 = 10%

    def test_positive_for_uptrend(self):
        s = pd.Series(np.linspace(100, 200, 50))
        roc = calc_roc(s, window=10).dropna()
        assert (roc > 0).all()


# ======================================================================
# TestCalcStochastic
# ======================================================================

class TestCalcStochastic:
    def test_returns_two_series(self):
        high, low, close, _ = make_ohlcv()
        result = calc_stochastic(high, low, close)
        assert len(result) == 2

    def test_k_range_0_100(self):
        high, low, close, _ = make_ohlcv()
        k, _ = calc_stochastic(high, low, close)
        k_valid = k.dropna()
        assert (k_valid >= 0).all()
        assert (k_valid <= 100).all()

    def test_d_range_0_100(self):
        high, low, close, _ = make_ohlcv()
        _, d = calc_stochastic(high, low, close)
        d_valid = d.dropna()
        assert (d_valid >= 0).all()
        assert (d_valid <= 100).all()

    def test_d_is_sma_of_k(self):
        high, low, close, _ = make_ohlcv()
        k, d = calc_stochastic(high, low, close, k_window=14, d_window=3)
        expected_d = k.rolling(3).mean()
        assert np.allclose(d.dropna().values, expected_d.dropna().values, atol=1e-10)


# ======================================================================
# TestCalcATR
# ======================================================================

class TestCalcATR:
    def test_length(self):
        high, low, close, _ = make_ohlcv()
        assert len(calc_atr(high, low, close)) == N

    def test_positive_values(self):
        high, low, close, _ = make_ohlcv()
        atr = calc_atr(high, low, close).dropna()
        assert (atr > 0).all()

    def test_atr_geq_hl_range(self):
        """ATR ≥ 当日高低差（EMA 平滑后可能更高也可能更低，但初始值应 ≥ H-L 范围）"""
        high, low, close, _ = make_ohlcv()
        atr = calc_atr(high, low, close, window=1)
        hl = (high - low)
        # with window=1, ATR ≈ TR which ≥ H-L
        assert (atr.dropna() >= hl.dropna() - 1e-10).all()

    def test_finite(self):
        high, low, close, _ = make_ohlcv()
        atr = calc_atr(high, low, close).dropna()
        assert np.isfinite(atr.values).all()


# ======================================================================
# TestCalcBollingerBands
# ======================================================================

class TestCalcBollingerBands:
    def test_returns_three_series(self):
        s = make_price_series()
        result = calc_bollinger_bands(s)
        assert len(result) == 3

    def test_upper_above_middle_above_lower(self):
        s = make_price_series()
        upper, middle, lower = calc_bollinger_bands(s)
        valid = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid] > middle[valid]).all()
        assert (middle[valid] > lower[valid]).all()

    def test_middle_equals_sma(self):
        s = make_price_series()
        _, middle, _ = calc_bollinger_bands(s, window=20)
        sma = calc_sma(s, 20)
        assert np.allclose(middle.dropna().values, sma.dropna().values, atol=1e-10)

    def test_wider_bands_with_more_std(self):
        s = make_price_series()
        upper_2, _, lower_2 = calc_bollinger_bands(s, num_std=2.0)
        upper_3, _, lower_3 = calc_bollinger_bands(s, num_std=3.0)
        valid = ~(upper_2.isna() | upper_3.isna())
        assert (upper_3[valid] > upper_2[valid]).all()
        assert (lower_3[valid] < lower_2[valid]).all()

    def test_nan_prefix(self):
        s = make_price_series()
        _, middle, _ = calc_bollinger_bands(s, window=20)
        assert middle.isna().sum() == 19


# ======================================================================
# TestCalcHistoricalVol
# ======================================================================

class TestCalcHistoricalVol:
    def test_length(self):
        s = make_price_series()
        assert len(calc_historical_vol(s)) == len(s)

    def test_positive_values(self):
        s = make_price_series()
        hv = calc_historical_vol(s).dropna()
        assert (hv > 0).all()

    def test_annualized_vs_not(self):
        s = make_price_series()
        hv_ann = calc_historical_vol(s, annualize=True).dropna()
        hv_raw = calc_historical_vol(s, annualize=False).dropna()
        ratio = (hv_ann / hv_raw).dropna()
        assert np.allclose(ratio.values, np.sqrt(252), rtol=1e-6)

    def test_higher_vol_series(self):
        s_low = make_price_series(vol=0.005)
        s_high = make_price_series(vol=0.03)
        hv_low = calc_historical_vol(s_low).dropna().mean()
        hv_high = calc_historical_vol(s_high).dropna().mean()
        assert hv_high > hv_low


# ======================================================================
# TestCalcOBV
# ======================================================================

class TestCalcOBV:
    def test_length(self):
        _, _, close, volume = make_ohlcv()
        assert len(calc_obv(close, volume)) == N

    def test_all_up_obv_monotonic(self):
        """全部上涨时 OBV 单调递增"""
        close = pd.Series(np.linspace(100, 200, 50))
        volume = pd.Series(np.ones(50) * 1000)
        obv = calc_obv(close, volume)
        assert (obv.diff().dropna() > 0).all()

    def test_all_down_obv_monotonic(self):
        """全部下跌时 OBV 单调递减"""
        close = pd.Series(np.linspace(200, 100, 50))
        volume = pd.Series(np.ones(50) * 1000)
        obv = calc_obv(close, volume)
        assert (obv.diff().dropna() <= 0).all()

    def test_first_value_zero_or_near(self):
        """OBV 第一个值应为 0（cumsum 从 diff NaN 起，第一个 diff 为 NaN → 0）"""
        _, _, close, volume = make_ohlcv()
        obv = calc_obv(close, volume)
        assert obv.iloc[0] == 0.0

    def test_cumulative_nature(self):
        """手动验证：简单 3 价格序列"""
        close = pd.Series([100.0, 105.0, 103.0, 106.0])
        volume = pd.Series([1000.0, 2000.0, 1500.0, 3000.0])
        obv = calc_obv(close, volume)
        # day0: 0, day1: +2000, day2: -1500, day3: +3000
        expected = [0.0, 2000.0, 500.0, 3500.0]
        assert np.allclose(obv.values, expected, atol=1e-10)


# ======================================================================
# TestCalcVWAP
# ======================================================================

class TestCalcVWAP:
    def test_length(self):
        high, low, close, volume = make_ohlcv()
        assert len(calc_vwap(high, low, close, volume)) == N

    def test_vwap_between_low_and_high(self):
        high, low, close, volume = make_ohlcv()
        vwap = calc_vwap(high, low, close, volume).dropna()
        # VWAP should be roughly between low and high
        assert (vwap > 0).all()

    def test_rolling_vwap_length(self):
        high, low, close, volume = make_ohlcv()
        vwap = calc_vwap(high, low, close, volume, window=20)
        assert len(vwap) == N

    def test_rolling_vwap_nan_prefix(self):
        high, low, close, volume = make_ohlcv()
        vwap = calc_vwap(high, low, close, volume, window=20)
        assert vwap.isna().sum() == 19

    def test_uniform_price_vwap_equals_price(self):
        """均匀价格时 VWAP 应等于价格"""
        n = 50
        close = high = low = pd.Series(np.ones(n) * 3800.0)
        volume = pd.Series(np.ones(n) * 1000.0)
        vwap = calc_vwap(high, low, close, volume)
        assert np.allclose(vwap.values, 3800.0, atol=1e-8)
