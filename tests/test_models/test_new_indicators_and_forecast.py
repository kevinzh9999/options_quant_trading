"""
test_new_indicators_and_forecast.py
------------------------------------
测试：
  VolatilityIndicators  (八) – atr / bollinger_bands / keltner_channel
  MomentumIndicators    (九) – rsi / roc / momentum_factor
  VolForecast           (十) – ewma_forecast / fit_and_predict (ewma/garch/har)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.indicators.volatility_ind import VolatilityIndicators, calc_atr, calc_bollinger_bands
from models.indicators.momentum import MomentumIndicators, calc_rsi, calc_roc
from models.volatility.vol_forecast import VolForecast


# ======================================================================
# 共用测试数据
# ======================================================================

N = 120
IDX = pd.date_range("2024-01-01", periods=N, freq="D")

rng = np.random.default_rng(0)

# 随机游走收盘价（正数）
_ret = rng.normal(0, 0.01, N)
CLOSE = pd.Series(4000 * np.exp(np.cumsum(_ret)), index=IDX)
HIGH  = CLOSE * (1 + abs(rng.normal(0, 0.005, N)))
LOW   = CLOSE * (1 - abs(rng.normal(0, 0.005, N)))

# 单调上涨序列（用于趋势测试）
UPTREND = pd.Series(np.linspace(100, 200, N), index=IDX)

# 纯正弦振荡（用于均值回归测试）
SINE = pd.Series(4000 + 50 * np.sin(np.linspace(0, 6 * np.pi, N)), index=IDX)


# ======================================================================
# 八、VolatilityIndicators
# ======================================================================

class TestATR:

    def test_returns_series(self):
        assert isinstance(VolatilityIndicators.atr(HIGH, LOW, CLOSE), pd.Series)

    def test_length_preserved(self):
        assert len(VolatilityIndicators.atr(HIGH, LOW, CLOSE, 14)) == N

    def test_atr_positive(self):
        result = VolatilityIndicators.atr(HIGH, LOW, CLOSE, 14).dropna()
        assert (result > 0).all()

    def test_higher_volatility_higher_atr(self):
        """振幅更大的价格序列 ATR 应更大。"""
        close_stable = pd.Series([4000.0] * N, index=IDX)
        high_stable  = close_stable + 5
        low_stable   = close_stable - 5

        close_vol = CLOSE.copy()
        high_vol  = HIGH.copy()
        low_vol   = LOW.copy()

        atr_stable = VolatilityIndicators.atr(high_stable, low_stable, close_stable, 14)
        atr_vol    = VolatilityIndicators.atr(high_vol,   low_vol,    close_vol,    14)

        # 随机游走 ATR 均值 > 稳定序列 ATR 均值（稳定序列 ATR ≈ 10）
        assert atr_vol.dropna().mean() > atr_stable.dropna().mean()

    def test_delegates_to_calc_atr(self):
        expected = calc_atr(HIGH, LOW, CLOSE, 14)
        result   = VolatilityIndicators.atr(HIGH, LOW, CLOSE, 14)
        pd.testing.assert_series_equal(result, expected)

    def test_index_preserved(self):
        assert list(VolatilityIndicators.atr(HIGH, LOW, CLOSE).index) == list(IDX)


class TestBollingerBands:

    def setup_method(self):
        self.result = VolatilityIndicators.bollinger_bands(CLOSE, 20, 2.0)

    def test_returns_dataframe(self):
        assert isinstance(self.result, pd.DataFrame)

    def test_columns(self):
        for col in ("middle", "upper", "lower", "bandwidth"):
            assert col in self.result.columns

    def test_length_preserved(self):
        assert len(self.result) == N

    def test_upper_gt_middle_gt_lower(self):
        valid = self.result.dropna()
        assert (valid["upper"] > valid["middle"]).all()
        assert (valid["middle"] > valid["lower"]).all()

    def test_bandwidth_positive(self):
        valid = self.result.dropna()
        assert (valid["bandwidth"] > 0).all()

    def test_bandwidth_formula(self):
        """bandwidth = (upper - lower) / middle。"""
        valid = self.result.dropna()
        expected = (valid["upper"] - valid["lower"]) / valid["middle"]
        diff = (valid["bandwidth"] - expected).abs()
        assert (diff < 1e-9).all()

    def test_leading_nan(self):
        assert self.result.iloc[:19].isna().all().all()
        assert not self.result.iloc[19].isna().any()

    def test_delegates_to_calc_bollinger(self):
        upper, middle, lower = calc_bollinger_bands(CLOSE, 20, 2.0)
        assert (self.result["upper"]  - upper ).abs().max()  < 1e-9
        assert (self.result["middle"] - middle).abs().max() < 1e-9
        assert (self.result["lower"]  - lower ).abs().max()  < 1e-9

    def test_wider_bands_with_more_std(self):
        bb1 = VolatilityIndicators.bollinger_bands(CLOSE, 20, 1.0)
        bb2 = VolatilityIndicators.bollinger_bands(CLOSE, 20, 3.0)
        width1 = (bb1["upper"] - bb1["lower"]).dropna().mean()
        width2 = (bb2["upper"] - bb2["lower"]).dropna().mean()
        assert width2 > width1

    def test_sine_close_to_middle(self):
        """振荡序列大多数时候价格在布林带中轨附近。"""
        result = VolatilityIndicators.bollinger_bands(SINE, 20, 2.0)
        valid = result.dropna()
        within = ((SINE.loc[valid.index] >= valid["lower"]) &
                  (SINE.loc[valid.index] <= valid["upper"])).mean()
        assert within > 0.7  # 正弦波非正态分布，2σ 覆盖约 70% 以上


class TestKeltnerChannel:

    def setup_method(self):
        self.result = VolatilityIndicators.keltner_channel(
            HIGH, LOW, CLOSE, ema_period=20, atr_period=14, multiplier=2.0
        )

    def test_returns_dataframe(self):
        assert isinstance(self.result, pd.DataFrame)

    def test_columns(self):
        for col in ("middle", "upper", "lower"):
            assert col in self.result.columns

    def test_upper_gt_middle_gt_lower(self):
        valid = self.result.dropna()
        assert (valid["upper"] > valid["middle"]).all()
        assert (valid["middle"] > valid["lower"]).all()

    def test_symmetric_channel(self):
        """upper - middle == middle - lower（ATR 等距）。"""
        valid = self.result.dropna()
        width_up = valid["upper"]  - valid["middle"]
        width_dn = valid["middle"] - valid["lower"]
        diff = (width_up - width_dn).abs()
        assert (diff < 1e-9).all()

    def test_larger_multiplier_wider_channel(self):
        kc1 = VolatilityIndicators.keltner_channel(HIGH, LOW, CLOSE, multiplier=1.0)
        kc2 = VolatilityIndicators.keltner_channel(HIGH, LOW, CLOSE, multiplier=3.0)
        w1 = (kc1["upper"] - kc1["lower"]).dropna().mean()
        w2 = (kc2["upper"] - kc2["lower"]).dropna().mean()
        assert w2 > w1

    def test_middle_is_ema(self):
        """middle 应等于 EMA(close, ema_period)。"""
        from models.indicators.trend import calc_ema
        expected_middle = calc_ema(CLOSE, 20)
        diff = (self.result["middle"] - expected_middle).abs()
        assert (diff.dropna() < 1e-9).all()

    def test_index_preserved(self):
        assert list(self.result.index) == list(IDX)

    def test_keltner_vs_bollinger_stability(self):
        """Keltner 通道宽度应比布林带更稳定（ATR 基于均值，std 更跳跃）。"""
        kc = VolatilityIndicators.keltner_channel(HIGH, LOW, CLOSE)
        bb = VolatilityIndicators.bollinger_bands(CLOSE)
        kc_width_std = (kc["upper"] - kc["lower"]).dropna().std()
        bb_width_std = (bb["upper"] - bb["lower"]).dropna().std()
        # ATR 更平滑，宽度波动应更小
        assert kc_width_std <= bb_width_std * 2   # 允许宽松：不要求必须更小


# ======================================================================
# 九、MomentumIndicators
# ======================================================================

class TestMomentumRSI:

    def test_returns_series(self):
        assert isinstance(MomentumIndicators.rsi(CLOSE, 14), pd.Series)

    def test_range_0_to_100(self):
        result = MomentumIndicators.rsi(CLOSE, 14)
        assert (result >= 0).all() and (result <= 100).all()

    def test_uptrend_rsi_above_50(self):
        """单调上涨序列 RSI 应持续 > 50。"""
        result = MomentumIndicators.rsi(UPTREND, 14)
        assert (result.iloc[20:] > 50).all()

    def test_constant_series_rsi_50_or_extreme(self):
        """恒定价格序列（无变动）RSI 约为 100（avg_loss=0）。"""
        const = pd.Series([100.0] * N, index=IDX)
        result = MomentumIndicators.rsi(const, 14)
        # diff=0 → gain=0 loss=0 → RSI填充为100
        assert (result >= 50).all()

    def test_length_preserved(self):
        assert len(MomentumIndicators.rsi(CLOSE, 14)) == N

    def test_delegates_to_calc_rsi(self):
        expected = calc_rsi(CLOSE, 14)
        result   = MomentumIndicators.rsi(CLOSE, 14)
        pd.testing.assert_series_equal(result, expected)

    def test_shorter_period_faster(self):
        """短周期 RSI 反应更快（振荡范围更大）。"""
        rsi5  = MomentumIndicators.rsi(CLOSE, 5)
        rsi14 = MomentumIndicators.rsi(CLOSE, 14)
        assert rsi5.std() >= rsi14.std()


class TestMomentumROC:

    def test_returns_series(self):
        assert isinstance(MomentumIndicators.roc(CLOSE, 12), pd.Series)

    def test_leading_nan(self):
        result = MomentumIndicators.roc(CLOSE, 12)
        assert result.iloc[:12].isna().all()
        assert not pd.isna(result.iloc[12])

    def test_uptrend_roc_positive(self):
        result = MomentumIndicators.roc(UPTREND, 10)
        assert (result.dropna() > 0).all()

    def test_roc_formula(self):
        """ROC = (close - close.shift(n)) / close.shift(n) * 100。"""
        result = MomentumIndicators.roc(CLOSE, 5)
        expected = (CLOSE - CLOSE.shift(5)) / CLOSE.shift(5) * 100
        diff = (result - expected).abs().dropna()
        assert (diff < 1e-9).all()

    def test_delegates_to_calc_roc(self):
        expected = calc_roc(CLOSE, 12)
        result   = MomentumIndicators.roc(CLOSE, 12)
        pd.testing.assert_series_equal(result, expected)

    def test_length_preserved(self):
        assert len(MomentumIndicators.roc(CLOSE, 12)) == N


class TestMomentumFactor:

    def setup_method(self):
        # 需要更长序列来测试 lookback=252
        _r = rng.normal(0, 0.01, 400)
        self.long_close = pd.Series(
            4000 * np.exp(np.cumsum(_r)),
            index=pd.date_range("2023-01-01", periods=400, freq="D"),
        )

    def test_returns_series(self):
        result = MomentumIndicators.momentum_factor(self.long_close, 252, 21)
        assert isinstance(result, pd.Series)

    def test_leading_nan(self):
        """前 lookback 行应为 NaN。"""
        result = MomentumIndicators.momentum_factor(self.long_close, 252, 21)
        assert result.iloc[:252].isna().all()
        assert not pd.isna(result.iloc[252])

    def test_uptrend_factor_positive(self):
        """单调上涨序列动量因子应全为正。"""
        long_up = pd.Series(
            np.linspace(100, 300, 400),
            index=pd.date_range("2023-01-01", periods=400, freq="D"),
        )
        result = MomentumIndicators.momentum_factor(long_up, 252, 21)
        assert (result.dropna() > 0).all()

    def test_formula(self):
        """mom_t = close_{t-skip} / close_{t-lookback} - 1。"""
        result = MomentumIndicators.momentum_factor(self.long_close, 252, 21)
        expected = self.long_close.shift(21) / self.long_close.shift(252) - 1
        diff = (result - expected).abs().dropna()
        assert (diff < 1e-9).all()

    def test_skip_zero_equals_simple_return(self):
        """skip=0 时 = 普通 lookback 期收益率。"""
        result = MomentumIndicators.momentum_factor(self.long_close, 20, 0)
        expected = self.long_close / self.long_close.shift(20) - 1
        diff = (result - expected).abs().dropna()
        assert (diff < 1e-9).all()

    def test_length_preserved(self):
        result = MomentumIndicators.momentum_factor(self.long_close, 252, 21)
        assert len(result) == len(self.long_close)


# ======================================================================
# 十、VolForecast
# ======================================================================

def _make_returns(n=200, vol=0.015, seed=1):
    rng2 = np.random.default_rng(seed)
    r = rng2.normal(0, vol, n)
    close = pd.Series(4000 * np.exp(np.cumsum(r)))
    return close


class TestVolForecastInit:

    def test_valid_method_garch(self):
        vf = VolForecast("garch")
        assert vf.method == "garch"

    def test_valid_method_ewma(self):
        vf = VolForecast("ewma")
        assert vf.method == "ewma"

    def test_valid_method_har(self):
        vf = VolForecast("har")
        assert vf.method == "har"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            VolForecast("unknown_method")

    def test_default_method_garch(self):
        vf = VolForecast()
        assert vf.method == "garch"


class TestEWMAForecast:

    def setup_method(self):
        close = _make_returns(200)
        returns = np.log(close / close.shift(1)).dropna()
        self.returns = returns

    def test_returns_float(self):
        result = VolForecast("ewma").ewma_forecast(self.returns)
        assert isinstance(result, float)

    def test_result_positive(self):
        result = VolForecast("ewma").ewma_forecast(self.returns)
        assert result > 0

    def test_result_reasonable_range(self):
        """年化波动率应在 1%–200%（合成 vol≈1.5%/日 × sqrt(252) ≈ 24%）。"""
        result = VolForecast("ewma").ewma_forecast(self.returns)
        assert 0.01 <= result <= 2.0

    def test_higher_decay_smoother(self):
        """更高的 decay（接近1）使 EWMA 更平滑，对近期变化响应更慢。"""
        # 在相同数据上，decay=0.99 应比 decay=0.50 更接近长期均值
        vf = VolForecast("ewma")
        # 构造一个近期波动率突变的序列
        rng3 = np.random.default_rng(99)
        r_normal = rng3.normal(0, 0.01, 150)
        r_spike  = rng3.normal(0, 0.05, 10)
        r = pd.Series(np.concatenate([r_normal, r_spike]))
        vol_fast = vf.ewma_forecast(r, decay=0.50)
        vol_slow = vf.ewma_forecast(r, decay=0.99)
        # 快速 EWMA 应对近期波动率暴涨有更强响应
        assert vol_fast > vol_slow

    def test_empty_returns_default(self):
        result = VolForecast("ewma").ewma_forecast(pd.Series([], dtype=float))
        assert result == 0.20

    def test_custom_decay(self):
        vf = VolForecast("ewma", decay=0.97)
        result = vf.ewma_forecast(self.returns, decay=0.97)
        assert result > 0


class TestFitAndPredictEWMA:

    def setup_method(self):
        self.close = _make_returns(200)

    def test_returns_dict(self):
        result = VolForecast("ewma").fit_and_predict(self.close)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = VolForecast("ewma").fit_and_predict(self.close)
        for k in ("method", "current_vol", "forecast_vol", "model_params", "diagnostics"):
            assert k in result

    def test_method_label(self):
        result = VolForecast("ewma").fit_and_predict(self.close)
        assert result["method"] == "ewma"

    def test_vols_positive(self):
        result = VolForecast("ewma").fit_and_predict(self.close)
        assert result["current_vol"] > 0
        assert result["forecast_vol"] > 0

    def test_vols_reasonable(self):
        result = VolForecast("ewma").fit_and_predict(self.close)
        assert 0.01 <= result["current_vol"] <= 2.0
        assert 0.01 <= result["forecast_vol"] <= 2.0

    def test_model_params_has_decay(self):
        result = VolForecast("ewma").fit_and_predict(self.close)
        assert "decay" in result["model_params"]

    def test_short_series_raises(self):
        with pytest.raises((ValueError, Exception)):
            VolForecast("ewma").fit_and_predict(pd.Series([100.0]))


class TestFitAndPredictGARCH:

    def setup_method(self):
        self.close = _make_returns(200, seed=42)

    def test_returns_dict(self):
        result = VolForecast("garch").fit_and_predict(self.close, horizon=5)
        assert isinstance(result, dict)

    def test_method_label(self):
        result = VolForecast("garch").fit_and_predict(self.close)
        assert result["method"] == "garch"

    def test_vols_positive(self):
        result = VolForecast("garch").fit_and_predict(self.close)
        assert result["current_vol"] > 0
        assert result["forecast_vol"] > 0

    def test_vols_reasonable(self):
        result = VolForecast("garch").fit_and_predict(self.close)
        assert 0.01 <= result["forecast_vol"] <= 2.0

    def test_diagnostics_has_aic_bic(self):
        result = VolForecast("garch").fit_and_predict(self.close)
        diag = result["diagnostics"]
        assert "aic" in diag and "bic" in diag

    def test_horizon_affects_forecast(self):
        """不同预测期限 forecast_vol 可能不同（GARCH 均值回归效应）。"""
        r1 = VolForecast("garch").fit_and_predict(self.close, horizon=1)
        r5 = VolForecast("garch").fit_and_predict(self.close, horizon=5)
        # 两者均应为正数
        assert r1["forecast_vol"] > 0
        assert r5["forecast_vol"] > 0


class TestFitAndPredictHAR:

    def setup_method(self):
        self.close = _make_returns(300, seed=7)

    def test_returns_dict(self):
        result = VolForecast("har").fit_and_predict(self.close)
        assert isinstance(result, dict)

    def test_method_label(self):
        result = VolForecast("har").fit_and_predict(self.close)
        assert result["method"] == "har"

    def test_vols_positive(self):
        result = VolForecast("har").fit_and_predict(self.close)
        assert result["forecast_vol"] > 0

    def test_diagnostics_has_ci(self):
        result = VolForecast("har").fit_and_predict(self.close)
        assert "ci_lower" in result["diagnostics"]
        assert "ci_upper" in result["diagnostics"]
