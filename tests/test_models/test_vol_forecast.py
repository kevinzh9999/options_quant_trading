"""
tests/test_models/test_vol_forecast.py
---------------------------------------
VolForecaster 单元测试。

覆盖：
- VolForecastResult dataclass 字段
- ForecastMethod 枚举
- VolForecaster.forecast() 四种方法
- _har_rv_forecast / _fit_har_ols
- 置信区间合理性
- 集成权重
- 参数校验（样本不足抛 ValueError）
- fit_har 缓存
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.volatility.vol_forecast import (
    ForecastMethod,
    VolForecastResult,
    VolForecaster,
)


# ======================================================================
# 测试固件
# ======================================================================

RNG = np.random.default_rng(42)


def make_returns(n: int = 300, vol: float = 0.20) -> pd.Series:
    """生成年化波动率约为 vol 的日度对数收益率序列。"""
    daily_vol = vol / np.sqrt(252)
    return pd.Series(RNG.normal(0, daily_vol, n))


def make_rv_series(n: int = 200, level: float = 0.20) -> pd.Series:
    """生成模拟的已实现波动率序列（年化，小数）。"""
    daily_vol = level / np.sqrt(252)
    raw = RNG.normal(0, daily_vol, n)
    rv = pd.Series(np.abs(raw) * np.sqrt(252)).clip(lower=0.01)
    rv.name = "rv"
    return rv


# ======================================================================
# TestVolForecastResult — dataclass 字段
# ======================================================================

class TestVolForecastResult:
    def test_fields_exist(self):
        r = VolForecastResult(
            trade_date="20240101",
            underlying="IC.CFX",
            forecast_vol=0.20,
        )
        assert r.trade_date == "20240101"
        assert r.underlying == "IC.CFX"
        assert r.forecast_vol == 0.20
        assert r.horizon == 5
        assert r.method == ForecastMethod.GARCH
        assert r.conf_interval_lower == 0.0
        assert r.conf_interval_upper == 0.0
        assert r.garch_vol == 0.0
        assert r.har_rv_vol == 0.0

    def test_custom_fields(self):
        r = VolForecastResult(
            trade_date="20240101",
            underlying="IO.CFX",
            forecast_vol=0.25,
            horizon=10,
            method=ForecastMethod.ENSEMBLE,
            garch_vol=0.22,
            har_rv_vol=0.28,
        )
        assert r.horizon == 10
        assert r.method == ForecastMethod.ENSEMBLE
        assert r.garch_vol == 0.22
        assert r.har_rv_vol == 0.28


# ======================================================================
# TestForecastMethodGARCH — 纯 GARCH 方法
# ======================================================================

class TestForecastMethodGARCH:
    def test_returns_vol_forecast_result(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert isinstance(result, VolForecastResult)

    def test_forecast_vol_positive(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.forecast_vol > 0

    def test_forecast_vol_reasonable(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns(vol=0.20))
        # 期望预测在 5%–80% 之间
        assert 0.05 < result.forecast_vol < 0.80

    def test_method_stored_in_result(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.method == ForecastMethod.GARCH

    def test_underlying_and_date_stored(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240315", "IF.CFX", make_returns())
        assert result.trade_date == "20240315"
        assert result.underlying == "IF.CFX"

    def test_horizon_stored(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns(), horizon=10)
        assert result.horizon == 10

    def test_garch_vol_populated(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.garch_vol > 0

    def test_har_rv_vol_zero_for_pure_garch(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.har_rv_vol == 0.0


# ======================================================================
# TestForecastMethodHARRV — 纯 HAR-RV 方法
# ======================================================================

class TestForecastMethodHARRV:
    def test_forecast_positive(self):
        fc = VolForecaster(method=ForecastMethod.HAR_RV)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.forecast_vol > 0

    def test_forecast_reasonable(self):
        fc = VolForecaster(method=ForecastMethod.HAR_RV)
        result = fc.forecast("20240101", "IC.CFX", make_returns(vol=0.20))
        assert 0.02 < result.forecast_vol < 1.0

    def test_har_rv_vol_populated(self):
        fc = VolForecaster(method=ForecastMethod.HAR_RV)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.har_rv_vol > 0

    def test_garch_vol_zero_for_pure_har(self):
        fc = VolForecaster(method=ForecastMethod.HAR_RV)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.garch_vol == 0.0

    def test_method_stored(self):
        fc = VolForecaster(method=ForecastMethod.HAR_RV)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.method == ForecastMethod.HAR_RV


# ======================================================================
# TestForecastMethodEnsemble — 集成方法
# ======================================================================

class TestForecastMethodEnsemble:
    def test_forecast_positive(self):
        fc = VolForecaster(method=ForecastMethod.ENSEMBLE)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.forecast_vol > 0

    def test_both_components_populated(self):
        fc = VolForecaster(method=ForecastMethod.ENSEMBLE)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.garch_vol > 0
        assert result.har_rv_vol > 0

    def test_ensemble_is_avg_of_components(self):
        fc = VolForecaster(method=ForecastMethod.ENSEMBLE)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        expected = 0.5 * result.garch_vol + 0.5 * result.har_rv_vol
        assert abs(result.forecast_vol - expected) < 1e-10

    def test_forecast_reasonable(self):
        fc = VolForecaster(method=ForecastMethod.ENSEMBLE)
        result = fc.forecast("20240101", "IC.CFX", make_returns(vol=0.20))
        assert 0.02 < result.forecast_vol < 1.0


# ======================================================================
# TestForecastMethodGARCHX — GARCH-X 方法
# ======================================================================

class TestForecastMethodGARCHX:
    def test_forecast_positive(self):
        fc = VolForecaster(method=ForecastMethod.GARCH_X)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.forecast_vol > 0

    def test_garch_vol_populated(self):
        fc = VolForecaster(method=ForecastMethod.GARCH_X)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.garch_vol > 0


# ======================================================================
# TestConfidenceInterval — 置信区间
# ======================================================================

class TestConfidenceInterval:
    def test_ci_lower_less_than_forecast(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.conf_interval_lower < result.forecast_vol

    def test_ci_upper_greater_than_forecast(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.conf_interval_upper > result.forecast_vol

    def test_ci_positive(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.conf_interval_lower > 0
        assert result.conf_interval_upper > 0

    def test_ci_width_positive(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.conf_interval_upper > result.conf_interval_lower


# ======================================================================
# TestInputValidation — 参数校验
# ======================================================================

class TestInputValidation:
    def test_insufficient_returns_raises(self):
        fc = VolForecaster(method=ForecastMethod.GARCH)
        with pytest.raises(ValueError, match="不足"):
            fc.forecast("20240101", "IC.CFX", make_returns(n=30))

    def test_returns_with_nan_handled(self):
        returns = make_returns(n=300)
        returns.iloc[10:20] = np.nan  # 注入 NaN
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", returns)
        assert result.forecast_vol > 0


# ======================================================================
# TestHARRVInternal — HAR-RV 内部方法
# ======================================================================

class TestHARRVInternal:
    def test_har_rv_forecast_positive(self):
        fc = VolForecaster()
        rv = make_rv_series(n=200)
        pred = fc._har_rv_forecast(rv, horizon=5)
        assert pred > 0

    def test_har_rv_forecast_reasonable(self):
        fc = VolForecaster()
        rv = make_rv_series(n=200, level=0.20)
        pred = fc._har_rv_forecast(rv, horizon=5)
        # Prediction should be in a plausible range
        assert 0.01 < pred < 1.0

    def test_har_rv_horizon_1_vs_5(self):
        """horizon=1 和 horizon=5 的预测应该接近（HAR 有持续性）。"""
        fc = VolForecaster()
        rv = make_rv_series(n=200)
        pred1 = fc._har_rv_forecast(rv, horizon=1)
        pred5 = fc._har_rv_forecast(rv, horizon=5)
        # Both positive and within 3x of each other
        assert pred1 > 0 and pred5 > 0
        assert abs(pred1 - pred5) / max(pred1, pred5) < 0.5

    def test_fit_har_betas_length(self):
        fc = VolForecaster()
        rv = make_rv_series(n=200)
        betas, residual_std = fc._fit_har_ols(rv)
        assert len(betas) == 4  # intercept + 3 components
        assert residual_std >= 0

    def test_fit_har_caches_betas(self):
        fc = VolForecaster()
        rv = make_rv_series(n=200)
        fc.fit_har(rv)
        assert fc._har_betas is not None
        assert len(fc._har_betas) == 4

    def test_fit_har_used_in_subsequent_forecast(self):
        """预先 fit_har 后再调用 forecast 应该使用缓存系数。"""
        fc = VolForecaster(method=ForecastMethod.HAR_RV)
        rv = make_rv_series(n=200)
        returns = make_returns(n=300)
        fc.fit_har(rv)
        result = fc.forecast("20240101", "IC.CFX", returns)
        assert result.forecast_vol > 0

    def test_har_fallback_small_sample(self):
        """样本不足 30 时回退到移动均值，仍返回正值。"""
        fc = VolForecaster()
        rv = make_rv_series(n=20)
        pred = fc._har_rv_forecast(rv, horizon=5)
        assert pred > 0


# ======================================================================
# TestForecastConsistency — 一致性测试
# ======================================================================

class TestForecastConsistency:
    def test_higher_vol_returns_higher_forecast(self):
        """高波动率时段应预测更高的未来波动率。"""
        fc_low = VolForecaster(method=ForecastMethod.GARCH)
        fc_high = VolForecaster(method=ForecastMethod.GARCH)
        returns_low = make_returns(n=300, vol=0.10)
        returns_high = make_returns(n=300, vol=0.40)
        result_low = fc_low.forecast("20240101", "IC.CFX", returns_low)
        result_high = fc_high.forecast("20240101", "IC.CFX", returns_high)
        assert result_high.forecast_vol > result_low.forecast_vol

    def test_all_methods_produce_positive_forecast(self):
        returns = make_returns(n=300)
        for method in ForecastMethod:
            fc = VolForecaster(method=method)
            result = fc.forecast("20240101", "IC.CFX", returns)
            assert result.forecast_vol > 0, f"Method {method} returned non-positive forecast"

    def test_forecast_vol_floor_at_1pct(self):
        """forecast_vol 最低为 1%（防止极端情况）。"""
        fc = VolForecaster(method=ForecastMethod.GARCH)
        result = fc.forecast("20240101", "IC.CFX", make_returns())
        assert result.forecast_vol >= 0.01
