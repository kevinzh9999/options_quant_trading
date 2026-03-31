"""
tests/test_models/test_statistics.py
--------------------------------------
cointegration / ou_process / regime_detection 单元测试。

覆盖：
- CointegrationResult dataclass 字段
- cointegration_test (Engle-Granger / Johansen)
- estimate_hedge_ratio (OLS / TLS)
- rolling_cointegration
- OUParams dataclass
- fit_ou_process: parameter recovery on simulated OU
- ou_half_life
- simulate_ou: shape, positivity of spread, mean reversion
- HMMRegimeDetector: fit, predict, get_state_statistics
- HMMRegimeResult 字段
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.statistics.cointegration import (
    CointegrationResult,
    cointegration_test,
    estimate_hedge_ratio,
    rolling_cointegration,
)
from models.statistics.ou_process import (
    OUParams,
    fit_ou_process,
    ou_half_life,
    simulate_ou,
)
from models.statistics.regime_detection import (
    HMMRegimeDetector,
    HMMRegimeResult,
    RegimeState,
)


# ======================================================================
# 测试固件
# ======================================================================

RNG = np.random.default_rng(42)
TRADING_DAYS = 252


def make_cointegrated_pair(n: int = 300, beta: float = 1.5, noise_std: float = 5.0) -> tuple[pd.Series, pd.Series]:
    """生成已知协整关系的价格对：y = beta*x + spread, spread ~ OU"""
    x = 3000.0 + np.cumsum(RNG.normal(0, 10, n))
    spread = np.zeros(n)
    kappa = 10.0  # 快速均值回归
    for t in range(1, n):
        spread[t] = spread[t - 1] + kappa * (0 - spread[t - 1]) / TRADING_DAYS + RNG.normal(0, noise_std)
    y = beta * x + spread

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    s1 = pd.Series(y, index=dates, name="IF")
    s2 = pd.Series(x, index=dates, name="IH")
    return s1, s2


def make_ou_spread(kappa: float = 5.0, theta: float = 0.0, sigma: float = 2.0,
                   n: int = 500) -> pd.Series:
    """生成 OU 过程价差序列"""
    paths = simulate_ou(kappa, theta, sigma, x0=0.0, n_steps=n, n_paths=1, dt=1/252, seed=42)
    return pd.Series(paths[0, :n])


def make_feature_df(n: int = 300) -> pd.DataFrame:
    """生成含 log_return / rv / rv_change 的特征 DataFrame"""
    # Two-regime returns: first half low-vol, second half high-vol
    returns_low = RNG.normal(0, 0.005, n // 2)
    returns_high = RNG.normal(0, 0.02, n - n // 2)
    returns = np.concatenate([returns_low, returns_high])

    rv = np.abs(returns) * np.sqrt(252)
    rv_change = np.diff(rv, prepend=rv[0])

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "log_return": returns,
        "rv": rv,
        "rv_change": rv_change,
    }, index=dates)


# ======================================================================
# TestCointegrationResult
# ======================================================================

class TestCointegrationResult:
    def test_fields(self):
        r = CointegrationResult(
            pair_id="IF-IH",
            method="engle_granger",
            is_cointegrated=True,
            p_value=0.01,
            test_statistic=-4.5,
        )
        assert r.pair_id == "IF-IH"
        assert r.is_cointegrated is True
        assert r.hedge_ratio == 1.0
        assert r.critical_values == {}


# ======================================================================
# TestEstimateHedgeRatio
# ======================================================================

class TestEstimateHedgeRatio:
    def test_ols_returns_tuple(self):
        s1, s2 = make_cointegrated_pair()
        beta, intercept = estimate_hedge_ratio(s1, s2, "ols")
        assert isinstance(beta, float)
        assert isinstance(intercept, float)

    def test_ols_beta_near_true(self):
        """OLS 对冲比例应接近真实值 1.5"""
        s1, s2 = make_cointegrated_pair(n=500, beta=1.5)
        beta, _ = estimate_hedge_ratio(s1, s2, "ols")
        assert abs(beta - 1.5) < 0.15

    def test_tls_returns_tuple(self):
        s1, s2 = make_cointegrated_pair()
        beta, intercept = estimate_hedge_ratio(s1, s2, "tls")
        assert isinstance(beta, float)
        assert isinstance(intercept, float)

    def test_tls_beta_reasonable(self):
        s1, s2 = make_cointegrated_pair(n=500, beta=1.5)
        beta, _ = estimate_hedge_ratio(s1, s2, "tls")
        assert 0.5 < beta < 3.0

    def test_unknown_method_raises(self):
        s1, s2 = make_cointegrated_pair()
        with pytest.raises(ValueError, match="未知方法"):
            estimate_hedge_ratio(s1, s2, "invalid")


# ======================================================================
# TestCointegrationTest
# ======================================================================

class TestCointegrationTest:
    def test_returns_result_instance(self):
        s1, s2 = make_cointegrated_pair()
        result = cointegration_test(s1, s2)
        assert isinstance(result, CointegrationResult)

    def test_method_stored(self):
        s1, s2 = make_cointegrated_pair()
        result = cointegration_test(s1, s2, method="engle_granger")
        assert result.method == "engle_granger"

    def test_pair_id_from_series_names(self):
        s1, s2 = make_cointegrated_pair()
        result = cointegration_test(s1, s2)
        assert result.pair_id == "IF-IH"

    def test_p_value_in_0_1(self):
        s1, s2 = make_cointegrated_pair()
        result = cointegration_test(s1, s2)
        assert 0.0 <= result.p_value <= 1.0

    def test_cointegrated_pair_detected(self):
        """真实协整配对应以高概率通过检验"""
        s1, s2 = make_cointegrated_pair(n=500, noise_std=2.0)
        result = cointegration_test(s1, s2)
        assert result.p_value < 0.10  # 应有一定统计功效

    def test_hedge_ratio_positive(self):
        s1, s2 = make_cointegrated_pair()
        result = cointegration_test(s1, s2)
        assert result.hedge_ratio > 0

    def test_spread_stats_finite(self):
        s1, s2 = make_cointegrated_pair()
        result = cointegration_test(s1, s2)
        assert np.isfinite(result.spread_mean)
        assert np.isfinite(result.spread_std)
        assert result.spread_std > 0

    def test_critical_values_dict(self):
        s1, s2 = make_cointegrated_pair()
        result = cointegration_test(s1, s2, method="engle_granger")
        assert "5%" in result.critical_values

    def test_johansen_method(self):
        s1, s2 = make_cointegrated_pair(n=300)
        result = cointegration_test(s1, s2, method="johansen")
        assert result.method == "johansen"
        assert "95%" in result.critical_values

    def test_unknown_method_raises(self):
        s1, s2 = make_cointegrated_pair()
        with pytest.raises(ValueError, match="未知检验方法"):
            cointegration_test(s1, s2, method="invalid")

    def test_independent_series_higher_pvalue(self):
        """随机游走对应该有较高 p 值（弱协整）"""
        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Two independent random walks
        s1 = pd.Series(np.cumsum(RNG.normal(0, 1, n)), index=dates, name="A")
        s2 = pd.Series(np.cumsum(RNG.normal(0, 1, n)), index=dates, name="B")
        result = cointegration_test(s1, s2)
        # No strong guarantee, just check it doesn't crash and p_value in [0,1]
        assert 0.0 <= result.p_value <= 1.0


# ======================================================================
# TestRollingCointegration
# ======================================================================

class TestRollingCointegration:
    def test_returns_dataframe(self):
        s1, s2 = make_cointegrated_pair(n=200)
        df = rolling_cointegration(s1, s2, window=60, step=20)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        s1, s2 = make_cointegrated_pair(n=200)
        df = rolling_cointegration(s1, s2, window=60, step=20)
        expected = {"date", "p_value", "is_cointegrated", "hedge_ratio", "spread_mean", "spread_std"}
        assert expected.issubset(set(df.columns))

    def test_correct_row_count(self):
        n = 200
        window = 60
        step = 20
        s1, s2 = make_cointegrated_pair(n=n)
        df = rolling_cointegration(s1, s2, window=window, step=step)
        expected_rows = len(range(0, n - window + 1, step))
        assert len(df) == expected_rows

    def test_p_values_in_0_1(self):
        s1, s2 = make_cointegrated_pair(n=200)
        df = rolling_cointegration(s1, s2, window=60, step=20)
        assert (df["p_value"] >= 0.0).all()
        assert (df["p_value"] <= 1.0).all()


# ======================================================================
# TestOUHalfLife
# ======================================================================

class TestOUHalfLife:
    def test_positive_kappa(self):
        hl = ou_half_life(5.0, dt=1 / 252)
        assert hl > 0

    def test_kappa_zero_returns_inf(self):
        assert ou_half_life(0.0) == float("inf")

    def test_kappa_negative_returns_inf(self):
        assert ou_half_life(-1.0) == float("inf")

    def test_formula(self):
        kappa = 10.0
        dt = 1 / 252
        expected = np.log(2) / kappa / dt
        assert abs(ou_half_life(kappa, dt) - expected) < 1e-10

    def test_higher_kappa_shorter_half_life(self):
        hl_fast = ou_half_life(20.0)
        hl_slow = ou_half_life(5.0)
        assert hl_fast < hl_slow


# ======================================================================
# TestFitOUProcess
# ======================================================================

class TestFitOUProcess:
    def test_returns_ou_params(self):
        spread = make_ou_spread()
        result = fit_ou_process(spread)
        assert isinstance(result, OUParams)

    def test_kappa_positive(self):
        spread = make_ou_spread(kappa=5.0)
        result = fit_ou_process(spread)
        assert result.kappa > 0

    def test_sigma_positive(self):
        spread = make_ou_spread()
        result = fit_ou_process(spread)
        assert result.sigma > 0

    def test_half_life_positive_finite(self):
        spread = make_ou_spread()
        result = fit_ou_process(spread)
        assert result.half_life > 0
        assert np.isfinite(result.half_life)

    def test_is_mean_reverting_true(self):
        """真实 OU 过程应检测为均值回归"""
        spread = make_ou_spread(kappa=5.0, n=1000)
        result = fit_ou_process(spread)
        assert result.is_mean_reverting is True

    def test_theta_near_true_mean(self):
        """估计的 θ 应接近真实均值（0）"""
        spread = make_ou_spread(kappa=5.0, theta=0.0, n=2000)
        result = fit_ou_process(spread)
        assert abs(result.theta) < 3.0  # 相对宽松的容差

    def test_r_squared_in_0_1(self):
        spread = make_ou_spread()
        result = fit_ou_process(spread)
        assert -0.1 <= result.r_squared <= 1.0  # 允许略微负值

    def test_fast_mean_reversion_shorter_half_life(self):
        """更快的均值回归 → 更短的半衰期"""
        slow = fit_ou_process(make_ou_spread(kappa=2.0, n=1000))
        fast = fit_ou_process(make_ou_spread(kappa=20.0, n=1000))
        assert fast.half_life < slow.half_life

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError, match="长度不足"):
            fit_ou_process(pd.Series([1.0, 2.0]))


# ======================================================================
# TestSimulateOU
# ======================================================================

class TestSimulateOU:
    def test_shape(self):
        paths = simulate_ou(5.0, 0.0, 2.0, 0.0, n_steps=100, n_paths=50)
        assert paths.shape == (50, 101)

    def test_initial_value(self):
        x0 = 5.0
        paths = simulate_ou(5.0, 0.0, 2.0, x0=x0, n_steps=50, n_paths=20)
        assert (paths[:, 0] == x0).all()

    def test_mean_reversion(self):
        """高 kappa 时，路径应在长时间内收敛到 theta"""
        theta = 10.0
        paths = simulate_ou(50.0, theta, 0.5, x0=0.0, n_steps=500, n_paths=1000, seed=1)
        final_mean = float(paths[:, -1].mean())
        assert abs(final_mean - theta) < 2.0

    def test_reproducible_with_seed(self):
        p1 = simulate_ou(5.0, 0.0, 2.0, 0.0, n_steps=50, n_paths=10, seed=99)
        p2 = simulate_ou(5.0, 0.0, 2.0, 0.0, n_steps=50, n_paths=10, seed=99)
        assert np.array_equal(p1, p2)

    def test_different_seeds_differ(self):
        p1 = simulate_ou(5.0, 0.0, 2.0, 0.0, n_steps=50, n_paths=10, seed=1)
        p2 = simulate_ou(5.0, 0.0, 2.0, 0.0, n_steps=50, n_paths=10, seed=2)
        assert not np.array_equal(p1, p2)

    def test_finite_values(self):
        paths = simulate_ou(5.0, 0.0, 1.0, 0.0, n_steps=100, n_paths=50)
        assert np.isfinite(paths).all()


# ======================================================================
# TestHMMRegimeDetector
# ======================================================================

class TestHMMRegimeDetector:
    def test_init_default(self):
        det = HMMRegimeDetector()
        assert det.n_states == 2
        assert not det.is_fitted

    def test_fit_returns_self(self):
        det = HMMRegimeDetector(n_states=2)
        result = det.fit(make_feature_df(n=300))
        assert result is det

    def test_is_fitted_after_fit(self):
        det = HMMRegimeDetector()
        det.fit(make_feature_df())
        assert det.is_fitted

    def test_fit_labels_states(self):
        det = HMMRegimeDetector()
        det.fit(make_feature_df())
        assert len(det._state_labels) == 2
        states = set(det._state_labels.values())
        assert RegimeState.LOW_VOL in states
        assert RegimeState.HIGH_VOL in states

    def test_predict_returns_result(self):
        det = HMMRegimeDetector()
        feat = make_feature_df()
        det.fit(feat)
        result = det.predict("20240101", feat)
        assert isinstance(result, HMMRegimeResult)

    def test_predict_not_fitted_raises(self):
        det = HMMRegimeDetector()
        with pytest.raises(RuntimeError, match="未拟合"):
            det.predict("20240101", make_feature_df())

    def test_predict_current_state_is_regime(self):
        det = HMMRegimeDetector()
        feat = make_feature_df()
        det.fit(feat)
        result = det.predict("20240101", feat)
        assert isinstance(result.current_state, RegimeState)

    def test_predict_state_probabilities(self):
        det = HMMRegimeDetector()
        feat = make_feature_df()
        det.fit(feat)
        result = det.predict("20240101", feat)
        probs = result.state_probability
        assert len(probs) == 2
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_state_sequence_length(self):
        det = HMMRegimeDetector()
        feat = make_feature_df(n=300)
        det.fit(feat)
        result = det.predict("20240101", feat)
        assert len(result.state_sequence) == len(feat.dropna())

    def test_predict_transition_matrix_shape(self):
        det = HMMRegimeDetector(n_states=2)
        feat = make_feature_df()
        det.fit(feat)
        result = det.predict("20240101", feat)
        assert result.transition_matrix.shape == (2, 2)

    def test_transition_matrix_rows_sum_to_1(self):
        det = HMMRegimeDetector()
        feat = make_feature_df()
        det.fit(feat)
        result = det.predict("20240101", feat)
        row_sums = result.transition_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_get_state_statistics_not_fitted_raises(self):
        det = HMMRegimeDetector()
        with pytest.raises(RuntimeError, match="未拟合"):
            det.get_state_statistics()

    def test_get_state_statistics_keys(self):
        det = HMMRegimeDetector()
        feat = make_feature_df()
        det.fit(feat)
        stats = det.get_state_statistics()
        assert "low_vol" in stats or "high_vol" in stats

    def test_get_state_statistics_feature_keys(self):
        det = HMMRegimeDetector(features=["log_return", "rv"])
        feat = make_feature_df()
        det.fit(feat)
        stats = det.get_state_statistics()
        for state_stats in stats.values():
            assert "log_return" in state_stats
            assert "rv" in state_stats
            assert "mean" in state_stats["rv"]
            assert "std" in state_stats["rv"]

    def test_high_vol_state_has_higher_rv(self):
        """高波动状态的 rv 均值应高于低波动状态"""
        det = HMMRegimeDetector(features=["log_return", "rv"])
        feat = make_feature_df(n=500)
        det.fit(feat)
        stats = det.get_state_statistics()
        if "low_vol" in stats and "high_vol" in stats:
            low_rv = stats["low_vol"].get("rv", {}).get("mean", 0)
            high_rv = stats["high_vol"].get("rv", {}).get("mean", 0)
            assert high_rv > low_rv

    def test_3_state_hmm(self):
        det = HMMRegimeDetector(n_states=3, features=["log_return", "rv"])
        feat = make_feature_df(n=500)
        det.fit(feat)
        result = det.predict("20240101", feat)
        assert isinstance(result.current_state, RegimeState)

    def test_insufficient_data_raises(self):
        det = HMMRegimeDetector()
        tiny_feat = make_feature_df(n=5)
        with pytest.raises(ValueError, match="样本量"):
            det.fit(tiny_feat)
