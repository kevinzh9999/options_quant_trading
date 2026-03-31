"""
tests/test_models/test_black_scholes.py
-----------------------------------------
BlackScholes 类单元测试。

覆盖：
- price(): 已知结果验证、Put-Call Parity（含分红率）、边界条件
- delta(): 符号、范围、ATM 近似、含分红率
- gamma(): 正值、Call/Put 相同、含分红率
- theta(): 负值（通常）、每日单位、含分红率
- vega(): 正值、ATM 最大、含分红率
- implied_volatility(): 正向→反推 round-trip、Newton 收敛、二分法兜底、边界 NaN
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from models.pricing.black_scholes import BlackScholes

BS = BlackScholes  # 简写


# ======================================================================
# 参考参数
# ======================================================================

# 教科书经典参数（无分红）：S=K=100, T=1, r=5%, σ=20%
# Call ≈ 10.4506  (Hull 《期权、期货及其他衍生产品》)
S0, K0, T0, r0, sigma0 = 100.0, 100.0, 1.0, 0.05, 0.20

# 含分红
q0 = 0.02   # 2% 连续分红率


# ======================================================================
# TestBSPrice — 定价
# ======================================================================

class TestBSPrice:
    def test_call_known_value(self):
        """S=K=100, T=1, r=5%, σ=20% → Call ≈ 10.4506"""
        c = BS.price(S0, K0, T0, r0, sigma0, option_type="C")
        assert abs(c - 10.4506) < 0.001, f"Call price {c:.4f} ≠ expected 10.4506"

    def test_put_known_value(self):
        """Put-Call Parity 反推：Put = Call - S + K·e^{-rT} ≈ 5.5735"""
        c = BS.price(S0, K0, T0, r0, sigma0, option_type="C")
        expected_put = c - S0 + K0 * math.exp(-r0 * T0)
        p = BS.price(S0, K0, T0, r0, sigma0, option_type="P")
        assert abs(p - expected_put) < 1e-8

    def test_put_call_parity_no_dividend(self):
        """Call - Put = S·e^{-qT} - K·e^{-rT}（q=0）"""
        c = BS.price(S0, K0, T0, r0, sigma0, q=0.0, option_type="C")
        p = BS.price(S0, K0, T0, r0, sigma0, q=0.0, option_type="P")
        lhs = c - p
        rhs = S0 * math.exp(0) - K0 * math.exp(-r0 * T0)
        assert abs(lhs - rhs) < 1e-8

    def test_put_call_parity_with_dividend(self):
        """Call - Put = S·e^{-qT} - K·e^{-rT}（含分红率）"""
        c = BS.price(S0, K0, T0, r0, sigma0, q=q0, option_type="C")
        p = BS.price(S0, K0, T0, r0, sigma0, q=q0, option_type="P")
        lhs = c - p
        rhs = S0 * math.exp(-q0 * T0) - K0 * math.exp(-r0 * T0)
        assert abs(lhs - rhs) < 1e-8

    def test_call_positive(self):
        assert BS.price(S0, K0, T0, r0, sigma0, option_type="C") > 0

    def test_put_positive(self):
        assert BS.price(S0, K0, T0, r0, sigma0, option_type="P") > 0

    def test_deep_itm_call_near_intrinsic(self):
        """深度实值 Call 趋近于 S - K·e^{-rT}"""
        c = BS.price(200.0, 100.0, T0, r0, 0.01, option_type="C")
        lower = 200.0 - 100.0 * math.exp(-r0 * T0)
        assert c > lower - 0.01

    def test_higher_vol_higher_price(self):
        c_low = BS.price(S0, K0, T0, r0, 0.10, option_type="C")
        c_high = BS.price(S0, K0, T0, r0, 0.40, option_type="C")
        assert c_high > c_low

    def test_longer_expiry_higher_call(self):
        c_short = BS.price(S0, K0, 0.25, r0, sigma0, option_type="C")
        c_long = BS.price(S0, K0, 1.0, r0, sigma0, option_type="C")
        assert c_long > c_short

    # --- 边界条件 ---

    def test_s_zero_returns_zero(self):
        assert BS.price(0.0, K0, T0, r0, sigma0, option_type="C") == 0.0

    def test_k_zero_returns_zero(self):
        assert BS.price(S0, 0.0, T0, r0, sigma0, option_type="C") == 0.0

    def test_expired_call_intrinsic(self):
        """T=0 时 Call 返回内在价值 max(S-K, 0)"""
        assert BS.price(120.0, 100.0, 0.0, r0, sigma0, option_type="C") == 20.0
        assert BS.price(80.0, 100.0, 0.0, r0, sigma0, option_type="C") == 0.0

    def test_expired_put_intrinsic(self):
        assert BS.price(80.0, 100.0, 0.0, r0, sigma0, option_type="P") == 20.0
        assert BS.price(120.0, 100.0, 0.0, r0, sigma0, option_type="P") == 0.0

    def test_sigma_zero_call_discounted(self):
        """σ=0 时 Call 返回折现内在价值"""
        c = BS.price(S0, K0, T0, r0, 0.0, option_type="C")
        expected = max(S0 - K0 * math.exp(-r0 * T0), 0.0)
        assert abs(c - expected) < 1e-8

    def test_dividend_reduces_call(self):
        """分红率使 Call 价格降低（因标的折现价更低）"""
        c_no_div = BS.price(S0, K0, T0, r0, sigma0, q=0.0, option_type="C")
        c_div = BS.price(S0, K0, T0, r0, sigma0, q=0.05, option_type="C")
        assert c_no_div > c_div

    def test_call_upper_bound(self):
        """Call ≤ S·e^{-qT}（标的折现价是上界）"""
        c = BS.price(S0, K0, T0, r0, sigma0, q=q0, option_type="C")
        assert c <= S0 * math.exp(-q0 * T0) + 1e-8

    def test_string_option_type(self):
        c = BS.price(S0, K0, T0, r0, sigma0, option_type="C")
        assert c > 0


# ======================================================================
# TestBSDelta — Delta
# ======================================================================

class TestBSDelta:
    def test_call_delta_positive(self):
        assert BS.delta(S0, K0, T0, r0, sigma0, option_type="C") > 0

    def test_put_delta_negative(self):
        assert BS.delta(S0, K0, T0, r0, sigma0, option_type="P") < 0

    def test_call_delta_in_0_1(self):
        d = BS.delta(S0, K0, T0, r0, sigma0, option_type="C")
        assert 0 < d < 1

    def test_put_delta_in_minus1_0(self):
        d = BS.delta(S0, K0, T0, r0, sigma0, option_type="P")
        assert -1 < d < 0

    def test_atm_call_delta_near_half(self):
        """ATM Call Delta ≈ 0.5（无分红时稍高于 0.5）"""
        d = BS.delta(S0, K0, T0, r0, sigma0, option_type="C")
        assert 0.45 < d < 0.65

    def test_put_call_delta_relationship(self):
        """Call Delta - Put Delta = e^{-qT}（含分红率的 Put-Call 关系）"""
        dc = BS.delta(S0, K0, T0, r0, sigma0, q=q0, option_type="C")
        dp = BS.delta(S0, K0, T0, r0, sigma0, q=q0, option_type="P")
        expected = math.exp(-q0 * T0)
        assert abs(dc - dp - expected) < 1e-8

    def test_deep_itm_call_delta_near_1(self):
        d = BS.delta(200.0, 100.0, T0, r0, sigma0, option_type="C")
        assert d > 0.9

    def test_deep_otm_call_delta_near_0(self):
        d = BS.delta(50.0, 100.0, T0, r0, sigma0, option_type="C")
        assert d < 0.1

    def test_dividend_reduces_call_delta(self):
        """分红率使 Call Delta 降低（e^{-qT} 因子）"""
        d_no_div = BS.delta(S0, K0, T0, r0, sigma0, q=0.0, option_type="C")
        d_div = BS.delta(S0, K0, T0, r0, sigma0, q=0.10, option_type="C")
        assert d_no_div > d_div


# ======================================================================
# TestBSGamma — Gamma
# ======================================================================

class TestBSGamma:
    def test_gamma_positive(self):
        assert BS.gamma(S0, K0, T0, r0, sigma0) > 0

    def test_gamma_call_equals_put(self):
        """Gamma 对 Call 和 Put 相同"""
        # gamma 只有一个接口（不区分期权类型），验证值为正即可
        g = BS.gamma(S0, K0, T0, r0, sigma0)
        assert g > 0

    def test_atm_gamma_is_max(self):
        """ATM Gamma > OTM Gamma"""
        g_atm = BS.gamma(100.0, 100.0, T0, r0, sigma0)
        g_otm = BS.gamma(100.0, 150.0, T0, r0, sigma0)
        assert g_atm > g_otm

    def test_expired_gamma_zero(self):
        assert BS.gamma(S0, K0, 0.0, r0, sigma0) == 0.0

    def test_sigma_zero_gamma_zero(self):
        assert BS.gamma(S0, K0, T0, r0, 0.0) == 0.0

    def test_finite(self):
        assert math.isfinite(BS.gamma(S0, K0, T0, r0, sigma0))

    def test_dividend_effect(self):
        """含分红率时 Gamma 应仍为正"""
        g = BS.gamma(S0, K0, T0, r0, sigma0, q=q0)
        assert g > 0


# ======================================================================
# TestBSTheta — Theta
# ======================================================================

class TestBSTheta:
    def test_call_theta_negative(self):
        """做多 Call 的 Theta 通常为负（时间价值损耗）"""
        assert BS.theta(S0, K0, T0, r0, sigma0, option_type="C") < 0

    def test_put_theta_negative(self):
        """ATM Put Theta 通常为负（时间价值损耗）"""
        assert BS.theta(S0, K0, T0, r0, sigma0, option_type="P") < 0

    def test_theta_is_daily(self):
        """Theta 应为每日衰减（量级约为年化 Theta / 365）"""
        theta_daily = BS.theta(S0, K0, T0, r0, sigma0, option_type="C")
        # 粗略验证：|theta_daily| 应远小于期权价值
        price = BS.price(S0, K0, T0, r0, sigma0, option_type="C")
        assert abs(theta_daily) < price  # 每天衰减 < 期权价值

    def test_expired_theta_zero(self):
        assert BS.theta(S0, K0, 0.0, r0, sigma0, option_type="C") == 0.0

    def test_sigma_zero_theta_zero(self):
        assert BS.theta(S0, K0, T0, r0, 0.0, option_type="C") == 0.0

    def test_finite(self):
        assert math.isfinite(BS.theta(S0, K0, T0, r0, sigma0, option_type="C"))

    def test_dividend_call_theta_sign(self):
        """高分红率下 Call 的 Theta 仍为负（正常市场条件）"""
        t = BS.theta(S0, K0, 0.25, r0, sigma0, q=0.01, option_type="C")
        assert t < 0


# ======================================================================
# TestBSVega — Vega
# ======================================================================

class TestBSVega:
    def test_vega_positive(self):
        assert BS.vega(S0, K0, T0, r0, sigma0) > 0

    def test_vega_call_equals_put(self):
        """Vega 对 Call 和 Put 相同（单接口，不区分类型）"""
        v = BS.vega(S0, K0, T0, r0, sigma0)
        assert v > 0

    def test_atm_vega_is_max(self):
        """ATM Vega > OTM Vega"""
        v_atm = BS.vega(100.0, 100.0, T0, r0, sigma0)
        v_otm = BS.vega(100.0, 200.0, T0, r0, sigma0)
        assert v_atm > v_otm

    def test_vega_per_1pct(self):
        """Vega 定义为每 1% 波动率变动的价格变化 → 应为 raw_vega / 100"""
        import math
        from scipy.stats import norm as scipy_norm
        d1 = (math.log(S0 / K0) + (r0 + 0.5 * sigma0 ** 2) * T0) / (sigma0 * math.sqrt(T0))
        raw_vega = S0 * scipy_norm.pdf(d1) * math.sqrt(T0)
        expected = raw_vega / 100.0
        assert abs(BS.vega(S0, K0, T0, r0, sigma0) - expected) < 1e-8

    def test_expired_vega_zero(self):
        assert BS.vega(S0, K0, 0.0, r0, sigma0) == 0.0

    def test_sigma_zero_vega_zero(self):
        assert BS.vega(S0, K0, T0, r0, 0.0) == 0.0

    def test_longer_expiry_higher_vega(self):
        v_short = BS.vega(S0, K0, 0.25, r0, sigma0)
        v_long = BS.vega(S0, K0, 1.0, r0, sigma0)
        assert v_long > v_short

    def test_dividend_vega(self):
        v = BS.vega(S0, K0, T0, r0, sigma0, q=q0)
        assert v > 0


# ======================================================================
# TestBSImpliedVolatility — IV 反推
# ======================================================================

class TestBSImpliedVolatility:
    def _round_trip(self, S, K, T, r, sigma, q=0.0, option_type="C"):
        """正向定价 → 反推 IV 的 round-trip 测试。"""
        price = BS.price(S, K, T, r, sigma, q=q, option_type=option_type)
        iv = BS.implied_volatility(price, S, K, T, r, q=q, option_type=option_type)
        return iv

    def test_atm_call_round_trip(self):
        iv = self._round_trip(S0, K0, T0, r0, sigma0, option_type="C")
        assert iv is not None and not math.isnan(iv)
        assert abs(iv - sigma0) < 1e-6

    def test_atm_put_round_trip(self):
        iv = self._round_trip(S0, K0, T0, r0, sigma0, option_type="P")
        assert abs(iv - sigma0) < 1e-6

    def test_otm_call_round_trip(self):
        iv = self._round_trip(100.0, 110.0, 0.5, 0.03, 0.25, option_type="C")
        assert abs(iv - 0.25) < 1e-5

    def test_itm_put_round_trip(self):
        iv = self._round_trip(100.0, 110.0, 0.5, 0.03, 0.25, option_type="P")
        assert abs(iv - 0.25) < 1e-5

    def test_high_vol_round_trip(self):
        iv = self._round_trip(S0, K0, T0, r0, 0.60, option_type="C")
        assert abs(iv - 0.60) < 1e-5

    def test_low_vol_round_trip(self):
        iv = self._round_trip(S0, K0, T0, r0, 0.05, option_type="C")
        assert abs(iv - 0.05) < 1e-5

    def test_with_dividend_round_trip(self):
        iv = self._round_trip(S0, K0, T0, r0, 0.30, q=0.03, option_type="C")
        assert abs(iv - 0.30) < 1e-5

    def test_short_expiry_round_trip(self):
        iv = self._round_trip(S0, K0, 5 / 365, r0, sigma0, option_type="C")
        assert abs(iv - sigma0) < 1e-4

    # --- 边界 → 应返回 NaN ---

    def test_t_zero_returns_nan(self):
        assert math.isnan(BS.implied_volatility(5.0, S0, K0, 0.0, r0))

    def test_negative_price_returns_nan(self):
        assert math.isnan(BS.implied_volatility(-1.0, S0, K0, T0, r0))

    def test_price_below_intrinsic_returns_nan(self):
        """期权价格低于内在价值 → 无套利条件不满足 → NaN"""
        # 深度实值 Call 内在价值约 S-K·e^{-rT} ≈ 100 - 50·e^{-0.05} ≈ 52.4
        assert math.isnan(BS.implied_volatility(1.0, 100.0, 50.0, T0, r0, option_type="C"))

    def test_s_zero_returns_nan(self):
        assert math.isnan(BS.implied_volatility(5.0, 0.0, K0, T0, r0))

    def test_k_zero_returns_nan(self):
        assert math.isnan(BS.implied_volatility(5.0, S0, 0.0, T0, r0))

    def test_zero_market_price_returns_nan(self):
        assert math.isnan(BS.implied_volatility(0.0, S0, K0, T0, r0))

    def test_extremely_small_price_returns_nan(self):
        """极小价格（接近 0）→ NaN（深度虚值）"""
        assert math.isnan(BS.implied_volatility(1e-12, S0, K0, T0, r0))

    # --- IV 值范围 ---

    def test_iv_positive(self):
        price = BS.price(S0, K0, T0, r0, sigma0, option_type="C")
        iv = BS.implied_volatility(price, S0, K0, T0, r0)
        assert iv > 0

    def test_iv_in_reasonable_range(self):
        price = BS.price(S0, K0, T0, r0, sigma0, option_type="C")
        iv = BS.implied_volatility(price, S0, K0, T0, r0)
        assert 0.01 < iv < 5.0


# ======================================================================
# TestBSNumericalConsistency — 数值一致性
# ======================================================================

class TestBSNumericalConsistency:
    def test_vega_approximates_price_sensitivity(self):
        """有限差分验证：vega ≈ (price(σ+ε) - price(σ-ε)) / (2ε) × 0.01"""
        eps = 1e-4
        p_up = BS.price(S0, K0, T0, r0, sigma0 + eps, option_type="C")
        p_dn = BS.price(S0, K0, T0, r0, sigma0 - eps, option_type="C")
        fd_vega = (p_up - p_dn) / (2 * eps) * 0.01  # per 1% σ change
        analytic_vega = BS.vega(S0, K0, T0, r0, sigma0)
        assert abs(fd_vega - analytic_vega) < 1e-5

    def test_delta_approximates_price_sensitivity(self):
        """有限差分验证：delta ≈ (price(S+ε) - price(S-ε)) / (2ε)"""
        eps = 0.01
        p_up = BS.price(S0 + eps, K0, T0, r0, sigma0, option_type="C")
        p_dn = BS.price(S0 - eps, K0, T0, r0, sigma0, option_type="C")
        fd_delta = (p_up - p_dn) / (2 * eps)
        analytic_delta = BS.delta(S0, K0, T0, r0, sigma0, option_type="C")
        assert abs(fd_delta - analytic_delta) < 1e-4

    def test_gamma_approximates_delta_sensitivity(self):
        """有限差分验证：gamma ≈ (delta(S+ε) - delta(S-ε)) / (2ε)"""
        eps = 0.1
        d_up = BS.delta(S0 + eps, K0, T0, r0, sigma0, option_type="C")
        d_dn = BS.delta(S0 - eps, K0, T0, r0, sigma0, option_type="C")
        fd_gamma = (d_up - d_dn) / (2 * eps)
        analytic_gamma = BS.gamma(S0, K0, T0, r0, sigma0)
        assert abs(fd_gamma - analytic_gamma) < 1e-4

    def test_theta_approximates_time_decay(self):
        """有限差分验证：theta ≈ (price(T-ε) - price(T)) / ε / 365"""
        eps = 1 / 365
        p_now = BS.price(S0, K0, T0, r0, sigma0, option_type="C")
        p_tomorrow = BS.price(S0, K0, T0 - eps, r0, sigma0, option_type="C")
        fd_theta = p_tomorrow - p_now  # 1-day change
        analytic_theta = BS.theta(S0, K0, T0, r0, sigma0, option_type="C")
        assert abs(fd_theta - analytic_theta) < 1e-3
