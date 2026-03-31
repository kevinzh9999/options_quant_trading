"""
test_implied_vol.py
-------------------
测试 BS 定价和隐含波动率计算（models/implied_vol.py）。
"""

from __future__ import annotations

import numpy as np
import pytest

from models.pricing.implied_vol import OptionType, bs_price, calc_implied_vol


# ======================================================================
# BS 定价测试
# ======================================================================

class TestBsPrice:

    def test_atm_call_put_parity(self):
        """
        ATM 平价验证：Call - Put = S*e^{-qT} - K*e^{-rT}
        这里假设无股息（q=0），则 Call - Put = S - K*e^{-rT}
        """
        S, K, T, r, sigma = 4000.0, 4000.0, 30 / 365, 0.02, 0.20
        call = bs_price(S, K, T, r, sigma, OptionType.CALL)
        put = bs_price(S, K, T, r, sigma, OptionType.PUT)
        expected_diff = S - K * np.exp(-r * T)
        assert abs((call - put) - expected_diff) < 1e-6, "Put-Call Parity 不成立"

    def test_call_price_positive(self):
        """期权价格应为正数"""
        price = bs_price(4000, 4200, 60 / 365, 0.02, 0.25, OptionType.CALL)
        assert price > 0

    def test_put_price_positive(self):
        """认沽期权价格应为正数"""
        price = bs_price(4000, 3800, 60 / 365, 0.02, 0.25, OptionType.PUT)
        assert price > 0

    def test_call_intrinsic_value_lower_bound(self):
        """深度实值 Call 价格应 >= max(0, S - K*e^{-rT})"""
        S, K, T, r, sigma = 5000.0, 4000.0, 30 / 365, 0.02, 0.20
        price = bs_price(S, K, T, r, sigma, OptionType.CALL)
        intrinsic = S - K * np.exp(-r * T)
        assert price >= intrinsic - 1e-6

    def test_put_intrinsic_value_lower_bound(self):
        """深度实值 Put 价格应 >= max(0, K*e^{-rT} - S)"""
        S, K, T, r, sigma = 3000.0, 4000.0, 30 / 365, 0.02, 0.20
        price = bs_price(S, K, T, r, sigma, OptionType.PUT)
        intrinsic = K * np.exp(-r * T) - S
        assert price >= intrinsic - 1e-6

    def test_higher_vol_higher_price(self):
        """波动率越高，期权价格越贵"""
        S, K, T, r = 4000.0, 4100.0, 45 / 365, 0.02
        price_low = bs_price(S, K, T, r, 0.15, OptionType.CALL)
        price_high = bs_price(S, K, T, r, 0.30, OptionType.CALL)
        assert price_high > price_low

    def test_longer_maturity_higher_price(self):
        """到期时间越长，期权价格越贵"""
        S, K, r, sigma = 4000.0, 4100.0, 0.02, 0.20
        price_short = bs_price(S, K, 15 / 365, r, sigma, OptionType.CALL)
        price_long = bs_price(S, K, 60 / 365, r, sigma, OptionType.CALL)
        assert price_long > price_short


# ======================================================================
# 隐含波动率测试
# ======================================================================

class TestCalcImpliedVol:

    def test_iv_roundtrip(self):
        """正向定价 -> 反推 IV -> 应恢复原始波动率"""
        S, K, T, r = 4000.0, 4000.0, 30 / 365, 0.02
        true_sigma = 0.20

        market_price = bs_price(S, K, T, r, true_sigma, OptionType.CALL)
        implied_sigma = calc_implied_vol(market_price, S, K, T, r, OptionType.CALL)

        assert implied_sigma is not None
        assert abs(implied_sigma - true_sigma) < 1e-4, (
            f"IV 反推误差过大: 真实={true_sigma:.4f}, 推算={implied_sigma:.4f}"
        )

    def test_iv_put_roundtrip(self):
        """认沽期权 IV 反推"""
        S, K, T, r = 4000.0, 3800.0, 45 / 365, 0.02
        true_sigma = 0.25

        market_price = bs_price(S, K, T, r, true_sigma, OptionType.PUT)
        implied_sigma = calc_implied_vol(market_price, S, K, T, r, OptionType.PUT)

        assert implied_sigma is not None
        assert abs(implied_sigma - true_sigma) < 1e-4

    def test_iv_positive(self):
        """IV 应为正数"""
        iv = calc_implied_vol(
            market_price=50.0, S=4000.0, K=4000.0,
            T=30 / 365, r=0.02, option_type=OptionType.CALL
        )
        assert iv is None or iv > 0

    def test_iv_returns_none_for_expired(self):
        """到期时间 T=0 时应返回 None"""
        iv = calc_implied_vol(
            market_price=50.0, S=4000.0, K=4000.0,
            T=0.0, r=0.02, option_type=OptionType.CALL
        )
        assert iv is None

    def test_iv_returns_none_below_intrinsic(self):
        """期权价格低于内在价值时应返回 None（无解）"""
        S, K, T, r = 4000.0, 3000.0, 30 / 365, 0.02
        intrinsic = S - K * np.exp(-r * T)
        iv = calc_implied_vol(
            market_price=intrinsic - 1,  # 低于内在价值
            S=S, K=K, T=T, r=r, option_type=OptionType.CALL
        )
        assert iv is None

    @pytest.mark.parametrize("method", ["newton", "bisect", "brent"])
    def test_all_methods_consistent(self, method: str):
        """所有求解方法应给出一致的 IV"""
        S, K, T, r, sigma = 4000.0, 4100.0, 45 / 365, 0.02, 0.22
        market_price = bs_price(S, K, T, r, sigma, OptionType.CALL)
        iv = calc_implied_vol(market_price, S, K, T, r, OptionType.CALL, method=method)
        assert iv is not None
        assert abs(iv - sigma) < 1e-3, f"方法 {method} IV 误差过大: {iv:.6f} vs {sigma:.6f}"
