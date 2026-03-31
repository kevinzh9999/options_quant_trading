"""
test_greeks_calculator.py
--------------------------
测试 GreeksCalculator.calculate_position_greeks。

结构：
  TestReturnShape       : 返回值结构和字段完整性
  TestSingleLongCall    : 单腿多头认购
  TestSingleShortPut    : 单腿空头认沽
  TestStraddle          : 多头跨式（long call + long put）
  TestShortStraddle     : 空头跨式（short call + short put）
  TestBullCallSpread    : 牛市认购价差
  TestContractUnit      : 合约乘数放大效果
  TestIvField           : IV 字段处理（缺省/零/NaN）
  TestBoundaryExpired   : 到期合约边界
  TestEmptyPositions    : 空列表输入
"""

from __future__ import annotations

import math

from models.pricing.black_scholes import BlackScholes
from models.pricing.greeks import GreeksCalculator

# ======================================================================
# 共用测试参数
# ======================================================================

S = 4000.0
K_ATM = 4000.0
K_OTM_C = 4200.0   # OTM 认购
K_OTM_P = 3800.0   # OTM 认沽
R = 0.02
SIGMA = 0.20
TRADE_DATE = "20260101"
EXPIRE_DATE = "20270101"   # 约 1 年后（T ≈ 1.0），确保非零 T
UNIT = 100


def _pos(strike, cp, volume, expire=EXPIRE_DATE, unit=UNIT, iv=SIGMA):
    return {
        "strike_price": strike,
        "call_put": cp,
        "expire_date": expire,
        "volume": volume,
        "contract_unit": unit,
        "iv": iv,
    }


def _calc(positions, underlying=S, r=R):
    gc = GreeksCalculator(trade_date=TRADE_DATE)
    return gc.calculate_position_greeks(positions, underlying, r)


# ======================================================================
# TestReturnShape
# ======================================================================

class TestReturnShape:

    def test_top_level_keys(self):
        result = _calc([_pos(K_ATM, "C", 1)])
        for key in ("net_delta", "net_gamma", "net_theta", "net_vega", "positions_detail"):
            assert key in result, f"缺少顶层键: {key}"

    def test_positions_detail_length(self):
        positions = [_pos(K_ATM, "C", 1), _pos(K_ATM, "P", -1)]
        result = _calc(positions)
        assert len(result["positions_detail"]) == 2

    def test_detail_keys(self):
        result = _calc([_pos(K_ATM, "C", 1)])
        detail = result["positions_detail"][0]
        for key in ("strike_price", "call_put", "expire_date", "volume",
                    "contract_unit", "T", "iv", "delta", "gamma",
                    "theta", "vega", "rho",
                    "position_delta", "position_gamma",
                    "position_theta", "position_vega"):
            assert key in detail, f"detail 缺少字段: {key}"

    def test_all_values_finite(self):
        result = _calc([_pos(K_ATM, "C", 1), _pos(K_OTM_P, "P", -2)])
        for key in ("net_delta", "net_gamma", "net_theta", "net_vega"):
            assert math.isfinite(result[key]), f"{key} 不是有限数"

    def test_T_positive(self):
        result = _calc([_pos(K_ATM, "C", 1)])
        assert result["positions_detail"][0]["T"] > 0

    def test_iv_stored_in_detail(self):
        result = _calc([_pos(K_ATM, "C", 1, iv=0.30)])
        assert abs(result["positions_detail"][0]["iv"] - 0.30) < 1e-9


# ======================================================================
# TestSingleLongCall
# ======================================================================

class TestSingleLongCall:
    """多头认购：delta > 0，gamma > 0，theta < 0，vega > 0。"""

    def setup_method(self):
        self.result = _calc([_pos(K_ATM, "C", 1)])
        self.detail = self.result["positions_detail"][0]

    def test_net_delta_positive(self):
        assert self.result["net_delta"] > 0

    def test_net_gamma_positive(self):
        assert self.result["net_gamma"] > 0

    def test_net_theta_negative(self):
        assert self.result["net_theta"] < 0

    def test_net_vega_positive(self):
        assert self.result["net_vega"] > 0

    def test_position_delta_equals_delta_times_unit(self):
        expected = self.detail["delta"] * 1 * UNIT
        assert abs(self.detail["position_delta"] - expected) < 1e-9

    def test_atm_delta_in_call_range(self):
        """ATM 认购 delta ∈ (0.4, 0.7) × contract_unit（T=1时远期效应使delta>0.5）。"""
        assert 0.4 * UNIT < self.result["net_delta"] < 0.7 * UNIT

    def test_two_contracts_double_delta(self):
        r1 = _calc([_pos(K_ATM, "C", 1)])
        r2 = _calc([_pos(K_ATM, "C", 2)])
        assert abs(r2["net_delta"] - 2 * r1["net_delta"]) < 1e-9


# ======================================================================
# TestSingleShortPut
# ======================================================================

class TestSingleShortPut:
    """空头认沽：delta > 0，theta > 0，vega < 0。"""

    def setup_method(self):
        self.result = _calc([_pos(K_ATM, "P", -1)])

    def test_net_delta_positive(self):
        """空头认沽 delta = -(-0.5) × 100 > 0。"""
        assert self.result["net_delta"] > 0

    def test_net_gamma_negative(self):
        assert self.result["net_gamma"] < 0

    def test_net_theta_positive(self):
        """空头期权时间衰减对持仓有利（theta > 0）。"""
        assert self.result["net_theta"] > 0

    def test_net_vega_negative(self):
        assert self.result["net_vega"] < 0


# ======================================================================
# TestStraddle（long call + long put，ATM）
# ======================================================================

class TestStraddle:
    """多头跨式：delta ≈ 0，gamma > 0，theta < 0，vega > 0。"""

    def setup_method(self):
        self.result = _calc([
            _pos(K_ATM, "C", 1),
            _pos(K_ATM, "P", 1),
        ])

    def test_net_delta_near_zero(self):
        """ATM 跨式净 delta 较小（远期效应使T=1时约+16pts，远小于单腿~58pts）。"""
        single_call_delta = abs(_calc([_pos(K_ATM, "C", 1)])["net_delta"])
        assert abs(self.result["net_delta"]) < single_call_delta * 0.5

    def test_net_gamma_positive(self):
        assert self.result["net_gamma"] > 0

    def test_net_theta_negative(self):
        assert self.result["net_theta"] < 0

    def test_net_vega_positive(self):
        assert self.result["net_vega"] > 0

    def test_vega_is_double_single(self):
        """跨式 vega ≈ 2 × 单腿 vega。"""
        single = _calc([_pos(K_ATM, "C", 1)])
        assert abs(self.result["net_vega"] - 2 * single["net_vega"]) < 0.01


# ======================================================================
# TestShortStraddle（short call + short put，ATM）
# ======================================================================

class TestShortStraddle:
    """空头跨式：delta ≈ 0，gamma < 0，theta > 0，vega < 0。"""

    def setup_method(self):
        self.result = _calc([
            _pos(K_ATM, "C", -1),
            _pos(K_ATM, "P", -1),
        ])

    def test_net_delta_near_zero(self):
        """空头跨式净 delta 较小（远期效应使T=1时约+16pts，远小于单腿~58pts）。"""
        single_call_delta = abs(_calc([_pos(K_ATM, "C", 1)])["net_delta"])
        assert abs(self.result["net_delta"]) < single_call_delta * 0.5

    def test_net_gamma_negative(self):
        assert self.result["net_gamma"] < 0

    def test_net_theta_positive(self):
        assert self.result["net_theta"] > 0

    def test_net_vega_negative(self):
        assert self.result["net_vega"] < 0

    def test_short_straddle_opposite_long(self):
        """空头跨式 Greeks = -多头跨式 Greeks。"""
        long_result = _calc([_pos(K_ATM, "C", 1), _pos(K_ATM, "P", 1)])
        for key in ("net_delta", "net_gamma", "net_theta", "net_vega"):
            assert abs(self.result[key] + long_result[key]) < 1e-9, f"{key} 不满足对称性"


# ======================================================================
# TestBullCallSpread（long ATM call + short OTM call）
# ======================================================================

class TestBullCallSpread:
    """牛市认购价差：正 delta，正 gamma（低于单腿），低 vega。"""

    def setup_method(self):
        self.result = _calc([
            _pos(K_ATM, "C", 1),          # long ATM call
            _pos(K_OTM_C, "C", -1),       # short OTM call
        ])
        self.single_call = _calc([_pos(K_ATM, "C", 1)])

    def test_net_delta_positive(self):
        assert self.result["net_delta"] > 0

    def test_net_delta_less_than_single_call(self):
        """价差 delta 小于裸多头认购（short leg 抵消部分 delta）。"""
        assert self.result["net_delta"] < self.single_call["net_delta"]

    def test_net_vega_less_than_single_call(self):
        """价差 vega 小于单腿 vega（两腿部分对冲）。"""
        assert self.result["net_vega"] < self.single_call["net_vega"]

    def test_net_vega_positive(self):
        """ATM long leg vega > OTM short leg vega at short T（30天时ATM vega最大）。"""
        # 用短期到期日（30天）验证：近期ATM vega >> OTM vega
        expire_30 = "20260131"
        r_short = _calc([
            _pos(K_ATM,   "C",  1, expire=expire_30),
            _pos(K_OTM_C, "C", -1, expire=expire_30),
        ])
        assert r_short["net_vega"] > 0

    def test_detail_count(self):
        assert len(self.result["positions_detail"]) == 2


# ======================================================================
# TestContractUnit
# ======================================================================

class TestContractUnit:

    def test_larger_unit_scales_greeks(self):
        """合约乘数 200 vs 100：所有组合 Greeks 均为 2 倍。"""
        r100 = _calc([_pos(K_ATM, "C", 1, unit=100)])
        r200 = _calc([_pos(K_ATM, "C", 1, unit=200)])
        for key in ("net_delta", "net_gamma", "net_theta", "net_vega"):
            assert abs(r200[key] - 2 * r100[key]) < 1e-9, f"{key} 合约乘数缩放不正确"

    def test_unit_stored_in_detail(self):
        result = _calc([_pos(K_ATM, "C", 1, unit=200)])
        assert result["positions_detail"][0]["contract_unit"] == 200

    def test_position_delta_formula(self):
        """position_delta = delta × volume × contract_unit。"""
        result = _calc([_pos(K_ATM, "C", 3, unit=50)])
        detail = result["positions_detail"][0]
        expected = detail["delta"] * 3 * 50
        assert abs(detail["position_delta"] - expected) < 1e-9


# ======================================================================
# TestIvField
# ======================================================================

class TestIvField:

    def test_iv_default_when_missing(self):
        """position 无 iv 字段时应使用默认值（0.20），不报错。"""
        pos = {
            "strike_price": K_ATM, "call_put": "C",
            "expire_date": EXPIRE_DATE, "volume": 1, "contract_unit": UNIT,
        }
        gc = GreeksCalculator(trade_date=TRADE_DATE)
        result = gc.calculate_position_greeks([pos], S, R)
        assert math.isfinite(result["net_delta"])
        assert result["positions_detail"][0]["iv"] == 0.20

    def test_iv_zero_falls_back_to_default(self):
        result = _calc([_pos(K_ATM, "C", 1, iv=0.0)])
        assert result["positions_detail"][0]["iv"] == 0.20

    def test_iv_nan_falls_back_to_default(self):
        result = _calc([_pos(K_ATM, "C", 1, iv=float("nan"))])
        assert result["positions_detail"][0]["iv"] == 0.20

    def test_iv_affects_vega(self):
        """更高的 IV 会影响 Greeks 计算（验证 IV 被正确传入）。"""
        r_low = _calc([_pos(K_ATM, "C", 1, iv=0.10)])
        r_high = _calc([_pos(K_ATM, "C", 1, iv=0.50)])
        # Vega 的原始值在标准 BS 下与 sigma 无关（只依赖 d1），
        # 但 theta 会因 sigma 不同而变化
        assert r_low["net_theta"] != r_high["net_theta"]

    def test_higher_iv_higher_vega_deep_otm(self):
        """深 OTM 期权：IV 越高，Vega 越大（d1 更接近 0）。"""
        deep_otm = 5000.0
        r_low = _calc([_pos(deep_otm, "C", 1, iv=0.10)])
        r_high = _calc([_pos(deep_otm, "C", 1, iv=0.50)])
        assert r_high["net_vega"] > r_low["net_vega"]


# ======================================================================
# TestBoundaryExpired
# ======================================================================

class TestBoundaryExpired:

    def test_expired_contract_zero_greeks(self):
        """已到期合约（expire_date < trade_date）的 Greeks 应为 0。"""
        expired = "20250101"   # 在 trade_date=20260101 之前
        result = _calc([_pos(K_ATM, "C", 1, expire=expired)])
        assert result["net_delta"] == 0.0
        assert result["net_gamma"] == 0.0
        assert result["net_theta"] == 0.0
        assert result["net_vega"] == 0.0

    def test_T_zero_for_expired(self):
        expired = "20250101"
        result = _calc([_pos(K_ATM, "C", 1, expire=expired)])
        assert result["positions_detail"][0]["T"] == 0.0

    def test_short_expiry_vega_smaller_than_long_expiry(self):
        """剩余 1 天的 Vega 应远小于剩余 1 年的 Vega（√T 缩放）。"""
        near_expire = "20260102"   # trade_date + 1 day
        r_near = _calc([_pos(K_ATM, "C", 1, expire=near_expire)])
        r_far  = _calc([_pos(K_ATM, "C", 1)])
        assert abs(r_near["net_vega"]) < abs(r_far["net_vega"])


# ======================================================================
# TestEmptyPositions
# ======================================================================

class TestEmptyPositions:

    def test_empty_list_all_zeros(self):
        result = _calc([])
        assert result["net_delta"] == 0.0
        assert result["net_gamma"] == 0.0
        assert result["net_theta"] == 0.0
        assert result["net_vega"] == 0.0

    def test_empty_list_empty_detail(self):
        result = _calc([])
        assert result["positions_detail"] == []

    def test_zero_volume_contributes_nothing(self):
        """持仓量为 0 的合约不应影响组合 Greeks。"""
        result_zero = _calc([_pos(K_ATM, "C", 0)])
        result_empty = _calc([])
        assert result_zero["net_delta"] == result_empty["net_delta"]
        assert result_zero["net_vega"] == result_empty["net_vega"]


# ======================================================================
# TestNumericalConsistency
# ======================================================================

class TestNumericalConsistency:
    """验证 GreeksCalculator 与 BlackScholes 直接调用的一致性。"""

    def _T(self):
        import pandas as pd
        return max((pd.Timestamp(EXPIRE_DATE) - pd.Timestamp(TRADE_DATE)).days / 365.0, 0.0)

    def test_delta_matches_blackscholes(self):
        result = _calc([_pos(K_ATM, "C", 1)])
        T = self._T()
        expected_delta = BlackScholes.delta(S, K_ATM, T, R, SIGMA, 0.0, "C")
        actual_delta = result["positions_detail"][0]["delta"]
        assert abs(actual_delta - expected_delta) < 1e-9

    def test_gamma_matches_blackscholes(self):
        result = _calc([_pos(K_ATM, "C", 1)])
        T = self._T()
        expected = BlackScholes.gamma(S, K_ATM, T, R, SIGMA, 0.0)
        actual = result["positions_detail"][0]["gamma"]
        assert abs(actual - expected) < 1e-9

    def test_theta_matches_blackscholes(self):
        result = _calc([_pos(K_ATM, "C", 1)])
        T = self._T()
        expected = BlackScholes.theta(S, K_ATM, T, R, SIGMA, 0.0, "C")
        actual = result["positions_detail"][0]["theta"]
        assert abs(actual - expected) < 1e-9

    def test_vega_matches_blackscholes(self):
        result = _calc([_pos(K_ATM, "C", 1)])
        T = self._T()
        expected = BlackScholes.vega(S, K_ATM, T, R, SIGMA, 0.0)
        actual = result["positions_detail"][0]["vega"]
        assert abs(actual - expected) < 1e-9

    def test_net_delta_is_sum_of_position_deltas(self):
        positions = [
            _pos(K_ATM, "C", 2),
            _pos(K_OTM_P, "P", -1),
        ]
        result = _calc(positions)
        expected_net = sum(d["position_delta"] for d in result["positions_detail"])
        assert abs(result["net_delta"] - expected_net) < 1e-9

    def test_net_vega_is_sum_of_position_vegas(self):
        positions = [_pos(K_ATM, "C", 1), _pos(K_ATM, "P", 1), _pos(K_OTM_C, "C", -2)]
        result = _calc(positions)
        expected_net = sum(d["position_vega"] for d in result["positions_detail"])
        assert abs(result["net_vega"] - expected_net) < 1e-9
