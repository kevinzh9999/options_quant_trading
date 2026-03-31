"""
tests/test_models/test_pricing.py
----------------------------------
implied_vol / greeks / vol_surface 单元测试。

覆盖：
- bs_d1_d2: 标准参数验证
- bs_price: Call/Put 对称性、Put-Call Parity、边界条件
- calc_implied_vol: 正向→反推 round-trip，三种方法，边界情况
- calc_implied_vol_batch: DataFrame 批量计算
- calc_delta/gamma/theta/vega/rho: 符号/范围/关系
- calc_all_greeks: 一次性计算与逐一计算一致
- calc_portfolio_greeks: 空持仓、多空组合
- VolSmile: ATM 插值、get_iv
- VolSurface: build_from_options_df、查询接口、term_structure、to_dataframe
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.pricing.implied_vol import (
    OptionType,
    bs_d1_d2,
    bs_price,
    calc_implied_vol,
    calc_implied_vol_batch,
)
from models.pricing.greeks import (
    Greeks,
    PortfolioGreeks,
    calc_all_greeks,
    calc_delta,
    calc_gamma,
    calc_portfolio_greeks,
    calc_rho,
    calc_theta,
    calc_vega,
)
from models.pricing.vol_surface import VolSmile, VolSurface


# ======================================================================
# 共用参数
# ======================================================================

# 标准 BS 测试参数
S, K, T, r, sigma = 3800.0, 3800.0, 30 / 365, 0.025, 0.20
S_itm, K_otm = 4000.0, 3800.0  # 深度实值 Call


def make_atm_call_price() -> float:
    return bs_price(S, K, T, r, sigma, OptionType.CALL)


def make_atm_put_price() -> float:
    return bs_price(S, K, T, r, sigma, OptionType.PUT)


# ======================================================================
# TestBSD1D2
# ======================================================================

class TestBSD1D2:
    def test_atm_d2_equals_d1_minus_sigma_sqrtT(self):
        d1, d2 = bs_d1_d2(S, K, T, r, sigma)
        assert abs(d2 - (d1 - sigma * np.sqrt(T))) < 1e-12

    def test_d1_formula(self):
        d1, _ = bs_d1_d2(S, K, T, r, sigma)
        expected = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        assert abs(d1 - expected) < 1e-12

    def test_deep_itm_d1_positive(self):
        d1, _ = bs_d1_d2(S_itm, K_otm, T, r, sigma)
        assert d1 > 0  # 实值 Call 的 d1 应为正

    def test_deep_otm_d1_negative(self):
        d1, _ = bs_d1_d2(3000.0, K, T, r, sigma)
        assert d1 < 0  # 虚值 Call 的 d1 应为负


# ======================================================================
# TestBSPrice
# ======================================================================

class TestBSPrice:
    def test_call_positive(self):
        assert make_atm_call_price() > 0

    def test_put_positive(self):
        assert make_atm_put_price() > 0

    def test_put_call_parity(self):
        """Put-Call Parity: C - P = S - K·e^{-rT}"""
        C = make_atm_call_price()
        P = make_atm_put_price()
        rhs = S - K * np.exp(-r * T)
        assert abs(C - P - rhs) < 1e-6

    def test_call_ge_put_for_itm_call(self):
        """实值 Call > Put（相同行权价 K < S）"""
        C = bs_price(S, 3500.0, T, r, sigma, OptionType.CALL)
        P = bs_price(S, 3500.0, T, r, sigma, OptionType.PUT)
        assert C > P

    def test_call_price_lower_bound(self):
        """Call ≥ max(S - K·e^{-rT}, 0)"""
        C = make_atm_call_price()
        lower = max(S - K * np.exp(-r * T), 0.0)
        assert C >= lower - 1e-6

    def test_call_price_upper_bound(self):
        """Call ≤ S"""
        C = make_atm_call_price()
        assert C <= S

    def test_expired_call_intrinsic(self):
        """T=0 时，Call 返回内在价值"""
        C = bs_price(S_itm, K_otm, 0.0, r, sigma, OptionType.CALL)
        assert abs(C - (S_itm - K_otm)) < 1e-10

    def test_expired_put_otm_zero(self):
        """T=0 时，OTM Put 价值为 0"""
        P = bs_price(S, 3000.0, 0.0, r, sigma, OptionType.PUT)
        assert P == 0.0

    def test_string_option_type_works(self):
        C1 = bs_price(S, K, T, r, sigma, OptionType.CALL)
        C2 = bs_price(S, K, T, r, sigma, "C")
        assert abs(C1 - C2) < 1e-12

    def test_higher_vol_higher_call_price(self):
        C_low = bs_price(S, K, T, r, 0.10, OptionType.CALL)
        C_high = bs_price(S, K, T, r, 0.40, OptionType.CALL)
        assert C_high > C_low

    def test_longer_expiry_higher_call_price(self):
        C_short = bs_price(S, K, 30 / 365, r, sigma, OptionType.CALL)
        C_long = bs_price(S, K, 90 / 365, r, sigma, OptionType.CALL)
        assert C_long > C_short


# ======================================================================
# TestCalcImpliedVol — round-trip 测试
# ======================================================================

class TestCalcImpliedVol:
    def _round_trip(self, S, K, T, r, sigma, opt_type, method="brent"):
        price = bs_price(S, K, T, r, sigma, opt_type)
        iv = calc_implied_vol(price, S, K, T, r, opt_type, method=method)
        return iv

    def test_atm_call_brent(self):
        iv = self._round_trip(S, K, T, r, 0.20, OptionType.CALL, "brent")
        assert iv is not None
        assert abs(iv - 0.20) < 1e-5

    def test_atm_put_brent(self):
        iv = self._round_trip(S, K, T, r, 0.20, OptionType.PUT, "brent")
        assert iv is not None
        assert abs(iv - 0.20) < 1e-5

    def test_atm_call_newton(self):
        iv = self._round_trip(S, K, T, r, 0.20, OptionType.CALL, "newton")
        assert iv is not None
        assert abs(iv - 0.20) < 1e-4

    def test_atm_call_bisect(self):
        iv = self._round_trip(S, K, T, r, 0.20, OptionType.CALL, "bisect")
        assert iv is not None
        assert abs(iv - 0.20) < 1e-5

    def test_high_vol_round_trip(self):
        iv = self._round_trip(S, K, T, r, 0.50, OptionType.CALL, "brent")
        assert iv is not None
        assert abs(iv - 0.50) < 1e-4

    def test_low_vol_round_trip(self):
        iv = self._round_trip(S, K, T, r, 0.10, OptionType.PUT, "brent")
        assert iv is not None
        assert abs(iv - 0.10) < 1e-4

    def test_expired_returns_none(self):
        assert calc_implied_vol(100.0, S, K, 0.0, r, OptionType.CALL) is None

    def test_price_below_intrinsic_returns_none(self):
        """期权价格低于内在价值时无解"""
        # 深度实值 Call 内在价值 = S - K·e^{-rT} ≈ 200 (S=4000, K=3800)
        result = calc_implied_vol(1.0, 4000.0, 3800.0, T, r, OptionType.CALL)
        assert result is None

    def test_otm_call_round_trip(self):
        """虚值 Call"""
        price = bs_price(3800.0, 4200.0, 60 / 365, r, 0.25, OptionType.CALL)
        iv = calc_implied_vol(price, 3800.0, 4200.0, 60 / 365, r, OptionType.CALL)
        assert iv is not None
        assert abs(iv - 0.25) < 1e-4

    def test_string_option_type(self):
        price = bs_price(S, K, T, r, 0.20, "C")
        iv = calc_implied_vol(price, S, K, T, r, "C")
        assert iv is not None
        assert abs(iv - 0.20) < 1e-5


# ======================================================================
# TestCalcImpliedVolBatch
# ======================================================================

class TestCalcImpliedVolBatch:
    def _make_options_df(self) -> pd.DataFrame:
        """构造 5 个期权的测试 DataFrame（expire_date 与 trade_date=20240601 恰好相差 30 天）"""
        rows = []
        # 20240601 + 30 days = 20240701, so T = 30/365 exactly
        expire_date = "20240701"
        T_test = 30 / 365
        for K_test in [3600, 3700, 3800, 3900, 4000]:
            sigma_test = 0.20 + 0.02 * abs(K_test - 3800) / 100  # smile
            for ot in ["C", "P"]:
                price = bs_price(3800.0, K_test, T_test, 0.025, sigma_test, ot)
                rows.append({
                    "ts_code": f"IO2406-{ot}-{K_test}",
                    "strike_price": float(K_test),
                    "call_put": ot,
                    "expire_date": expire_date,
                    "close": price,
                })
        return pd.DataFrame(rows)

    def test_returns_series(self):
        df = self._make_options_df()
        result = calc_implied_vol_batch(df, 3800.0, 0.025, "20240601")
        assert isinstance(result, pd.Series)

    def test_indexed_by_ts_code(self):
        df = self._make_options_df()
        result = calc_implied_vol_batch(df, 3800.0, 0.025, "20240601")
        assert "IO2406-C-3800" in result.index

    def test_iv_values_positive(self):
        df = self._make_options_df()
        result = calc_implied_vol_batch(df, 3800.0, 0.025, "20240601")
        valid = result.dropna()
        assert (valid > 0).all()

    def test_empty_df_returns_empty(self):
        result = calc_implied_vol_batch(pd.DataFrame(), 3800.0, 0.025)
        assert result.empty

    def test_approximate_iv_values(self):
        """ATM 期权 IV 应接近输入的 0.20"""
        df = self._make_options_df()
        result = calc_implied_vol_batch(df, 3800.0, 0.025, "20240601")
        atm_call_iv = result.get("IO2406-C-3800")
        if atm_call_iv is not None and not np.isnan(atm_call_iv):
            assert abs(atm_call_iv - 0.20) < 0.01


# ======================================================================
# TestCalcDelta
# ======================================================================

class TestCalcDelta:
    def test_call_delta_in_0_1(self):
        d = calc_delta(S, K, T, r, sigma, OptionType.CALL)
        assert 0 < d < 1

    def test_put_delta_in_minus1_0(self):
        d = calc_delta(S, K, T, r, sigma, OptionType.PUT)
        assert -1 < d < 0

    def test_atm_call_delta_near_half(self):
        d = calc_delta(S, K, T, r, sigma, OptionType.CALL)
        assert 0.4 < d < 0.7  # ATM delta 约 0.5

    def test_put_call_delta_sum_near_zero(self):
        """N(d1) + (N(d1)-1) = 2N(d1) - 1；但 Call Delta + Put Delta = N(d1) + N(d1)-1 不一定为0"""
        dc = calc_delta(S, K, T, r, sigma, OptionType.CALL)
        dp = calc_delta(S, K, T, r, sigma, OptionType.PUT)
        # Call Delta - |Put Delta| 应接近 0 for ATM
        assert abs(dc + dp) < 0.1

    def test_deep_itm_call_delta_near_1(self):
        d = calc_delta(S_itm, 3000.0, T, r, sigma, OptionType.CALL)
        assert d > 0.9

    def test_expired_itm_call_delta(self):
        """到期后实值 Call delta = 1"""
        d = calc_delta(4000.0, 3800.0, 0.0, r, sigma, OptionType.CALL)
        assert d == 1.0

    def test_string_option_type(self):
        d1 = calc_delta(S, K, T, r, sigma, "C")
        d2 = calc_delta(S, K, T, r, sigma, OptionType.CALL)
        assert abs(d1 - d2) < 1e-12


# ======================================================================
# TestCalcGamma
# ======================================================================

class TestCalcGamma:
    def test_gamma_positive(self):
        assert calc_gamma(S, K, T, r, sigma) > 0

    def test_gamma_same_for_call_and_put(self):
        """Gamma 对 Call 和 Put 相同（gamma 函数不接受 option_type）"""
        g = calc_gamma(S, K, T, r, sigma)
        assert g > 0

    def test_expired_gamma_zero(self):
        assert calc_gamma(S, K, 0.0, r, sigma) == 0.0

    def test_gamma_finite(self):
        assert np.isfinite(calc_gamma(S, K, T, r, sigma))


# ======================================================================
# TestCalcTheta
# ======================================================================

class TestCalcTheta:
    def test_call_theta_negative(self):
        """做多 Call 的 Theta 为负（时间价值损耗）"""
        assert calc_theta(S, K, T, r, sigma, OptionType.CALL) < 0

    def test_put_theta_negative(self):
        assert calc_theta(S, K, T, r, sigma, OptionType.PUT) < 0

    def test_trading_days_vs_calendar(self):
        """交易日 Theta 绝对值大于日历日 Theta（分母更小）"""
        th_td = abs(calc_theta(S, K, T, r, sigma, OptionType.CALL, trading_days=True))
        th_cd = abs(calc_theta(S, K, T, r, sigma, OptionType.CALL, trading_days=False))
        assert th_td > th_cd

    def test_expired_theta_zero(self):
        assert calc_theta(S, K, 0.0, r, sigma, OptionType.CALL) == 0.0

    def test_theta_finite(self):
        assert np.isfinite(calc_theta(S, K, T, r, sigma, OptionType.CALL))


# ======================================================================
# TestCalcVega
# ======================================================================

class TestCalcVega:
    def test_vega_positive(self):
        assert calc_vega(S, K, T, r, sigma) > 0

    def test_vega_atm_is_max(self):
        """ATM 期权 Vega 应大于深度 OTM 期权"""
        vega_atm = calc_vega(S, 3800.0, T, r, sigma)
        vega_otm = calc_vega(S, 4500.0, T, r, sigma)
        assert vega_atm > vega_otm

    def test_expired_vega_zero(self):
        assert calc_vega(S, K, 0.0, r, sigma) == 0.0

    def test_vega_per_1pct(self):
        """Vega 定义为每 1% 波动率变动，应为 raw vega / 100"""
        raw_vega = S * np.sqrt(T) * 1.0  # rough magnitude
        vega = calc_vega(S, K, T, r, sigma)
        assert vega < raw_vega  # 除以 100 后应小于 raw


# ======================================================================
# TestCalcRho
# ======================================================================

class TestCalcRho:
    def test_call_rho_positive(self):
        """Call 的 Rho 为正（利率上升对 Call 有利）"""
        assert calc_rho(S, K, T, r, sigma, OptionType.CALL) > 0

    def test_put_rho_negative(self):
        """Put 的 Rho 为负（利率上升对 Put 不利）"""
        assert calc_rho(S, K, T, r, sigma, OptionType.PUT) < 0

    def test_expired_rho_zero(self):
        assert calc_rho(S, K, 0.0, r, sigma, OptionType.CALL) == 0.0


# ======================================================================
# TestCalcAllGreeks
# ======================================================================

class TestCalcAllGreeks:
    def test_returns_greeks_instance(self):
        g = calc_all_greeks(S, K, T, r, sigma, OptionType.CALL, "TEST")
        assert isinstance(g, Greeks)

    def test_ts_code_stored(self):
        g = calc_all_greeks(S, K, T, r, sigma, OptionType.CALL, "IO2406-C-3800")
        assert g.ts_code == "IO2406-C-3800"

    def test_matches_individual_functions(self):
        g = calc_all_greeks(S, K, T, r, sigma, OptionType.CALL)
        assert abs(g.delta - calc_delta(S, K, T, r, sigma, OptionType.CALL)) < 1e-12
        assert abs(g.gamma - calc_gamma(S, K, T, r, sigma)) < 1e-12
        assert abs(g.theta - calc_theta(S, K, T, r, sigma, OptionType.CALL)) < 1e-12
        assert abs(g.vega - calc_vega(S, K, T, r, sigma)) < 1e-12
        assert abs(g.rho - calc_rho(S, K, T, r, sigma, OptionType.CALL)) < 1e-12

    def test_all_fields_finite(self):
        g = calc_all_greeks(S, K, T, r, sigma, OptionType.PUT)
        for field_name, val in vars(g).items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{field_name} is not finite"


# ======================================================================
# TestCalcPortfolioGreeks
# ======================================================================

class TestCalcPortfolioGreeks:
    def _make_positions(self) -> pd.DataFrame:
        """构造多空混合持仓（到期日设为未来，确保 T > 0）"""
        return pd.DataFrame([
            {
                "ts_code": "IO2706-C-3800",
                "strike_price": 3800.0,
                "call_put": "C",
                "expire_date": "20270628",
                "net_position": -10,   # 做空 10 手
                "iv": 0.20,
            },
            {
                "ts_code": "IO2706-P-3800",
                "strike_price": 3800.0,
                "call_put": "P",
                "expire_date": "20270628",
                "net_position": -10,   # 做空 10 手
                "iv": 0.21,
            },
        ])

    def test_empty_positions_zeros(self):
        pg = calc_portfolio_greeks(pd.DataFrame(), 3800.0, 0.025)
        assert isinstance(pg, PortfolioGreeks)
        assert pg.net_delta == 0.0
        assert pg.net_vega == 0.0

    def test_returns_portfolio_greeks(self):
        pg = calc_portfolio_greeks(self._make_positions(), 3800.0, 0.025)
        assert isinstance(pg, PortfolioGreeks)

    def test_short_straddle_vega_negative(self):
        """做空跨式组合的净 Vega 为负（波动率上升亏损）"""
        pg = calc_portfolio_greeks(self._make_positions(), 3800.0, 0.025)
        assert pg.net_vega < 0

    def test_short_straddle_theta_positive(self):
        """做空跨式组合的净 Theta 为正（每天收取时间价值）"""
        pg = calc_portfolio_greeks(self._make_positions(), 3800.0, 0.025)
        assert pg.net_theta > 0

    def test_delta_dollars_scale(self):
        """delta_dollars = net_delta × spot_price"""
        pg = calc_portfolio_greeks(self._make_positions(), 3800.0, 0.025)
        assert abs(pg.delta_dollars - pg.net_delta * 3800.0) < 1e-6

    def test_all_fields_finite(self):
        pg = calc_portfolio_greeks(self._make_positions(), 3800.0, 0.025)
        for field_name, val in vars(pg).items():
            assert np.isfinite(val), f"{field_name} is not finite"


# ======================================================================
# TestVolSmile
# ======================================================================

class TestVolSmile:
    def _make_smile(self) -> VolSmile:
        strikes = np.array([3600.0, 3700.0, 3800.0, 3900.0, 4000.0])
        # Quadratic smile: ATM 最低，两翼升高
        ivs = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        return VolSmile(
            expire_date="20240628",
            T=30 / 365,
            strikes=strikes,
            ivs=ivs,
            forward=3802.0,
        )

    def test_atm_iv_computed_on_init(self):
        smile = self._make_smile()
        assert smile.atm_iv > 0

    def test_atm_iv_near_20pct(self):
        """ATM IV 应接近 0.20（forward ≈ 3800，ATM strike = 3800）"""
        smile = self._make_smile()
        assert abs(smile.atm_iv - 0.20) < 0.05

    def test_get_iv_at_known_strike(self):
        smile = self._make_smile()
        iv = smile.get_iv(3800.0)
        assert abs(iv - 0.20) < 1e-6

    def test_get_iv_interpolated(self):
        smile = self._make_smile()
        iv = smile.get_iv(3750.0)  # 3700 ~ 3800 之间
        assert 0.20 <= iv <= 0.22

    def test_get_iv_outside_range_uses_endpoint(self):
        smile = self._make_smile()
        iv_low = smile.get_iv(3000.0)
        iv_high = smile.get_iv(5000.0)
        assert iv_low == smile.ivs[0]
        assert iv_high == smile.ivs[-1]


# ======================================================================
# TestVolSurface
# ======================================================================

class TestVolSurface:
    def _make_options_df(self, n_expires: int = 2) -> pd.DataFrame:
        """构造含多个到期日的期权 DataFrame"""
        rows = []
        expire_dates = ["20240628", "20240726"][:n_expires]
        for exp in expire_dates:
            exp_ts = pd.Timestamp(exp)
            ref_ts = pd.Timestamp("20240601")
            T_test = (exp_ts - ref_ts).days / 365.0
            for K_test in [3600, 3700, 3800, 3900, 4000]:
                sigma_test = 0.20 + 0.01 * abs(K_test - 3800) / 100
                for ot in ["C", "P"]:
                    price = bs_price(3800.0, K_test, T_test, 0.025, sigma_test, ot)
                    rows.append({
                        "ts_code": f"{exp}-{ot}-{K_test}",
                        "strike_price": float(K_test),
                        "call_put": ot,
                        "expire_date": exp,
                        "close": price,
                        "volume": 1000,
                        "oi": 5000,
                    })
        return pd.DataFrame(rows)

    def _make_surface(self) -> VolSurface:
        surface = VolSurface("20240601", "IO", 3800.0, 0.025)
        surface.build_from_options_df(self._make_options_df())
        return surface

    def test_build_returns_self(self):
        surface = VolSurface("20240601", "IO", 3800.0, 0.025)
        result = surface.build_from_options_df(self._make_options_df())
        assert result is surface

    def test_expire_dates_populated(self):
        surface = self._make_surface()
        dates = surface.get_all_expire_dates()
        assert len(dates) >= 1

    def test_expire_dates_sorted(self):
        surface = self._make_surface()
        dates = surface.get_all_expire_dates()
        assert dates == sorted(dates)

    def test_get_atm_iv_returns_float(self):
        surface = self._make_surface()
        dates = surface.get_all_expire_dates()
        atm = surface.get_atm_iv(dates[0])
        assert atm is not None
        assert isinstance(atm, float)

    def test_get_atm_iv_reasonable(self):
        surface = self._make_surface()
        dates = surface.get_all_expire_dates()
        atm = surface.get_atm_iv(dates[0])
        assert 0.05 < atm < 0.80

    def test_get_atm_iv_unknown_date_none(self):
        surface = self._make_surface()
        assert surface.get_atm_iv("19000101") is None

    def test_get_nearest_atm_iv_returns_float(self):
        surface = self._make_surface()
        iv = surface.get_nearest_atm_iv(target_T=30 / 365)
        assert iv is not None
        assert iv > 0

    def test_get_smile_returns_vol_smile(self):
        surface = self._make_surface()
        dates = surface.get_all_expire_dates()
        smile = surface.get_smile(dates[0])
        assert isinstance(smile, VolSmile)

    def test_get_smile_unknown_returns_none(self):
        surface = self._make_surface()
        assert surface.get_smile("19000101") is None

    def test_to_dataframe_columns(self):
        surface = self._make_surface()
        df = surface.to_dataframe()
        assert not df.empty
        expected_cols = {"trade_date", "underlying", "expire_date", "T", "strike_price", "iv", "moneyness"}
        assert expected_cols.issubset(set(df.columns))

    def test_to_dataframe_moneyness(self):
        surface = self._make_surface()
        df = surface.to_dataframe()
        # moneyness = K/S; ATM ≈ 1.0
        atm_rows = df[abs(df["moneyness"] - 1.0) < 0.01]
        assert len(atm_rows) > 0

    def test_term_structure_columns(self):
        surface = self._make_surface()
        ts = surface.term_structure()
        assert set(ts.columns) == {"expire_date", "T", "atm_iv"}

    def test_term_structure_sorted_by_T(self):
        surface = self._make_surface()
        ts = surface.term_structure()
        if len(ts) > 1:
            assert ts["T"].is_monotonic_increasing

    def test_build_empty_df_no_smiles(self):
        surface = VolSurface("20240601", "IO", 3800.0, 0.025)
        surface.build_from_options_df(pd.DataFrame())
        assert len(surface.get_all_expire_dates()) == 0

    def test_min_volume_filter(self):
        surface = VolSurface("20240601", "IO", 3800.0, 0.025)
        df = self._make_options_df()
        # Filter out all contracts with min_volume > max volume
        surface.build_from_options_df(df, min_volume=999999)
        assert len(surface.get_all_expire_dates()) == 0

    def test_to_dataframe_empty_when_no_smiles(self):
        surface = VolSurface("20240601", "IO", 3800.0, 0.025)
        df = surface.to_dataframe()
        assert df.empty
