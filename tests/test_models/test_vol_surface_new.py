"""
test_vol_surface_new.py
-----------------------
测试 VolSurface 新增方法：
  build_surface / get_skew / get_term_structure /
  get_risk_reversal / get_butterfly

所有合成数据测试均基于 BlackScholes 生成期权价格，
再通过 ImpliedVolCalculator 反推 IV，最后传入 VolSurface。

TestBuildSurface      : build_surface 枢轴表结构
TestGetSkew           : get_skew 微笑提取
TestGetTermStructure  : get_term_structure 期限结构
TestGetRiskReversal   : get_risk_reversal 风险逆转
TestGetButterfly      : get_butterfly 蝶式
TestRealDataSurface   : 真实数据库集成（无数据时 skip）
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from models.pricing.black_scholes import BlackScholes
from models.pricing.implied_vol import ImpliedVolCalculator
from models.pricing.vol_surface import VolSurface


# ======================================================================
# 测试固件 & 辅助函数
# ======================================================================

S = 4000.0
R = 0.02
TRADE_DATE = "20260101"


def _expire(days: int) -> str:
    ts = pd.Timestamp(TRADE_DATE) + pd.Timedelta(days=days)
    return ts.strftime("%Y%m%d")


def _build_chain(
    S: float,
    strikes: list[float],
    T: float,
    sigma: float,
    trade_date: str,
    expire_date: str,
    q: float = 0.0,
) -> pd.DataFrame:
    """用 BS 公式生成认购+认沽期权链（含成交量）。"""
    rows = []
    for K in strikes:
        for cp in ("C", "P"):
            price = BlackScholes.price(S, K, T, R, sigma, q, cp)
            rows.append({
                "exercise_price": float(K),
                "call_put": cp,
                "expire_date": expire_date,
                "close": price,
                "volume": 200,
            })
    return pd.DataFrame(rows)


def _calc_iv(chain: pd.DataFrame, trade_date: str = TRADE_DATE) -> pd.DataFrame:
    calc = ImpliedVolCalculator(risk_free_rate=R)
    return calc.calculate_iv_for_chain(chain, S, trade_date)


def _surface() -> VolSurface:
    return VolSurface(TRADE_DATE, "IO", S, R)


def _multi_expire_iv(
    strikes: list[float],
    sigmas_by_days: dict[int, float],
) -> pd.DataFrame:
    """生成多到期日的 options_with_iv DataFrame。"""
    calc = ImpliedVolCalculator(risk_free_rate=R)
    pieces = []
    for days, sigma in sigmas_by_days.items():
        T = days / 365.0
        expire = _expire(days)
        chain = _build_chain(S, strikes, T, sigma, TRADE_DATE, expire)
        piece = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


# 典型行权价（围绕 ATM 4000）
STRIKES = [3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0, 4300.0, 4400.0]
# 单到期日（30 天）
EXPIRE_30 = _expire(30)
CHAIN_30 = _build_chain(S, STRIKES, 30 / 365, 0.20, TRADE_DATE, EXPIRE_30)
IV_30 = _calc_iv(CHAIN_30)

# 多到期日（近月 30 天 sigma=0.18，次月 60 天 sigma=0.20，季度 90 天 sigma=0.22）
MULTI_IV = _multi_expire_iv(
    STRIKES,
    {30: 0.18, 60: 0.20, 90: 0.22},
)


# ======================================================================
# TestBuildSurface
# ======================================================================

class TestBuildSurface:

    def test_returns_dataframe(self):
        vs = _surface()
        result = vs.build_surface(IV_30)
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_expire_dates(self):
        vs = _surface()
        result = vs.build_surface(IV_30)
        assert EXPIRE_30 in result.columns

    def test_index_is_moneyness(self):
        """index 应为 moneyness 值（浮点，围绕 1.0）。"""
        vs = _surface()
        result = vs.build_surface(IV_30)
        assert result.index.name == "moneyness"
        assert result.index.dtype in (float, "float64")

    def test_moneyness_range(self):
        """ATM 周围行权价对应 moneyness 应在 0.8–1.25 之间。"""
        vs = _surface()
        result = vs.build_surface(IV_30)
        assert result.index.min() >= 0.7
        assert result.index.max() <= 1.5

    def test_iv_values_positive(self):
        vs = _surface()
        result = vs.build_surface(IV_30)
        assert (result.dropna() > 0).all().all()

    def test_iv_values_in_range(self):
        """IV 应在 1%–100% 之间（合成 sigma=20%）。"""
        vs = _surface()
        result = vs.build_surface(IV_30)
        assert (result.dropna() > 0.01).all().all()
        assert (result.dropna() < 1.0).all().all()

    def test_multi_expire_columns(self):
        """多到期日时每个到期日对应一列。"""
        vs = _surface()
        result = vs.build_surface(MULTI_IV)
        expire_dates = {_expire(d) for d in (30, 60, 90)}
        for exp in expire_dates:
            assert exp in result.columns

    def test_empty_input_returns_empty(self):
        vs = _surface()
        empty = pd.DataFrame(columns=["moneyness", "expire_date", "iv", "is_valid"])
        result = vs.build_surface(empty)
        assert result.empty

    def test_all_invalid_returns_empty(self):
        vs = _surface()
        df = IV_30.copy()
        df["is_valid"] = False
        result = vs.build_surface(df)
        assert result.empty

    def test_index_is_sorted(self):
        vs = _surface()
        result = vs.build_surface(IV_30)
        assert list(result.index) == sorted(result.index)


# ======================================================================
# TestGetSkew
# ======================================================================

class TestGetSkew:

    def test_returns_dataframe(self):
        vs = _surface()
        result = vs.get_skew(IV_30, EXPIRE_30)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        vs = _surface()
        result = vs.get_skew(IV_30, EXPIRE_30)
        for col in ("exercise_price", "moneyness", "iv_call", "iv_put"):
            assert col in result.columns

    def test_sorted_by_moneyness(self):
        vs = _surface()
        result = vs.get_skew(IV_30, EXPIRE_30)
        assert list(result["moneyness"]) == sorted(result["moneyness"])

    def test_iv_call_put_both_present(self):
        """所有行权价都有认购和认沽。"""
        vs = _surface()
        result = vs.get_skew(IV_30, EXPIRE_30)
        assert result["iv_call"].notna().all()
        assert result["iv_put"].notna().all()

    def test_call_put_parity_approx(self):
        """认购和认沽的 IV 反推值应非常接近（同一 σ 生成）。"""
        vs = _surface()
        result = vs.get_skew(IV_30, EXPIRE_30)
        diff = (result["iv_call"] - result["iv_put"]).abs()
        assert (diff < 0.005).all(), f"Call/Put IV 差异过大: {diff.max():.4f}"

    def test_smile_structure(self):
        """微笑结构：OTM 两侧 IV 高于 ATM（使用手工构造微笑数据）。"""
        calc = ImpliedVolCalculator(risk_free_rate=R)
        sigma_by_K = {3700: 0.24, 3800: 0.22, 3900: 0.21, 4000: 0.20,
                      4100: 0.21, 4200: 0.22, 4300: 0.24}
        rows = []
        for K, sig in sigma_by_K.items():
            for cp in ("C", "P"):
                price = BlackScholes.price(S, float(K), 30 / 365, R, sig, 0.0, cp)
                rows.append({"exercise_price": float(K), "call_put": cp,
                              "expire_date": EXPIRE_30, "close": price, "volume": 100})
        chain = pd.DataFrame(rows)
        iv_df = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        vs = _surface()
        skew = vs.get_skew(iv_df, EXPIRE_30)
        atm_iv = skew[skew["exercise_price"] == 4000.0]["iv_call"].iloc[0]
        otm_ivs = skew[skew["exercise_price"].isin([3700.0, 4300.0])]["iv_call"]
        assert otm_ivs.mean() > atm_iv, "OTM IV 应高于 ATM IV"

    def test_wrong_expire_returns_empty(self):
        vs = _surface()
        result = vs.get_skew(IV_30, "99991231")
        assert result.empty

    def test_all_invalid_returns_empty(self):
        vs = _surface()
        df = IV_30.copy()
        df["is_valid"] = False
        result = vs.get_skew(df, EXPIRE_30)
        assert result.empty

    def test_row_count_equals_strikes(self):
        """每个行权价一行（合并认购认沽）。"""
        vs = _surface()
        result = vs.get_skew(IV_30, EXPIRE_30)
        assert len(result) == len(STRIKES)


# ======================================================================
# TestGetTermStructure
# ======================================================================

class TestGetTermStructure:

    def test_returns_dataframe(self):
        vs = _surface()
        result = vs.get_term_structure(MULTI_IV, S)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        vs = _surface()
        result = vs.get_term_structure(MULTI_IV, S)
        for col in ("expire_date", "T", "atm_iv"):
            assert col in result.columns

    def test_row_count_equals_expire_dates(self):
        vs = _surface()
        result = vs.get_term_structure(MULTI_IV, S)
        assert len(result) == 3

    def test_sorted_by_T(self):
        vs = _surface()
        result = vs.get_term_structure(MULTI_IV, S)
        assert list(result["T"]) == sorted(result["T"])

    def test_atm_iv_close_to_true_sigma(self):
        """ATM IV 应接近生成时的 sigma（误差 < 1e-3）。"""
        vs = _surface()
        result = vs.get_term_structure(MULTI_IV, S)
        expected = {_expire(30): 0.18, _expire(60): 0.20, _expire(90): 0.22}
        for _, row in result.iterrows():
            exp_sigma = expected[row["expire_date"]]
            assert abs(row["atm_iv"] - exp_sigma) < 1e-3, (
                f"{row['expire_date']}: atm_iv={row['atm_iv']:.4f} ≠ {exp_sigma}"
            )

    def test_upward_term_structure(self):
        """近月 sigma < 中月 < 远月 → 期限结构向上。"""
        vs = _surface()
        result = vs.get_term_structure(MULTI_IV, S)
        ivs = result["atm_iv"].values
        assert ivs[0] < ivs[1] < ivs[2], "期限结构应单调递增"

    def test_T_values_approx(self):
        """T 值应接近 30/365, 60/365, 90/365。"""
        vs = _surface()
        result = vs.get_term_structure(MULTI_IV, S)
        expected_Ts = [30 / 365, 60 / 365, 90 / 365]
        for row_T, expected_T in zip(sorted(result["T"]), expected_Ts):
            assert abs(row_T - expected_T) < 2 / 365

    def test_empty_input_returns_empty(self):
        vs = _surface()
        empty = pd.DataFrame(columns=["exercise_price", "expire_date", "T", "iv", "is_valid"])
        result = vs.get_term_structure(empty, S)
        assert result.empty

    def test_all_invalid_returns_empty(self):
        vs = _surface()
        df = MULTI_IV.copy()
        df["is_valid"] = False
        result = vs.get_term_structure(df, S)
        assert result.empty


# ======================================================================
# TestGetRiskReversal
# ======================================================================

class TestGetRiskReversal:

    def _smile_iv(self, sigma_by_K: dict[float, float]) -> pd.DataFrame:
        """从行权价-sigma 映射构建 options_with_iv。"""
        calc = ImpliedVolCalculator(risk_free_rate=R)
        rows = []
        for K, sig in sigma_by_K.items():
            for cp in ("C", "P"):
                price = BlackScholes.price(S, K, 30 / 365, R, sig, 0.0, cp)
                rows.append({"exercise_price": K, "call_put": cp,
                              "expire_date": EXPIRE_30, "close": price, "volume": 100})
        chain = pd.DataFrame(rows)
        return calc.calculate_iv_for_chain(chain, S, TRADE_DATE)

    def test_flat_smile_rr_near_zero(self):
        """平坦微笑（所有行权价 sigma 相同）时 RR ≈ 0。"""
        sigma_map = {k: 0.20 for k in STRIKES}
        iv_df = self._smile_iv(sigma_map)
        vs = _surface()
        rr = vs.get_risk_reversal(iv_df, EXPIRE_30, delta_target=0.25)
        assert not math.isnan(rr)
        assert abs(rr) < 0.02, f"平坦微笑 RR={rr:.4f} 应接近 0"

    def test_negative_skew_rr_negative(self):
        """负偏斜（put 更贵）时 RR < 0（A 股典型特征）。"""
        # OTM put（低行权价）IV 更高
        sigma_map = {3700: 0.26, 3800: 0.23, 3900: 0.21,
                     4000: 0.20, 4100: 0.20, 4200: 0.20, 4300: 0.20}
        iv_df = self._smile_iv(sigma_map)
        vs = _surface()
        rr = vs.get_risk_reversal(iv_df, EXPIRE_30, delta_target=0.25)
        assert not math.isnan(rr)
        assert rr < 0, f"负偏斜时 RR 应 < 0，实际 RR={rr:.4f}"

    def test_positive_skew_rr_positive(self):
        """正偏斜（call 更贵）时 RR > 0。"""
        sigma_map = {3700: 0.20, 3800: 0.20, 3900: 0.20,
                     4000: 0.20, 4100: 0.21, 4200: 0.23, 4300: 0.26}
        iv_df = self._smile_iv(sigma_map)
        vs = _surface()
        rr = vs.get_risk_reversal(iv_df, EXPIRE_30, delta_target=0.25)
        assert not math.isnan(rr)
        assert rr > 0, f"正偏斜时 RR 应 > 0，实际 RR={rr:.4f}"

    def test_wrong_expire_returns_nan(self):
        vs = _surface()
        rr = vs.get_risk_reversal(IV_30, "99991231", delta_target=0.25)
        assert math.isnan(rr)

    def test_no_delta_column_returns_nan(self):
        vs = _surface()
        df = IV_30.copy()
        df["delta"] = float("nan")
        rr = vs.get_risk_reversal(df, EXPIRE_30, delta_target=0.25)
        assert math.isnan(rr)

    def test_custom_delta_target(self):
        """10-delta RR 也能计算（绝对值应比 25-delta RR 更大）。"""
        sigma_map = {3700: 0.28, 3800: 0.24, 3900: 0.22,
                     4000: 0.20, 4100: 0.20, 4200: 0.20, 4300: 0.20}
        iv_df = self._smile_iv(sigma_map)
        vs = _surface()
        rr_25 = vs.get_risk_reversal(iv_df, EXPIRE_30, delta_target=0.25)
        rr_10 = vs.get_risk_reversal(iv_df, EXPIRE_30, delta_target=0.10)
        assert not math.isnan(rr_25) and not math.isnan(rr_10)
        # 10-delta 更接近极端行权价，skew 应更大（绝对值）
        assert abs(rr_10) >= abs(rr_25) - 0.01


# ======================================================================
# TestGetButterfly
# ======================================================================

class TestGetButterfly:

    def _smile_iv(self, sigma_by_K: dict[float, float]) -> pd.DataFrame:
        calc = ImpliedVolCalculator(risk_free_rate=R)
        rows = []
        for K, sig in sigma_by_K.items():
            for cp in ("C", "P"):
                price = BlackScholes.price(S, K, 30 / 365, R, sig, 0.0, cp)
                rows.append({"exercise_price": K, "call_put": cp,
                              "expire_date": EXPIRE_30, "close": price, "volume": 100})
        return calc.calculate_iv_for_chain(pd.DataFrame(rows), S, TRADE_DATE)

    def test_smile_bf_positive(self):
        """微笑（两端高中间低）时 BF > 0。"""
        sigma_map = {3700: 0.25, 3800: 0.23, 3900: 0.21,
                     4000: 0.20, 4100: 0.21, 4200: 0.23, 4300: 0.25}
        iv_df = self._smile_iv(sigma_map)
        vs = _surface()
        bf = vs.get_butterfly(iv_df, EXPIRE_30, delta_target=0.25)
        assert not math.isnan(bf)
        assert bf > 0, f"微笑时 BF 应 > 0，实际 BF={bf:.4f}"

    def test_flat_smile_bf_near_zero(self):
        """平坦微笑时 BF ≈ 0。"""
        sigma_map = {k: 0.20 for k in STRIKES}
        iv_df = self._smile_iv(sigma_map)
        vs = _surface()
        bf = vs.get_butterfly(iv_df, EXPIRE_30, delta_target=0.25)
        assert not math.isnan(bf)
        assert abs(bf) < 0.01, f"平坦微笑 BF={bf:.4f} 应接近 0"

    def test_inverted_smile_bf_negative(self):
        """倒置微笑（中间高两端低）时 BF < 0。"""
        sigma_map = {3700: 0.18, 3800: 0.19, 3900: 0.20,
                     4000: 0.22, 4100: 0.20, 4200: 0.19, 4300: 0.18}
        iv_df = self._smile_iv(sigma_map)
        vs = _surface()
        bf = vs.get_butterfly(iv_df, EXPIRE_30, delta_target=0.25)
        assert not math.isnan(bf)
        assert bf < 0, f"倒置微笑时 BF 应 < 0，实际 BF={bf:.4f}"

    def test_wrong_expire_returns_nan(self):
        vs = _surface()
        bf = vs.get_butterfly(IV_30, "99991231")
        assert math.isnan(bf)

    def test_bf_less_than_wing_iv(self):
        """BF 的量级应远小于 wing IV（BF 是差值的一半）。"""
        sigma_map = {3700: 0.25, 3800: 0.23, 3900: 0.21,
                     4000: 0.20, 4100: 0.21, 4200: 0.23, 4300: 0.25}
        iv_df = self._smile_iv(sigma_map)
        vs = _surface()
        bf = vs.get_butterfly(iv_df, EXPIRE_30, delta_target=0.25)
        assert abs(bf) < 0.15, f"BF={bf:.4f} 量级应合理（< 15%）"


# ======================================================================
# TestRealDataSurface（真实数据库集成）
# ======================================================================

def _get_real_db_and_date():
    """尝试打开真实数据库，返回 (db, latest_date) 或 (None, None)。"""
    try:
        import os
        db_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "market_data.db")
        )
        if not os.path.exists(db_path):
            return None, None
        from data.storage.db_manager import DBManager
        db = DBManager(db_path)
        latest = db.get_latest_date("options_daily", "%IO%")
        if not latest:
            db.close()
            return None, None
        return db, latest
    except Exception:
        return None, None


@pytest.fixture(scope="module")
def real_surface_data():
    db, trade_date = _get_real_db_and_date()
    if db is None:
        pytest.skip("数据库无 IO 期权数据，跳过真实数据测试")

    try:
        chain = db.get_options_chain("IO", trade_date)
        underlying = _get_futures_price(db, trade_date)
        if chain.empty or underlying is None:
            db.close()
            pytest.skip("无有效期权链或期货价格")
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        iv_df = calc.calculate_iv_for_chain(chain, underlying, trade_date)
        vs = VolSurface(trade_date, "IO", underlying, 0.02)
        yield vs, iv_df, underlying, trade_date
    finally:
        db.close()


def _get_futures_price(db, trade_date: str):
    try:
        df = db.get_futures_daily("%IF%", trade_date, trade_date)
        if df.empty:
            return None
        if "oi" in df.columns:
            df = df.sort_values("oi", ascending=False)
        return float(df.iloc[0]["close"])
    except Exception:
        return None


class TestRealDataSurface:

    def test_build_surface_not_empty(self, real_surface_data):
        vs, iv_df, underlying, trade_date = real_surface_data
        result = vs.build_surface(iv_df)
        assert not result.empty, "真实数据构建曲面不应为空"

    def test_skew_not_empty(self, real_surface_data):
        vs, iv_df, underlying, trade_date = real_surface_data
        expire_dates = iv_df["expire_date"].unique()
        assert len(expire_dates) > 0
        # 取第一个到期日
        skew = vs.get_skew(iv_df, str(expire_dates[0]))
        assert not skew.empty

    def test_term_structure_upward_or_flat(self, real_surface_data):
        vs, iv_df, underlying, trade_date = real_surface_data
        ts = vs.get_term_structure(iv_df, underlying)
        assert not ts.empty
        print(f"\n=== 期限结构 ({trade_date}) ===")
        print(ts.to_string(index=False))
        # 至少有一个有效 ATM IV
        assert ts["atm_iv"].notna().any()

    def test_print_skew(self, real_surface_data):
        vs, iv_df, underlying, trade_date = real_surface_data
        expire_dates = sorted(iv_df["expire_date"].unique())
        # 打印近月 skew
        skew = vs.get_skew(iv_df, expire_dates[0])
        print(f"\n=== IV 微笑 ({trade_date}, expire={expire_dates[0]}, S={underlying:.0f}) ===")
        print(f"{'行权价':>10} {'Moneyness':>10} {'IV(C)':>8} {'IV(P)':>8}")
        for _, row in skew.iterrows():
            c_str = f"{row['iv_call']:.4f}" if not pd.isna(row['iv_call']) else "  ---"
            p_str = f"{row['iv_put']:.4f}" if not pd.isna(row['iv_put']) else "  ---"
            print(f"{row['exercise_price']:>10.0f} {row['moneyness']:>10.4f} {c_str:>8} {p_str:>8}")

    def test_rr_and_bf_finite(self, real_surface_data):
        vs, iv_df, underlying, trade_date = real_surface_data
        expire_dates = sorted(iv_df["expire_date"].unique())
        rr = vs.get_risk_reversal(iv_df, expire_dates[0])
        bf = vs.get_butterfly(iv_df, expire_dates[0])
        print(f"\n[真实数据] {trade_date} 近月: RR={rr:.4f}, BF={bf:.4f}")
        # A 股通常 RR < 0（put skew），BF > 0（smile）
        assert not (math.isnan(rr) and math.isnan(bf)), "RR 和 BF 不能同时为 NaN"

    def test_atm_iv_reasonable(self, real_surface_data):
        vs, iv_df, underlying, trade_date = real_surface_data
        ts = vs.get_term_structure(iv_df, underlying)
        for _, row in ts.iterrows():
            assert 0.05 <= row["atm_iv"] <= 0.80, (
                f"{row['expire_date']} ATM IV={row['atm_iv']:.4f} 不合理"
            )
