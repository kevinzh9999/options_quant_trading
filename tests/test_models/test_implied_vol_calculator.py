"""
test_implied_vol_calculator.py
------------------------------
测试 ImpliedVolCalculator 类的所有功能。

结构：
- TestCalculateIvForChain   : calculate_iv_for_chain 核心逻辑
- TestGetAtmIv              : get_atm_iv 提取
- TestBuildIvHistory        : build_iv_history（使用 mock DBManager）
- TestRealDataIntegration   : 真实数据测试（数据库有数据时运行，否则 skip）
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pandas as pd
import pytest

from models.pricing.black_scholes import BlackScholes
from models.pricing.implied_vol import ImpliedVolCalculator


# ======================================================================
# 测试固件
# ======================================================================

# 典型 A 股股指期权参数
S = 4000.0
r = 0.02
q = 0.0
sigma_true = 0.20
TRADE_DATE = "20260101"


def _make_chain(
    S: float,
    strikes: list[float],
    T: float,
    r: float,
    sigma: float,
    _trade_date: str,
    expire_date: str,
    include_volume: bool = True,
    q: float = 0.0,
) -> pd.DataFrame:
    """生成合成期权链（认购+认沽，每个行权价各一条）。"""
    rows = []
    for K in strikes:
        for cp in ("C", "P"):
            price = BlackScholes.price(S, K, T, r, sigma, q, cp)
            row = {
                "exercise_price": K,
                "call_put": cp,
                "expire_date": expire_date,
                "close": price,
            }
            if include_volume:
                row["volume"] = 100
            rows.append(row)
    return pd.DataFrame(rows)


def _expire_date_for_T(trade_date: str, T_years: float) -> str:
    """由 trade_date 和 T（年化）推算 expire_date 字符串。"""
    days = int(round(T_years * 365))
    ts = pd.Timestamp(trade_date) + pd.Timedelta(days=days)
    return ts.strftime("%Y%m%d")


# ======================================================================
# TestCalculateIvForChain
# ======================================================================

class TestCalculateIvForChain:
    """calculate_iv_for_chain 的核心逻辑测试。"""

    T = 30 / 365
    expire = _expire_date_for_T(TRADE_DATE, T)
    strikes = [3800.0, 3900.0, 4000.0, 4100.0, 4200.0]

    def _calc(self, **kwargs) -> pd.DataFrame:
        calc = ImpliedVolCalculator(risk_free_rate=r, dividend_yield=q)
        chain = _make_chain(S, self.strikes, self.T, r, sigma_true, TRADE_DATE, self.expire, **kwargs)
        return calc.calculate_iv_for_chain(chain, S, TRADE_DATE)

    def test_returns_dataframe(self):
        result = self._calc()
        assert isinstance(result, pd.DataFrame)

    def test_new_columns_present(self):
        result = self._calc()
        for col in ("iv", "T", "moneyness", "delta", "is_valid"):
            assert col in result.columns, f"缺少列: {col}"

    def test_row_count_unchanged(self):
        result = self._calc()
        assert len(result) == len(self.strikes) * 2

    def test_iv_roundtrip_atm(self):
        """ATM 合约 IV 反推误差 < 1e-4。"""
        result = self._calc()
        atm_calls = result[(result["exercise_price"] == 4000.0) & (result["call_put"] == "C")]
        assert len(atm_calls) == 1
        iv = atm_calls.iloc[0]["iv"]
        assert abs(iv - sigma_true) < 1e-4, f"ATM Call IV={iv:.6f} ≠ {sigma_true}"

    def test_iv_roundtrip_all_strikes(self):
        """所有合约 IV 反推误差均 < 1e-3。"""
        result = self._calc()
        valid = result[result["is_valid"]]
        for _, row in valid.iterrows():
            assert abs(row["iv"] - sigma_true) < 1e-3, (
                f"K={row['exercise_price']} {row['call_put']} IV={row['iv']:.6f}"
            )

    def test_T_column_positive(self):
        result = self._calc()
        assert (result["T"] > 0).all(), "T 应全部 > 0"

    def test_T_column_approx(self):
        result = self._calc()
        # T 应在 30/365 附近（±2天）
        expected_T = self.T
        assert (abs(result["T"] - expected_T) < 3 / 365).all()

    def test_moneyness_call_otm(self):
        """OTM 认购（K > S）：moneyness = S/K < 1。"""
        result = self._calc()
        otm_call = result[(result["exercise_price"] == 4200.0) & (result["call_put"] == "C")]
        assert otm_call.iloc[0]["moneyness"] < 1.0

    def test_moneyness_put_otm(self):
        """OTM 认沽（K < S）：moneyness = K/S < 1。"""
        result = self._calc()
        otm_put = result[(result["exercise_price"] == 3800.0) & (result["call_put"] == "P")]
        assert otm_put.iloc[0]["moneyness"] < 1.0

    def test_moneyness_atm_near_one(self):
        """ATM 合约 moneyness ≈ 1。"""
        result = self._calc()
        atm = result[result["exercise_price"] == 4000.0]
        assert (abs(atm["moneyness"] - 1.0) < 0.01).all()

    def test_delta_call_range(self):
        """认购 Delta ∈ (0, 1)。"""
        result = self._calc()
        calls = result[(result["call_put"] == "C") & result["is_valid"]]
        assert ((calls["delta"] > 0) & (calls["delta"] < 1)).all()

    def test_delta_put_range(self):
        """认沽 Delta ∈ (-1, 0)。"""
        result = self._calc()
        puts = result[(result["call_put"] == "P") & result["is_valid"]]
        assert ((puts["delta"] > -1) & (puts["delta"] < 0)).all()

    def test_delta_atm_call_near_half(self):
        """ATM 认购 Delta ≈ 0.5。"""
        result = self._calc()
        atm_call = result[(result["exercise_price"] == 4000.0) & (result["call_put"] == "C")]
        assert abs(atm_call.iloc[0]["delta"] - 0.5) < 0.05

    def test_is_valid_all_true_normal(self):
        """正常合约（T≥7天，成交量>0）should all be valid。"""
        result = self._calc()
        assert result["is_valid"].all(), "正常合约应全部 is_valid=True"

    def test_is_valid_false_when_volume_zero(self):
        """成交量为0的合约 is_valid=False。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        chain = _make_chain(S, self.strikes, self.T, r, sigma_true, TRADE_DATE, self.expire)
        chain["volume"] = 0
        result = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        assert not result["is_valid"].any()

    def test_is_valid_false_when_close_zero(self):
        """close=0 的合约 is_valid=False（深度虚值价格为0）。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        chain = _make_chain(S, [4000.0], self.T, r, sigma_true, TRADE_DATE, self.expire)
        chain["close"] = 0.0
        result = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        assert not result["is_valid"].any()

    def test_is_valid_false_when_T_less_than_7_days(self):
        """剩余 < 7 天的合约 is_valid=False。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        short_T = 3 / 365
        short_expire = _expire_date_for_T(TRADE_DATE, short_T)
        chain = _make_chain(S, [4000.0], short_T, r, sigma_true, TRADE_DATE, short_expire)
        result = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        assert not result["is_valid"].any()

    def test_empty_chain_returns_empty(self):
        """空 DataFrame 输入返回空 DataFrame（含 5 个新列）。"""
        calc = ImpliedVolCalculator()
        empty = pd.DataFrame(columns=["exercise_price", "call_put", "expire_date", "close"])
        result = calc.calculate_iv_for_chain(empty, S, TRADE_DATE)
        assert result.empty
        for col in ("iv", "T", "moneyness", "delta", "is_valid"):
            assert col in result.columns

    def test_missing_close_nan_iv(self):
        """close 为 NaN 的合约 IV 应为 NaN，is_valid=False。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        chain = _make_chain(S, [4000.0], self.T, r, sigma_true, TRADE_DATE, self.expire)
        chain["close"] = float("nan")
        result = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        assert result["iv"].isna().all()
        assert not result["is_valid"].any()

    def test_with_dividend_yield(self):
        """含分红率时 IV 反推仍然准确（误差 < 1e-3）。"""
        q_div = 0.03
        calc = ImpliedVolCalculator(risk_free_rate=r, dividend_yield=q_div)
        chain = _make_chain(S, [4000.0], self.T, r, sigma_true, TRADE_DATE, self.expire, q=q_div)
        result = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        assert abs(result.iloc[0]["iv"] - sigma_true) < 1e-3

    def test_iv_smile_structure(self):
        """虚值合约 IV 应高于平值（波动率微笑基本结构）。

        使用真实波动率微笑数据：OTM IV > ATM IV。
        这里用更高的 sigma 给 OTM 合约，验证结构。
        实际上：用同一 sigma 生成价格，IV 反推后所有合约 IV 相同（平坦微笑）。
        改用不同 sigma 生成每个合约价格来模拟微笑。
        """
        calc = ImpliedVolCalculator(risk_free_rate=r)
        # 手动构造有微笑的期权链：OTM sigma 更高
        T = self.T
        expire = self.expire
        rows = []
        sigma_by_strike = {3800: 0.22, 3900: 0.21, 4000: 0.20, 4100: 0.21, 4200: 0.22}
        for K, sig in sigma_by_strike.items():
            for cp in ("C", "P"):
                price = BlackScholes.price(S, K, T, r, sig, 0.0, cp)
                rows.append({"exercise_price": float(K), "call_put": cp,
                              "expire_date": expire, "close": price, "volume": 100})
        chain = pd.DataFrame(rows)
        result = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        valid = result[result["is_valid"]]

        atm_iv = valid[valid["exercise_price"] == 4000.0]["iv"].mean()
        otm_wing_iv = valid[valid["exercise_price"].isin([3800.0, 4200.0])]["iv"].mean()
        assert otm_wing_iv > atm_iv, "微笑结构：OTM IV 应 > ATM IV"


# ======================================================================
# TestGetAtmIv
# ======================================================================

class TestGetAtmIv:
    """get_atm_iv 提取测试。"""

    T = 30 / 365
    expire = _expire_date_for_T(TRADE_DATE, T)

    def _chain_with_iv(self, strikes=None, sigma=sigma_true) -> pd.DataFrame:
        strikes = strikes or [3800.0, 3900.0, 4000.0, 4100.0, 4200.0]
        calc = ImpliedVolCalculator(risk_free_rate=r)
        chain = _make_chain(S, strikes, self.T, r, sigma, TRADE_DATE, self.expire)
        return calc.calculate_iv_for_chain(chain, S, TRADE_DATE)

    def test_atm_iv_in_reasonable_range(self):
        """ATM IV 应在合理范围（10%-50%）。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        result = self._chain_with_iv()
        atm_iv = calc.get_atm_iv(result, S)
        assert 0.10 <= atm_iv <= 0.50, f"ATM IV={atm_iv:.4f} 不在合理范围"

    def test_atm_iv_close_to_true_sigma(self):
        """ATM IV 应接近真实 sigma（误差 < 1e-3）。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        result = self._chain_with_iv()
        atm_iv = calc.get_atm_iv(result, S)
        assert abs(atm_iv - sigma_true) < 1e-3, f"ATM IV={atm_iv:.6f} ≠ {sigma_true}"

    def test_atm_iv_averages_call_and_put(self):
        """ATM IV 是认购和认沽 IV 的均值。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        result = self._chain_with_iv(strikes=[4000.0])
        atm_iv = calc.get_atm_iv(result, S)
        call_iv = result[(result["exercise_price"] == 4000.0) & (result["call_put"] == "C")].iloc[0]["iv"]
        put_iv = result[(result["exercise_price"] == 4000.0) & (result["call_put"] == "P")].iloc[0]["iv"]
        expected = (call_iv + put_iv) / 2
        assert abs(atm_iv - expected) < 1e-8

    def test_atm_picks_nearest_strike(self):
        """标的价格 4050 时，ATM 应选行权价 4000（最近）。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        result = self._chain_with_iv()
        # 将 underlying_price 设为 4050，最近行权价是 4000
        atm_iv = calc.get_atm_iv(result, 4050.0)
        # 取 4000 行权价的 IV 均值
        k4000 = result[result["exercise_price"] == 4000.0]
        expected = k4000["iv"].mean()
        assert abs(atm_iv - expected) < 1e-6

    def test_atm_iv_empty_chain(self):
        """空 DataFrame 返回 NaN。"""
        calc = ImpliedVolCalculator()
        empty = pd.DataFrame(columns=["exercise_price", "call_put", "expire_date", "close",
                                       "iv", "is_valid"])
        assert math.isnan(calc.get_atm_iv(empty, S))

    def test_atm_iv_all_invalid(self):
        """所有合约 is_valid=False 时返回 NaN。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        result = self._chain_with_iv()
        result["is_valid"] = False
        assert math.isnan(calc.get_atm_iv(result, S))

    def test_atm_iv_all_volume_zero(self):
        """所有成交量为0时 ATM IV 返回 NaN。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        chain = _make_chain(S, [4000.0], self.T, r, sigma_true, TRADE_DATE, self.expire)
        chain["volume"] = 0
        result = calc.calculate_iv_for_chain(chain, S, TRADE_DATE)
        assert math.isnan(calc.get_atm_iv(result, S))

    def test_atm_iv_with_high_sigma(self):
        """高波动率（sigma=60%）时 ATM IV 仍能正确提取。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        result = self._chain_with_iv(sigma=0.60)
        atm_iv = calc.get_atm_iv(result, S)
        assert abs(atm_iv - 0.60) < 1e-3

    def test_atm_iv_without_is_valid_column(self):
        """没有 is_valid 列时也能运行（全部视为有效）。"""
        calc = ImpliedVolCalculator(risk_free_rate=r)
        chain = _make_chain(S, [4000.0], self.T, r, sigma_true, TRADE_DATE, self.expire)
        chain["iv"] = sigma_true
        # 不加 is_valid 列
        result = calc.get_atm_iv(chain, S)
        assert not math.isnan(result)


# ======================================================================
# TestBuildIvHistory（使用 mock DBManager）
# ======================================================================

class TestBuildIvHistory:
    """build_iv_history 使用 mock DB 测试。"""

    T = 30 / 365
    start = "20260101"
    end = "20260110"

    def _make_mock_db(self, has_options=True, has_futures=True):
        """构造有数据的 mock DBManager。"""
        db = MagicMock()

        # 交易日历：3 个交易日
        trade_dates = ["20260102", "20260105", "20260106"]
        db.get_trade_calendar.return_value = pd.DataFrame({"trade_date": trade_dates})

        if has_futures:
            def get_futures_daily(ts_code, start_date, end_date):  # noqa: ARG001
                del ts_code, start_date, end_date
                return pd.DataFrame({"close": [4000.0], "oi": [10000]})
            db.get_futures_daily.side_effect = get_futures_daily
        else:
            db.get_futures_daily.return_value = pd.DataFrame()

        if has_options:
            def get_options_chain(underlying, trade_date, expire_date=None):  # noqa: ARG001
                del underlying, expire_date
                expire = _expire_date_for_T(trade_date, 30 / 365)
                return _make_chain(4000.0, [3900.0, 4000.0, 4100.0], 30 / 365,
                                   0.02, 0.20, trade_date, expire)
            db.get_options_chain.side_effect = get_options_chain
        else:
            db.get_options_chain.return_value = pd.DataFrame()

        return db

    def test_returns_dataframe(self):
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db()
        result = calc.build_iv_history(db, "IO", self.start, self.end)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db()
        result = calc.build_iv_history(db, "IO", self.start, self.end)
        for col in ("trade_date", "atm_iv", "underlying_price"):
            assert col in result.columns

    def test_row_count_matches_calendar(self):
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db()
        result = calc.build_iv_history(db, "IO", self.start, self.end)
        assert len(result) == 3  # 3 个交易日

    def test_atm_iv_valid_values(self):
        """有数据时 atm_iv 应为合理值（0.10–0.50）。"""
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db()
        result = calc.build_iv_history(db, "IO", self.start, self.end)
        valid = result["atm_iv"].dropna()
        assert len(valid) > 0
        assert ((valid >= 0.10) & (valid <= 0.50)).all()

    def test_underlying_price_populated(self):
        """underlying_price 列应被填充。"""
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db()
        result = calc.build_iv_history(db, "IO", self.start, self.end)
        assert (result["underlying_price"].dropna() == 4000.0).all()

    def test_no_futures_data_returns_nan_iv(self):
        """无期货数据时 atm_iv 全为 NaN。"""
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db(has_futures=False)
        result = calc.build_iv_history(db, "IO", self.start, self.end)
        assert result["atm_iv"].isna().all()

    def test_no_options_data_returns_nan_iv(self):
        """无期权数据时 atm_iv 全为 NaN。"""
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db(has_options=False)
        result = calc.build_iv_history(db, "IO", self.start, self.end)
        assert result["atm_iv"].isna().all()

    def test_empty_calendar_fallback(self):
        """无交易日历时仍能运行（按自然工作日降级）。"""
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db()
        db.get_trade_calendar.return_value = pd.DataFrame()
        db.get_futures_daily.return_value = pd.DataFrame({"close": [4000.0], "oi": [10000]})
        def get_options_chain(underlying, trade_date, expire_date=None):  # noqa: ARG001
            del underlying, expire_date
            expire = _expire_date_for_T(trade_date, 30 / 365)
            return _make_chain(4000.0, [4000.0], 30 / 365, 0.02, 0.20, trade_date, expire)
        db.get_options_chain.side_effect = get_options_chain
        result = calc.build_iv_history(db, "IO", "20260105", "20260109")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_underlying_mapping_io_to_if(self):
        """IO 标的应查询 IF 期货数据。"""
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db()
        calc.build_iv_history(db, "IO", self.start, self.end)
        calls = db.get_futures_daily.call_args_list
        assert len(calls) > 0
        # _get_underlying_price 以关键字参数调用
        ts_code_arg = calls[0].kwargs.get("ts_code") or calls[0][0][0] if calls[0][0] else calls[0].kwargs["ts_code"]
        assert "IF" in ts_code_arg

    def test_underlying_mapping_mo_to_im(self):
        """MO 标的应查询 IM 期货数据。"""
        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        db = self._make_mock_db()
        calc.build_iv_history(db, "MO", self.start, self.end)
        calls = db.get_futures_daily.call_args_list
        assert len(calls) > 0
        ts_code_arg = calls[0].kwargs.get("ts_code") or calls[0][0][0] if calls[0][0] else calls[0].kwargs["ts_code"]
        assert "IM" in ts_code_arg


# ======================================================================
# TestRealDataIntegration（真实数据库）
# ======================================================================

def _get_real_db():
    """尝试打开真实数据库，失败返回 None。"""
    try:
        import os
        # 相对于项目根目录
        db_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "market_data.db"
        )
        db_path = os.path.abspath(db_path)
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
def real_data():
    """真实数据库 fixture，无数据则 skip 整个类。"""
    db, latest_date = _get_real_db()
    if db is None:
        pytest.skip("数据库无 IO 期权数据，跳过真实数据测试")
    yield db, latest_date
    db.close()


class TestRealDataIntegration:
    """使用真实数据库的集成测试（数据库有数据时运行）。"""

    def test_iv_chain_not_empty(self, real_data):
        """真实期权链 IV 计算结果非空。"""
        db, trade_date = real_data
        chain = db.get_options_chain("IO", trade_date)
        assert not chain.empty, "期权链应有数据"
        underlying = _get_spot_from_db(db, trade_date)
        if underlying is None:
            pytest.skip("无对应期货价格")

        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        result = calc.calculate_iv_for_chain(chain, underlying, trade_date)
        assert len(result) == len(chain)
        valid = result[result["is_valid"]]
        assert len(valid) > 0, "应至少有一个有效 IV"

    def test_iv_smile_otm_higher_than_atm(self, real_data):
        """真实 IV 微笑：OTM 合约 IV 一般 >= ATM IV。"""
        db, trade_date = real_data
        chain = db.get_options_chain("IO", trade_date)
        underlying = _get_spot_from_db(db, trade_date)
        if underlying is None:
            pytest.skip("无对应期货价格")

        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        result = calc.calculate_iv_for_chain(chain, underlying, trade_date)
        valid = result[result["is_valid"] & result["iv"].notna()]
        if len(valid) < 5:
            pytest.skip("有效合约太少，无法验证微笑")

        # 找 ATM 行权价
        strikes = valid["exercise_price"].unique()
        atm_strike = min(strikes, key=lambda k: abs(float(k) - underlying))
        atm_iv = valid[valid["exercise_price"] == atm_strike]["iv"].mean()

        # 找最远 OTM（行权价偏离最大）
        far_otm_strikes = sorted(strikes, key=lambda k: abs(float(k) - underlying), reverse=True)
        if len(far_otm_strikes) < 2:
            pytest.skip("行权价太少")
        far_iv = valid[valid["exercise_price"].isin(far_otm_strikes[:2])]["iv"].mean()
        print(f"\n[真实数据] {trade_date} ATM IV={atm_iv:.4f}, Far OTM IV={far_iv:.4f}")
        assert far_iv >= atm_iv * 0.8, "OTM IV 应接近或高于 ATM IV（允许 20% 容差）"

    def test_atm_iv_in_reasonable_range(self, real_data):
        """真实 ATM IV 应在 10%-50% 范围内。"""
        db, trade_date = real_data
        chain = db.get_options_chain("IO", trade_date)
        underlying = _get_spot_from_db(db, trade_date)
        if underlying is None:
            pytest.skip("无对应期货价格")

        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        result = calc.calculate_iv_for_chain(chain, underlying, trade_date)
        atm_iv = calc.get_atm_iv(result, underlying)
        print(f"\n[真实数据] {trade_date} ATM IV={atm_iv:.4f}")
        assert 0.10 <= atm_iv <= 0.50, f"ATM IV={atm_iv:.4f} 不合理"

    def test_print_iv_by_strike(self, real_data):
        """打印各行权价 IV，人工检查微笑结构。"""
        db, trade_date = real_data
        chain = db.get_options_chain("IO", trade_date)
        underlying = _get_spot_from_db(db, trade_date)
        if underlying is None:
            pytest.skip("无对应期货价格")

        calc = ImpliedVolCalculator(risk_free_rate=0.02)
        result = calc.calculate_iv_for_chain(chain, underlying, trade_date)
        valid = result[result["is_valid"]].sort_values(["exercise_price", "call_put"])

        print(f"\n=== IO 期权 IV 微笑 ({trade_date}, S={underlying:.0f}) ===")
        print(f"{'行权价':>10} {'类型':>4} {'IV':>8} {'Delta':>8} {'Moneyness':>10}")
        for _, row in valid.iterrows():
            print(f"{row['exercise_price']:>10.0f} {row['call_put']:>4} "
                  f"{row['iv']:>8.4f} {row['delta']:>8.4f} {row['moneyness']:>10.4f}")


def _get_spot_from_db(db, trade_date: str):
    """从 IF 期货获取当日标的价格（主力合约收盘价）。"""
    try:
        df = db.get_futures_daily("%IF%", trade_date, trade_date)
        if df.empty:
            return None
        if "oi" in df.columns:
            df = df.sort_values("oi", ascending=False)
        return float(df.iloc[0]["close"])
    except Exception:
        return None
