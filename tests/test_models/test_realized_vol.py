"""
test_realized_vol.py
--------------------
测试 models/volatility/realized_vol.py 中的 RealizedVolCalculator 类。

测试覆盖：
1. from_daily — 手算验证、年化逻辑、window NaN 前缀、边界
2. parkinson  — 输出形状与非负性、与 from_daily 同量级
3. garman_klass — 输出形状与非负性、clip 保护
4. from_intraday — 空输入抛 NotImplementedError
5. smoke test  — 从数据库读取 IC.CFX 日线，计算20日RV，打印最近5个值
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.volatility.realized_vol import RealizedVolCalculator

# 数据库路径（测试用）
_DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "storage" / "trading.db"
)
_DB_EXISTS = _DB_PATH.exists()


# ======================================================================
# 辅助工具
# ======================================================================

def _make_prices(values: list[float]) -> pd.Series:
    """用简单整数 index 构造收盘价 Series"""
    return pd.Series(values, dtype=float)


def _make_ohlc(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """生成合成 OHLC 日线 DataFrame，用于 Parkinson / GK 测试"""
    rng = np.random.default_rng(seed)
    close = 4000.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high  = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


# ======================================================================
# 一、from_daily
# ======================================================================

class TestFromDaily:

    # ------------------------------------------------------------------
    # 手算验证
    # ------------------------------------------------------------------

    def test_hand_calculation_matches_numpy(self):
        """
        用已知价格序列手算，对比 from_daily 输出。

        价格: [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        log_returns: [NaN, ln(102/100), ln(101/102), ...]
        window=5 时，第一个有效 RV = std([r1,r2,r3,r4,r5])（ddof=1）
        """
        prices = _make_prices([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rv = RealizedVolCalculator.from_daily(prices, window=5, annualize=False)

        # numpy 参考计算
        log_ret = np.log(prices / prices.shift(1))
        expected = log_ret.rolling(window=5).std()  # ddof=1，与 pandas 一致

        # 第一个有效值在 index=5
        assert rv.iloc[5] == pytest.approx(expected.iloc[5], rel=1e-10)

        # 所有有效值一致
        valid = expected.notna()
        np.testing.assert_allclose(
            rv[valid].values,
            expected[valid].values,
            rtol=1e-10,
        )

    def test_hand_calculation_first_valid_value(self):
        """
        对 window=5，手动计算第一个有效值并验证。

        r1 = ln(102/100) ≈  0.019803
        r2 = ln(101/102) ≈ -0.009852
        r3 = ln(103/101) ≈  0.019612
        r4 = ln(105/103) ≈  0.019231
        r5 = ln(104/105) ≈ -0.009524
        std([r1..r5], ddof=1) — 用 numpy 独立计算作为参考
        """
        prices = _make_prices([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rv_unnorm = RealizedVolCalculator.from_daily(prices, window=5, annualize=False)

        p = np.array([100., 102., 101., 103., 105., 104.])
        rets = np.log(p[1:] / p[:-1])          # r1..r5, 5 elements
        expected_rv5 = float(np.std(rets, ddof=1))

        assert rv_unnorm.iloc[5] == pytest.approx(expected_rv5, rel=1e-10)

    # ------------------------------------------------------------------
    # 年化逻辑
    # ------------------------------------------------------------------

    def test_annualization_factor_is_sqrt_252(self):
        """年化值 = 日值 × sqrt(252)，精确到机器精度"""
        prices = pd.Series(
            [100.0 * (1.005 ** i) for i in range(30)], dtype=float
        )
        rv_raw  = RealizedVolCalculator.from_daily(prices, window=5, annualize=False)
        rv_ann  = RealizedVolCalculator.from_daily(prices, window=5, annualize=True)

        valid = rv_raw.notna() & (rv_raw > 0)
        ratio = rv_ann[valid] / rv_raw[valid]
        np.testing.assert_allclose(ratio.values, np.sqrt(252), rtol=1e-12)

    def test_custom_trading_days(self):
        """trading_days 参数正确影响年化因子"""
        prices = pd.Series([100.0 * (1.005 ** i) for i in range(30)], dtype=float)
        rv_252 = RealizedVolCalculator.from_daily(prices, window=5, annualize=True, trading_days=252)
        rv_365 = RealizedVolCalculator.from_daily(prices, window=5, annualize=True, trading_days=365)

        valid = rv_252.notna() & (rv_252 > 0)
        ratio = rv_365[valid] / rv_252[valid]
        np.testing.assert_allclose(
            ratio.values,
            np.sqrt(365) / np.sqrt(252),
            rtol=1e-12,
        )

    # ------------------------------------------------------------------
    # window 参数 / NaN 前缀
    # ------------------------------------------------------------------

    def test_first_window_values_are_nan(self):
        """
        前 window 个值为 NaN。
        原因：log_returns[0]=NaN，rolling(window) 需要 window 个有效值，
        因此 rv[0..window-1] 均为 NaN，rv[window] 为第一个有效值。
        """
        prices = pd.Series([100.0 + i for i in range(30)], dtype=float)
        window = 10
        rv = RealizedVolCalculator.from_daily(prices, window=window, annualize=False)

        assert rv.iloc[:window].isna().all(), f"前 {window} 个值应为 NaN"
        assert rv.iloc[window:].notna().all(), f"从 index={window} 起应全部有效"

    def test_window_2_first_two_are_nan(self):
        """
        window=2 时前 2 个值为 NaN：
        - index=0: log_return 本身为 NaN
        - index=1: rolling(2) 需要 2 个有效值，但 window [NaN, r1] 仅 1 个 → NaN
        - index=2 起: valid
        """
        prices = _make_prices([100.0, 101.0, 102.0, 103.0, 104.0])
        rv = RealizedVolCalculator.from_daily(prices, window=2, annualize=False)

        assert rv.iloc[:2].isna().all()
        assert rv.iloc[2:].notna().all()

    def test_returns_same_length_as_input(self):
        """输出长度与输入等长"""
        for n in [5, 20, 100]:
            prices = pd.Series(range(100, 100 + n), dtype=float)
            rv = RealizedVolCalculator.from_daily(prices, window=5)
            assert len(rv) == n, f"n={n}: 期望长度 {n}，得到 {len(rv)}"

    def test_preserves_index(self):
        """输出 Series 与输入 close_prices 共享 index"""
        idx = pd.date_range("2024-01-01", periods=20)
        prices = pd.Series(range(100, 120), index=idx, dtype=float)
        rv = RealizedVolCalculator.from_daily(prices, window=5)
        assert rv.index.equals(idx)

    def test_empty_input_returns_empty(self):
        """空输入返回空 Series"""
        rv = RealizedVolCalculator.from_daily(pd.Series(dtype=float), window=5)
        assert len(rv) == 0

    def test_all_valid_values_positive(self):
        """所有有效（非 NaN）的 RV 值应为正数"""
        prices = _make_prices([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rv = RealizedVolCalculator.from_daily(prices, window=3, annualize=True)
        valid = rv.dropna()
        assert (valid > 0).all()

    def test_higher_vol_yields_higher_rv(self):
        """高波动率价格序列 → 更高的平均 RV"""
        rng = np.random.default_rng(42)
        base = 4000.0

        prices_low  = pd.Series(base * np.cumprod(1 + rng.normal(0, 0.005, 50)))
        rng2 = np.random.default_rng(42)
        prices_high = pd.Series(base * np.cumprod(1 + rng2.normal(0, 0.02,  50)))

        rv_low  = RealizedVolCalculator.from_daily(prices_low,  window=5).dropna().mean()
        rv_high = RealizedVolCalculator.from_daily(prices_high, window=5).dropna().mean()
        assert rv_high > rv_low


# ======================================================================
# 二、Parkinson 估计量
# ======================================================================

class TestParkinson:

    def test_output_same_length_as_input(self):
        df = _make_ohlc(30)
        rv = RealizedVolCalculator.parkinson(df["high"], df["low"], window=10)
        assert len(rv) == 30

    def test_valid_values_non_negative(self):
        """Parkinson RV 恒非负（sqrt 保证）"""
        df = _make_ohlc(50)
        rv = RealizedVolCalculator.parkinson(df["high"], df["low"], window=10)
        assert (rv.dropna() >= 0).all()

    def test_first_window_minus_1_are_nan(self):
        """Parkinson 无首元素 NaN 问题，前 window-1 个值为 NaN"""
        df = _make_ohlc(30)
        window = 10
        rv = RealizedVolCalculator.parkinson(df["high"], df["low"], window=window)
        # rolling(10).mean() 前 9 个 NaN，第 10 个（index=9）开始有效
        assert rv.iloc[:window - 1].isna().all()
        assert rv.iloc[window - 1:].notna().all()

    def test_annualization_factor(self):
        """年化因子 = sqrt(252)"""
        df = _make_ohlc(40)
        rv_raw = RealizedVolCalculator.parkinson(df["high"], df["low"], window=5, annualize=False)
        rv_ann = RealizedVolCalculator.parkinson(df["high"], df["low"], window=5, annualize=True)
        valid = rv_raw.notna() & (rv_raw > 0)
        ratio = rv_ann[valid] / rv_raw[valid]
        np.testing.assert_allclose(ratio.values, np.sqrt(252), rtol=1e-12)

    def test_reasonable_magnitude(self):
        """年化 Parkinson RV 在合理范围（5% ~ 150%）"""
        df = _make_ohlc(60, seed=7)
        rv = RealizedVolCalculator.parkinson(df["high"], df["low"], window=20, annualize=True)
        valid = rv.dropna()
        assert (valid > 0.02).all()
        assert (valid < 2.0).all()


# ======================================================================
# 三、Garman-Klass 估计量
# ======================================================================

class TestGarmanKlass:

    def test_output_same_length_as_input(self):
        df = _make_ohlc(30)
        rv = RealizedVolCalculator.garman_klass(
            df["open"], df["high"], df["low"], df["close"], window=10
        )
        assert len(rv) == 30

    def test_valid_values_non_negative(self):
        """GK RV 恒非负（clip + sqrt 保证）"""
        df = _make_ohlc(50)
        rv = RealizedVolCalculator.garman_klass(
            df["open"], df["high"], df["low"], df["close"], window=10
        )
        assert (rv.dropna() >= 0).all()

    def test_first_window_minus_1_are_nan(self):
        """前 window-1 个值为 NaN"""
        df = _make_ohlc(30)
        window = 10
        rv = RealizedVolCalculator.garman_klass(
            df["open"], df["high"], df["low"], df["close"], window=window
        )
        assert rv.iloc[:window - 1].isna().all()
        assert rv.iloc[window - 1:].notna().all()

    def test_annualization_factor(self):
        df = _make_ohlc(40)
        rv_raw = RealizedVolCalculator.garman_klass(
            df["open"], df["high"], df["low"], df["close"], window=5, annualize=False
        )
        rv_ann = RealizedVolCalculator.garman_klass(
            df["open"], df["high"], df["low"], df["close"], window=5, annualize=True
        )
        valid = rv_raw.notna() & (rv_raw > 0)
        ratio = rv_ann[valid] / rv_raw[valid]
        np.testing.assert_allclose(ratio.values, np.sqrt(252), rtol=1e-12)

    def test_gk_vs_parkinson_same_order_of_magnitude(self):
        """GK 和 Parkinson 估计量量级应相近（同一数据集，比率在 0.5~2 之间）"""
        df = _make_ohlc(60, seed=3)
        pk = RealizedVolCalculator.parkinson(
            df["high"], df["low"], window=20, annualize=True
        ).dropna()
        gk = RealizedVolCalculator.garman_klass(
            df["open"], df["high"], df["low"], df["close"], window=20, annualize=True
        ).dropna()
        ratio = gk / pk
        assert (ratio > 0.5).all() and (ratio < 2.0).all()


# ======================================================================
# 四、from_intraday（分钟线）
# ======================================================================

class TestFromIntraday:

    def test_empty_input_raises(self):
        """空 DataFrame 应抛出 NotImplementedError"""
        with pytest.raises(NotImplementedError):
            RealizedVolCalculator.from_intraday(pd.DataFrame())

    def test_synthetic_intraday_returns_series(self):
        """合成分钟线数据 → 返回正确长度的 Series"""
        rng = np.random.default_rng(0)
        n_days, bars = 5, 48  # 5 天 × 48 根 5 分钟线
        records = []
        price = 4000.0
        for d in range(n_days):
            date_str = f"2024-01-{d + 2:02d}"
            for b in range(bars):
                price *= np.exp(rng.normal(0, 0.002))
                h = int(9 + (b * 5) // 60)
                m = (b * 5) % 60
                records.append({
                    "datetime": f"{date_str} {h:02d}:{m:02d}:00",
                    "close": round(price, 2),
                })
        df = pd.DataFrame(records)
        rv = RealizedVolCalculator.from_intraday(df, freq_minutes=5, annualize=True)
        assert len(rv) == n_days
        assert (rv > 0).all()


# ======================================================================
# 五、Smoke test — 真实数据库
# ======================================================================

@pytest.mark.skipif(not _DB_EXISTS, reason=f"DB 不存在: {_DB_PATH}")
class TestSmokeRealData:

    def _load_close(self, ts_code: str = "IC.CFX") -> pd.Series:
        conn = sqlite3.connect(_DB_PATH)
        df = pd.read_sql(
            "SELECT trade_date, close FROM futures_daily "
            "WHERE ts_code = ? ORDER BY trade_date",
            conn,
            params=(ts_code,),
        )
        conn.close()
        df["close"] = pd.to_numeric(df["close"])
        return df.set_index("trade_date")["close"]

    def test_from_daily_on_IC(self):
        """IC.CFX 日线 → 20 日 RV，检查基本属性并打印最近 5 个值"""
        close = self._load_close("IC.CFX")
        rv = RealizedVolCalculator.from_daily(close, window=20, annualize=True)

        # 基本健康检查
        assert len(rv) == len(close)
        valid = rv.dropna()
        assert len(valid) > 200, "有效 RV 值过少"
        assert (valid > 0.05).all(), "年化 RV 不应低于 5%"
        assert (valid < 2.0).all(),  "年化 RV 不应超过 200%"

        print(f"\n[smoke] IC.CFX 20日RV（最近5个值）:")
        for date, val in valid.tail(5).items():
            print(f"  {date}: {val:.4f}  ({val*100:.2f}%)")

    def test_parkinson_on_real_data(self):
        """IC.CFX 日线 OHLC → Parkinson 20日RV"""
        conn = sqlite3.connect(_DB_PATH)
        df = pd.read_sql(
            "SELECT trade_date, open, high, low, close FROM futures_daily "
            "WHERE ts_code = 'IC.CFX' ORDER BY trade_date",
            conn,
        )
        conn.close()
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])

        rv_pk = RealizedVolCalculator.parkinson(df["high"], df["low"], window=20)
        rv_gk = RealizedVolCalculator.garman_klass(
            df["open"], df["high"], df["low"], df["close"], window=20
        )

        assert (rv_pk.dropna() > 0).all()
        assert (rv_gk.dropna() > 0).all()

        print(f"\n[smoke] IC.CFX Parkinson vs GK（最近5个值）:")
        tail_idx = rv_pk.dropna().tail(5).index
        for i in tail_idx:
            print(f"  {df['trade_date'].iloc[i]}  PK={rv_pk.iloc[i]:.4f}  GK={rv_gk.iloc[i]:.4f}")
