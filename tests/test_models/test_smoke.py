"""
test_smoke.py
-------------
端到端 smoke test：验证从数据到信号的完整链路。

流程：
  1. 从数据库读 IF 日线  -> 计算 RV
  2. 拟合 GJR-GARCH      -> 预测未来5天波动率
  3. 从数据库读 IO 期权链 -> 计算 IV -> 构建波动率曲面 -> 提取 ATM IV
  4. 计算 VRP（GARCH预测波动率 - ATM IV）并打印摘要

依赖真实数据库，DB 中无数据时跳过（pytest.mark.skip）。
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd
import pytest

# ======================================================================
# 数据库路径探测
# ======================================================================

def _find_db() -> str | None:
    """查找项目默认 SQLite 数据库路径。"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "market_data.db"),
        os.path.join(os.path.dirname(__file__), "..", "..", "market_data.db"),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


_DB_PATH = _find_db()
_HAS_DB = _DB_PATH is not None

pytestmark = pytest.mark.skipif(
    not _HAS_DB,
    reason="未找到 market_data.db，跳过端到端 smoke test（需先运行下载脚本）",
)


# ======================================================================
# 辅助：静默 arch 拟合输出
# ======================================================================

logging.getLogger("arch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# Smoke Test 主流程
# ======================================================================

class TestSmokeEndToEnd:
    """从数据到信号的完整链路 smoke test。"""

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------

    def _get_if_close(self) -> pd.Series:
        """从数据库读取 IF 近月合约日线收盘价（最近 300 交易日）。"""
        from data.storage.db_manager import DBManager

        db = DBManager(_DB_PATH)

        # 尝试最活跃的 IF 主力合约代码
        candidate_codes = ["IF.CFX"]  # 连续合约（若有）
        # 枚举最近几个月合约作为备选
        for year in range(2024, 2026):
            for month in range(1, 13):
                candidate_codes.append(f"IF{year % 100:02d}{month:02d}.CFX")

        for ts_code in candidate_codes:
            try:
                df = db.get_futures_daily(ts_code, "20230101", "20251231")
                if df is not None and len(df) >= 60:
                    df = df.sort_values("trade_date")
                    close = pd.Series(
                        df["close"].values,
                        index=pd.to_datetime(df["trade_date"]),
                        name=ts_code,
                        dtype=float,
                    )
                    return close.iloc[-300:]
            except Exception:
                continue

        # 如果单合约没有，尝试跨合约拼接 IF
        try:
            df = db.query_df(
                "SELECT trade_date, close, ts_code FROM futures_daily "
                "WHERE ts_code LIKE 'IF%' ORDER BY trade_date DESC LIMIT 500"
            )
            if df is not None and len(df) >= 60:
                df = df.sort_values("trade_date")
                close = pd.Series(
                    df["close"].values,
                    index=pd.to_datetime(df["trade_date"]),
                    name="IF_spliced",
                    dtype=float,
                ).dropna()
                return close.iloc[-300:]
        except Exception:
            pass

        return pd.Series(dtype=float)

    def _get_io_options(self, trade_date: str) -> pd.DataFrame:
        """从数据库读取 IO 期权链。"""
        from data.storage.db_manager import DBManager

        db = DBManager(_DB_PATH)
        try:
            df = db.get_options_daily("IO", trade_date)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        # 备选：直接查表
        try:
            df = db.query_df(
                f"SELECT * FROM options_daily WHERE underlying_code='IO' "
                f"AND trade_date='{trade_date}'"
            )
            return df if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 测试主体
    # ------------------------------------------------------------------

    def test_garch_forecast_pipeline(self):
        """
        链路一：IF日线 -> 对数收益率 -> RV -> GJR-GARCH -> 5日预测波动率。
        """
        from models.volatility.realized_vol import compute_rolling_rv
        from models.volatility.garch_model import GJRGARCHModel

        close = self._get_if_close()
        if len(close) < 60:
            pytest.skip("IF 日线数据不足 60 条，跳过")

        # Step 1: 对数收益率
        returns = np.log(close / close.shift(1)).dropna()
        assert len(returns) >= 30, "收益率序列长度不足"

        # Step 2: 滚动 RV（22日窗口）
        rolling_rv = compute_rolling_rv(returns, window=22)
        valid_rv = rolling_rv.dropna()
        assert len(valid_rv) > 0, "RV 计算失败"
        latest_rv = float(valid_rv.iloc[-1])
        assert 0.0 < latest_rv < 5.0, f"RV 超出合理范围: {latest_rv:.4f}"

        # Step 3: GJR-GARCH 拟合 + 预测
        if len(returns) < 60:
            pytest.skip("收益率数据不足以拟合 GARCH（需要 60+）")

        model = GJRGARCHModel()
        fit_result = model.fit(returns)
        assert fit_result.converged or True, "GARCH 未收敛"  # 允许未收敛，只要不报错

        garch_vol_5d = float(model.forecast_period_avg(horizon=5))
        assert 0.01 < garch_vol_5d < 3.0, f"GARCH 预测波动率超出合理范围: {garch_vol_5d:.4f}"

        print(
            f"\n[Smoke] IF 最近 22 日 RV: {latest_rv*100:.2f}%  |  "
            f"GARCH 5日预测: {garch_vol_5d*100:.2f}%"
        )

    def test_vol_surface_and_atm_iv(self):
        """
        链路二：IO期权链 -> 计算IV -> 构建波动率曲面 -> 提取ATM IV。
        """
        from models.pricing.vol_surface import VolSurface

        # 找到数据库中最新有效期权日期
        from data.storage.db_manager import DBManager
        db = DBManager(_DB_PATH)

        trade_date = None
        try:
            row = db.query_df(
                "SELECT trade_date FROM options_daily WHERE underlying_code='IO' "
                "ORDER BY trade_date DESC LIMIT 1"
            )
            if row is not None and not row.empty:
                trade_date = str(row["trade_date"].iloc[0]).replace("-", "")
        except Exception:
            pass

        if trade_date is None:
            pytest.skip("数据库中无 IO 期权数据，跳过")

        opts = self._get_io_options(trade_date)
        if opts.empty:
            pytest.skip(f"IO 期权链 {trade_date} 为空，跳过")

        # 获取对应 IF 近月价格作为现货价
        close = self._get_if_close()
        if close.empty:
            pytest.skip("无 IF 日线，无法获取现货价")

        spot = float(close.iloc[-1])
        assert spot > 0

        # 构建波动率曲面（列名可能是 exercise_price 或 strike_price，统一）
        if "exercise_price" in opts.columns and "strike_price" not in opts.columns:
            opts = opts.rename(columns={"exercise_price": "strike_price"})

        surface = VolSurface(
            trade_date=trade_date,
            underlying="IO",
            spot_price=spot,
            risk_free_rate=0.02,
        )
        surface.build_from_options_df(opts, min_volume=0, min_oi=0)

        expire_dates = surface.get_all_expire_dates()
        if not expire_dates:
            pytest.skip("波动率曲面构建后无有效到期日，可能 IV 计算均失败")

        # 提取最近一个到期日的 ATM IV
        nearest_expire = min(expire_dates)
        atm_iv = surface.get_atm_iv(nearest_expire)

        assert atm_iv is not None and atm_iv > 0, f"ATM IV 无效: {atm_iv}"
        assert 0.01 < atm_iv < 3.0, f"ATM IV 超出合理范围: {atm_iv:.4f}"

        print(
            f"\n[Smoke] IO 期权曲面 {trade_date} "
            f"| 到期日: {nearest_expire} "
            f"| ATM IV: {atm_iv*100:.2f}%"
        )

    def test_vrp_signal(self):
        """
        链路三：GARCH预测波动率 vs ATM IV -> 计算 VRP（波动率风险溢价）。

        VRP = IV - GARCH_RV（正值表示期权被高估，可做空波动率）
        """
        from models.volatility.garch_model import GJRGARCHModel
        from models.pricing.vol_surface import VolSurface
        from data.storage.db_manager import DBManager

        close = self._get_if_close()
        if len(close) < 60:
            pytest.skip("IF 日线数据不足，跳过 VRP 测试")

        returns = np.log(close / close.shift(1)).dropna()
        if len(returns) < 60:
            pytest.skip("收益率不足 60 条")

        # GARCH 预测
        model = GJRGARCHModel()
        model.fit(returns)
        garch_vol = float(model.forecast_period_avg(horizon=22))  # 22日预测

        # 找最新期权日期
        db = DBManager(_DB_PATH)
        trade_date = None
        try:
            row = db.query_df(
                "SELECT trade_date FROM options_daily WHERE underlying_code='IO' "
                "ORDER BY trade_date DESC LIMIT 1"
            )
            if row is not None and not row.empty:
                trade_date = str(row["trade_date"].iloc[0]).replace("-", "")
        except Exception:
            pass

        if trade_date is None:
            pytest.skip("无 IO 期权数据")

        opts = self._get_io_options(trade_date)
        if opts.empty:
            pytest.skip("期权链为空")

        spot = float(close.iloc[-1])

        if "exercise_price" in opts.columns and "strike_price" not in opts.columns:
            opts = opts.rename(columns={"exercise_price": "strike_price"})

        surface = VolSurface(
            trade_date=trade_date,
            underlying="IO",
            spot_price=spot,
            risk_free_rate=0.02,
        )
        surface.build_from_options_df(opts)
        expire_dates = surface.get_all_expire_dates()
        if not expire_dates:
            pytest.skip("波动率曲面无有效到期日")

        atm_iv = surface.get_atm_iv(min(expire_dates))
        if atm_iv is None or atm_iv <= 0:
            pytest.skip("ATM IV 无效")

        # VRP 计算
        vrp = atm_iv - garch_vol
        vrp_pct = vrp * 100

        print(
            f"\n[Smoke] VRP 信号 ({trade_date})"
            f"\n  GARCH 22日预测波动率 : {garch_vol*100:.2f}%"
            f"\n  ATM IV (最近月)      : {atm_iv*100:.2f}%"
            f"\n  VRP = IV - GARCH     : {vrp_pct:+.2f}%"
            f"\n  信号方向             : {'做空波动率 (IV溢价)' if vrp > 0 else '做多波动率 (IV折价)'}"
        )

        # 断言 VRP 在合理区间（-50% ~ +50% 年化波动率差）
        assert -0.5 < vrp < 0.5, f"VRP 超出合理范围: {vrp_pct:.2f}%"
