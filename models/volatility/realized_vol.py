"""
realized_vol.py
---------------
已实现波动率（Realized Volatility, RV）计算模块。

提供两套接口：
1. RealizedVolCalculator（类，静态方法）— 面向日线数据，当前主要使用
2. 模块级函数 compute_realized_vol / compute_rolling_rv — 面向分钟线数据，
   等分钟线到位后完整使用

支持估计量：
- Close-to-Close（滚动标准差）
- Parkinson（高低价，1980）
- Garman-Klass（OHLC，1980）
- 日内简单 RV / Rogers-Satchell（分钟线接口用）
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 股指期货每日交易分钟数（9:30-11:30 + 13:00-15:00 = 240 分钟）
TRADING_MINUTES_PER_DAY = 240
# 每年交易日（A 股约 252 个交易日）
TRADING_DAYS_PER_YEAR = 252


# ======================================================================
# 分钟线估计量枚举（与 compute_realized_vol 搭配使用）
# ======================================================================

class RVEstimator(str, Enum):
    """已实现波动率估计量类型（用于分钟线接口）"""
    SIMPLE = "simple"                      # 日内收益率平方和
    PARKINSON = "parkinson"                # Parkinson 高低价估计
    ROGERS_SATCHELL = "rogers_satchell"    # Rogers-Satchell OHLC 估计


# ======================================================================
# RealizedVolCalculator — 面向日线数据的主力类
# ======================================================================

class RealizedVolCalculator:
    """
    已实现波动率计算器。

    支持两种模式：
    1. 日频模式（当前可用）：用日线收盘价计算 Close-to-Close 滚动 RV
    2. 日内模式（分钟线到位后）：用日内分钟线数据计算每日 RV

    所有方法均为静态方法，无需实例化。
    """

    # ------------------------------------------------------------------
    # 1. Close-to-Close 滚动 RV（日线）
    # ------------------------------------------------------------------

    @staticmethod
    def from_daily(
        close_prices: pd.Series,
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        用日线收盘价序列计算滚动已实现波动率。

        数学公式：
            r_t = ln(P_t / P_{t-1})
            RV_t = std(r_{t-window+1}, ..., r_t)          [sample std, ddof=1]
            RV_annualized = RV * sqrt(trading_days)

        参数
        ----
        close_prices : pd.Series
            收盘价序列，index 为日期（任意可排序 index 均可）
        window : int
            滚动窗口天数，默认 20（约一个月交易日）
        annualize : bool
            是否年化，默认 True
        trading_days : int
            年交易日数，默认 252

        返回
        ----
        pd.Series
            已实现波动率序列，与 close_prices 等长；
            前 window 个值为 NaN（第 0 个 log_return 本身为 NaN，
            再加上 window-1 个窗口不足的值，共 window 个 NaN）。
        """
        if close_prices.empty:
            return pd.Series(dtype=float)

        # Step 1: 对数收益率 r_t = ln(P_t / P_{t-1})
        log_returns = np.log(close_prices / close_prices.shift(1))

        # Step 2: 滚动标准差（sample std, ddof=1）
        rv = log_returns.rolling(window=window).std()

        # Step 3: 年化
        if annualize:
            rv = rv * np.sqrt(trading_days)

        return rv

    # ------------------------------------------------------------------
    # 2. 日内 RV（分钟线）
    # ------------------------------------------------------------------

    @staticmethod
    def from_intraday(
        minute_bars: pd.DataFrame,
        freq_minutes: int = 5,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        用日内分钟线数据计算每日已实现波动率。

        数学公式：
            r_i = ln(P_i / P_{i-1})   (日内第 i 个 bar 相对于前一个 bar)
            RV_t = sqrt(sum(r_i^2))   (当日所有 bar 的收益率平方和开方)
            RV_annualized = RV_t * sqrt(trading_days)

        参数
        ----
        minute_bars : pd.DataFrame
            分钟线 DataFrame，须含 datetime（可解析为 Timestamp）和 close 列
        freq_minutes : int
            K 线频率（分钟），默认 5
        annualize : bool
            是否年化
        trading_days : int
            年交易日数

        返回
        ----
        pd.Series
            每日已实现波动率，index 为交易日期（pd.Timestamp）

        注意
        ----
        - 每日第一根 bar 的收益率使用相对于前一根 bar 的价格，
          第一根 bar 本身跳过（避免隔夜跳空污染日内 RV）
        - 分钟线数据尚未下载时传入空 DataFrame 会抛出 NotImplementedError
        """
        if minute_bars.empty:
            raise NotImplementedError(
                "分钟线 DataFrame 为空，请先运行 download_futures_min.py 下载数据"
            )

        df = minute_bars.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        # 如果原始数据粒度细于 freq_minutes，按频率重采样（取每段末 bar 的 close）
        if freq_minutes > 1:
            df = (
                df.set_index("datetime")["close"]
                .resample(f"{freq_minutes}min", closed="right", label="right")
                .last()
                .dropna()
                .reset_index()
            )

        df["_date"] = df["datetime"].dt.date

        def _daily_rv(day_df: pd.DataFrame) -> float:
            closes = day_df["close"].values
            if len(closes) < 2:
                return float("nan")
            # 跳过第一根 bar（避免隔夜跳空），从第 2 根开始计算日内收益率
            returns = np.log(closes[1:] / closes[:-1])
            # RV_t = sqrt(sum(r_i^2))
            return float(np.sqrt(np.sum(returns ** 2)))

        rvs = df.groupby("_date").apply(_daily_rv, include_groups=False)
        rvs.index = pd.to_datetime(rvs.index)

        if annualize:
            rvs = rvs * np.sqrt(trading_days)

        return rvs

    # ------------------------------------------------------------------
    # 3. Parkinson 估计量（日线 OHLC）
    # ------------------------------------------------------------------

    @staticmethod
    def parkinson(
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        Parkinson（1980）波动率估计，利用日内最高/最低价。

        比 Close-to-Close 更高效（利用了日内价格范围信息），
        但假设价格过程无漂移项（μ=0）。

        数学公式（逐日方差 → 滚动均值 → 开方 → 年化）：
            σ²_pk_t = (1 / (4 * ln2)) * ln(H_t / L_t)²
            σ²_pk_roll = mean(σ²_pk_{t-window+1}, ..., σ²_pk_t)
            σ_pk = sqrt(σ²_pk_roll) * sqrt(trading_days)

        参数
        ----
        high : pd.Series   日最高价
        low  : pd.Series   日最低价
        window : int        滚动窗口
        annualize : bool    是否年化
        trading_days : int  年交易日数

        返回
        ----
        pd.Series   Parkinson 波动率序列，与输入等长，前 window-1 个值为 NaN
        """
        log_hl_sq = np.log(high / low) ** 2
        # 日方差 → 滚动均值 → 缩放
        pk_var = (1.0 / (4.0 * np.log(2))) * log_hl_sq.rolling(window=window).mean()
        rv = np.sqrt(pk_var)

        if annualize:
            rv = rv * np.sqrt(trading_days)

        return rv

    # ------------------------------------------------------------------
    # 4. Garman-Klass 估计量（日线 OHLC）
    # ------------------------------------------------------------------

    @staticmethod
    def garman_klass(
        open_p: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """
        Garman-Klass（1980）波动率估计，利用 OHLC 四价。

        比 Parkinson 更高效，额外利用了开盘价和收盘价信息，
        在允许漂移项时仍表现良好。

        数学公式（逐日方差 → 滚动均值 → 开方 → 年化）：
            σ²_GK_t = 0.5 * ln(H/L)² - (2*ln2 - 1) * ln(C/O)²
            σ²_GK_roll = mean(σ²_GK_{t-window+1}, ..., σ²_GK_t)
            σ_GK = sqrt(max(σ²_GK_roll, 0)) * sqrt(trading_days)

        注意：GK 日方差可能因开收价跳空出现微小负值，clip(lower=0) 处理。

        参数
        ----
        open_p : pd.Series  开盘价
        high   : pd.Series  最高价
        low    : pd.Series  最低价
        close  : pd.Series  收盘价
        window : int         滚动窗口
        annualize : bool     是否年化
        trading_days : int   年交易日数

        返回
        ----
        pd.Series   Garman-Klass 波动率序列，与输入等长，前 window-1 个值为 NaN
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_p)

        # 逐日 GK 方差
        gk_daily = 0.5 * log_hl ** 2 - (2.0 * np.log(2) - 1.0) * log_co ** 2

        # 滚动均值，clip 防止极小负值导致 sqrt 出 NaN
        gk_var = gk_daily.rolling(window=window).mean().clip(lower=0.0)
        rv = np.sqrt(gk_var)

        if annualize:
            rv = rv * np.sqrt(trading_days)

        return rv


# ======================================================================
# 模块级函数（面向分钟线，向后兼容旧接口）
# ======================================================================

def compute_realized_vol(
    min_df: pd.DataFrame,
    estimator: RVEstimator = RVEstimator.SIMPLE,
    freq_minutes: int = 5,
    annualize: bool = True,
) -> pd.Series:
    """
    从分钟线数据计算日度已实现波动率。

    Parameters
    ----------
    min_df : pd.DataFrame
        分钟线数据，含 datetime, open, high, low, close 列
    estimator : RVEstimator
        估计量类型
    freq_minutes : int
        K 线频率（分钟）
    annualize : bool
        是否年化

    Returns
    -------
    pd.Series
        日度 RV，index 为日期；数据不足的日期返回 NaN

    Notes
    -----
    - 每日有效 bar 数量 < 正常数量 80% 时，该日 RV 标记为 NaN
    - 第一根 bar 收益率基于前一根 bar 收盘价，避免隔夜跳空污染
    """
    if min_df.empty:
        return pd.Series(dtype=float)

    df = min_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["_date"] = df["datetime"].dt.date
    df = df.sort_values("datetime")

    expected_bars = TRADING_MINUTES_PER_DAY // freq_minutes  # 默认 48

    def _compute_day(day_df: pd.DataFrame) -> float:
        if len(day_df) < expected_bars * 0.8:
            return float("nan")
        day_df = day_df.sort_values("datetime")
        if estimator == RVEstimator.SIMPLE:
            return _simple_rv(day_df)
        if estimator == RVEstimator.PARKINSON:
            return _parkinson_rv(day_df)
        if estimator == RVEstimator.ROGERS_SATCHELL:
            return _rogers_satchell_rv(day_df)
        raise ValueError(f"未知估计量: {estimator}")

    daily_rv = df.groupby("_date").apply(_compute_day, include_groups=False)
    daily_rv.index = pd.to_datetime(daily_rv.index)

    if annualize:
        daily_rv = daily_rv * np.sqrt(TRADING_DAYS_PER_YEAR)

    return daily_rv.rename("rv")


def compute_rolling_rv(
    min_df: pd.DataFrame,
    window: int = 20,
    estimator: RVEstimator = RVEstimator.SIMPLE,
    freq_minutes: int = 5,
) -> pd.Series:
    """
    计算滚动窗口平均已实现波动率（用于 GARCH 输入和 VRP 计算）。

    等价于 HAR-RV 模型的月度分量。

    Parameters
    ----------
    min_df : pd.DataFrame
        分钟线数据
    window : int
        滚动窗口（交易日），默认 20
    estimator : RVEstimator
        估计量类型
    freq_minutes : int
        K 线频率（分钟）

    Returns
    -------
    pd.Series
        滚动平均 RV，前 window-1 个值为 NaN
    """
    daily_rv = compute_realized_vol(
        min_df, estimator=estimator, freq_minutes=freq_minutes, annualize=True
    )
    return daily_rv.rolling(window=window).mean()


# ======================================================================
# 内部辅助函数（单日，未年化）
# ======================================================================

def _simple_rv(day_df: pd.DataFrame) -> float:
    """
    单日简单已实现波动率（未年化）。

    RV = sqrt(sum(r_i^2))，r_i 为日内对数收益率（跳过首根 bar）。
    """
    closes = day_df["close"].values
    if len(closes) < 2:
        return float("nan")
    returns = np.log(closes[1:] / closes[:-1])
    return float(np.sqrt(np.sum(returns ** 2)))


def _parkinson_rv(day_df: pd.DataFrame) -> float:
    """
    Parkinson 高低价估计量（单日，未年化）。

    Parkinson(1980): RV = sqrt(1/(4*n*ln2) * sum(ln(H_i/L_i)^2))
    """
    log_hl = np.log(day_df["high"].values / day_df["low"].values)
    n = len(log_hl)
    if n == 0:
        return float("nan")
    pk_var = np.sum(log_hl ** 2) / (4.0 * n * np.log(2))
    return float(np.sqrt(pk_var))


def _rogers_satchell_rv(day_df: pd.DataFrame) -> float:
    """
    Rogers-Satchell OHLC 估计量（单日，未年化）。

    Rogers & Satchell(1991): 对有漂移的价格过程无偏。
    RS = mean(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O))
    """
    h = day_df["high"].values
    l = day_df["low"].values
    o = day_df["open"].values
    c = day_df["close"].values

    rs_terms = (
        np.log(h / c) * np.log(h / o)
        + np.log(l / c) * np.log(l / o)
    )
    rs_var = np.mean(rs_terms)
    return float(np.sqrt(max(rs_var, 0.0)))
