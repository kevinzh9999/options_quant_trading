"""
trend.py
--------
职责：趋势类技术指标计算。

提供：MA、EMA、MACD、ADX
所有函数接受 pd.Series（收盘价或 OHLCV DataFrame）
并返回 pd.Series（与输入同索引）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_sma(close: pd.Series, window: int) -> pd.Series:
    """简单移动平均（SMA），前 window-1 个值为 NaN。"""
    return close.rolling(window=window).mean()


def calc_ema(close: pd.Series, span: int) -> pd.Series:
    """指数移动平均（EMA），使用 pandas ewm。"""
    return close.ewm(span=span, adjust=False).mean()


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 指标（异同移动平均线）。

    Returns
    -------
    (macd_line, signal_line, histogram)
    - macd_line = EMA(fast) - EMA(slow)
    - signal_line = EMA(macd_line, signal)
    - histogram = macd_line - signal_line
    """
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    ADX（平均趋向指数）—— 衡量趋势强度（不区分方向）。

    ADX > 25：趋势明显；ADX > 50：强趋势；ADX < 20：震荡市。

    Notes
    -----
    TR = max(H-L, |H-prev_C|, |L-prev_C|)
    +DM = max(H_t - H_{t-1}, 0) if > |L_t - L_{t-1}| else 0
    -DM = max(L_{t-1} - L_t, 0) if > |H_t - H_{t-1}| else 0
    DI± = 100 * Wilder_EMA(DM±) / ATR
    DX  = 100 * |DI+ - DI-| / (DI+ + DI-)
    ADX = Wilder_EMA(DX, window)
    """
    close_prev = close.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    dm_plus = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    dm_minus = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )

    # Wilder smoothing: alpha = 1/window
    alpha = 1.0 / window
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    ema_dm_plus = dm_plus.ewm(alpha=alpha, adjust=False).mean()
    ema_dm_minus = dm_minus.ewm(alpha=alpha, adjust=False).mean()

    atr_safe = atr.replace(0, np.nan)
    di_plus = 100.0 * ema_dm_plus / atr_safe
    di_minus = 100.0 * ema_dm_minus / atr_safe

    di_sum = (di_plus + di_minus).replace(0, np.nan)
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


# ======================================================================
# TrendIndicators 类：面向策略层的面向对象接口
# ======================================================================

class TrendIndicators:
    """
    趋势类技术指标计算。

    所有方法为静态方法，输入输出均为 pd.Series / pd.DataFrame。
    内部委托给模块级函数，提供统一的类接口。
    """

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        简单移动平均（SMA）。

        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int
            窗口期

        Returns
        -------
        pd.Series
            SMA 序列，前 period-1 个值为 NaN
        """
        return calc_sma(series, period)

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        指数移动平均（EMA）。

        使用 pandas ewm，adjust=False（Wilder 风格递推）。

        Parameters
        ----------
        series : pd.Series
            价格序列
        period : int
            span 周期

        Returns
        -------
        pd.Series
            EMA 序列
        """
        return calc_ema(series, period)

    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        MACD 指标（异同移动平均线）。

        macd_line  = EMA(fast) - EMA(slow)
        signal_line = EMA(macd_line, signal)
        histogram   = macd_line - signal_line

        Parameters
        ----------
        close  : pd.Series
        fast   : int, 快线周期，默认 12
        slow   : int, 慢线周期，默认 26
        signal : int, 信号线平滑周期，默认 9

        Returns
        -------
        pd.DataFrame
            列：macd_line, signal_line, histogram
        """
        macd_line, signal_line, histogram = calc_macd(close, fast, slow, signal)
        return pd.DataFrame(
            {"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram},
            index=close.index,
        )

    @staticmethod
    def donchian_channel(
        high: pd.Series,
        low: pd.Series,
        period: int = 20,
    ) -> pd.DataFrame:
        """
        唐奇安通道（Donchian Channel）。

        upper  = period 日最高价的滚动最大值
        lower  = period 日最低价的滚动最小值
        middle = (upper + lower) / 2

        常用于突破策略：价格突破 upper → 做多，跌破 lower → 做空。

        Parameters
        ----------
        high   : pd.Series  最高价
        low    : pd.Series  最低价
        period : int        回看周期，默认 20

        Returns
        -------
        pd.DataFrame
            列：upper, lower, middle；前 period-1 行为 NaN
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2.0
        return pd.DataFrame(
            {"upper": upper, "lower": lower, "middle": middle},
            index=high.index,
        )

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        ADX（平均趋向指数）—— 衡量趋势强度，不区分方向。

        ADX > 25：趋势明显；ADX > 50：强趋势；ADX < 20：震荡市。

        Parameters
        ----------
        high, low, close : pd.Series  OHLCV 中的 HLC 三条序列
        period : int                  Wilder 平滑窗口，默认 14

        Returns
        -------
        pd.Series
            ADX 序列，值域 [0, 100]
        """
        return calc_adx(high, low, close, period)
