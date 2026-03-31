"""
momentum.py
-----------
职责：动量类技术指标计算。

提供：RSI、ROC（变化率）、Stochastic（随机指标）
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI（相对强弱指数）。

    值域 [0, 100]；RSI > 70：超买；RSI < 30：超卖。

    Notes
    -----
    RSI = 100 - 100 / (1 + RS)
    RS = 平均上涨幅度 / 平均下跌幅度（使用 Wilder 平滑：alpha=1/window）
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(100.0)  # avg_loss=0 时 RSI=100


def calc_roc(close: pd.Series, window: int = 10) -> pd.Series:
    """
    ROC（变化率）—— 衡量价格动量（百分比）。

    ROC = (close_t - close_{t-window}) / close_{t-window} × 100
    """
    return (close - close.shift(window)) / close.shift(window) * 100.0


def calc_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    随机指标（KD 指标）。

    值域 [0, 100]；%K > 80：超买；%K < 20：超卖。

    Returns
    -------
    (%K, %D)
        %K = (close - lowest_low) / (highest_high - lowest_low) × 100
        %D = SMA(%K, d_window)
    """
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()

    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100.0 * (close - lowest_low) / denom
    d = k.rolling(window=d_window).mean()

    return k, d


# ======================================================================
# MomentumIndicators 类：面向策略层的面向对象接口
# ======================================================================

class MomentumIndicators:
    """
    动量类技术指标计算。

    所有方法为静态方法，输入输出均为 pd.Series。
    """

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI（相对强弱指数）。

        RSI = 100 - 100 / (1 + RS)
        RS  = Wilder_EMA(gain) / Wilder_EMA(loss)  [alpha=1/period]

        值域 [0, 100]；RSI > 70 超买，RSI < 30 超卖。

        Parameters
        ----------
        close  : pd.Series
        period : int, Wilder 平滑窗口，默认 14

        Returns
        -------
        pd.Series
        """
        return calc_rsi(close, period)

    @staticmethod
    def roc(close: pd.Series, period: int = 12) -> pd.Series:
        """
        ROC（Rate of Change，变动率）。

        ROC = (close_t - close_{t-period}) / close_{t-period} × 100

        Parameters
        ----------
        close  : pd.Series
        period : int, 回看期，默认 12

        Returns
        -------
        pd.Series
            百分比变化率，前 period 行为 NaN
        """
        return calc_roc(close, period)

    @staticmethod
    def momentum_factor(
        close: pd.Series,
        lookback: int = 252,
        skip: int = 21,
    ) -> pd.Series:
        """
        动量因子（学术定义，Jegadeesh & Titman 1993）。

        mom_t = close_{t-skip} / close_{t-lookback} - 1

        跳过最近 skip 天以避免短期反转效应（通常 1 个月 = 21 交易日）。

        Parameters
        ----------
        close    : pd.Series
        lookback : int, 回看总窗口（交易日），默认 252（1年）
        skip     : int, 跳过最近天数，默认 21（1个月）

        Returns
        -------
        pd.Series
            动量得分（浮点，百分比小数形式）；前 lookback 行为 NaN
        """
        return close.shift(skip) / close.shift(lookback) - 1.0
