"""
volatility_ind.py
-----------------
职责：波动率类技术指标计算。

注意：本模块计算基于价格的技术指标（ATR、布林带），
与 models/volatility/ 中基于统计模型的波动率预测不同。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    ATR（真实波幅）—— 衡量价格波动范围（点位）。

    True Range = max(H-L, |H-Prev_C|, |L-Prev_C|)
    ATR = Wilder EMA(TR, window)  [alpha = 1/window]
    """
    close_prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / window
    return tr.ewm(alpha=alpha, adjust=False).mean()


def calc_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    布林带（Bollinger Bands）。

    Returns
    -------
    (upper_band, middle_band, lower_band)
    - upper  = SMA + num_std × rolling_std
    - middle = SMA
    - lower  = SMA - num_std × rolling_std
    """
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def calc_historical_vol(
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """
    历史波动率（基于收盘价对数收益率的滚动标准差）。

    Returns
    -------
    pd.Series
        历史波动率（年化小数，若 annualize=True）

    Notes
    -----
    对更精确的 RV 估计请使用 models.volatility.compute_realized_vol。
    """
    log_returns = np.log(close / close.shift(1))
    hv = log_returns.rolling(window=window).std()
    if annualize:
        hv = hv * np.sqrt(252)
    return hv


# ======================================================================
# VolatilityIndicators 类：面向策略层的面向对象接口
# ======================================================================

class VolatilityIndicators:
    """
    波动率类技术指标计算。

    所有方法为静态方法，输入输出均为 pd.Series / pd.DataFrame。
    """

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        ATR（Average True Range，真实波幅均值）。

        True Range = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
        ATR = Wilder EMA(TR, period)  [alpha = 1/period]

        用途：仓位管理（波动率目标策略用 ATR 计算每手风险）。

        Parameters
        ----------
        high, low, close : pd.Series
        period : int, 默认 14

        Returns
        -------
        pd.Series
            ATR 序列（与 close 同索引）
        """
        return calc_atr(high, low, close, period)

    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """
        布林带（Bollinger Bands）。

        middle = SMA(close, period)
        upper  = middle + num_std × rolling_std
        lower  = middle - num_std × rolling_std
        bandwidth = (upper - lower) / middle  （相对带宽）

        Parameters
        ----------
        close    : pd.Series
        period   : int, 均值和标准差的回看窗口，默认 20
        num_std  : float, 标准差倍数，默认 2.0

        Returns
        -------
        pd.DataFrame
            列：middle, upper, lower, bandwidth
        """
        upper, middle, lower = calc_bollinger_bands(close, period, num_std)
        bandwidth = (upper - lower) / middle.replace(0, np.nan)
        return pd.DataFrame(
            {"middle": middle, "upper": upper, "lower": lower, "bandwidth": bandwidth},
            index=close.index,
        )

    @staticmethod
    def keltner_channel(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ema_period: int = 20,
        atr_period: int = 14,
        multiplier: float = 2.0,
    ) -> pd.DataFrame:
        """
        Keltner Channel（肯特纳通道）。

        middle = EMA(close, ema_period)
        upper  = middle + multiplier × ATR(atr_period)
        lower  = middle - multiplier × ATR(atr_period)

        比布林带更稳定（基于 ATR 而非价格标准差），常用于过滤布林带假突破。

        Parameters
        ----------
        high, low, close : pd.Series
        ema_period : int, EMA 中轨周期，默认 20
        atr_period : int, ATR 计算周期，默认 14
        multiplier : float, ATR 倍数，默认 2.0

        Returns
        -------
        pd.DataFrame
            列：middle, upper, lower
        """
        from .trend import calc_ema
        middle = calc_ema(close, ema_period)
        atr = calc_atr(high, low, close, atr_period)
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        return pd.DataFrame(
            {"middle": middle, "upper": upper, "lower": lower},
            index=close.index,
        )
