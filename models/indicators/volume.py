"""
volume.py
---------
职责：成交量类技术指标计算。

提供：OBV、VWAP
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV（能量潮）—— 将成交量与价格方向结合。

    若收盘价上涨，OBV += volume；下跌，OBV -= volume；不变，OBV 不变。

    Returns
    -------
    pd.Series
        OBV 累积序列
    """
    price_change = close.diff()
    signed_vol = pd.Series(
        np.where(price_change > 0, volume,
                 np.where(price_change < 0, -volume, 0.0)),
        index=close.index,
    )
    return signed_vol.cumsum()


def calc_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: Optional[int] = None,
) -> pd.Series:
    """
    VWAP（成交量加权平均价）。

    Parameters
    ----------
    window : int, optional
        滚动窗口（交易日）。
        None → 累积 VWAP（从序列起点累计）；
        指定整数 → 滚动 VWAP（日线级别分析）。

    Notes
    -----
    典型价格 = (high + low + close) / 3
    VWAP = sum(典型价格 × 成交量) / sum(成交量)
    """
    typical_price = (high + low + close) / 3.0
    tp_volume = typical_price * volume

    if window is None:
        vwap = tp_volume.cumsum() / volume.cumsum()
    else:
        vwap = tp_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()

    return vwap
