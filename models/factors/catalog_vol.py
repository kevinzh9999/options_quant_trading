"""
Layer 1: 波动率类因子（V分替代候选）
====================================
当前系统 V分: ratio = ATR_short(5) / ATR_long(40) → VolATRRatio
V分是逆向指标：ratio低=蓄势待发=高分
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.factors.base import Factor
from models.factors.operators import (
    atr, true_range, ts_mean, ts_stddev, returns, log, bollinger_band,
)


class VolATRRatio(Factor):
    """当前V分使用的因子：ATR短期/长期比值。"""
    category = "volatility"

    def __init__(self, short=5, long=40):
        self.short = short
        self.long = long

    @property
    def name(self):
        return f"vol_atr_ratio_{self.short}_{self.long}"

    def compute_series(self, bar_5m, **kwargs):
        atr_s = atr(bar_5m['high'], bar_5m['low'], bar_5m['close'], self.short)
        atr_l = atr(bar_5m['high'], bar_5m['low'], bar_5m['close'], self.long)
        return atr_s / atr_l.replace(0, np.nan)


class VolATRTrend(Factor):
    """ATR变化趋势：ATR在扩张还是收缩。
    捕捉波动率的方向（V分只看水平）。
    """
    category = "volatility"

    def __init__(self, short=3, long=8):
        self.short = short
        self.long = long

    @property
    def name(self):
        return f"vol_atr_trend_{self.short}_{self.long}"

    def compute_series(self, bar_5m, **kwargs):
        tr = true_range(bar_5m['high'], bar_5m['low'], bar_5m['close'])
        atr_now = ts_mean(tr, self.short)
        atr_prev = ts_mean(tr, self.long)
        return (atr_now - atr_prev) / atr_prev.replace(0, np.nan)


class VolParkinson(Factor):
    """Parkinson波动率：只用High-Low估计（理论效率是Close-Close的5倍）。"""
    category = "volatility"

    def __init__(self, lookback=20):
        self.lookback = lookback

    @property
    def name(self):
        return f"vol_parkinson_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        hl_ratio = log(bar_5m['high'] / bar_5m['low'].replace(0, np.nan))
        return (hl_ratio ** 2).rolling(self.lookback).mean().apply(
            lambda x: np.sqrt(x / (4 * np.log(2)))
        )


class VolReturnStd(Factor):
    """收益率标准差比值：短期std / 长期std。"""
    category = "volatility"

    def __init__(self, short=10, long=40):
        self.short = short
        self.long = long

    @property
    def name(self):
        return f"vol_ret_std_{self.short}_{self.long}"

    def compute_series(self, bar_5m, **kwargs):
        ret = returns(bar_5m['close'], 1)
        std_s = ts_stddev(ret, self.short)
        std_l = ts_stddev(ret, self.long)
        return std_s / std_l.replace(0, np.nan)


class VolBBWidth(Factor):
    """布林带宽度：(上轨-下轨)/中轨。"""
    category = "volatility"

    def __init__(self, n=20, k=2.0):
        self.n = n
        self.k = k

    @property
    def name(self):
        return f"vol_bb_width_{self.n}"

    def compute_series(self, bar_5m, **kwargs):
        _, _, _, width, _ = bollinger_band(bar_5m['close'], self.n, self.k)
        return width
