"""
Layer 1: 成交量类因子（Q分替代候选）
====================================
当前系统 Q分: ratio = volume[-1] / mean(volume[-20:]) → VolRatio
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.factors.base import Factor
from models.factors.operators import (
    returns, sign, ts_mean, ts_sum, ts_corr,
)


class QtyRatio(Factor):
    """当前Q分使用的因子：当前bar量 / 20根均量。"""
    category = "volume"

    def __init__(self, lookback=20):
        self.lookback = lookback

    @property
    def name(self):
        return f"qty_ratio_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        return bar_5m['volume'] / ts_mean(bar_5m['volume'], self.lookback).replace(0, np.nan)


class QtyTrend(Factor):
    """成交量趋势：短期均量 / 长期均量。连续放量比单根放量更可靠。"""
    category = "volume"

    def __init__(self, short=3, long=10):
        self.short = short
        self.long = long

    @property
    def name(self):
        return f"qty_trend_{self.short}_{self.long}"

    def compute_series(self, bar_5m, **kwargs):
        vol_s = ts_mean(bar_5m['volume'], self.short)
        vol_l = ts_mean(bar_5m['volume'], self.long)
        return vol_s / vol_l.replace(0, np.nan)


class QtyPriceCorr(Factor):
    """量价相关性：价格和成交量的滚动相关系数。
    正相关=量价齐升（健康趋势），负相关=量价背离。
    101 Alphas: Alpha#6 = -ts_corr(open, volume, 10)
    """
    category = "volume"

    def __init__(self, lookback=10):
        self.lookback = lookback

    @property
    def name(self):
        return f"qty_price_corr_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        return ts_corr(bar_5m['close'], bar_5m['volume'], self.lookback)


class QtySignedFlow(Factor):
    """带方向的资金流：sign(return) * volume。
    粗略估计主动买入/卖出（没有tick数据时的OFI近似）。
    """
    category = "volume"

    def __init__(self, lookback=10):
        self.lookback = lookback

    @property
    def name(self):
        return f"qty_signed_flow_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        ret = returns(bar_5m['close'], 1)
        signed_vol = sign(ret) * bar_5m['volume']
        total_vol = ts_sum(bar_5m['volume'], self.lookback)
        return ts_sum(signed_vol, self.lookback) / total_vol.replace(0, np.nan)
