"""
Layer 1: 结构类因子（B分替代候选）
====================================
覆盖当前 B分 的布林带突破逻辑以及K线内部结构。
"""
from __future__ import annotations

import pandas as pd

from models.factors.base import Factor
from models.factors.operators import (
    bollinger_band, body_ratio, ts_max, ts_min, rsi,
)


class BollBreakout(Factor):
    """布林带突破强度：%B = (close-lower)/(upper-lower)。
    对应当前系统B分。%B>1=在上轨以上，%B<0=在下轨以下。
    """
    category = "structure"

    def __init__(self, n=20, k=2.0):
        self.n = n
        self.k = k

    @property
    def name(self):
        return f"boll_breakout_{self.n}"

    def compute_series(self, bar_5m, **kwargs):
        _, _, _, _, pct_b = bollinger_band(bar_5m['close'], self.n, self.k)
        return pct_b


class BodyRatioFactor(Factor):
    """K线实体占比 = (C-O)/(H-L)。范围-1到+1。"""
    category = "structure"

    @property
    def name(self):
        return "body_ratio"

    def compute_series(self, bar_5m, **kwargs):
        return body_ratio(bar_5m['open'], bar_5m['high'],
                         bar_5m['low'], bar_5m['close'])


class PricePosition(Factor):
    """价格在N期范围中的位置（0=最低, 1=最高）。"""
    category = "structure"

    def __init__(self, lookback=240):  # 240 bars = 1 trading day
        self.lookback = lookback

    @property
    def name(self):
        return f"price_pos_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        high_n = ts_max(bar_5m['high'], self.lookback)
        low_n = ts_min(bar_5m['low'], self.lookback)
        return (bar_5m['close'] - low_n) / (high_n - low_n + 1e-10)


class RSIFactor(Factor):
    """RSI指标。"""
    category = "structure"

    def __init__(self, n=14):
        self.n = n

    @property
    def name(self):
        return f"rsi_{self.n}"

    def compute_series(self, bar_5m, **kwargs):
        return rsi(bar_5m['close'], self.n)
