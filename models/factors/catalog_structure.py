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


class HorizontalReversalFactor(Factor):
    """横盘反转因子：趋势运行后出现连续横盘，预示反向行情。

    reversal_strength = prior_trend × horizontal_score × low_vol_score × (-trend_sign)
    正值=看多反转，负值=看空反转，绝对值=信号强度。

    镜像自 ExitEvaluator 的 MomentumExhaustedCondition，但：
    - 去掉持仓依赖（hold_time、position direction）
    - 加上前置趋势强度
    - 输出连续值而非二元信号
    """
    category = "structure"

    def __init__(self, trend_lookback=12, horizontal_window=3, vol_window=3):
        self.K = trend_lookback      # 前置趋势lookback
        self.N = horizontal_window   # 横盘窗口
        self.M = vol_window          # 低波动窗口

    @property
    def name(self):
        return f"horizontal_reversal_{self.K}_{self.N}_{self.M}"

    @property
    def description(self):
        return (f"横盘反转因子: 前置趋势{self.K}bar + {self.N}bar横盘 + "
                f"{self.M}bar低波动 → 反向信号")

    @property
    def params(self):
        return {"K": self.K, "N": self.N, "M": self.M}

    def compute_series(self, bar_5m, **kwargs):
        close = bar_5m['close']
        high = bar_5m['high']
        low = bar_5m['low']
        n = len(close)

        # 前置趋势强度：K bar前到N bar前的净变化
        # 即 close[-N-1] - close[-K-1] 的方向和幅度
        prior_close = close.shift(self.N)
        prior_start = close.shift(self.K)
        prior_trend = (prior_close - prior_start) / prior_start.replace(0, 1e-10)

        # 横盘score：最近N bar的range / 布林带宽度的倒数
        # range越窄→score越高
        recent_high = ts_max(high, self.N)
        recent_low = ts_min(low, self.N)
        recent_range = recent_high - recent_low

        # 布林带宽度做归一化基准
        boll_std = ts_stddev(close, 20)
        boll_width = 4 * boll_std

        # horizontal_score = 1 - (recent_range / boll_width)，clamp到[0,1]
        range_ratio = recent_range / boll_width.replace(0, 1e-10)
        horizontal_score = (1 - range_ratio).clip(0, 1)

        # 低波动score：最近M bar的平均bar range vs 长期平均
        bar_range = high - low
        recent_avg_range = ts_mean(bar_range, self.M)
        long_avg_range = ts_mean(bar_range, 40)
        vol_ratio = recent_avg_range / long_avg_range.replace(0, 1e-10)
        low_vol_score = (1 - vol_ratio).clip(0, 1)

        # 趋势方向（用于翻转）
        trend_sign = prior_trend.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # 组合：绝对趋势强度 × 横盘程度 × 低波动程度 × 反向
        reversal_strength = prior_trend.abs() * horizontal_score * low_vol_score * (-trend_sign)

        return reversal_strength


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
