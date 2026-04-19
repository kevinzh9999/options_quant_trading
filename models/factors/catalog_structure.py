"""
Layer 1: 结构类因子（B分替代候选）
====================================
覆盖当前 B分 的布林带突破逻辑以及K线内部结构。
"""
from __future__ import annotations

import pandas as pd

from models.factors.base import Factor
from models.factors.operators import (
    bollinger_band, body_ratio, ts_max, ts_min, ts_mean, ts_stddev, rsi,
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

    【评估结论：归档不上线 — 2026-04-11】
    全局Daily IC强(IM=0.111, IC=0.089)，但分组验证后实用强度不足：
    - 高振幅日Pos-Neg均值差: IM=-3.2bps IC=-5.3bps（<10bps实用门槛）
    - 两品种在高振幅日都是趋势延续模式，不是反转
    - ME对照：exit后反向MFE/MAE=0.66-0.78，反向胜率35-45%，不支持反转假设
    - 结论：Daily IC数值被全局drift高估，因子无实盘价值
    保留代码作为研究记录，避免未来重复评估。

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


class HorizontalReversalSimple(Factor):
    """简化版横盘反转因子。完全忠于主观观察，没有额外条件。

    【评估结论：观察池待定 — 2026-04-11】
    900天IM验证结果：
    - 低振幅日Pos-Neg收益差在fwd=2-24全部为正，单调递增
    - fwd=12-24形成+8.7~+9.6bps平坦高原
    - 相比219天峰值+18.8bps有明显衰减，但形态完整保留
    - 状态：观察池待定，非归档
    - 不进入实盘（边际收益扣除成本后太薄），也不放弃（形态稳健）
    - 重新评估触发条件：低振幅日样本>=1500笔，或启动regime-aware策略时
    方法论教训：见docs/factor_research_lessons.md原则9

    逻辑：
    1. 过去K根bar的前置动量判断趋势方向
    2. 最近N根bar没有创出趋势方向的新极值→停滞
    3. 停滞→预测反转

    输出：+1=下跌后横盘预测反转上涨, -1=上涨后横盘预测反转下跌, 0=无信号
    """
    category = "structure"

    def __init__(self, K=12, N=3):
        self.K = K
        self.N = N

    @property
    def name(self):
        return f"hr_simple_K{self.K}_N{self.N}"

    @property
    def params(self):
        return {"K": self.K, "N": self.N}

    def compute_series(self, bar_5m, **kwargs):
        close = bar_5m['close']
        high = bar_5m['high']
        low = bar_5m['low']

        prior_mom = close - close.shift(self.K)
        recent_high = ts_max(high, self.N)
        recent_low = ts_min(low, self.N)
        ref_high = high.shift(self.N)
        ref_low = low.shift(self.N)

        up_stall = (prior_mom > 0) & (recent_high <= ref_high)
        down_stall = (prior_mom < 0) & (recent_low >= ref_low)

        signal = pd.Series(0.0, index=close.index)
        signal[up_stall] = -1.0
        signal[down_stall] = 1.0
        return signal


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
