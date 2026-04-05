"""
Layer 1: 价格/动量类因子（M分替代候选）
========================================
覆盖当前 M分 的所有计算变体。

当前系统 M分:
  mom_5m = (close[-1] - close[-13]) / close[-13]  → MomSimple(12)
  方向需要5min和15min一致                           → MomMultiScale
"""
from __future__ import annotations

import pandas as pd

from models.factors.base import Factor
from models.factors.operators import (
    returns, decay_linear, decay_exp, ts_rank, sign, linreg_slope, ts_corr,
)


class MomSimple(Factor):
    """当前M分使用的因子：简单N根bar收益率。
    对应代码：mom_5m = (close[-1] - close[-13]) / close[-13]
    """
    category = "momentum"
    description = "简单N期收益率，当前系统M分的基础"

    def __init__(self, lookback=12):
        self.lookback = lookback

    @property
    def name(self):
        return f"mom_simple_{self.lookback}"

    @property
    def params(self):
        return {"lookback": self.lookback}

    def compute_series(self, bar_5m, **kwargs):
        return returns(bar_5m['close'], self.lookback)


class MomEMA(Factor):
    """EMA交叉度：快EMA vs 慢EMA 的偏离。
    优势：对近期价格赋予更多权重，比简单return更平滑。
    """
    category = "momentum"

    def __init__(self, fast=5, slow=20):
        self.fast = fast
        self.slow = slow

    @property
    def name(self):
        return f"mom_ema_{self.fast}_{self.slow}"

    def compute_series(self, bar_5m, **kwargs):
        ema_f = decay_exp(bar_5m['close'], self.fast)
        ema_s = decay_exp(bar_5m['close'], self.slow)
        return (ema_f - ema_s) / ema_s


class MomLinReg(Factor):
    """线性回归斜率：用最小二乘法拟合N根bar的趋势线，取斜率。
    优势：不受lookback两端异常值影响。
    """
    category = "momentum"

    def __init__(self, lookback=12):
        self.lookback = lookback

    @property
    def name(self):
        return f"mom_linreg_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        return linreg_slope(bar_5m['close'], self.lookback)


class MomDecayLinear(Factor):
    """线性衰减加权动量：近期bar权重线性递增。
    101 Alphas 灵感：decay_linear 是高频使用的算子。
    """
    category = "momentum"

    def __init__(self, lookback=12):
        self.lookback = lookback

    @property
    def name(self):
        return f"mom_decay_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        ret = returns(bar_5m['close'], 1)
        return decay_linear(ret, self.lookback)


class MomRank(Factor):
    """当前价格在过去N期中的百分位排名。
    101 Alphas: ts_rank 是核心算子。
    天然标准化到0-1。
    """
    category = "momentum"

    def __init__(self, lookback=48):
        self.lookback = lookback

    @property
    def name(self):
        return f"mom_rank_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        return ts_rank(bar_5m['close'], self.lookback)


class MomMultiScale(Factor):
    """多尺度动量：5min + 15min 动量的加权组合。
    对应当前系统：M分要求5min和15min方向一致。
    这里用连续值替代二值判断。
    """
    category = "momentum"

    def __init__(self, fast=6, slow=18):
        self.fast = fast
        self.slow = slow

    @property
    def name(self):
        return f"mom_multiscale_{self.fast}_{self.slow}"

    def compute_series(self, bar_5m, **kwargs):
        mom_fast = returns(bar_5m['close'], self.fast)
        mom_slow = returns(bar_5m['close'], self.slow)
        agreement = sign(mom_fast) * sign(mom_slow)  # +1=一致, -1=矛盾
        return (mom_fast + mom_slow) / 2 * (0.5 + 0.5 * agreement)


class MomRiskAdjusted(Factor):
    """风险调整动量：return / volatility。
    类似Sharpe ratio的日内版本。
    优势：在高波动环境下自动降低信号，低波动环境下增强。
    """
    category = "momentum"

    def __init__(self, mom_lookback=12, vol_lookback=20):
        self.mom_lookback = mom_lookback
        self.vol_lookback = vol_lookback

    @property
    def name(self):
        return f"mom_risk_adj_{self.mom_lookback}_{self.vol_lookback}"

    def compute_series(self, bar_5m, **kwargs):
        from models.factors.operators import ts_stddev
        mom = returns(bar_5m['close'], self.mom_lookback)
        vol = ts_stddev(returns(bar_5m['close'], 1), self.vol_lookback)
        return mom / vol.replace(0, float('nan'))
