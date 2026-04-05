"""
Layer 1: 跨品种因子
====================
利用IM/IC/IF/IH之间的相对强弱和联动关系。
当前系统是纯单品种时间序列，跨品种信息完全未使用。

数据源：index_min 4个现货指数（000852/000905/000300/000016）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.factors.base import Factor
from models.factors.operators import returns, ts_corr, ts_stddev, ts_mean, atr


class CrossMomentumSpread(Factor):
    """IM vs IH 超额动量：IM涨且跑赢IH = 中小盘资金流入信号。

    理论：A股大小盘轮动是核心风格因子。当IM跑赢IH时，
    说明资金从大盘蓝筹流向中小盘成长，IM的动量信号更可信。
    """
    category = "cross_asset"

    def __init__(self, lookback=12, other_col='close_IH'):
        self.lookback = lookback
        self.other_col = other_col

    @property
    def name(self):
        sym = self.other_col.split('_')[-1]
        return f"cross_mom_{sym}_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        if self.other_col not in bar_5m.columns:
            return pd.Series(np.nan, index=bar_5m.index)
        ret_self = returns(bar_5m['close'], self.lookback)
        ret_other = returns(bar_5m[self.other_col], self.lookback)
        return ret_self - ret_other


class CrossVolRatio(Factor):
    """IM vs IF 相对波动率：IM波动率/IF波动率。

    高比值说明中小盘波动放大（可能有行情），低比值说明市场平淡。
    """
    category = "cross_asset"

    def __init__(self, lookback=20, other_col='close_IF'):
        self.lookback = lookback
        self.other_col = other_col

    @property
    def name(self):
        sym = self.other_col.split('_')[-1]
        return f"cross_vol_{sym}_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        if self.other_col not in bar_5m.columns:
            return pd.Series(np.nan, index=bar_5m.index)
        std_self = ts_stddev(returns(bar_5m['close'], 1), self.lookback)
        std_other = ts_stddev(returns(bar_5m[self.other_col], 1), self.lookback)
        return std_self / std_other.replace(0, np.nan)


class CrossCorrelation(Factor):
    """IM-IH 滚动相关性。

    相关性下降 = 风格分化（大小盘走势脱钩），可能是IM独立行情的信号。
    """
    category = "cross_asset"

    def __init__(self, lookback=48, other_col='close_IH'):
        self.lookback = lookback
        self.other_col = other_col

    @property
    def name(self):
        sym = self.other_col.split('_')[-1]
        return f"cross_corr_{sym}_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        if self.other_col not in bar_5m.columns:
            return pd.Series(np.nan, index=bar_5m.index)
        return ts_corr(bar_5m['close'], bar_5m[self.other_col], self.lookback)


class CrossRank(Factor):
    """IM在4品种中的动量排名（1=最强，4=最弱）。

    101 Alphas核心思想：rank()。如果IM动量排第一，说明资金主线在中小盘。
    """
    category = "cross_asset"

    def __init__(self, lookback=12):
        self.lookback = lookback

    @property
    def name(self):
        return f"cross_rank_{self.lookback}"

    def compute_series(self, bar_5m, **kwargs):
        cols = ['close', 'close_IF', 'close_IH', 'close_IC']
        available = [c for c in cols if c in bar_5m.columns]
        if len(available) < 2:
            return pd.Series(np.nan, index=bar_5m.index)

        rets = pd.DataFrame({c: returns(bar_5m[c], self.lookback) for c in available})
        # IM的排名（0=最弱，1=最强）
        ranks = rets.rank(axis=1, pct=True)
        return ranks['close']  # IM's rank among all
