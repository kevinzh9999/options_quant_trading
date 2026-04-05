"""
Layer 1: 日线级别因子（多时间尺度）
====================================
用日线特征预测当日日内振幅——比开盘30min等待更早（盘前可知）。

215天研究结论：日内振幅是策略PnL最强预测因子(r=0.43)。
如果能盘前预测振幅，可以在开盘前就决定今天的交易强度。

数据源：index_daily（2015~2026），映射到每根5分钟bar。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.factors.base import Factor
from models.factors.operators import ts_stddev, ts_mean, ts_max, ts_min, returns


class DailyBBWidth(Factor):
    """日线布林带宽度。

    215天波段研究发现：长趋势入场时BB width=3.71% vs 短趋势2.51%。
    高BB width = 近期波动大 = 日内策略有利环境。
    """
    category = "daily"

    def __init__(self, n=20):
        self.n = n

    @property
    def name(self):
        return f"daily_bb_width_{self.n}"

    def compute_series(self, bar_5m, daily=None, **kwargs):
        if daily is None or 'bb_width' not in bar_5m.columns:
            return pd.Series(np.nan, index=bar_5m.index)
        return bar_5m['bb_width']


class DailyRangeMA(Factor):
    """过去N天的平均日线振幅。

    最直接的振幅预测因子——昨天振幅大，今天往往也大（波动率聚集效应）。
    """
    category = "daily"

    def __init__(self, n=5):
        self.n = n

    @property
    def name(self):
        return f"daily_range_ma_{self.n}"

    def compute_series(self, bar_5m, **kwargs):
        if 'daily_range' not in bar_5m.columns:
            return pd.Series(np.nan, index=bar_5m.index)
        return bar_5m['daily_range']


class DailyConsecDays(Factor):
    """连续同方向天数。

    连续涨/跌天数多 = 趋势中 = 日内策略不利（低振幅单边）。
    连续少 = 震荡 = 日内策略有利。
    """
    category = "daily"

    @property
    def name(self):
        return "daily_consec"

    def compute_series(self, bar_5m, **kwargs):
        if 'consec_days' not in bar_5m.columns:
            return pd.Series(np.nan, index=bar_5m.index)
        return bar_5m['consec_days']


class DailyGapSize(Factor):
    """隔夜gap绝对值（今开-昨收）/昨收。

    大gap = 高波动日的强信号（开盘就有方向），日内策略可能有利。
    """
    category = "daily"

    @property
    def name(self):
        return "daily_gap_size"

    def compute_series(self, bar_5m, **kwargs):
        if 'gap_pct' not in bar_5m.columns:
            return pd.Series(np.nan, index=bar_5m.index)
        return bar_5m['gap_pct'].abs()
