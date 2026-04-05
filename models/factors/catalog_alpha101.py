"""
Layer 1: 101 Alphas 经典因子
==============================
从论文中选取与日内动量策略最相关的因子。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.factors.base import Factor
from models.factors.operators import (
    sign, delta, returns, log, ts_corr, ts_stddev, ts_rank, ts_argmax,
    abs_, typical_price, vwap_cumulative,
)


class Alpha001(Factor):
    """Alpha#1: (rank(ts_argmax(sign(delta(close,1))^2, 5)) - 0.5) * (-sign(delta(close,1)))
    含义：如果最近5天有大涨，且今天收跌，做空。
    """
    category = "composite"

    @property
    def name(self):
        return "alpha001"

    def compute_series(self, bar_5m, **kwargs):
        c = bar_5m['close']
        inner = sign(delta(c, 1)) ** 2
        return (ts_rank(ts_argmax(inner, 5), 5) - 0.5) * (-sign(delta(c, 1)))


class Alpha002(Factor):
    """Alpha#2: -ts_corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)
    含义：成交量变化和价格变化的负相关性。
    """
    category = "composite"

    @property
    def name(self):
        return "alpha002"

    def compute_series(self, bar_5m, **kwargs):
        c, o, v = bar_5m['close'], bar_5m['open'], bar_5m['volume']
        d_log_vol = delta(log(v), 2)
        price_move = (c - o) / o
        return -ts_corr(d_log_vol.rank(pct=True), price_move.rank(pct=True), 6)


class Alpha006(Factor):
    """Alpha#6: -ts_corr(open, volume, 10)
    含义：开盘价和成交量的负相关性。简单但有效。
    """
    category = "composite"

    @property
    def name(self):
        return "alpha006"

    def compute_series(self, bar_5m, **kwargs):
        return -ts_corr(bar_5m['open'], bar_5m['volume'], 10)


class Alpha012(Factor):
    """Alpha#12: sign(delta(volume,1)) * (-delta(close,1))
    含义：量增价跌→做多（吸筹），量增价涨→做空（出货）。
    """
    category = "composite"

    @property
    def name(self):
        return "alpha012"

    def compute_series(self, bar_5m, **kwargs):
        return sign(delta(bar_5m['volume'], 1)) * (-delta(bar_5m['close'], 1))


class Alpha018(Factor):
    """Alpha#18: -rank(ts_stddev(abs(close-open), 5) + (close-open) + ts_corr(close,open,10))
    含义：波动率+趋势+开收相关性的组合。
    """
    category = "composite"

    @property
    def name(self):
        return "alpha018"

    def compute_series(self, bar_5m, **kwargs):
        c, o = bar_5m['close'], bar_5m['open']
        part1 = ts_stddev(abs_(c - o), 5)
        part2 = c - o
        part3 = ts_corr(c, o, 10)
        return -(part1 + part2 + part3).rank(pct=True)


class Alpha041(Factor):
    """Alpha#41: ((high * low)^0.5) - vwap
    含义：几何均价 vs VWAP 的偏离。
    """
    category = "composite"

    @property
    def name(self):
        return "alpha041"

    def compute_series(self, bar_5m, **kwargs):
        geo_mean = (bar_5m['high'] * bar_5m['low']) ** 0.5
        tp = typical_price(bar_5m['high'], bar_5m['low'], bar_5m['close'])
        vwap = vwap_cumulative(tp, bar_5m['volume'])
        return geo_mean - vwap


class Alpha101(Factor):
    """Alpha#101: (close - open) / ((high - low) + .001)
    含义：K线实体占比（论文原始定义）。
    """
    category = "composite"

    @property
    def name(self):
        return "alpha101"

    def compute_series(self, bar_5m, **kwargs):
        return (bar_5m['close'] - bar_5m['open']) / (bar_5m['high'] - bar_5m['low'] + 0.001)
