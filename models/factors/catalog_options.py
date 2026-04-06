"""
期权因子目录（catalog_options.py）
==================================
期权波动率 / Skew / 期限结构 / Greeks / 情绪类因子。

与 catalog_price/vol/volume 不同：
  - 数据频率：日频（daily_model_output），不是5分钟bar
  - Target：已实现VRP / IV变化（卖方盈亏），不是标的价格return
  - 用途：VRP策略、仓位管理、象限判断

所有因子接受 daily_data (DataFrame)，必须包含对应字段。
"""
from __future__ import annotations

import pandas as pd

from models.factors.base import Factor
from models.factors.operators import (
    abs_, delta, iv_percentile, iv_term_spread, ts_rank, vrp, vrp_regime,
)


# ── 波动率水平类 ────────────────────────────────────────────

class VRPLevel(Factor):
    """VRP绝对值：ATM IV - Blended RV。正值=卖方有利。"""
    name = "vrp_level"
    category = "options_vol"

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        return vrp(d["atm_iv"], d["blended_rv"])


class VRPPercentile(Factor):
    """VRP在过去N日中的分位（0-1）。"""
    name = "vrp_percentile_60"
    category = "options_vol"

    def __init__(self, n: int = 60):
        self.n = n
        self._name = f"vrp_percentile_{n}"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        v = vrp(d["atm_iv"], d["blended_rv"])
        return vrp_regime(v, self.n)


class IVPercentile(Factor):
    """ATM IV的历史分位（0-1）。"""
    name = "iv_percentile"
    category = "options_vol"

    def __init__(self, n: int = 252):
        self.n = n

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        return iv_percentile(d["atm_iv"], self.n)


class IVRVRatio(Factor):
    """IV/RV比值。>1=卖方有利，<1=买方有利。"""
    name = "iv_rv_ratio"
    category = "options_vol"

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        rv = d["blended_rv"].replace(0, float("nan"))
        return d["atm_iv"] / rv


class IVChange(Factor):
    """IV的N日变化速度。正=IV上升。"""
    name = "iv_change_5d"
    category = "options_vol"

    def __init__(self, n: int = 5):
        self.n = n
        self._name = f"iv_change_{n}d"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        return delta(d["atm_iv"], self.n)


# ── 期限结构类 ──────────────────────────────────────────────

class IVTermSpread(Factor):
    """近月IV - 远月IV。正值=倒挂=短期恐慌。"""
    name = "iv_term_spread"
    category = "options_structure"

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        return iv_term_spread(d["iv_near"], d["iv_far"])


# ── Skew类 ──────────────────────────────────────────────────

class SkewRR(Factor):
    """25D Risk Reversal（Put skew）。负值=Put比Call贵=看跌情绪。"""
    name = "skew_rr_25d"
    category = "options_skew"

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        return d["rr_25d"]


class SkewChange(Factor):
    """RR的N日变化速度。"""
    name = "skew_rr_change_5d"
    category = "options_skew"

    def __init__(self, n: int = 5):
        self.n = n
        self._name = f"skew_rr_change_{n}d"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        return delta(d["rr_25d"], self.n)


# ── Greeks类 ────────────────────────────────────────────────

class ThetaVegaRatio(Factor):
    """组合Theta/Vega比值。衡量每单位vega风险收获多少theta。"""
    name = "theta_vega_ratio"
    category = "options_greeks"

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        vega = abs_(d["net_vega"]).replace(0, float("nan"))
        return d["net_theta"] / vega


# ── 情绪类 ──────────────────────────────────────────────────

class PCRVolume(Factor):
    """Put/Call成交量比。>1=看跌情绪偏重（逆向看多信号）。"""
    name = "pcr_volume"
    category = "options_sentiment"

    def compute_series(self, bar_5m=None, daily: pd.DataFrame = None, **kw):
        d = daily if daily is not None else bar_5m
        call_vol = d["call_volume"].replace(0, float("nan"))
        return d["put_volume"] / call_vol
