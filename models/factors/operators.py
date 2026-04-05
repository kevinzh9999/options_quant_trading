"""
Layer 0: Operators（算子库）
============================
标准化的数据变换函数，所有因子的构建积木。
全部向量化计算，一次算完整个时间序列。

命名参考 101 Formulaic Alphas 论文。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 时间序列算子 ──────────────────────────────────────────

def delay(series: pd.Series, n: int) -> pd.Series:
    """n期前的值。"""
    return series.shift(n)


def delta(series: pd.Series, n: int) -> pd.Series:
    """当前值 - n期前的值。"""
    return series - series.shift(n)


def returns(series: pd.Series, n: int = 1) -> pd.Series:
    """n期收益率 = (当前 - n期前) / n期前。"""
    return series.pct_change(n)


def ts_max(series: pd.Series, n: int) -> pd.Series:
    """过去n期最大值。"""
    return series.rolling(n).max()


def ts_min(series: pd.Series, n: int) -> pd.Series:
    """过去n期最小值。"""
    return series.rolling(n).min()


def ts_argmax(series: pd.Series, n: int) -> pd.Series:
    """过去n期中最大值出现的位置（距今多少期）。"""
    return series.rolling(n).apply(lambda x: x.argmax(), raw=True)


def ts_argmin(series: pd.Series, n: int) -> pd.Series:
    """过去n期中最小值出现的位置。"""
    return series.rolling(n).apply(lambda x: x.argmin(), raw=True)


def ts_rank(series: pd.Series, n: int) -> pd.Series:
    """当前值在过去n期中的分位数（0-1）。"""
    return series.rolling(n).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
    )


def ts_stddev(series: pd.Series, n: int) -> pd.Series:
    """过去n期标准差。"""
    return series.rolling(n).std()


def ts_mean(series: pd.Series, n: int) -> pd.Series:
    """过去n期均值。"""
    return series.rolling(n).mean()


def ts_sum(series: pd.Series, n: int) -> pd.Series:
    """过去n期累计和。"""
    return series.rolling(n).sum()


def ts_corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    """过去n期的滚动Pearson相关系数。"""
    return x.rolling(n).corr(y)


def ts_covariance(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    """过去n期的滚动协方差。"""
    return x.rolling(n).cov(y)


def ts_skewness(series: pd.Series, n: int) -> pd.Series:
    """过去n期的偏度。"""
    return series.rolling(n).skew()


def ts_kurtosis(series: pd.Series, n: int) -> pd.Series:
    """过去n期的峰度。"""
    return series.rolling(n).kurt()


def ts_product(series: pd.Series, n: int) -> pd.Series:
    """过去n期的连乘积。"""
    return series.rolling(n).apply(lambda x: x.prod(), raw=True)


# ── 加权/衰减算子 ────────────────────────────────────────

def decay_linear(series: pd.Series, n: int) -> pd.Series:
    """线性衰减加权均值（近期权重线性递增）。"""
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return series.rolling(n).apply(lambda x: np.dot(x, weights), raw=True)


def decay_exp(series: pd.Series, span: int) -> pd.Series:
    """指数衰减均值（EMA）。"""
    return series.ewm(span=span).mean()


# ── 截面算子 ─────────────────────────────────────────────

def rank(series: pd.Series) -> pd.Series:
    """时序排名标准化到 0-1。"""
    return series.rank(pct=True)


def cross_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面排名：每个时间点，对所有品种排名（0-1）。"""
    return df.rank(axis=1, pct=True)


def scale(series: pd.Series, target: float = 1.0) -> pd.Series:
    """缩放使绝对值之和 = target。"""
    s = series.abs().sum()
    return series * target / s if s > 0 else series


def normalize(series: pd.Series, n: int) -> pd.Series:
    """Z-Score 标准化（过去n期的均值和标准差）。"""
    mean = ts_mean(series, n)
    std = ts_stddev(series, n)
    return (series - mean) / std.replace(0, np.nan)


# ── 数学/逻辑算子 ────────────────────────────────────────

def sign(series: pd.Series) -> pd.Series:
    """符号函数：正=1，负=-1，零=0。"""
    return np.sign(series)


def log(series: pd.Series) -> pd.Series:
    """自然对数。"""
    return np.log(series.clip(lower=1e-10))


def abs_(series: pd.Series) -> pd.Series:
    """绝对值。"""
    return series.abs()


def max_(a, b) -> pd.Series:
    """逐元素取大值。"""
    return np.maximum(a, b)


def min_(a, b) -> pd.Series:
    """逐元素取小值。"""
    return np.minimum(a, b)


def clamp(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """限幅。"""
    return series.clip(lower=lower, upper=upper)


def if_else(condition: pd.Series, true_val, false_val) -> pd.Series:
    """条件选择。"""
    return pd.Series(np.where(condition, true_val, false_val), index=condition.index)


# ── K线结构算子 ──────────────────────────────────────────

def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """典型价格 = (H+L+C)/3。"""
    return (high + low + close) / 3.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真实波幅 = max(H-L, |H-prevC|, |L-prevC|)。"""
    prev_c = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low - prev_c).abs()
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """平均真实波幅（N期ATR）。"""
    tr = true_range(high, low, close)
    return ts_mean(tr, n)


def body_ratio(open_: pd.Series, high: pd.Series,
               low: pd.Series, close: pd.Series) -> pd.Series:
    """K线实体占比 = (C-O)/(H-L)。范围 -1 到 +1。"""
    return (close - open_) / (high - low + 1e-10)


def vwap_cumulative(typical: pd.Series, volume: pd.Series) -> pd.Series:
    """日内累计 VWAP（注意：需要按日分组）。"""
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def bollinger_band(close: pd.Series, n: int = 20, k: float = 2.0):
    """布林带。返回 (upper, middle, lower, width, %b)。"""
    middle = ts_mean(close, n)
    std = ts_stddev(close, n)
    upper = middle + k * std
    lower = middle - k * std
    width = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper, middle, lower, width, pct_b


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI（相对强弱指标）。"""
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def linreg_slope(series: pd.Series, n: int) -> pd.Series:
    """过去n期线性回归斜率（标准化为收益率）。"""
    def _slope(vals):
        x = np.arange(len(vals))
        slope = np.polyfit(x, vals, 1)[0]
        return slope / vals[-1] if vals[-1] != 0 else 0
    return series.rolling(n).apply(_slope, raw=True)
