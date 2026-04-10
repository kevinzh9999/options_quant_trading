"""
原子因子层 — 纯计算，无交易逻辑。

每个函数接收市场数据，返回一个数值或状态。
Entry策略和Exit策略都基于这些原子因子的组合。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 价格动量
# ---------------------------------------------------------------------------

def momentum(close: np.ndarray, lookback: int) -> float:
    """价格动量：(close[-1] - close[-lb-1]) / close[-lb-1]。

    Returns: 百分比变化（正=上涨，负=下跌），数据不足返回0.0。
    """
    if len(close) < lookback + 1 or close[-lookback - 1] == 0:
        return 0.0
    return (close[-1] - close[-lookback - 1]) / close[-lookback - 1]


def momentum_direction(mom: float) -> str:
    """动量方向：LONG/SHORT/空。"""
    if mom > 0:
        return "LONG"
    elif mom < 0:
        return "SHORT"
    return ""


def amplitude(close: np.ndarray, n_bars: int = 48) -> float:
    """振幅：最近n_bars的(max-min)/first。"""
    if len(close) < 2:
        return 0.0
    recent = close[-min(n_bars, len(close)):]
    if recent[0] <= 0:
        return 0.0
    return (max(recent) - min(recent)) / recent[0]


# ---------------------------------------------------------------------------
# 布林带
# ---------------------------------------------------------------------------

def boll_params(close_series: pd.Series, period: int = 20) -> Tuple[float, float]:
    """布林带参数：(mid, std)。数据不足返回(nan, nan)。"""
    if len(close_series) < period:
        return float("nan"), float("nan")
    mid = float(close_series.iloc[-period:].mean())
    std = float(close_series.iloc[-period:].std())
    return mid, std


def boll_zone(price: float, mid: float, std: float) -> str:
    """布林带zone判定。

    Returns: ABOVE_UPPER / UPPER_ZONE / MID_UPPER / MID_LOWER / LOWER_ZONE / BELOW_LOWER / ""
    """
    if np.isnan(mid) or std <= 0:
        return ""
    upper = mid + 2 * std
    lower = mid - 2 * std
    if price >= upper:
        return "ABOVE_UPPER"
    elif price >= mid + std:
        return "UPPER_ZONE"
    elif price >= mid:
        return "MID_UPPER"
    elif price >= mid - std:
        return "MID_LOWER"
    elif price >= lower:
        return "LOWER_ZONE"
    else:
        return "BELOW_LOWER"


def boll_width(std: float) -> float:
    """布林带宽度 = 4 * std。"""
    return 4 * std if std > 0 else 0.0


# ---------------------------------------------------------------------------
# 波动率
# ---------------------------------------------------------------------------

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int) -> float:
    """Average True Range。"""
    n = len(high)
    if n < period + 1:
        return 0.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    return float(np.mean(tr[-period:]))


def atr_ratio(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              short_period: int = 5, long_period: int = 40) -> float:
    """ATR短/长比率。>1=波动收缩, <1=波动扩张。"""
    atr_s = atr(high, low, close, short_period)
    atr_l = atr(high, low, close, long_period)
    if atr_l <= 0 or atr_s <= 0:
        return 1.0
    return atr_s / atr_l


# ---------------------------------------------------------------------------
# 成交量
# ---------------------------------------------------------------------------

def volume_percentile(cur_vol: float, hist_vols: list) -> float:
    """成交量在历史同时段的分位数（0.0-1.0）。"""
    if not hist_vols or len(hist_vols) < 5:
        return -1.0  # 数据不足
    return sum(1 for v in hist_vols if v <= cur_vol) / len(hist_vols)


def volume_ratio(cur_vol: float, avg_vol: float) -> float:
    """成交量与均值的比率。"""
    if avg_vol <= 0:
        return 1.0
    return cur_vol / avg_vol


# ---------------------------------------------------------------------------
# K线形态
# ---------------------------------------------------------------------------

def narrow_range(bar_5m: pd.DataFrame, n_bars: int = 3,
                 boll_std: float = 0.0) -> float:
    """最近n根bar的range占布林带宽度的比例。

    Returns: range / boll_width, 值越小=越收窄。boll_std=0时返回-1。
    """
    if bar_5m is None or len(bar_5m) < n_bars or boll_std <= 0:
        return -1.0
    last_h = bar_5m["high"].astype(float).iloc[-n_bars:]
    last_l = bar_5m["low"].astype(float).iloc[-n_bars:]
    total_range = float(last_h.max() - last_l.min())
    bw = boll_width(boll_std)
    if bw <= 0:
        return -1.0
    return total_range / bw


def price_trending(bar_5m: pd.DataFrame, n_bars: int = 3,
                   boll_std: float = 0.0, direction: str = "") -> bool:
    """最近n根bar是否仍在趋势中（close变化>布林带宽度的5%）。"""
    if bar_5m is None or len(bar_5m) < n_bars or boll_std <= 0:
        return False
    last_c = bar_5m["close"].astype(float).iloc[-n_bars:]
    close_change = float(last_c.iloc[-1]) - float(last_c.iloc[0])
    bw = boll_width(boll_std)
    if direction == "LONG":
        return close_change > bw * 0.05
    elif direction == "SHORT":
        return close_change < -bw * 0.05
    return abs(close_change) > bw * 0.05


def breakout_prev_range(close: np.ndarray, high: np.ndarray,
                        low: np.ndarray, n_bars: int = 5) -> Tuple[bool, str]:
    """是否突破前n根bar的high/low。

    Returns: (is_breakout, direction)
    """
    if len(close) < n_bars + 1:
        return False, ""
    prev_high = float(high[-n_bars - 1:-1].max())
    prev_low = float(low[-n_bars - 1:-1].min())
    cur = float(close[-1])
    if cur > prev_high:
        return True, "LONG"
    elif cur < prev_low:
        return True, "SHORT"
    return False, ""


# ---------------------------------------------------------------------------
# 持仓状态
# ---------------------------------------------------------------------------

def hold_time(entry_time_utc: str, current_time_utc: str) -> int:
    """持仓时间（分钟）。"""
    try:
        h1, m1 = int(entry_time_utc[:2]), int(entry_time_utc[3:5])
        h2, m2 = int(current_time_utc[:2]), int(current_time_utc[3:5])
        return (h2 * 60 + m2) - (h1 * 60 + m1)
    except Exception:
        return 0


def pnl_pct(current_price: float, entry_price: float,
            direction: str) -> float:
    """持仓盈亏百分比（正=盈利，负=亏损）。"""
    if entry_price <= 0:
        return 0.0
    if direction == "LONG":
        return (current_price - entry_price) / entry_price
    else:
        return (entry_price - current_price) / entry_price


def trailing_drawdown(current_price: float, extreme_price: float,
                      direction: str) -> float:
    """从极值回撤的百分比（正值=有回撤）。"""
    if direction == "LONG" and extreme_price > 0:
        return (extreme_price - current_price) / extreme_price
    elif direction == "SHORT" and extreme_price > 0:
        return (current_price - extreme_price) / extreme_price
    return 0.0


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(close: np.ndarray, period: int = 14) -> float:
    """RSI指标（0-100）。"""
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
