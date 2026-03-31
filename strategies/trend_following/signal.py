"""
signal.py — 趋势信号生成器
基于双均线、唐奇安通道和 ADX 过滤器生成方向性信号。
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from models.indicators.trend import TrendIndicators, calc_adx
from models.indicators.volatility_ind import calc_atr

logger = logging.getLogger(__name__)


class TrendSignalGenerator:
    """
    Trend signal generator using dual MA + Donchian channel + ADX filter.

    Parameters
    ----------
    fast_period : int
        Fast SMA window.
    slow_period : int
        Slow SMA window.
    donchian_period : int
        Donchian channel lookback.
    adx_period : int
        ADX smoothing window.
    adx_threshold : float
        Minimum ADX to confirm trend entry.
    atr_period : int
        ATR smoothing window.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        donchian_period: int = 20,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        atr_period: int = 20,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.donchian_period = donchian_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all trend and volatility indicators.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close (volume optional).

        Returns
        -------
        pd.DataFrame
            Original df plus: sma_fast, sma_slow, donchian_upper,
            donchian_lower, adx, atr, atr_pct
        """
        result = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        result["sma_fast"] = TrendIndicators.sma(close, self.fast_period)
        result["sma_slow"] = TrendIndicators.sma(close, self.slow_period)

        dc = TrendIndicators.donchian_channel(high, low, self.donchian_period)
        # Shift by 1 so today's close is compared against YESTERDAY's N-day max/min high/low.
        # This is the standard Donchian breakout rule: close > prev-period channel high → entry.
        # Without shift, don_upper includes today's high, making close >= don_upper essentially
        # impossible (close ≤ high always, so close >= max_high requires close = today's high =
        # all-time 20-day high simultaneously — triggers only 0.2% of days instead of ~7%).
        result["donchian_upper"] = dc["upper"].shift(1)
        result["donchian_lower"] = dc["lower"].shift(1)

        result["adx"] = calc_adx(high, low, close, self.adx_period)
        result["atr"] = calc_atr(high, low, close, self.atr_period)
        result["atr_pct"] = result["atr"] / close.replace(0, np.nan)

        return result

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def get_signal(
        self,
        df_with_indicators: pd.DataFrame,
        current_position: str = "FLAT",
        entry_price: float = 0.0,
        atr_stop_multiplier: float = 2.5,
    ) -> Dict:
        """
        Generate a trading signal from the latest row of the indicator DataFrame.

        Entry rules:
          LONG  : close >= donchian_upper AND sma_fast > sma_slow AND adx > threshold
          SHORT : close <= donchian_lower AND sma_fast < sma_slow AND adx > threshold

        Exit rules:
          EXIT long  : close < sma_slow
          EXIT short : close > sma_slow

        Stop-loss rules (checked before regular exit):
          STOP long  : close < entry_price - atr_stop_multiplier * atr
          STOP short : close > entry_price + atr_stop_multiplier * atr

        Parameters
        ----------
        df_with_indicators : pd.DataFrame
            Output of compute_indicators().
        current_position : str
            "FLAT", "LONG", or "SHORT".
        entry_price : float
            Entry price of current position (used for stop-loss).
        atr_stop_multiplier : float
            ATR multiple for stop-loss distance.

        Returns
        -------
        dict with keys:
            direction, signal_type, strength, stop_loss_price,
            adx, atr_pct, trend_aligned
        """
        min_rows = max(self.slow_period, self.adx_period) + 2
        if len(df_with_indicators) < min_rows:
            return {
                "direction": "FLAT",
                "signal_type": "HOLD",
                "strength": 0.0,
                "stop_loss_price": 0.0,
                "adx": 0.0,
                "atr_pct": 0.0,
                "trend_aligned": False,
            }

        row = df_with_indicators.iloc[-1]

        close = float(row["close"])
        sma_fast = row.get("sma_fast", float("nan"))
        sma_slow = row.get("sma_slow", float("nan"))
        don_upper = row.get("donchian_upper", float("nan"))
        don_lower = row.get("donchian_lower", float("nan"))
        adx = row.get("adx", float("nan"))
        atr = row.get("atr", float("nan"))
        atr_pct = row.get("atr_pct", float("nan"))

        # Default safe values for NaN
        if np.isnan(adx):
            adx = 0.0
        if np.isnan(atr):
            atr = 0.0
        if np.isnan(atr_pct):
            atr_pct = 0.0

        result = {
            "direction": "FLAT",
            "signal_type": "HOLD",
            "strength": 0.0,
            "stop_loss_price": 0.0,
            "adx": adx,
            "atr_pct": atr_pct,
            "trend_aligned": False,
        }

        # --- Check stop-loss FIRST (highest priority) ---
        if current_position == "LONG" and entry_price > 0 and atr > 0:
            stop_price = entry_price - atr_stop_multiplier * atr
            if close < stop_price:
                result["signal_type"] = "STOP_LOSS"
                result["direction"] = "FLAT"
                result["stop_loss_price"] = stop_price
                return result

        if current_position == "SHORT" and entry_price > 0 and atr > 0:
            stop_price = entry_price + atr_stop_multiplier * atr
            if close > stop_price:
                result["signal_type"] = "STOP_LOSS"
                result["direction"] = "FLAT"
                result["stop_loss_price"] = stop_price
                return result

        # --- Check regular exits ---
        if current_position == "LONG" and not np.isnan(sma_slow):
            if close < sma_slow:
                result["signal_type"] = "EXIT"
                result["direction"] = "FLAT"
                return result

        if current_position == "SHORT" and not np.isnan(sma_slow):
            if close > sma_slow:
                result["signal_type"] = "EXIT"
                result["direction"] = "FLAT"
                return result

        # --- Entry signals (only when FLAT) ---
        if current_position != "FLAT":
            return result

        trend_strong = adx >= self.adx_threshold

        long_entry = (
            not np.isnan(don_upper)
            and not np.isnan(sma_fast)
            and not np.isnan(sma_slow)
            and close >= don_upper
            and sma_fast > sma_slow
            and trend_strong
        )

        short_entry = (
            not np.isnan(don_lower)
            and not np.isnan(sma_fast)
            and not np.isnan(sma_slow)
            and close <= don_lower
            and sma_fast < sma_slow
            and trend_strong
        )

        if long_entry:
            stop_price = close - atr_stop_multiplier * atr if atr > 0 else 0.0
            strength = min(1.0, (adx - self.adx_threshold) / 20.0) if adx > self.adx_threshold else 0.5
            result.update({
                "direction": "LONG",
                "signal_type": "ENTRY",
                "strength": strength,
                "stop_loss_price": stop_price,
                "trend_aligned": True,
            })
        elif short_entry:
            stop_price = close + atr_stop_multiplier * atr if atr > 0 else 0.0
            strength = min(1.0, (adx - self.adx_threshold) / 20.0) if adx > self.adx_threshold else 0.5
            result.update({
                "direction": "SHORT",
                "signal_type": "ENTRY",
                "strength": strength,
                "stop_loss_price": stop_price,
                "trend_aligned": True,
            })

        return result
