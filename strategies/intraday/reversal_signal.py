"""
reversal_signal.py
------------------
反转信号检测器。

两种实现：
1. ReversalDetector（v1）：K线阴阳计数，4连阴确认趋势+3连阳检测反转
2. ReversalDetectorSlope（v2）：线性回归斜率，长窗口确认趋势+短窗口检测反转
   - 趋势确认：长窗口（8根bar）斜率 < bear_thr → BEAR
   - 反转检测：短窗口（6根bar）斜率 > rev_thr → LONG反转
   - 无NONE退出：BEAR/BULL只能通过反转互相转换
   - depth过滤：confirm_high - trend_low >= min_depth

Per-symbol配置：enabled + 方法特定参数。
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np


REVERSAL_CONFIG: Dict[str, Dict] = {
    "IM": {
        "enabled": True, "method": "slope",
        "long_n": 8, "short_n": 6, "bear_thr": -1.5, "rev_thr": 3.0, "min_depth": 25,
        # F1 filter: skip 1st reversal of day if BJ time < this threshold.
        # Rationale: slope params long_n=8 + short_n=6 make earliest possible
        # reversal fire ≈10:30 BJ, but that captures "breakout's first pullback"
        # rather than a true reversal. 900d IS/OOS backtest shows delay to ≥11:00
        # yields +156K yuan (+0.86pt/day). Empty/None disables.
        "first_trade_min_bj": "11:00",
    },
    "IC": {"enabled": False},
    "IF": {"enabled": False},
    "IH": {"enabled": False},
}


class ReversalSignal:
    """Represents a triggered reversal signal."""

    def __init__(self, direction: str, depth: float):
        self.direction = direction  # "LONG" or "SHORT"
        self.depth = depth          # absolute points


class ReversalDetector:
    """Per-symbol candle-based reversal state machine.

    Tracks consecutive bearish/bullish candles to confirm trend,
    then detects reversal when opposite candles appear.

    State machine:
      NONE -> BEAR_CONFIRMED (4 consecutive bearish)
      NONE -> BULL_CONFIRMED (4 consecutive bullish)
      BEAR_CONFIRMED -> reversal LONG (3 consecutive bullish, depth in range)
      BULL_CONFIRMED -> reversal SHORT (3 consecutive bearish, depth in range)

    After reversal triggers, state resets to the new trend direction.
    """

    def __init__(self, symbol: str, config: Dict | None = None):
        self.symbol = symbol
        cfg = config or REVERSAL_CONFIG.get(symbol, {})
        self.enabled = cfg.get("enabled", False)
        self.min_depth = cfg.get("min_depth", 10)
        self.max_depth = cfg.get("max_depth", 30)

        # State machine
        self.state: str = "NONE"  # NONE / BEAR_CONFIRMED / BULL_CONFIRMED
        self.consec_bear: int = 0
        self.consec_bull: int = 0
        self.reversal_bull_count: int = 0
        self.reversal_bear_count: int = 0
        self.trend_confirm_price: float = 0.0
        self.trend_extreme_price: float = 0.0

    def reset(self) -> None:
        """Reset all state (e.g. at start of new day)."""
        self.state = "NONE"
        self.consec_bear = 0
        self.consec_bull = 0
        self.reversal_bull_count = 0
        self.reversal_bear_count = 0
        self.trend_confirm_price = 0.0
        self.trend_extreme_price = 0.0

    def update(self, bar_open: float, bar_high: float,
               bar_low: float, bar_close: float) -> Optional[ReversalSignal]:
        """Feed one completed 5-min bar. Returns ReversalSignal if triggered.

        Args:
            bar_open: bar open price
            bar_high: bar high price
            bar_low: bar low price
            bar_close: bar close price

        Returns:
            ReversalSignal if reversal detected and depth in range, else None.
        """
        if not self.enabled:
            return None

        # Classify candle
        if bar_close > bar_open:
            candle = "L"  # bullish
        elif bar_close < bar_open:
            candle = "S"  # bearish
        else:
            candle = "-"  # doji

        # Update consecutive counts
        if candle == "L":
            self.consec_bull += 1
            self.consec_bear = 0
        elif candle == "S":
            self.consec_bear += 1
            self.consec_bull = 0
        # doji: don't reset either count (matches backtest behavior)

        # Check for new trend confirmation
        if self.consec_bear >= 4 and self.state != "BEAR_CONFIRMED":
            self.state = "BEAR_CONFIRMED"
            self.trend_confirm_price = bar_close
            self.trend_extreme_price = bar_close
            self.reversal_bull_count = 0

        if self.consec_bull >= 4 and self.state != "BULL_CONFIRMED":
            self.state = "BULL_CONFIRMED"
            self.trend_confirm_price = bar_close
            self.trend_extreme_price = bar_close
            self.reversal_bear_count = 0

        # Update trend extreme
        if self.state == "BEAR_CONFIRMED":
            self.trend_extreme_price = min(self.trend_extreme_price, bar_low)
        elif self.state == "BULL_CONFIRMED":
            self.trend_extreme_price = max(self.trend_extreme_price, bar_high)

        # Check for reversal
        reversal_signal = None

        if self.state == "BEAR_CONFIRMED":
            if candle == "L":
                self.reversal_bull_count += 1
            elif candle == "S":
                self.reversal_bull_count = 0
            if self.reversal_bull_count >= 3:
                depth = self.trend_confirm_price - self.trend_extreme_price
                if self.min_depth <= depth <= self.max_depth:
                    reversal_signal = ReversalSignal(
                        direction="LONG", depth=depth)
                # Reset: transition to BULL_CONFIRMED
                self.state = "BULL_CONFIRMED"
                self.trend_confirm_price = bar_close
                self.trend_extreme_price = bar_high
                self.reversal_bear_count = 0
                self.consec_bull = 3

        elif self.state == "BULL_CONFIRMED":
            if candle == "S":
                self.reversal_bear_count += 1
            elif candle == "L":
                self.reversal_bear_count = 0
            if self.reversal_bear_count >= 3:
                depth = self.trend_extreme_price - self.trend_confirm_price
                if self.min_depth <= depth <= self.max_depth:
                    reversal_signal = ReversalSignal(
                        direction="SHORT", depth=depth)
                # Reset: transition to BEAR_CONFIRMED
                self.state = "BEAR_CONFIRMED"
                self.trend_confirm_price = bar_close
                self.trend_extreme_price = bar_low
                self.reversal_bull_count = 0
                self.consec_bear = 3

        return reversal_signal


class ReversalDetectorSlope:
    """Slope-based reversal detector using linear regression.

    Trend confirmation: slope of closes over long_n bars.
    Reversal detection: slope of closes over short_n bars flips.
    No exit to NONE — BEAR/BULL only transition via reversal.

    Parameters (from config):
        long_n: bars for trend slope (default 8 = 40min)
        short_n: bars for reversal slope (default 6 = 30min)
        bear_thr: slope threshold for BEAR confirmation (default -1.5 pt/bar)
        rev_thr: slope threshold for reversal detection (default 3.0 pt/bar)
        min_depth: minimum depth (confirm_high - trend_low) to fire signal
    """

    def __init__(self, symbol: str, config: Dict | None = None):
        self.symbol = symbol
        cfg = config or REVERSAL_CONFIG.get(symbol, {})
        self.enabled = cfg.get("enabled", False)
        self.long_n = cfg.get("long_n", 8)
        self.short_n = cfg.get("short_n", 6)
        self.bear_thr = cfg.get("bear_thr", -1.5)
        self.rev_thr = cfg.get("rev_thr", 3.0)
        self.min_depth = cfg.get("min_depth", 25)
        self.reset()

    def reset(self) -> None:
        self.state: str = "NONE"
        self._closes: list = []
        self._highs: list = []
        self._lows: list = []
        self.confirm_high: float = 0.0
        self.confirm_low: float = 0.0
        self.trend_high: float = 0.0
        self.trend_low: float = 0.0

    def update(self, bar_open: float, bar_high: float,
               bar_low: float, bar_close: float) -> Optional[ReversalSignal]:
        if not self.enabled:
            return None

        self._closes.append(bar_close)
        self._highs.append(bar_high)
        self._lows.append(bar_low)
        n = len(self._closes)

        if n < self.long_n:
            return None

        sL = np.polyfit(range(min(n, self.long_n)),
                        self._closes[-min(n, self.long_n):], 1)[0]
        sS = np.polyfit(range(min(n, self.short_n)),
                        self._closes[-min(n, self.short_n):], 1)[0]

        signal = None

        if self.state == "NONE":
            if sL <= self.bear_thr:
                self.state = "BEAR"
                self.confirm_high = max(self._highs[-self.long_n:])
                self.confirm_low = min(self._lows[-self.long_n:])
                self.trend_low = self.confirm_low
                self.trend_high = self.confirm_high
            elif sL >= -self.bear_thr:
                self.state = "BULL"
                self.confirm_high = max(self._highs[-self.long_n:])
                self.confirm_low = min(self._lows[-self.long_n:])
                self.trend_high = self.confirm_high
                self.trend_low = self.confirm_low

        elif self.state == "BEAR":
            self.trend_low = min(self.trend_low, bar_low)
            self.trend_high = max(self.trend_high, bar_high)
            if sS >= self.rev_thr:
                depth = self.confirm_high - self.trend_low
                if depth >= self.min_depth:
                    signal = ReversalSignal(direction="LONG", depth=depth)
                self.state = "BULL"
                self.confirm_low = self.trend_low
                self.confirm_high = bar_high
                self.trend_high = bar_high
                self.trend_low = bar_low

        elif self.state == "BULL":
            self.trend_high = max(self.trend_high, bar_high)
            self.trend_low = min(self.trend_low, bar_low)
            if sS <= -self.rev_thr:
                depth = self.trend_high - self.confirm_low
                if depth >= self.min_depth:
                    signal = ReversalSignal(direction="SHORT", depth=depth)
                self.state = "BEAR"
                self.confirm_high = self.trend_high
                self.confirm_low = bar_low
                self.trend_low = bar_low
                self.trend_high = bar_high

        return signal
