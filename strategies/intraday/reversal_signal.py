"""
reversal_signal.py
------------------
K线反转信号检测器。

检测逻辑：
  - 4根连续阴线 → BEAR_CONFIRMED 趋势
  - 4根连续阳线 → BULL_CONFIRMED 趋势
  - 趋势确认后，3根连续反向K线 → 反转信号
  - depth = trend_confirm_price - trend_extreme_price（绝对点数）
  - 反转触发后重置状态机，进入新趋势

Per-symbol配置：enabled/min_depth/max_depth。
"""

from __future__ import annotations
from typing import Dict, Optional


REVERSAL_CONFIG: Dict[str, Dict] = {
    "IM": {"enabled": True, "min_depth": 10, "max_depth": 30},
    "IC": {"enabled": False, "min_depth": 10, "max_depth": 30},  # disabled until verified
    "IF": {"enabled": False, "min_depth": 10, "max_depth": 30},
    "IH": {"enabled": False, "min_depth": 10, "max_depth": 30},
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
