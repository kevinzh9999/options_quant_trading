"""
strategy.py — 趋势跟踪策略
双均线 + 唐奇安通道 + ADX 过滤器。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from strategies.base import BaseStrategy, Signal, StrategyConfig, SignalDirection, SignalStrength
from strategies.trend_following.signal import TrendSignalGenerator
from strategies.trend_following.position import TrendPositionSizer

logger = logging.getLogger(__name__)


@dataclass
class TrendConfig(StrategyConfig):
    fast_period: int = 10
    slow_period: int = 30
    donchian_period: int = 20
    adx_period: int = 14
    adx_threshold: float = 20.0
    atr_period: int = 20
    atr_stop_multiplier: float = 2.5
    vol_target: float = 0.15
    capital_allocation: float = 0.8
    max_position_per_symbol: float = 0.20
    contract_multipliers: dict = field(default_factory=lambda: {
        "IF.CFX": 300, "IH.CFX": 300, "IC.CFX": 200, "IM.CFX": 200
    })


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following on futures using dual MA + Donchian + ADX filter.
    """

    def __init__(self, config: TrendConfig) -> None:
        super().__init__(config)
        self.trend_config = config
        self.sig_gen = TrendSignalGenerator(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
            donchian_period=config.donchian_period,
            adx_period=config.adx_period,
            adx_threshold=config.adx_threshold,
            atr_period=config.atr_period,
        )
        self.sizer = TrendPositionSizer(
            vol_target=config.vol_target,
            max_position_per_symbol=config.max_position_per_symbol,
        )
        # symbol -> {direction, entry_price, stop_loss, lots, entry_date}
        # "__equity__" key holds latest broker balance for sizing
        self._positions: Dict[str, dict] = {}

    def generate_signals(self, trade_date: str, market_data: dict) -> List[Signal]:
        """
        For each symbol in universe:
        1. Get history df from market_data
        2. Compute indicators
        3. Get signal (stop-loss checked first)
        4. Generate entry/exit signals
        """
        signals: List[Signal] = []

        n_active = sum(1 for s in self.config.universe if s in market_data)
        if n_active == 0:
            return signals

        for symbol in self.config.universe:
            df = market_data.get(symbol)
            if df is None or df.empty or len(df) < self.trend_config.slow_period + 10:
                continue

            needed_cols = ["open", "high", "low", "close"]
            if not all(c in df.columns for c in needed_cols):
                continue

            try:
                df_ind = self.sig_gen.compute_indicators(df)
            except Exception as exc:
                logger.warning("compute_indicators failed for %s: %s", symbol, exc)
                continue

            # Skip if ATR is NaN
            atr_val = df_ind["atr"].iloc[-1]
            if pd.isna(atr_val):
                continue

            current_pos = self._positions.get(symbol, {})
            pos_dir = current_pos.get("direction", "FLAT")
            entry_price = current_pos.get("entry_price", 0.0)

            sig_info = self.sig_gen.get_signal(
                df_ind,
                current_position=pos_dir,
                entry_price=entry_price,
                atr_stop_multiplier=self.trend_config.atr_stop_multiplier,
            )

            close = float(df_ind["close"].iloc[-1])
            atr = float(atr_val)
            multiplier = self.trend_config.contract_multipliers.get(symbol, 200)

            # Handle exit / stop-loss
            if pos_dir != "FLAT" and sig_info["signal_type"] in ("EXIT", "STOP_LOSS"):
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    signal_date=trade_date,
                    instrument=symbol,
                    direction=SignalDirection.CLOSE,
                    strength=(
                        SignalStrength.STRONG
                        if sig_info["signal_type"] == "STOP_LOSS"
                        else SignalStrength.MODERATE
                    ),
                    target_volume=0,
                    price_ref=close,
                    notes=f"{sig_info['signal_type']}: pos={pos_dir} close={close:.1f}",
                    metadata={
                        "contract_multiplier": multiplier,
                        "margin_rate": 0.15,
                        **sig_info,
                    },
                ))
                self._positions.pop(symbol, None)
                continue

            # Handle entry
            if pos_dir == "FLAT" and sig_info["direction"] in ("LONG", "SHORT"):
                account_equity = self._positions.get("__equity__", {}).get(
                    "balance", 1_000_000
                )
                lots = self.sizer.calculate_lots(
                    symbol=symbol,
                    current_price=close,
                    atr=atr,
                    account_equity=account_equity,
                    capital_allocation=self.trend_config.capital_allocation,
                    n_symbols=max(n_active, 1),
                    contract_multiplier=multiplier,
                )
                if lots < 1:
                    lots = 1

                direction = (
                    SignalDirection.LONG
                    if sig_info["direction"] == "LONG"
                    else SignalDirection.SHORT
                )
                strength = (
                    SignalStrength.STRONG
                    if sig_info.get("adx", 0) > 30
                    else SignalStrength.MODERATE
                )
                stop_loss = sig_info.get("stop_loss_price", 0.0)

                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    signal_date=trade_date,
                    instrument=symbol,
                    direction=direction,
                    strength=strength,
                    target_volume=lots,
                    price_ref=close,
                    notes=f"Entry {sig_info['direction']}: ADX={sig_info.get('adx', 0):.1f}",
                    metadata={
                        "contract_multiplier": multiplier,
                        "margin_rate": 0.15,
                        "stop_loss_price": stop_loss,
                        "atr": atr,
                        **sig_info,
                    },
                ))
                self._positions[symbol] = {
                    "direction": sig_info["direction"],
                    "entry_price": close,
                    "stop_loss": stop_loss,
                    "lots": lots,
                    "entry_date": trade_date,
                }

        return signals

    def on_fill(
        self,
        order_id: str,
        instrument: str,
        direction: str,
        volume: int,
        price: float,
        trade_date: str,
    ) -> None:
        self.logger.info(
            "Fill %s %s %d @ %.2f", instrument, direction, volume, price
        )

    def get_status(self) -> Dict:
        return {sym: info for sym, info in self._positions.items() if sym != "__equity__"}
