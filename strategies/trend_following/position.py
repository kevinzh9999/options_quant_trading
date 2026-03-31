"""
position.py — 趋势跟踪仓位管理
波动率目标化仓位计算和 ATR 止损。
"""

from __future__ import annotations

import numpy as np


class TrendPositionSizer:
    """
    Volatility-targeted position sizer.

    Sizes positions so that the annualized volatility contribution of each
    symbol approximately equals (vol_target / n_symbols) of the portfolio,
    subject to a per-symbol cap.

    Parameters
    ----------
    vol_target : float
        Annualized portfolio volatility target (e.g. 0.15 for 15%).
    max_position_per_symbol : float
        Maximum fraction of account equity in a single symbol (e.g. 0.20).
    """

    def __init__(
        self,
        vol_target: float = 0.15,
        max_position_per_symbol: float = 0.20,
    ) -> None:
        self.vol_target = vol_target
        self.max_position_per_symbol = max_position_per_symbol

    def calculate_lots(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        account_equity: float,
        capital_allocation: float,
        n_symbols: int,
        contract_multiplier: int = 200,
    ) -> int:
        """
        Calculate target position size in lots.

        Uses volatility targeting:
            symbol_ann_vol = (atr / current_price) * sqrt(252)
            target_value   = (vol_target / symbol_ann_vol)
                             * (account_equity * capital_allocation / n_symbols)
            lots           = target_value / (current_price * contract_multiplier)

        Capped at:
            max_lots = max_position_per_symbol * account_equity
                       / (current_price * contract_multiplier)

        Parameters
        ----------
        symbol : str
            Symbol name (used for logging only).
        current_price : float
            Latest close price.
        atr : float
            ATR value in price points.
        account_equity : float
            Total account equity.
        capital_allocation : float
            Fraction of equity to allocate to this strategy (e.g. 0.8).
        n_symbols : int
            Number of active symbols (for equal-weight allocation).
        contract_multiplier : int
            Contract multiplier.

        Returns
        -------
        int
            Target lots (>= 0).
        """
        if current_price <= 0 or atr <= 0 or account_equity <= 0 or n_symbols <= 0:
            return 0

        notional_per_lot = current_price * contract_multiplier
        if notional_per_lot <= 0:
            return 0

        # Annualized vol estimate from ATR
        symbol_ann_vol = (atr / current_price) * np.sqrt(252)
        if symbol_ann_vol <= 0:
            return 0

        allocated_capital = account_equity * capital_allocation / n_symbols
        target_value = (self.vol_target / symbol_ann_vol) * allocated_capital
        lots = target_value / notional_per_lot

        # Apply per-symbol cap
        max_lots = self.max_position_per_symbol * account_equity / notional_per_lot

        lots = min(lots, max_lots)
        return max(0, int(lots))

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        multiplier: float = 2.5,
    ) -> float:
        """
        Compute ATR-based stop-loss price.

        Parameters
        ----------
        entry_price : float
            Trade entry price.
        atr : float
            ATR value in price points.
        direction : str
            "LONG" or "SHORT".
        multiplier : float
            ATR distance multiplier.

        Returns
        -------
        float
            Stop-loss price.
        """
        if direction == "LONG":
            return entry_price - multiplier * atr
        else:
            return entry_price + multiplier * atr
