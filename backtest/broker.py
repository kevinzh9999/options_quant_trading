"""
broker.py
---------
回测模拟撮合引擎。

接收策略信号，模拟开平仓成交，计算手续费、滑点，更新持仓和账户状态。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    direction: str          # "LONG" / "SHORT"
    volume: int
    entry_price: float
    entry_date: str
    margin_rate: float = 0.15
    contract_multiplier: int = 200
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    margin: float = 0.0


@dataclass
class Trade:
    trade_date: str
    symbol: str
    direction: str          # "BUY" / "SELL"
    offset: str             # "OPEN" / "CLOSE"
    volume: int
    price: float
    commission: float
    slippage: float
    strategy_name: str


@dataclass
class AccountState:
    trade_date: str
    balance: float
    available: float
    margin: float
    unrealized_pnl: float
    realized_pnl: float
    commission_total: float
    positions: dict = field(default_factory=dict)


class SimBroker:
    """
    Simulated broker for backtesting.

    Parameters
    ----------
    initial_capital : float
        Starting capital in yuan.
    commission_rate : float
        Commission as fraction of notional value.
    slippage_points : float
        Slippage in price points per order.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.00005,
        slippage_points: float = 1.0,
    ) -> None:
        self._initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_points = slippage_points

        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._realized_pnl: float = 0.0
        self._current_state: Optional[AccountState] = None

        # Track available capital separately from balance
        # Initially all capital is available (no positions)
        self._available: float = initial_capital

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        direction: str,
        offset: str,
        volume: int,
        price: float,
        strategy_name: str,
        contract_multiplier: int = 200,
        margin_rate: float = 0.15,
    ) -> Optional[Trade]:
        """
        Submit an order.

        Parameters
        ----------
        symbol : str
            Contract symbol.
        direction : str
            "BUY" or "SELL".
        offset : str
            "OPEN" or "CLOSE".
        volume : int
            Number of lots.
        price : float
            Reference price.
        strategy_name : str
            Name of the strategy.
        contract_multiplier : int
            Contract multiplier (e.g. 200 for IM).
        margin_rate : float
            Margin rate (e.g. 0.15 for 15%).

        Returns
        -------
        Trade | None
            Executed trade, or None if order was rejected.
        """
        if volume <= 0:
            logger.warning("submit_order: volume must be > 0, got %d", volume)
            return None

        if offset == "OPEN":
            return self._open_position(
                symbol, direction, volume, price, strategy_name,
                contract_multiplier, margin_rate
            )
        elif offset == "CLOSE":
            return self._close_position(
                symbol, direction, volume, price, strategy_name,
                contract_multiplier, margin_rate
            )
        else:
            logger.warning("submit_order: unknown offset %s", offset)
            return None

    def _open_position(
        self,
        symbol: str,
        direction: str,
        volume: int,
        price: float,
        strategy_name: str,
        contract_multiplier: int,
        margin_rate: float,
    ) -> Optional[Trade]:
        """Open a new position."""
        # Apply slippage: BUY pays more, SELL receives less
        if direction == "BUY":
            actual_price = price + self.slippage_points
        else:
            actual_price = price - self.slippage_points

        commission = actual_price * volume * contract_multiplier * self.commission_rate
        required_margin = actual_price * volume * contract_multiplier * margin_rate
        total_cost = required_margin + commission

        if self._available < total_cost:
            logger.warning(
                "Insufficient margin for %s %s %d@%.2f: need %.0f, have %.0f",
                symbol, direction, volume, price, total_cost, self._available
            )
            return None

        # Deduct from available
        self._available -= total_cost

        # Create or update position
        pos_key = symbol
        if pos_key in self._positions:
            # Average into existing position (same direction assumed)
            existing = self._positions[pos_key]
            total_volume = existing.volume + volume
            avg_price = (
                (existing.entry_price * existing.volume + actual_price * volume)
                / total_volume
            )
            existing.entry_price = avg_price
            existing.volume = total_volume
            existing.contract_multiplier = contract_multiplier
            existing.margin_rate = margin_rate
        else:
            pos_direction = "LONG" if direction == "BUY" else "SHORT"
            self._positions[pos_key] = Position(
                symbol=symbol,
                direction=pos_direction,
                volume=volume,
                entry_price=actual_price,
                entry_date="",  # will be set by engine
                margin_rate=margin_rate,
                contract_multiplier=contract_multiplier,
                current_price=actual_price,
            )

        trade = Trade(
            trade_date="",
            symbol=symbol,
            direction=direction,
            offset="OPEN",
            volume=volume,
            price=actual_price,
            commission=commission,
            slippage=abs(actual_price - price),
            strategy_name=strategy_name,
        )
        self._trades.append(trade)
        logger.info(
            "OPEN %s %s %d @ %.2f  margin=%.0f  commission=%.2f",
            symbol, direction, volume, actual_price, required_margin, commission
        )
        return trade

    def _close_position(
        self,
        symbol: str,
        direction: str,
        volume: int,
        price: float,
        strategy_name: str,
        contract_multiplier: int,
        margin_rate: float,
    ) -> Optional[Trade]:
        """Close an existing position."""
        pos = self._positions.get(symbol)
        if pos is None:
            logger.warning("CLOSE %s: no position found", symbol)
            return None

        close_volume = min(volume, pos.volume)
        if close_volume <= 0:
            return None

        # For close: BUY covers short (actual_price = price + slippage for short cover)
        # SELL closes long (actual_price = price - slippage for long close)
        # The spec says:
        #   actual_price = price - slippage if direction=="BUY" else price + slippage
        # But note: closing BUY means covering short, sells at higher = actual_price + slippage
        # Re-reading spec: "SELL CLOSE: actual_price = price + slippage" seems reversed.
        # Logic: when CLOSING, slippage hurts the closer.
        # BUY CLOSE (cover short): you pay more → price + slippage
        # SELL CLOSE (close long): you receive less → price - slippage
        if direction == "BUY":
            actual_price = price + self.slippage_points
        else:
            actual_price = max(0.0, price - self.slippage_points)

        commission = actual_price * close_volume * contract_multiplier * self.commission_rate

        # Calculate PnL
        if direction == "BUY":
            # BUY CLOSE = covering a short position
            pnl = (pos.entry_price - actual_price) * close_volume * contract_multiplier
        else:
            # SELL CLOSE = closing a long position
            pnl = (actual_price - pos.entry_price) * close_volume * contract_multiplier

        self._realized_pnl += pnl

        # Release margin for closed portion
        released_margin = pos.entry_price * close_volume * contract_multiplier * pos.margin_rate
        self._available += released_margin + pnl - commission

        # Reduce or remove position
        pos.volume -= close_volume
        if pos.volume <= 0:
            del self._positions[symbol]

        trade = Trade(
            trade_date="",
            symbol=symbol,
            direction=direction,
            offset="CLOSE",
            volume=close_volume,
            price=actual_price,
            commission=commission,
            slippage=abs(actual_price - price),
            strategy_name=strategy_name,
        )
        self._trades.append(trade)
        logger.info(
            "CLOSE %s %s %d @ %.2f  pnl=%.0f  commission=%.2f",
            symbol, direction, close_volume, actual_price, pnl, commission
        )
        return trade

    # ------------------------------------------------------------------
    # Daily mark-to-market update
    # ------------------------------------------------------------------

    def update_daily(self, trade_date: str, prices: Dict[str, float]) -> AccountState:
        """
        Mark positions to market at end of day.

        Parameters
        ----------
        trade_date : str
            Current trading date (YYYYMMDD).
        prices : dict
            Symbol → close price mapping.

        Returns
        -------
        AccountState
            End-of-day account state snapshot.
        """
        total_unrealized = 0.0
        total_margin = 0.0

        for symbol, pos in self._positions.items():
            current_price = prices.get(symbol, pos.current_price)
            pos.current_price = current_price

            mult = pos.contract_multiplier
            if pos.direction == "LONG":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.volume * mult
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.volume * mult

            pos.margin = current_price * pos.volume * mult * pos.margin_rate

            total_unrealized += pos.unrealized_pnl
            total_margin += pos.margin

        # Clean up any zero-volume ghost positions
        for sym in [s for s, p in self._positions.items() if p.volume <= 0]:
            del self._positions[sym]

        total_commission = sum(t.commission for t in self._trades)

        balance = self._initial_capital + self._realized_pnl + total_unrealized - total_commission
        available = balance - total_margin

        state = AccountState(
            trade_date=trade_date,
            balance=balance,
            available=available,
            margin=total_margin,
            unrealized_pnl=total_unrealized,
            realized_pnl=self._realized_pnl,
            commission_total=total_commission,
            positions={k: v for k, v in self._positions.items()},
        )
        self._current_state = state
        # Sync available to computed value
        self._available = available
        return state

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return position for symbol, or None."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Return all open positions."""
        return dict(self._positions)

    def get_account_state(self) -> AccountState:
        """Return current account state (last computed)."""
        if self._current_state is None:
            # Return initial state
            return AccountState(
                trade_date="",
                balance=self._initial_capital,
                available=self._initial_capital,
                margin=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                commission_total=0.0,
            )
        return self._current_state

    def get_trade_history(self) -> List[Trade]:
        """Return list of all executed trades."""
        return list(self._trades)

    def force_close_all(
        self, prices: Dict[str, float], strategy_name: str,
    ) -> List[Trade]:
        """Force close all positions at current prices."""
        trades: List[Trade] = []
        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            price = prices.get(symbol, pos.current_price)
            close_dir = "SELL" if pos.direction == "LONG" else "BUY"
            trade = self.submit_order(
                symbol=symbol,
                direction=close_dir,
                offset="CLOSE",
                volume=pos.volume,
                price=price,
                strategy_name=strategy_name,
                contract_multiplier=pos.contract_multiplier,
                margin_rate=pos.margin_rate,
            )
            if trade:
                trades.append(trade)
        return trades

    def force_close_partial(
        self,
        symbol: str,
        volume: int,
        price: float,
        strategy_name: str,
    ) -> Optional[Trade]:
        """Force close partial position for a symbol."""
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        close_dir = "SELL" if pos.direction == "LONG" else "BUY"
        return self.submit_order(
            symbol=symbol,
            direction=close_dir,
            offset="CLOSE",
            volume=min(volume, pos.volume),
            price=price,
            strategy_name=strategy_name,
            contract_multiplier=pos.contract_multiplier,
            margin_rate=pos.margin_rate,
        )


# Backward-compat alias
SimulatedBroker = SimBroker
