"""
engine.py
---------
回测引擎核心。事件驱动，逐日推进。
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalDirection
from backtest.data_feed import DataFeed
from backtest.broker import SimBroker, AccountState
from backtest.report import BacktestReport

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtest engine. Iterates day by day, feeds market data to
    strategies, executes signals through the broker, and collects results.

    Parameters
    ----------
    data_feed : DataFeed
        Preloaded data feed.
    broker : SimBroker
        Simulated broker for order execution.
    """

    def __init__(self, data_feed: DataFeed, broker: SimBroker) -> None:
        self.data_feed = data_feed
        self.broker = broker
        self.strategies: List[BaseStrategy] = []
        self.daily_states: List[AccountState] = []
        self.daily_signals: List[Dict] = []
        self._symbol_multipliers: Dict[str, int] = {}
        self._symbol_margin_rates: Dict[str, float] = {}

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy to run in the backtest."""
        self.strategies.append(strategy)
        logger.info("Added strategy: %s", strategy.strategy_id)

    def set_symbol_params(
        self,
        symbol: str,
        contract_multiplier: int = 200,
        margin_rate: float = 0.15,
    ) -> None:
        """Set contract parameters for a symbol."""
        self._symbol_multipliers[symbol] = contract_multiplier
        self._symbol_margin_rates[symbol] = margin_rate

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, show_progress: bool = True) -> BacktestReport:
        """
        Run the full backtest.

        Returns
        -------
        BacktestReport
            Completed backtest report.
        """
        # 1. Preload data
        logger.info("Preloading data...")
        self.data_feed.preload()

        # 2. Get trading dates
        trading_dates = self.data_feed.get_trading_dates()
        if not trading_dates:
            logger.warning("No trading dates found in data feed.")
            return self._build_report()

        logger.info("Running backtest: %d trading days", len(trading_dates))
        n_strats = len(self.strategies)
        if n_strats == 0:
            logger.warning("No strategies registered.")

        # 3. Day loop
        for i, trade_date in enumerate(trading_dates):
            # a. Build current prices dict
            prices: Dict[str, float] = {}
            for sym in self.data_feed.symbols:
                bar = self.data_feed.get_daily_bar(sym, trade_date)
                if bar is not None and "close" in bar.index:
                    val = bar["close"]
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        prices[sym] = float(val)

            # b. Inject option prices into prices dict for mark-to-market
            for sym in self.data_feed.symbols:
                for prefix in ("MO", "IO"):
                    if sym.startswith(prefix):
                        chain = self.data_feed.get_options_chain_on_date(prefix, trade_date)
                        if chain is not None and not chain.empty:
                            if "ts_code" in chain.columns and "close" in chain.columns:
                                for _, row in chain[["ts_code", "close"]].iterrows():
                                    px = row["close"]
                                    if px is not None and not (isinstance(px, float) and pd.isna(px)):
                                        px_f = float(px)
                                        if px_f > 0:
                                            prices[str(row["ts_code"])] = px_f
                        break

            # Mark positions to market (now includes option prices)
            self.broker.update_daily(trade_date, prices)

            # c. Run each strategy
            for strategy in self.strategies:
                # Inject current broker equity and margin so strategies can size positions
                if hasattr(strategy, "_positions"):
                    state_now = self.broker.get_account_state()
                    strategy._positions["__equity__"] = {
                        "balance": state_now.balance,
                        "margin": state_now.margin,
                    }
                    # Inject broker positions so strategy can verify execution
                    strategy._positions["__broker_positions__"] = {
                        sym: {"direction": pos.direction, "volume": pos.volume}
                        for sym, pos in self.broker.get_all_positions().items()
                    }

                # Build market_data dict: symbol -> history df
                market_data: Dict[str, pd.DataFrame] = {}
                for sym in strategy.config.universe:
                    hist = self.data_feed.get_history(sym, trade_date, lookback=600)
                    if not hist.empty:
                        market_data[sym] = hist

                # Inject options chain if the strategy wants it
                # Look for option underlyings from the strategy's universe
                for sym in strategy.config.universe:
                    for prefix in ("MO", "IO"):
                        if sym.startswith(prefix):
                            chain = self.data_feed.get_options_chain_on_date(prefix, trade_date)
                            if chain is not None and not chain.empty:
                                market_data["options_chain"] = chain
                            break

                # Generate signals
                try:
                    signals = strategy.generate_signals(trade_date, market_data)
                except Exception as exc:
                    logger.exception(
                        "Strategy %s failed on %s: %s",
                        strategy.strategy_id, trade_date, exc
                    )
                    signals = []

                # d. Execute signals
                if signals:
                    self._execute_signals(signals, prices)
                    for sig in signals:
                        self.daily_signals.append({
                            "trade_date": trade_date,
                            "strategy_id": strategy.strategy_id,
                            **sig.to_dict(),
                        })

            # e. Safety net: force-close any expired option positions
            self._settle_expired_options(trade_date, prices)

            # f. Record daily state after execution
            state = self.broker.get_account_state()
            self.daily_states.append(state)

            # g. Progress reporting
            if show_progress and (i + 1) % 20 == 0:
                print(
                    f"  [{trade_date}] {i+1}/{len(trading_dates)}  "
                    f"balance={state.balance:,.0f}  "
                    f"positions={len(state.positions)}"
                )

        logger.info("Backtest complete. Final balance: %.0f", self.daily_states[-1].balance if self.daily_states else 0)
        return self._build_report()

    # ------------------------------------------------------------------
    # Expired option settlement (safety net)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_option_expiry(symbol: str) -> str | None:
        """Parse expiry YYYYMMDD from option symbol like MO2412-C-7000.CFX.

        MO contracts: MO{YYMM} → expiry is 3rd Friday of that month (approx day 20).
        Returns YYYYMMDD string or None if not an option.
        """
        import re
        m = re.match(r"MO(\d{4})-[CP]-\d+", symbol)
        if not m:
            return None
        yymm = m.group(1)
        year = 2000 + int(yymm[:2])
        month = int(yymm[2:])
        # Approximate expiry as 3rd Friday ≈ day 20 of the month
        return f"{year}{month:02d}20"

    def _settle_expired_options(
        self, trade_date: str, prices: Dict[str, float],
    ) -> None:
        """Force-close any option positions past their expiry date."""
        for symbol in list(self.broker._positions.keys()):
            expiry = self._parse_option_expiry(symbol)
            if expiry is None:
                continue
            if trade_date <= expiry:
                continue
            # This option has expired — force close at last known price (or 0)
            pos = self.broker._positions[symbol]
            close_price = prices.get(symbol, 0.0)
            close_dir = "SELL" if pos.direction == "LONG" else "BUY"
            logger.warning(
                "ENGINE EXPIRY CLEANUP: %s expired(%s) on %s, force close %d @ %.2f",
                symbol, expiry, trade_date, pos.volume, close_price,
            )
            self.broker.submit_order(
                symbol=symbol,
                direction=close_dir,
                offset="CLOSE",
                volume=pos.volume,
                price=close_price,
                strategy_name="engine_expiry",
                contract_multiplier=pos.contract_multiplier,
                margin_rate=pos.margin_rate,
            )

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------

    def _execute_signals(
        self,
        signals: List[Signal],
        prices: Dict[str, float],
    ) -> None:
        """
        Convert signals to broker orders and execute them.

        Signal semantics:
        - LONG + target_volume > 0  → BUY OPEN
        - SHORT + target_volume > 0 → SELL OPEN
        - CLOSE                     → close all positions for this instrument
        - NEUTRAL                   → skip
        """
        for signal in signals:
            instrument = signal.instrument
            direction = signal.direction

            if direction == SignalDirection.NEUTRAL:
                continue

            # Determine execution price
            exec_price = signal.price_ref if signal.price_ref > 0 else prices.get(instrument, 0.0)
            if exec_price <= 0:
                if direction == SignalDirection.CLOSE:
                    # Allow closing at zero — expired worthless options settle at 0
                    exec_price = 0.0
                else:
                    logger.warning("No price for %s, skipping signal", instrument)
                    continue

            # Contract params: check signal metadata first, then engine defaults
            multiplier = int(
                signal.metadata.get(
                    "contract_multiplier",
                    self._symbol_multipliers.get(instrument, 200)
                )
            )
            margin_rate = float(
                signal.metadata.get(
                    "margin_rate",
                    self._symbol_margin_rates.get(instrument, 0.15)
                )
            )

            if direction == SignalDirection.CLOSE:
                pos = self.broker.get_position(instrument)
                if pos is None:
                    logger.debug("CLOSE signal for %s but no position", instrument)
                    continue
                close_dir = "SELL" if pos.direction == "LONG" else "BUY"
                # Support partial close: use target_volume if > 0, else full position
                close_vol = signal.target_volume if signal.target_volume > 0 else pos.volume
                close_vol = min(close_vol, pos.volume)
                self.broker.submit_order(
                    symbol=instrument,
                    direction=close_dir,
                    offset="CLOSE",
                    volume=close_vol,
                    price=exec_price,
                    strategy_name=signal.strategy_id,
                    contract_multiplier=multiplier,
                    margin_rate=margin_rate,
                )

            elif direction == SignalDirection.LONG and signal.target_volume > 0:
                pos = self.broker.get_position(instrument)
                if pos is not None and pos.direction == "LONG":
                    logger.debug("Already LONG %s, skipping", instrument)
                    continue
                # Close existing opposite-direction position first
                if pos is not None and pos.direction == "SHORT":
                    self.broker.submit_order(
                        symbol=instrument, direction="BUY", offset="CLOSE",
                        volume=pos.volume, price=exec_price,
                        strategy_name=signal.strategy_id,
                        contract_multiplier=multiplier, margin_rate=margin_rate,
                    )
                self.broker.submit_order(
                    symbol=instrument,
                    direction="BUY",
                    offset="OPEN",
                    volume=signal.target_volume,
                    price=exec_price,
                    strategy_name=signal.strategy_id,
                    contract_multiplier=multiplier,
                    margin_rate=margin_rate,
                )

            elif direction == SignalDirection.SHORT and signal.target_volume > 0:
                pos = self.broker.get_position(instrument)
                if pos is not None and pos.direction == "SHORT":
                    logger.debug("Already SHORT %s, skipping", instrument)
                    continue
                # Close existing opposite-direction position first
                if pos is not None and pos.direction == "LONG":
                    self.broker.submit_order(
                        symbol=instrument, direction="SELL", offset="CLOSE",
                        volume=pos.volume, price=exec_price,
                        strategy_name=signal.strategy_id,
                        contract_multiplier=multiplier, margin_rate=margin_rate,
                    )
                self.broker.submit_order(
                    symbol=instrument,
                    direction="SELL",
                    offset="OPEN",
                    volume=signal.target_volume,
                    price=exec_price,
                    strategy_name=signal.strategy_id,
                    contract_multiplier=multiplier,
                    margin_rate=margin_rate,
                )

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------

    def _build_report(self) -> BacktestReport:
        """Construct the BacktestReport from recorded states and trades."""
        strategy_name = (
            self.strategies[0].strategy_id if self.strategies else "backtest"
        )
        report = BacktestReport(
            daily_states=self.daily_states,
            trades=self.broker.get_trade_history(),
            initial_capital=self.broker._initial_capital,
            strategy_name=strategy_name,
        )
        return report
