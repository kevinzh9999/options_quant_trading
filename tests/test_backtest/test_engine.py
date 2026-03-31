"""Tests for BacktestEngine using a mock DataFeed and mock strategy."""

from __future__ import annotations

from typing import Dict, List
import pandas as pd
import pytest

from strategies.base import BaseStrategy, StrategyConfig, Signal, SignalDirection, SignalStrength
from backtest.broker import SimBroker, AccountState
from backtest.engine import BacktestEngine
from backtest.report import BacktestReport


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, start_price: float = 5000.0) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with n rows."""
    import numpy as np
    prices = start_price + np.cumsum(np.random.randn(n) * 20)
    prices = np.maximum(prices, 100.0)
    df = pd.DataFrame({
        "trade_date": [f"2024{str(i+1).zfill(4)}" for i in range(n)],
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": 10000,
    })
    return df


class MockDataFeed:
    """In-memory DataFeed that requires no DB access."""

    def __init__(self, symbols: List[str], n_days: int = 60):
        self.symbols = symbols
        self._data: Dict[str, pd.DataFrame] = {}
        self._trading_dates: List[str] = []
        self._loaded = False
        self._n_days = n_days

    def preload(self) -> None:
        if self._loaded:
            return
        import numpy as np
        base_dates = [f"20240{str(i+101)[1:]}" for i in range(self._n_days)]
        self._trading_dates = base_dates
        for sym in self.symbols:
            df = _make_ohlcv(self._n_days)
            df["trade_date"] = base_dates
            self._data[sym] = df
        self._loaded = True

    def get_trading_dates(self) -> List[str]:
        if not self._loaded:
            self.preload()
        return list(self._trading_dates)

    def get_daily_bar(self, symbol: str, trade_date: str):
        df = self._data.get(symbol)
        if df is None:
            return None
        rows = df[df["trade_date"] == trade_date]
        if rows.empty:
            return None
        return rows.iloc[0]

    def get_history(self, symbol: str, trade_date: str, lookback: int = 60) -> pd.DataFrame:
        df = self._data.get(symbol, pd.DataFrame())
        if df.empty:
            return df
        mask = df["trade_date"] <= trade_date
        return df[mask].tail(lookback).reset_index(drop=True)

    def get_options_chain_on_date(self, underlying: str, trade_date: str):
        return None

    def get_index_close(self, index_code: str, trade_date: str):
        return None


class NeutralStrategy(BaseStrategy):
    """Strategy that always emits NEUTRAL signals — never trades."""

    def generate_signals(self, trade_date: str, market_data: dict) -> List[Signal]:
        return []

    def on_fill(self, order_id, instrument, direction, volume, price, trade_date):
        pass


class BuyAndHoldStrategy(BaseStrategy):
    """Buy on the first day, hold forever."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._bought = False

    def generate_signals(self, trade_date: str, market_data: dict) -> List[Signal]:
        if self._bought:
            return []
        sym = self.config.universe[0]
        df = market_data.get(sym)
        if df is None or df.empty:
            return []
        price = float(df["close"].iloc[-1])
        self._bought = True
        return [Signal(
            strategy_id=self.strategy_id,
            signal_date=trade_date,
            instrument=sym,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            target_volume=1,
            price_ref=price,
            metadata={"contract_multiplier": 200, "margin_rate": 0.15},
        )]

    def on_fill(self, order_id, instrument, direction, volume, price, trade_date):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_engine(symbols=None, n_days=60, initial_capital=1_000_000):
    if symbols is None:
        symbols = ["IM.CFX"]
    feed = MockDataFeed(symbols, n_days=n_days)
    broker = SimBroker(initial_capital=initial_capital, commission_rate=0.0, slippage_points=0.0)
    engine = BacktestEngine(feed, broker)
    return engine, feed, broker


def test_engine_runs_with_no_strategies():
    engine, _, _ = _make_engine()
    report = engine.run(show_progress=False)
    assert isinstance(report, BacktestReport)


def test_engine_records_daily_states():
    engine, feed, _ = _make_engine(n_days=30)
    config = StrategyConfig(strategy_id="neutral", universe=["IM.CFX"])
    engine.add_strategy(NeutralStrategy(config))
    report = engine.run(show_progress=False)
    # Should have one state per trading day
    assert len(report.daily_states) == 30


def test_engine_buy_and_hold_increases_balance():
    """Balance should increase when price trends up."""
    import numpy as np
    symbols = ["IM.CFX"]
    feed = MockDataFeed(symbols, n_days=50)
    # Manually set trending-up data
    dates = [f"20240{str(i+101)[1:]}" for i in range(50)]
    prices = 5000.0 + np.arange(50) * 10  # strictly increasing
    df = pd.DataFrame({
        "trade_date": dates,
        "open": prices,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "volume": 10000,
    })
    feed._data["IM.CFX"] = df
    feed._trading_dates = dates
    feed._loaded = True

    broker = SimBroker(initial_capital=1_000_000, commission_rate=0.0, slippage_points=0.0)
    engine = BacktestEngine(feed, broker)
    config = StrategyConfig(strategy_id="bah", universe=["IM.CFX"])
    engine.add_strategy(BuyAndHoldStrategy(config))
    engine.set_symbol_params("IM.CFX", contract_multiplier=200, margin_rate=0.15)

    report = engine.run(show_progress=False)
    final_balance = report.daily_states[-1].balance
    assert final_balance > 1_000_000, f"Expected profit, got balance={final_balance}"


def test_engine_signal_execution():
    """Verify that a LONG signal creates a position in the broker."""
    symbols = ["IM.CFX"]
    feed = MockDataFeed(symbols, n_days=5)
    feed.preload()
    broker = SimBroker(initial_capital=1_000_000, commission_rate=0.0, slippage_points=0.0)
    engine = BacktestEngine(feed, broker)
    config = StrategyConfig(strategy_id="bah", universe=["IM.CFX"])
    engine.add_strategy(BuyAndHoldStrategy(config))
    engine.set_symbol_params("IM.CFX", contract_multiplier=200, margin_rate=0.15)
    engine.run(show_progress=False)

    # Position should have been opened
    pos = broker.get_position("IM.CFX")
    assert pos is not None
    assert pos.direction == "LONG"
    assert pos.volume == 1


def test_engine_set_symbol_params():
    engine, _, _ = _make_engine()
    engine.set_symbol_params("IM.CFX", contract_multiplier=200, margin_rate=0.12)
    assert engine._symbol_multipliers["IM.CFX"] == 200
    assert engine._symbol_margin_rates["IM.CFX"] == pytest.approx(0.12)


def test_engine_returns_backtest_report():
    engine, _, _ = _make_engine()
    config = StrategyConfig(strategy_id="neutral", universe=["IM.CFX"])
    engine.add_strategy(NeutralStrategy(config))
    report = engine.run(show_progress=False)
    assert isinstance(report, BacktestReport)
    assert report.initial_capital == 1_000_000
