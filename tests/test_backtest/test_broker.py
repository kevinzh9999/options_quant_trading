"""Tests for SimBroker."""

from __future__ import annotations

import pytest
from backtest.broker import SimBroker, Position, Trade, AccountState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_broker(**kwargs) -> SimBroker:
    defaults = dict(initial_capital=1_000_000, commission_rate=0.0001, slippage_points=0.0)
    defaults.update(kwargs)
    return SimBroker(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_open_long():
    broker = make_broker()
    trade = broker.submit_order(
        "IF.CFX", "BUY", "OPEN", 1, 3000.0, "test",
        contract_multiplier=300, margin_rate=0.15
    )
    assert trade is not None
    assert trade.direction == "BUY"
    assert trade.offset == "OPEN"
    assert trade.volume == 1
    assert trade.price == 3000.0

    pos = broker.get_position("IF.CFX")
    assert pos is not None
    assert pos.volume == 1
    assert pos.direction == "LONG"
    assert pos.entry_price == pytest.approx(3000.0)


def test_open_short():
    broker = make_broker()
    trade = broker.submit_order(
        "IM.CFX", "SELL", "OPEN", 2, 5000.0, "test",
        contract_multiplier=200, margin_rate=0.15
    )
    assert trade is not None
    assert trade.direction == "SELL"
    assert trade.offset == "OPEN"

    pos = broker.get_position("IM.CFX")
    assert pos is not None
    assert pos.direction == "SHORT"
    assert pos.volume == 2


def test_close_long_pnl():
    """Open long then close at higher price — should have positive PnL."""
    broker = make_broker(commission_rate=0.0, slippage_points=0.0)
    broker.submit_order("IF.CFX", "BUY", "OPEN", 1, 3000.0, "test",
                        contract_multiplier=300, margin_rate=0.15)

    trade = broker.submit_order("IF.CFX", "SELL", "CLOSE", 1, 3100.0, "test",
                                contract_multiplier=300, margin_rate=0.15)
    assert trade is not None
    assert trade.offset == "CLOSE"

    # PnL = (3100 - 3000) * 1 * 300 = 30_000
    assert broker._realized_pnl == pytest.approx(30_000.0)
    assert broker.get_position("IF.CFX") is None


def test_close_short_pnl():
    """Open short then close at lower price — should have positive PnL."""
    broker = make_broker(commission_rate=0.0, slippage_points=0.0)
    broker.submit_order("IF.CFX", "SELL", "OPEN", 1, 3000.0, "test",
                        contract_multiplier=300, margin_rate=0.15)
    broker.submit_order("IF.CFX", "BUY", "CLOSE", 1, 2900.0, "test",
                        contract_multiplier=300, margin_rate=0.15)

    # PnL = (3000 - 2900) * 1 * 300 = 30_000
    assert broker._realized_pnl == pytest.approx(30_000.0)
    assert broker.get_position("IF.CFX") is None


def test_insufficient_margin():
    """Opening a position requiring more margin than available should be rejected."""
    broker = make_broker(initial_capital=1_000)
    # Required margin = 5000 * 1 * 200 * 0.15 = 150_000 >> 1_000
    trade = broker.submit_order("IM.CFX", "BUY", "OPEN", 1, 5000.0, "test",
                                contract_multiplier=200, margin_rate=0.15)
    assert trade is None
    assert broker.get_position("IM.CFX") is None


def test_daily_settlement():
    """update_daily should reflect unrealized PnL in balance."""
    broker = make_broker(commission_rate=0.0, slippage_points=0.0)
    broker.submit_order("IF.CFX", "BUY", "OPEN", 1, 3000.0, "test",
                        contract_multiplier=300, margin_rate=0.15)

    state = broker.update_daily("20240101", {"IF.CFX": 3200.0})
    assert isinstance(state, AccountState)
    # unrealized = (3200 - 3000) * 1 * 300 = 60_000
    assert state.unrealized_pnl == pytest.approx(60_000.0)
    # balance = 1_000_000 + 60_000 = 1_060_000
    assert state.balance == pytest.approx(1_060_000.0)


def test_multiple_positions():
    """Can hold positions in multiple symbols simultaneously."""
    broker = make_broker(commission_rate=0.0, slippage_points=0.0)
    broker.submit_order("IF.CFX", "BUY", "OPEN", 1, 3000.0, "test",
                        contract_multiplier=300, margin_rate=0.15)
    broker.submit_order("IM.CFX", "SELL", "OPEN", 2, 5000.0, "test",
                        contract_multiplier=200, margin_rate=0.15)

    positions = broker.get_all_positions()
    assert len(positions) == 2
    assert "IF.CFX" in positions
    assert "IM.CFX" in positions


def test_short_open_and_close():
    """Open short and verify closing it produces correct realized PnL."""
    broker = make_broker(commission_rate=0.0, slippage_points=0.0)
    broker.submit_order("IM.CFX", "SELL", "OPEN", 3, 5000.0, "test",
                        contract_multiplier=200, margin_rate=0.15)

    # Close at a loss (price went up)
    broker.submit_order("IM.CFX", "BUY", "CLOSE", 3, 5100.0, "test",
                        contract_multiplier=200, margin_rate=0.15)

    # PnL = (5000 - 5100) * 3 * 200 = -60_000
    assert broker._realized_pnl == pytest.approx(-60_000.0)
    assert broker.get_position("IM.CFX") is None


def test_slippage_applied():
    """Slippage should worsen fills: BUY pays more, SELL receives less."""
    broker = make_broker(commission_rate=0.0, slippage_points=5.0)
    trade = broker.submit_order("IM.CFX", "BUY", "OPEN", 1, 5000.0, "test",
                                contract_multiplier=200, margin_rate=0.15)
    assert trade is not None
    assert trade.price == pytest.approx(5005.0)  # 5000 + 5

    broker2 = make_broker(commission_rate=0.0, slippage_points=5.0)
    trade2 = broker2.submit_order("IM.CFX", "SELL", "OPEN", 1, 5000.0, "test",
                                  contract_multiplier=200, margin_rate=0.15)
    assert trade2 is not None
    assert trade2.price == pytest.approx(4995.0)  # 5000 - 5


def test_commission_deducted():
    """Commission should reduce available capital."""
    broker = make_broker(commission_rate=0.001, slippage_points=0.0)
    broker.submit_order("IF.CFX", "BUY", "OPEN", 1, 3000.0, "test",
                        contract_multiplier=300, margin_rate=0.15)

    state = broker.update_daily("20240101", {"IF.CFX": 3000.0})
    # commission = 3000 * 1 * 300 * 0.001 = 900
    assert state.commission_total == pytest.approx(900.0)


def test_trade_history():
    """Executed trades are recorded in trade history."""
    broker = make_broker(commission_rate=0.0, slippage_points=0.0)
    broker.submit_order("IF.CFX", "BUY", "OPEN", 1, 3000.0, "test",
                        contract_multiplier=300, margin_rate=0.15)
    broker.submit_order("IF.CFX", "SELL", "CLOSE", 1, 3100.0, "test",
                        contract_multiplier=300, margin_rate=0.15)

    history = broker.get_trade_history()
    assert len(history) == 2
    assert history[0].offset == "OPEN"
    assert history[1].offset == "CLOSE"


def test_close_nonexistent_position():
    """Closing a position that doesn't exist should return None (not raise)."""
    broker = make_broker()
    result = broker.submit_order("NOTEXIST.CFX", "SELL", "CLOSE", 1, 3000.0, "test")
    assert result is None
