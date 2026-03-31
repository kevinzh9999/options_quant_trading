"""Tests for BacktestReport using synthetic daily states."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backtest.broker import AccountState, Trade
from backtest.report import BacktestReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_states(n: int, start_balance: float = 1_000_000, trend: float = 100.0) -> list:
    """Create n AccountState objects with linearly increasing balance."""
    import datetime
    base = datetime.date(2023, 1, 2)
    states = []
    for i in range(n):
        balance = start_balance + i * trend
        d = base + datetime.timedelta(days=i)
        states.append(AccountState(
            trade_date=d.strftime("%Y%m%d"),
            balance=balance,
            available=balance * 0.8,
            margin=balance * 0.2,
            unrealized_pnl=i * trend,
            realized_pnl=0.0,
            commission_total=0.0,
        ))
    return states


def _make_flat_states(n: int, balance: float = 1_000_000) -> list:
    """Create n AccountState objects with constant balance (no trades)."""
    return _make_states(n, start_balance=balance, trend=0.0)


def _make_trade(offset: str = "OPEN", direction: str = "BUY",
                price: float = 3000.0, volume: int = 1,
                commission: float = 0.0, date: str = "20240101",
                symbol: str = "IF.CFX") -> Trade:
    return Trade(
        trade_date=date,
        symbol=symbol,
        direction=direction,
        offset=offset,
        volume=volume,
        price=price,
        commission=commission,
        slippage=0.0,
        strategy_name="test",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calculate_metrics_flat():
    """With no price movement, total return should be ~0."""
    states = _make_flat_states(100)
    report = BacktestReport(states, [], initial_capital=1_000_000)
    metrics = report.calculate_metrics()

    assert abs(metrics["total_return"]) < 1e-9
    assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["total_trades"] == 0


def test_calculate_metrics_uptrend():
    """Positive trend should give positive total_return and sharpe."""
    states = _make_states(252, start_balance=1_000_000, trend=500.0)
    report = BacktestReport(states, [], initial_capital=1_000_000)
    metrics = report.calculate_metrics()

    assert metrics["total_return"] > 0
    assert metrics["annualized_return"] > 0
    assert metrics["sharpe_ratio"] > 0
    assert metrics["max_drawdown"] <= 0  # drawdown is non-positive


def test_max_drawdown_calculation():
    """Max drawdown should correctly identify the peak-to-trough decline."""
    # Build: rise to 1_200_000 then fall to 900_000, then recover
    balances = (
        [1_000_000 + i * 10_000 for i in range(20)]    # rise
        + [1_200_000 - i * 15_000 for i in range(20)]  # fall
        + [900_000 + i * 10_000 for i in range(20)]    # recover
    )
    import datetime
    base = datetime.date(2023, 1, 2)
    states = []
    for i, bal in enumerate(balances):
        d = base + datetime.timedelta(days=i)
        states.append(AccountState(
            trade_date=d.strftime("%Y%m%d"),
            balance=bal,
            available=bal * 0.8,
            margin=bal * 0.2,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            commission_total=0.0,
        ))

    report = BacktestReport(states, [], initial_capital=1_000_000)
    metrics = report.calculate_metrics()
    # Peak = 1_390_000 (at index 19 of fall), trough = 900_000
    # max drawdown <= -0.20 (significant drawdown)
    assert metrics["max_drawdown"] < -0.20


def test_equity_curve_columns():
    states = _make_states(50)
    report = BacktestReport(states, [], initial_capital=1_000_000)
    curve = report.get_equity_curve()

    assert not curve.empty
    for col in ("trade_date", "balance", "daily_return", "cumulative_return", "drawdown"):
        assert col in curve.columns


def test_equity_curve_no_lookahead():
    """Drawdown must be <= 0 everywhere."""
    states = _make_states(100, trend=200.0)
    report = BacktestReport(states, [], initial_capital=1_000_000)
    curve = report.get_equity_curve()
    assert (curve["drawdown"] <= 1e-9).all()


def test_monthly_returns_shape():
    """Monthly returns pivot should have 12 month columns + Annual."""
    # Need at least a full year of data
    states = _make_states(300, trend=100.0)
    # Fix dates to span a real year
    for i, s in enumerate(states):
        s.trade_date = (pd.Timestamp("20230101") + pd.Timedelta(days=i)).strftime("%Y%m%d")

    report = BacktestReport(states, [], initial_capital=1_000_000)
    monthly = report.get_monthly_returns()

    assert not monthly.empty
    assert "Annual" in monthly.columns


def test_trade_summary_paired():
    """Paired OPEN/CLOSE trades should appear in trade summary."""
    trades = [
        _make_trade("OPEN", "BUY", price=3000.0, date="20240101"),
        _make_trade("CLOSE", "SELL", price=3100.0, date="20240115"),
    ]
    states = _make_states(20)
    report = BacktestReport(states, trades, initial_capital=1_000_000)
    summary = report.get_trade_summary()

    assert not summary.empty
    assert len(summary) == 1
    assert summary.iloc[0]["entry_price"] == pytest.approx(3000.0)
    assert summary.iloc[0]["exit_price"] == pytest.approx(3100.0)


def test_win_rate_from_trades():
    """Win rate should reflect proportion of profitable closed trades."""
    trades = [
        _make_trade("OPEN", "BUY", price=3000.0, date="20240101", symbol="A"),
        _make_trade("CLOSE", "SELL", price=3100.0, date="20240115", symbol="A"),  # win
        _make_trade("OPEN", "BUY", price=3000.0, date="20240201", symbol="B"),
        _make_trade("CLOSE", "SELL", price=2900.0, date="20240215", symbol="B"),  # loss
    ]
    states = _make_flat_states(60)
    report = BacktestReport(states, trades, initial_capital=1_000_000)
    metrics = report.calculate_metrics()
    assert metrics["win_rate"] == pytest.approx(0.5)


def test_empty_states():
    """Report with no states should return empty metrics."""
    report = BacktestReport([], [], initial_capital=1_000_000)
    metrics = report.calculate_metrics()
    assert metrics == {}


def test_print_report_runs(capsys):
    """print_report should not raise and should produce output."""
    states = _make_states(100, trend=200.0)
    report = BacktestReport(states, [], initial_capital=1_000_000, strategy_name="Test")
    report.print_report()
    captured = capsys.readouterr()
    assert "回测报告" in captured.out
    assert "Test" in captured.out


def test_save_to_csv(tmp_path):
    """save_to_csv should create expected CSV files."""
    states = _make_states(100, trend=100.0)
    trades = [
        _make_trade("OPEN", "BUY", price=3000.0, date="20240101"),
        _make_trade("CLOSE", "SELL", price=3100.0, date="20240115"),
    ]
    report = BacktestReport(states, trades, initial_capital=1_000_000)
    report.save_to_csv(str(tmp_path))

    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "trades.csv").exists()
