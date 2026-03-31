"""Tests for TrendSignalGenerator and TrendPositionSizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.trend_following.signal import TrendSignalGenerator
from strategies.trend_following.position import TrendPositionSizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, start: float = 5000.0, trend: float = 10.0,
             noise: float = 5.0) -> pd.DataFrame:
    """
    Create a synthetic OHLCV DataFrame with a trend and noise.

    Parameters
    ----------
    n : int
        Number of rows.
    start : float
        Starting close price.
    trend : float
        Daily price increment (positive = uptrend).
    noise : float
        Random noise amplitude.
    """
    rng = np.random.default_rng(42)
    closes = start + np.arange(n) * trend + rng.normal(0, noise, n)
    closes = np.maximum(closes, 10.0)
    # Use close as both high and low so Donchian breakout can trigger on close
    df = pd.DataFrame({
        "trade_date": [f"2024{str(i+101)[1:]}" for i in range(n)],
        "open": closes * 0.999,
        "high": closes,       # high == close so Donchian upper can be reached
        "low": closes * 0.99,
        "close": closes,
        "volume": 10000,
    })
    return df


def _make_flat_df(n: int = 100, price: float = 5000.0) -> pd.DataFrame:
    """Create a flat (no trend) OHLCV DataFrame."""
    return _make_df(n, start=price, trend=0.0, noise=1.0)


def _make_downtrend_df(n: int = 100) -> pd.DataFrame:
    """Downtrend df where low == close so Donchian lower breakout can trigger."""
    rng = np.random.default_rng(42)
    closes = 5000.0 + np.arange(n) * (-10.0) + rng.normal(0, 5.0, n)
    closes = np.maximum(closes, 10.0)
    df = pd.DataFrame({
        "trade_date": [f"2024{str(i+101)[1:]}" for i in range(n)],
        "open": closes * 1.001,
        "high": closes * 1.01,
        "low": closes,          # low == close so Donchian lower is reachable
        "close": closes,
        "volume": 10000,
    })
    return df


# ---------------------------------------------------------------------------
# TrendSignalGenerator tests
# ---------------------------------------------------------------------------

class TestTrendSignalGenerator:

    def setup_method(self):
        self.gen = TrendSignalGenerator(
            fast_period=5,
            slow_period=20,
            donchian_period=10,
            adx_period=7,
            adx_threshold=15.0,
            atr_period=10,
        )

    def test_compute_indicators_columns(self):
        df = _make_df(60)
        result = self.gen.compute_indicators(df)
        for col in ("sma_fast", "sma_slow", "donchian_upper", "donchian_lower",
                    "adx", "atr", "atr_pct"):
            assert col in result.columns, f"Missing column: {col}"

    def test_compute_indicators_preserves_input_columns(self):
        df = _make_df(60)
        result = self.gen.compute_indicators(df)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_indicator_lengths_match(self):
        df = _make_df(60)
        result = self.gen.compute_indicators(df)
        assert len(result) == len(df)

    def test_sma_fast_smaller_than_slow_in_downtrend(self):
        """In a strong downtrend, fast SMA should be below slow SMA."""
        df = _make_downtrend_df(80)
        result = self.gen.compute_indicators(df)
        # Check the last row after warmup
        last = result.iloc[-1]
        assert last["sma_fast"] < last["sma_slow"], (
            f"Expected sma_fast < sma_slow in downtrend, "
            f"got {last['sma_fast']:.2f} vs {last['sma_slow']:.2f}"
        )

    def test_sma_fast_larger_than_slow_in_uptrend(self):
        """In a strong uptrend, fast SMA should be above slow SMA."""
        df = _make_df(80, trend=20.0)
        result = self.gen.compute_indicators(df)
        last = result.iloc[-1]
        assert last["sma_fast"] > last["sma_slow"]

    def test_get_signal_returns_dict_with_required_keys(self):
        df = _make_df(80)
        result = self.gen.compute_indicators(df)
        sig = self.gen.get_signal(result)
        for key in ("direction", "signal_type", "strength", "stop_loss_price",
                    "adx", "atr_pct", "trend_aligned"):
            assert key in sig, f"Missing key in signal: {key}"

    def test_insufficient_data_returns_hold(self):
        """Too few rows should return HOLD signal."""
        df = _make_df(5)  # way less than slow_period
        result = self.gen.compute_indicators(df)
        sig = self.gen.get_signal(result)
        assert sig["signal_type"] == "HOLD"
        assert sig["direction"] == "FLAT"

    def test_long_entry_in_uptrend(self):
        """Strong uptrend with breakout should eventually generate LONG signal."""
        df = _make_df(200, trend=30.0, noise=2.0)
        # Scan through history slices, recomputing indicators each time to avoid
        # look-ahead bias — same pattern the engine uses
        found_long = False
        for i in range(50, len(df)):
            result = self.gen.compute_indicators(df.iloc[:i+1])
            sig = self.gen.get_signal(result)
            if sig["direction"] == "LONG" and sig["signal_type"] == "ENTRY":
                found_long = True
                break
        assert found_long, "Expected a LONG entry signal in a strong uptrend"

    def test_short_entry_in_downtrend(self):
        """Strong downtrend should eventually generate SHORT signal."""
        df = _make_downtrend_df(200)
        found_short = False
        for i in range(50, len(df)):
            result = self.gen.compute_indicators(df.iloc[:i+1])
            sig = self.gen.get_signal(result)
            if sig["direction"] == "SHORT" and sig["signal_type"] == "ENTRY":
                found_short = True
                break
        assert found_short, "Expected a SHORT entry signal in a strong downtrend"

    def test_stop_loss_triggered_for_long(self):
        """Stop loss should fire when price drops far below entry."""
        df = _make_df(60, trend=5.0)
        result = self.gen.compute_indicators(df)
        entry_price = float(result["close"].iloc[-1])
        atr = float(result["atr"].iloc[-1])

        # Simulate price crashing far below entry
        crash_price = entry_price - 10 * atr  # way below stop
        result2 = result.copy()
        result2.at[result2.index[-1], "close"] = crash_price
        result2.at[result2.index[-1], "atr"] = atr

        sig = self.gen.get_signal(
            result2,
            current_position="LONG",
            entry_price=entry_price,
            atr_stop_multiplier=2.5,
        )
        assert sig["signal_type"] == "STOP_LOSS"

    def test_exit_long_when_price_below_slow_ma(self):
        """Long position should be exited when close drops below slow SMA."""
        df = _make_df(80, trend=5.0)
        result = self.gen.compute_indicators(df)
        slow_ma = float(result["sma_slow"].iloc[-1])

        # Force close below slow MA
        result2 = result.copy()
        result2.at[result2.index[-1], "close"] = slow_ma * 0.95

        sig = self.gen.get_signal(
            result2,
            current_position="LONG",
            entry_price=slow_ma * 1.1,  # entry was above, no stop triggered
            atr_stop_multiplier=100.0,  # very wide stop so only exit fires
        )
        assert sig["signal_type"] == "EXIT"

    def test_atr_pct_is_positive(self):
        df = _make_df(60)
        result = self.gen.compute_indicators(df)
        # After warmup, ATR% should be positive
        valid = result["atr_pct"].dropna()
        assert (valid > 0).all()

    def test_donchian_upper_geq_lower(self):
        df = _make_df(60)
        result = self.gen.compute_indicators(df)
        valid_upper = result["donchian_upper"].dropna()
        valid_lower = result["donchian_lower"].dropna()
        # Align by index
        aligned = pd.concat([valid_upper, valid_lower], axis=1).dropna()
        assert (aligned["donchian_upper"] >= aligned["donchian_lower"]).all()


# ---------------------------------------------------------------------------
# TrendPositionSizer tests
# ---------------------------------------------------------------------------

class TestTrendPositionSizer:

    def setup_method(self):
        self.sizer = TrendPositionSizer(vol_target=0.15, max_position_per_symbol=0.20)

    def test_calculate_lots_basic(self):
        """Lots should be a non-negative integer."""
        lots = self.sizer.calculate_lots(
            symbol="IM.CFX",
            current_price=5000.0,
            atr=50.0,
            account_equity=1_000_000,
            capital_allocation=0.8,
            n_symbols=1,
            contract_multiplier=200,
        )
        assert isinstance(lots, int)
        assert lots >= 0

    def test_calculate_lots_zero_price(self):
        lots = self.sizer.calculate_lots(
            symbol="X", current_price=0.0, atr=50.0,
            account_equity=1_000_000, capital_allocation=0.8,
            n_symbols=1, contract_multiplier=200,
        )
        assert lots == 0

    def test_calculate_lots_zero_atr(self):
        lots = self.sizer.calculate_lots(
            symbol="X", current_price=5000.0, atr=0.0,
            account_equity=1_000_000, capital_allocation=0.8,
            n_symbols=1, contract_multiplier=200,
        )
        assert lots == 0

    def test_calculate_lots_more_equity_gives_more_lots(self):
        """Larger equity should result in equal or more lots."""
        lots_small = self.sizer.calculate_lots(
            symbol="X", current_price=5000.0, atr=50.0,
            account_equity=500_000, capital_allocation=0.8,
            n_symbols=1, contract_multiplier=200,
        )
        lots_large = self.sizer.calculate_lots(
            symbol="X", current_price=5000.0, atr=50.0,
            account_equity=5_000_000, capital_allocation=0.8,
            n_symbols=1, contract_multiplier=200,
        )
        assert lots_large >= lots_small

    def test_calculate_lots_capped_by_max_position(self):
        """
        With very low vol and very large equity, lots should be capped by
        max_position_per_symbol.
        """
        sizer = TrendPositionSizer(vol_target=0.15, max_position_per_symbol=0.10)
        equity = 100_000_000
        price = 5000.0
        mult = 200
        lots = sizer.calculate_lots(
            symbol="X", current_price=price, atr=1.0,  # tiny atr = high vol_target ratio
            account_equity=equity, capital_allocation=1.0,
            n_symbols=1, contract_multiplier=mult,
        )
        max_lots = int(sizer.max_position_per_symbol * equity / (price * mult))
        assert lots <= max_lots

    def test_calculate_stop_loss_long(self):
        stop = self.sizer.calculate_stop_loss(
            entry_price=5000.0, atr=50.0, direction="LONG", multiplier=2.5
        )
        expected = 5000.0 - 2.5 * 50.0
        assert stop == pytest.approx(expected)

    def test_calculate_stop_loss_short(self):
        stop = self.sizer.calculate_stop_loss(
            entry_price=5000.0, atr=50.0, direction="SHORT", multiplier=2.5
        )
        expected = 5000.0 + 2.5 * 50.0
        assert stop == pytest.approx(expected)

    def test_stop_loss_long_below_entry(self):
        stop = self.sizer.calculate_stop_loss(5000.0, 50.0, "LONG")
        assert stop < 5000.0

    def test_stop_loss_short_above_entry(self):
        stop = self.sizer.calculate_stop_loss(5000.0, 50.0, "SHORT")
        assert stop > 5000.0

    def test_n_symbols_reduces_lots(self):
        """More symbols means less allocation per symbol."""
        lots_1 = self.sizer.calculate_lots(
            symbol="X", current_price=5000.0, atr=50.0,
            account_equity=1_000_000, capital_allocation=0.8,
            n_symbols=1, contract_multiplier=200,
        )
        lots_4 = self.sizer.calculate_lots(
            symbol="X", current_price=5000.0, atr=50.0,
            account_equity=1_000_000, capital_allocation=0.8,
            n_symbols=4, contract_multiplier=200,
        )
        assert lots_4 <= lots_1
