"""
test_base_strategy.py
---------------------
测试策略框架基类和注册表。
"""

from __future__ import annotations

import pytest
import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalDirection, SignalStrength, StrategyConfig
from strategies.registry import StrategyRegistry


# ======================================================================
# 测试辅助：最小可运行的具体策略
# ======================================================================

class DummyStrategy(BaseStrategy):
    """用于测试的最简策略实现"""

    def generate_signals(
        self,
        trade_date: str,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        return [
            Signal(
                strategy_id=self.strategy_id,
                signal_date=trade_date,
                instrument="IF2406.CFX",
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,
            )
        ]

    def on_fill(self, order_id, instrument, direction, volume, price, trade_date):
        pass


# ======================================================================
# Signal 数据结构测试
# ======================================================================

class TestSignal:

    def test_is_actionable_for_non_neutral(self):
        s = Signal(
            strategy_id="test",
            signal_date="20240101",
            instrument="IF",
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
        )
        assert s.is_actionable is True

    def test_neutral_not_actionable(self):
        s = Signal(
            strategy_id="test",
            signal_date="20240101",
            instrument="IF",
            direction=SignalDirection.NEUTRAL,
            strength=SignalStrength.WEAK,
        )
        assert s.is_actionable is False

    def test_to_dict_contains_required_keys(self):
        s = Signal(
            strategy_id="test",
            signal_date="20240101",
            instrument="IF",
            direction=SignalDirection.SHORT,
            strength=SignalStrength.MODERATE,
        )
        d = s.to_dict()
        for key in ("strategy_id", "signal_date", "instrument", "direction", "strength"):
            assert key in d


# ======================================================================
# StrategyConfig 测试
# ======================================================================

class TestStrategyConfig:

    def test_validate_empty_strategy_id_raises(self):
        config = StrategyConfig(strategy_id="")
        with pytest.raises(ValueError, match="strategy_id"):
            config.validate()

    def test_validate_invalid_max_position_raises(self):
        config = StrategyConfig(strategy_id="test", max_position=0)
        with pytest.raises(ValueError, match="max_position"):
            config.validate()

    def test_valid_config_passes(self):
        config = StrategyConfig(strategy_id="test_strategy", max_position=5)
        config.validate()  # 不应抛出异常


# ======================================================================
# BaseStrategy 测试
# ======================================================================

class TestBaseStrategy:

    @pytest.fixture
    def strategy(self):
        config = StrategyConfig(
            strategy_id="dummy_test",
            enabled=True,
            universe=["IF"],
        )
        return DummyStrategy(config)

    def test_run_returns_signals(self, strategy):
        signals = strategy.run("20240101", {"IF": pd.DataFrame()})
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.LONG

    def test_disabled_strategy_returns_empty(self):
        config = StrategyConfig(strategy_id="disabled", enabled=False)
        s = DummyStrategy(config)
        signals = s.run("20240101", {})
        assert signals == []

    def test_signal_history_accumulates(self, strategy):
        strategy.run("20240101", {})
        strategy.run("20240102", {})
        assert len(strategy.get_signal_history()) == 2


# ======================================================================
# StrategyRegistry 测试
# ======================================================================

class TestStrategyRegistry:

    def test_register_and_create(self):
        registry = StrategyRegistry()
        registry.register("dummy", DummyStrategy)
        config = StrategyConfig(strategy_id="dummy_01")
        s = registry.create("dummy", config)
        assert isinstance(s, DummyStrategy)

    def test_create_unregistered_raises(self):
        registry = StrategyRegistry()
        with pytest.raises(KeyError, match="未注册"):
            registry.create("nonexistent", StrategyConfig(strategy_id="x"))

    def test_register_non_strategy_raises(self):
        registry = StrategyRegistry()
        with pytest.raises(TypeError):
            registry.register("bad", object)  # type: ignore

    def test_list_strategies(self):
        registry = StrategyRegistry()
        registry.register("aaa", DummyStrategy)
        registry.register("bbb", DummyStrategy)
        assert registry.list_strategies() == ["aaa", "bbb"]

    def test_contains(self):
        registry = StrategyRegistry()
        registry.register("my_strat", DummyStrategy)
        assert "my_strat" in registry
        assert "other" not in registry
