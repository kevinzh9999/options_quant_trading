"""strategies 包：多策略框架"""
from .base import BaseStrategy, Signal, StrategyConfig
from .registry import StrategyRegistry

__all__ = ["BaseStrategy", "Signal", "StrategyConfig", "StrategyRegistry"]
