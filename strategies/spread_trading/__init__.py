"""strategies.spread_trading 包：价差交易策略"""
from .strategy import SpreadTradingStrategy, SpreadTradingConfig
from .signal import SpreadSignal, SpreadSignalGenerator
from .pairs import PairConfig, SpreadStats

__all__ = [
    "SpreadTradingStrategy", "SpreadTradingConfig",
    "SpreadSignal", "SpreadSignalGenerator",
    "PairConfig", "SpreadStats",
]
