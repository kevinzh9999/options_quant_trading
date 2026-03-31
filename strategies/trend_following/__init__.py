"""strategies.trend_following 包：趋势跟踪策略"""
from .strategy import TrendFollowingStrategy, TrendConfig
from .signal import TrendSignalGenerator

__all__ = [
    "TrendFollowingStrategy", "TrendConfig",
    "TrendSignalGenerator",
]
