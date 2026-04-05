"""
Layer 1: Factor 基类 + 评估接口
================================
所有因子继承此类，实现 compute_series()。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """因子基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """因子唯一标识符，如 'mom_simple_12'。"""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """因子类别：'momentum', 'volatility', 'volume', 'structure', 'composite'。"""
        pass

    @property
    def description(self) -> str:
        return ""

    @property
    def params(self) -> dict:
        return {}

    @abstractmethod
    def compute_series(self, bar_5m: pd.DataFrame,
                       bar_15m: pd.DataFrame = None,
                       daily: pd.DataFrame = None) -> pd.Series:
        """
        向量化计算完整时间序列的因子值。

        输入：
          bar_5m: DataFrame with columns [open, high, low, close, volume]
          bar_15m: 可选，15分钟K线
          daily: 可选，日线数据
        输出：
          pd.Series，index 与 bar_5m 对齐
        """
        pass

    def compute(self, bar_5m: pd.DataFrame, **kwargs) -> float:
        """计算最新一根bar的因子值（用于实盘）。"""
        series = self.compute_series(bar_5m, **kwargs)
        return float(series.iloc[-1]) if len(series) > 0 else 0.0

    def __repr__(self):
        return f"<Factor {self.name} [{self.category}]>"
