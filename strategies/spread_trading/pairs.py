"""
pairs.py
--------
职责：价差对管理和统计分析。

维护协整配对（如 IF/IH、IC/IM）的价差统计状态，
计算 Z-score 和协整关系强度。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PairConfig:
    """
    价差对配置。

    Attributes
    ----------
    leg1 : str
        腿1品种代码（如 'IF'）
    leg2 : str
        腿2品种代码（如 'IH'）
    hedge_ratio : float
        对冲比例（leg1 + hedge_ratio × leg2 构成价差）
    lookback : int
        滚动统计窗口（交易日）
    entry_zscore : float
        入场 Z-score 阈值（>此值开空价差）
    exit_zscore : float
        出场 Z-score 阈值（<此值平仓）
    stop_zscore : float
        止损 Z-score 阈值
    """
    leg1: str
    leg2: str
    hedge_ratio: float = 1.0
    lookback: int = 60
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_zscore: float = 3.5


@dataclass
class SpreadStats:
    """
    价差统计状态（滚动更新）。

    Attributes
    ----------
    pair_id : str
        配对标识符，如 'IF-IH'
    spread_mean : float
        价差滚动均值
    spread_std : float
        价差滚动标准差
    current_spread : float
        当前价差值
    zscore : float
        当前价差 Z-score
    half_life : float
        均值回归半衰期（交易日）—— 由 OU 过程估计
    coint_pvalue : float
        协整检验 p 值（<0.05 表示协整关系显著）
    is_cointegrated : bool
        是否通过协整检验
    """
    pair_id: str
    spread_mean: float = 0.0
    spread_std: float = 0.0
    current_spread: float = 0.0
    zscore: float = 0.0
    half_life: float = float("inf")
    coint_pvalue: float = 1.0
    is_cointegrated: bool = False

    @property
    def is_overvalued(self) -> bool:
        """价差高估（做空价差机会）"""
        return self.zscore > 0

    @property
    def abs_zscore(self) -> float:
        return abs(self.zscore)


class PairsManager:
    """
    价差对管理器。

    负责维护多个配对的协整状态，
    提供每日价差 Z-score 更新。
    """

    def __init__(self, pairs: list[PairConfig]) -> None:
        self.pairs = {f"{p.leg1}-{p.leg2}": p for p in pairs}
        self._stats: dict[str, SpreadStats] = {}

    def update(
        self,
        trade_date: str,
        price_data: dict[str, float],
    ) -> dict[str, SpreadStats]:
        """
        更新所有配对的价差统计。

        Parameters
        ----------
        price_data : dict[str, float]
            当日各品种收盘价（键为品种代码）

        Returns
        -------
        dict[str, SpreadStats]
            更新后的统计字典（键为 pair_id）
        """
        raise NotImplementedError("TODO: 实现价差统计更新")

    def _calc_zscore(
        self,
        spread_series: pd.Series,
        lookback: int,
    ) -> float:
        """计算滚动 Z-score"""
        raise NotImplementedError("TODO: 实现 Z-score 计算")

    def _test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series,
    ) -> tuple[float, float]:
        """
        Engle-Granger 协整检验。

        Returns
        -------
        tuple
            (hedge_ratio, p_value)
        """
        raise NotImplementedError("TODO: 实现协整检验（使用 statsmodels）")
