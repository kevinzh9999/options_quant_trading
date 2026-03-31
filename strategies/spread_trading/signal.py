"""
signal.py
---------
职责：价差交易信号生成。

基于 Z-score 和协整检验生成价差套利信号。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .pairs import PairConfig, SpreadStats, PairsManager

logger = logging.getLogger(__name__)


class SpreadDirection(str, Enum):
    """价差信号方向"""
    BUY_SPREAD = "buy_spread"    # 做多价差（leg1多/leg2空）
    SELL_SPREAD = "sell_spread"  # 做空价差（leg1空/leg2多）
    NEUTRAL = "neutral"
    CLOSE = "close"              # 平仓信号


@dataclass
class SpreadSignal:
    """
    价差套利信号。

    Attributes
    ----------
    signal_date : str
        信号日期
    pair_id : str
        配对标识符，如 'IF-IH'
    direction : SpreadDirection
        价差交易方向
    zscore : float
        当前 Z-score
    spread_value : float
        当前价差值
    leg1_instrument : str
        腿1合约代码
    leg2_instrument : str
        腿2合约代码
    hedge_ratio : float
        对冲比例
    half_life : float
        均值回归半衰期（交易日）
    notes : str
        备注
    """
    signal_date: str
    pair_id: str
    direction: SpreadDirection
    zscore: float
    spread_value: float
    leg1_instrument: str
    leg2_instrument: str
    hedge_ratio: float
    half_life: float = float("inf")
    notes: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.direction not in (SpreadDirection.NEUTRAL,)


class SpreadSignalGenerator:
    """
    价差套利信号生成器。

    Parameters
    ----------
    pairs_manager : PairsManager
        配对管理器
    """

    def __init__(self, pairs_manager: PairsManager) -> None:
        self.pairs_manager = pairs_manager

    def generate(
        self,
        trade_date: str,
        spread_stats: SpreadStats,
        pair_config: PairConfig,
        leg1_contract: str,
        leg2_contract: str,
    ) -> SpreadSignal:
        """
        生成单对价差信号。

        Notes
        -----
        - Z-score > entry_zscore 且协整显著 → SELL_SPREAD
        - Z-score < -entry_zscore 且协整显著 → BUY_SPREAD
        - |Z-score| < exit_zscore → CLOSE（平仓信号）
        - |Z-score| > stop_zscore → CLOSE（止损平仓）
        """
        raise NotImplementedError("TODO: 实现价差信号生成")

    def generate_batch(
        self,
        trade_date: str,
        price_data: dict[str, float],
        contracts: dict[str, str],
    ) -> list[SpreadSignal]:
        """批量生成所有配对的价差信号"""
        raise NotImplementedError("TODO: 实现批量价差信号生成")
