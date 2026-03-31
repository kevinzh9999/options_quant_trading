"""
strategy.py
-----------
职责：价差交易策略主类。

基于协整配对（如 IF/IH、IC/IM）捕捉价差均值回归机会。
使用 Engle-Granger 协整检验和 Z-score 入场/出场。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalDirection, SignalStrength, StrategyConfig
from .pairs import PairConfig, PairsManager
from .signal import SpreadSignalGenerator, SpreadSignal, SpreadDirection

logger = logging.getLogger(__name__)


@dataclass
class SpreadTradingConfig(StrategyConfig):
    """
    价差交易策略配置。

    Attributes
    ----------
    pairs : list[PairConfig]
        交易配对列表
    coint_lookback : int
        协整检验历史窗口（交易日）
    recheck_coint_days : int
        重新检验协整关系的频率（交易日）
    """
    pairs: list[PairConfig] = field(default_factory=list)
    coint_lookback: int = 120
    recheck_coint_days: int = 20


class SpreadTradingStrategy(BaseStrategy):
    """
    价差套利策略（股指期货跨品种）。

    典型配对：IF-IH（沪深300 vs 上证50）、IC-IM（中证500 vs 中证1000）

    Parameters
    ----------
    config : SpreadTradingConfig
        策略配置
    """

    def __init__(self, config: SpreadTradingConfig) -> None:
        super().__init__(config)
        self.spread_config: SpreadTradingConfig = config
        self._pairs_manager = PairsManager(config.pairs)
        self._signal_generator = SpreadSignalGenerator(self._pairs_manager)

    @property
    def name(self) -> str:
        return "价差套利策略"

    def generate_signals(
        self,
        trade_date: str,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """
        生成当日价差信号。

        Notes
        -----
        - 提取各品种当日收盘价
        - 更新价差统计（Z-score）
        - 对每个配对生成信号
        - 将 SpreadSignal 转换为通用 Signal
        """
        raise NotImplementedError("TODO: 实现价差信号生成主流程")

    def on_fill(
        self,
        order_id: str,
        instrument: str,
        direction: str,
        volume: int,
        price: float,
        trade_date: str,
    ) -> None:
        """成交回调"""
        raise NotImplementedError("TODO: 实现成交回调")

    @staticmethod
    def _spread_signal_to_signals(
        spread_signal: SpreadSignal,
        strategy_id: str,
        volume: int,
    ) -> list[Signal]:
        """
        将 SpreadSignal 转换为两个 Signal（leg1 + leg2）。

        Returns
        -------
        list[Signal]
            长度为 2 的信号列表
        """
        raise NotImplementedError("TODO: 实现价差信号分解")
