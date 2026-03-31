"""
backtest.py
-----------
职责：价差交易策略回测适配器。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from .strategy import SpreadTradingStrategy

logger = logging.getLogger(__name__)


@dataclass
class SpreadBacktestResult:
    """价差交易策略回测结果"""
    strategy_id: str
    start_date: str
    end_date: str
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_days: float = 0.0
    avg_entry_zscore: float = 0.0   # 平均入场 Z-score
    avg_exit_zscore: float = 0.0    # 平均出场 Z-score
    pnl_series: pd.Series = field(default_factory=pd.Series)


class SpreadTradingBacktester:
    """价差交易策略回测器"""

    def __init__(self, strategy: SpreadTradingStrategy) -> None:
        self.strategy = strategy

    def run(
        self,
        start_date: str,
        end_date: str,
        market_data: dict[str, pd.DataFrame],
        initial_capital: float = 1_000_000.0,
    ) -> SpreadBacktestResult:
        """
        运行价差交易回测。

        Notes
        -----
        - 每日收盘后计算价差 Z-score
        - Z-score 超过阈值时次日开仓
        - Z-score 回到 exit_zscore 内时平仓
        - 超过 stop_zscore 时止损平仓
        """
        raise NotImplementedError("TODO: 实现价差交易回测主流程")
