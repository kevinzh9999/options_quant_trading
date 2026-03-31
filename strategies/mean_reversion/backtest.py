"""
backtest.py
-----------
职责：均值回归策略回测适配器。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from .strategy import MeanReversionStrategy

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionBacktestResult:
    """均值回归策略回测结果"""
    strategy_id: str
    start_date: str
    end_date: str
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_days: float = 0.0
    avg_entry_rsi: float = 0.0
    avg_exit_bb_zscore: float = 0.0
    pnl_series: pd.Series = field(default_factory=pd.Series)


class MeanReversionBacktester:
    """均值回归策略回测器"""

    def __init__(self, strategy: MeanReversionStrategy) -> None:
        self.strategy = strategy

    def run(
        self,
        start_date: str,
        end_date: str,
        market_data: dict[str, pd.DataFrame],
        initial_capital: float = 1_000_000.0,
        contract_multiplier: int = 300,
    ) -> MeanReversionBacktestResult:
        """
        运行均值回归回测。

        Notes
        -----
        - 超卖/超买信号次日开仓
        - 价格回归均线时平仓（止盈）
        - 超过止损幅度时强制平仓
        - 持仓期间每日更新止损价格
        """
        raise NotImplementedError("TODO: 实现均值回归回测主流程")
