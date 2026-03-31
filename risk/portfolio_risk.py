"""
portfolio_risk.py
-----------------
职责：多策略组合级别的风险管理。

在单策略风控（risk_checker.py）之上，
提供跨策略的整体风险监控和限额分配。

功能：
1. 组合总体风险预算分配（各策略的风险敞口上限）
2. 跨策略相关性监控（防止策略间过度相关导致集中风险）
3. 压力测试：历史极端情景下的组合损益模拟
4. 动态再平衡触发判断
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskBudget:
    """
    组合风险预算配置。

    Attributes
    ----------
    max_total_margin_ratio : float
        组合总保证金占净值比例上限
    max_single_strategy_weight : float
        单策略风险权重上限（如 0.5 = 不超过 50% 风险来自单策略）
    max_correlation_threshold : float
        策略间相关性告警阈值（>此值发出警告）
    daily_var_limit : float
        每日 VaR 上限（占净值百分比，95% 置信水平）
    """
    max_total_margin_ratio: float = 0.60
    max_single_strategy_weight: float = 0.50
    max_correlation_threshold: float = 0.70
    daily_var_limit: float = 0.03


@dataclass
class PortfolioRiskSnapshot:
    """
    组合风险快照（每日更新）。

    Attributes
    ----------
    trade_date : str
        快照日期
    total_equity : float
        组合总净值（元）
    total_margin : float
        组合总保证金占用（元）
    margin_ratio : float
        保证金比率
    strategy_pnl : dict[str, float]
        各策略当日盈亏
    strategy_weights : dict[str, float]
        各策略净值贡献比例
    net_delta : float
        组合总净 Delta 名义敞口（元）
    net_vega : float
        组合总净 Vega 敞口（元/1%波动率）
    daily_pnl : float
        当日组合总盈亏
    """
    trade_date: str
    total_equity: float
    total_margin: float
    margin_ratio: float
    strategy_pnl: dict[str, float] = field(default_factory=dict)
    strategy_weights: dict[str, float] = field(default_factory=dict)
    net_delta: float = 0.0
    net_vega: float = 0.0
    daily_pnl: float = 0.0


class PortfolioRiskManager:
    """
    多策略组合风险管理器。

    Parameters
    ----------
    config : Config
        系统配置
    risk_budget : PortfolioRiskBudget, optional
        风险预算配置，默认使用保守参数
    """

    def __init__(
        self,
        config: Config,
        risk_budget: Optional[PortfolioRiskBudget] = None,
    ) -> None:
        self.config = config
        self.risk_budget = risk_budget or PortfolioRiskBudget()
        self._snapshots: list[PortfolioRiskSnapshot] = []

    def check_portfolio_risk(
        self,
        trade_date: str,
        strategy_positions: dict[str, pd.DataFrame],
        account_info: dict,
    ) -> tuple[bool, list[str]]:
        """
        检查组合整体风险是否超限。

        Parameters
        ----------
        strategy_positions : dict[str, pd.DataFrame]
            各策略持仓（键为 strategy_id）
        account_info : dict
            账户资金信息

        Returns
        -------
        tuple[bool, list[str]]
            (是否通过风控, 告警信息列表)
            False 表示需要减仓或暂停交易
        """
        raise NotImplementedError("TODO: 实现组合级风险检查")

    def calc_strategy_correlations(
        self,
        pnl_history: dict[str, pd.Series],
        lookback: int = 60,
    ) -> pd.DataFrame:
        """
        计算各策略盈亏序列的相关性矩阵。

        Parameters
        ----------
        pnl_history : dict[str, pd.Series]
            各策略历史日盈亏（键为 strategy_id）
        lookback : int
            计算窗口（交易日）

        Returns
        -------
        pd.DataFrame
            策略间相关性矩阵
        """
        raise NotImplementedError("TODO: 实现策略相关性计算")

    def calc_portfolio_var(
        self,
        returns_history: dict[str, pd.Series],
        weights: dict[str, float],
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        计算组合 VaR（历史模拟法或参数法）。

        Parameters
        ----------
        method : str
            'historical'（历史模拟）/ 'parametric'（正态分布假设）

        Returns
        -------
        float
            VaR 值（正数，表示可能损失占净值比例）
        """
        raise NotImplementedError("TODO: 实现组合 VaR 计算")

    def suggest_rebalancing(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        rebalance_threshold: float = 0.05,
    ) -> dict[str, float]:
        """
        判断是否需要再平衡，返回建议调整量。

        Parameters
        ----------
        rebalance_threshold : float
            权重偏差超过此值时触发再平衡

        Returns
        -------
        dict[str, float]
            各策略建议调整量（正数=增加，负数=减少）
        """
        raise NotImplementedError("TODO: 实现再平衡建议")
