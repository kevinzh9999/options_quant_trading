"""
performance.py
--------------
职责：策略绩效统计和评估。
计算常用的策略绩效指标：
- 年化收益率、年化波动率
- 夏普比率（Sharpe Ratio）
- 最大回撤（Max Drawdown）
- 卡尔玛比率（Calmar Ratio）
- 胜率和盈亏比
- VaR（在险价值）和 CVaR（条件在险价值）

输入：逐笔交易记录或日度净值序列
输出：绩效指标字典或报告 DataFrame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """策略绩效指标"""
    # 收益类
    total_return: float           # 总收益率
    annualized_return: float      # 年化收益率
    annualized_vol: float         # 年化波动率
    sharpe_ratio: float           # 夏普比率（无风险利率已扣除）
    sortino_ratio: float          # 索提诺比率（只计下行波动率）

    # 回撤类
    max_drawdown: float           # 最大回撤（负值）
    max_drawdown_duration: int    # 最大回撤持续天数
    calmar_ratio: float           # 卡尔玛比率 = 年化收益/最大回撤绝对值

    # 交易类
    num_trades: int               # 总交易次数（按开平仓对计）
    win_rate: float               # 胜率
    avg_win: float                # 平均盈利（元）
    avg_loss: float               # 平均亏损（元，负值）
    profit_factor: float          # 盈亏比 = avg_win / |avg_loss|

    # 风险类
    var_95: float                 # 95% VaR（日度，负值）
    cvar_95: float                # 95% CVaR（负值）


class PerformanceAnalyzer:
    """
    策略绩效分析器。

    Parameters
    ----------
    risk_free_rate : float
        年化无风险利率（小数），用于夏普比率计算，默认 0.02
    """

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # 主分析入口
    # ------------------------------------------------------------------

    def analyze(self, equity_curve: pd.Series) -> PerformanceMetrics:
        """
        基于净值曲线计算全量绩效指标。

        Parameters
        ----------
        equity_curve : pd.Series
            日度净值序列，索引为日期，值为净值（初始值通常为 1.0 或账户初始金额）

        Returns
        -------
        PerformanceMetrics
            全量绩效指标

        Notes
        -----
        - 净值曲线须按日期升序排列
        - 若净值为账户金额（而非比率），内部自动归一化
        """
        raise NotImplementedError("TODO: 实现基于净值曲线的绩效分析")

    def analyze_trades(self, trades: pd.DataFrame) -> dict[str, float]:
        """
        基于逐笔交易记录计算交易层面指标。

        Parameters
        ----------
        trades : pd.DataFrame
            交易记录，需含 open_date, close_date, pnl（每笔盈亏，元）列

        Returns
        -------
        dict[str, float]
            胜率、平均盈利、平均亏损、盈亏比等
        """
        raise NotImplementedError("TODO: 实现逐笔交易绩效分析")

    # ------------------------------------------------------------------
    # 各指标计算
    # ------------------------------------------------------------------

    def calc_sharpe(self, returns: pd.Series) -> float:
        """
        计算年化夏普比率。

        Parameters
        ----------
        returns : pd.Series
            日度收益率序列（小数）

        Returns
        -------
        float
            年化夏普比率 = (mean(r) - rf/252) / std(r) × sqrt(252)
        """
        raise NotImplementedError("TODO: 实现夏普比率计算")

    def calc_max_drawdown(self, equity_curve: pd.Series) -> tuple[float, int]:
        """
        计算最大回撤和回撤持续时间。

        Parameters
        ----------
        equity_curve : pd.Series
            净值曲线

        Returns
        -------
        tuple[float, int]
            (最大回撤比率（负值），最大回撤持续交易日数)
        """
        raise NotImplementedError("TODO: 实现最大回撤计算")

    def calc_var_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        计算历史 VaR 和 CVaR。

        Parameters
        ----------
        returns : pd.Series
            日度收益率序列
        confidence : float
            置信度，默认 0.95

        Returns
        -------
        tuple[float, float]
            (VaR（负值）, CVaR（负值）)

        Notes
        -----
        使用历史模拟法（非参数方法），对分布无正态假设
        """
        raise NotImplementedError("TODO: 实现 VaR/CVaR 计算")

    def monthly_returns_table(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        生成月度收益率矩阵（行=年份，列=月份）。

        Parameters
        ----------
        equity_curve : pd.Series
            日度净值曲线

        Returns
        -------
        pd.DataFrame
            月度收益率矩阵，便于查看季节性规律
        """
        raise NotImplementedError("TODO: 实现月度收益率矩阵")

    def to_report(self, metrics: PerformanceMetrics) -> str:
        """将绩效指标格式化为可读字符串报告"""
        raise NotImplementedError("TODO: 实现绩效报告格式化")
