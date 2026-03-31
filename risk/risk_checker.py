"""
risk_checker.py
---------------
职责：事前（pre-trade）风控检查。
在实际下单前对以下维度进行检查：
1. 保证金占用检查（新开仓后保证金总占用不超过阈值）
2. 单日亏损限额检查（当日浮亏超过上限则停止交易）
3. Delta 敞口限额检查（防止方向性风险过大）
4. Vega 敞口限额检查（防止隐含波动率风险过大）
5. 合约流动性检查（成交量/持仓量不足时拒绝下单）

所有检查返回结构化的结果对象，支持生成审计日志。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import pandas as pd

from config import Config
from models.pricing.greeks import PortfolioGreeks

if TYPE_CHECKING:
    from data.sources.account_manager import AccountManager

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    """风控检查状态"""
    PASS = "pass"
    WARN = "warn"      # 接近阈值（如超过 80%）
    FAIL = "fail"      # 超过阈值，拒绝交易


@dataclass
class CheckResult:
    """单项风控检查结果"""
    name: str              # 检查项名称
    status: CheckStatus
    current_value: float   # 当前值
    limit_value: float     # 限制值
    message: str = ""      # 说明信息

    @property
    def passed(self) -> bool:
        return self.status != CheckStatus.FAIL


@dataclass
class RiskCheckReport:
    """完整风控检查报告"""
    trade_date: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """所有检查项均通过（PASS 或 WARN）"""
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        """返回所有失败的检查项"""
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        """返回人类可读的摘要"""
        status = "✓ 通过" if self.all_passed else "✗ 拒绝"
        lines = [f"[风控检查] {self.trade_date} - {status}"]
        for c in self.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[c.status.value]
            lines.append(f"  {icon} {c.name}: {c.current_value:.4f} / {c.limit_value:.4f} - {c.message}")
        return "\n".join(lines)


class RiskChecker:
    """
    事前风控检查器。

    Parameters
    ----------
    config : Config
        系统配置，读取各风控阈值
    account_manager : AccountManager, optional
        账户持仓管理器，注入后可直接从天勤读取实时资金和保证金数据，
        不注入时 run_all_checks() 需调用方手动传入 account_info 字典
    """

    def __init__(
        self,
        config: Config,
        account_manager: Optional["AccountManager"] = None,
    ) -> None:
        self.config = config
        self.account_manager = account_manager

    # ------------------------------------------------------------------
    # 主检查入口
    # ------------------------------------------------------------------

    def run_all_checks(
        self,
        trade_date: str,
        proposed_orders: list[dict],
        portfolio_greeks: PortfolioGreeks,
        account_info: Optional[dict] = None,
        current_positions: Optional[pd.DataFrame] = None,
        account_manager: Optional["AccountManager"] = None,
    ) -> RiskCheckReport:
        """
        运行所有风控检查。

        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD
        proposed_orders : list[dict]
            拟下单列表，每项包含 ts_code, direction, volume, price 等
        portfolio_greeks : PortfolioGreeks
            新增订单后的组合 Greeks（预计算）
        account_info : dict, optional
            账户资金快照（balance, available, margin 等）。
            若为 None 则优先使用 account_manager 或 self.account_manager 读取实时数据。
        current_positions : pd.DataFrame, optional
            当前持仓（含浮盈亏）。
            若为 None 则优先使用 account_manager 读取实时持仓。
        account_manager : AccountManager, optional
            本次调用使用的账户管理器，覆盖构造函数注入的实例。

        Returns
        -------
        RiskCheckReport
            包含所有检查项结果的报告

        Notes
        -----
        - 任一 FAIL 检查项将阻止全部订单执行
        - WARN 级别记录日志但不阻止交易
        - account_info 和 account_manager 至少提供一个，否则保证金检查将失败
        """
        raise NotImplementedError("TODO: 实现全量风控检查")

    # ------------------------------------------------------------------
    # 各项检查
    # ------------------------------------------------------------------

    def check_margin(
        self,
        account_balance: float,
        current_margin: float,
        additional_margin: float,
    ) -> CheckResult:
        """
        保证金占用检查。

        Parameters
        ----------
        account_balance : float
            账户总权益（元）
        current_margin : float
            当前保证金占用（元）
        additional_margin : float
            拟新增订单的保证金需求（元）

        Returns
        -------
        CheckResult
            检查结果，limit_value 为 max_margin_ratio × balance
        """
        raise NotImplementedError("TODO: 实现保证金检查")

    def check_daily_loss(
        self,
        account_balance: float,
        daily_pnl: float,
    ) -> CheckResult:
        """
        单日亏损限额检查。

        Parameters
        ----------
        account_balance : float
            账户总权益（元）
        daily_pnl : float
            当日已实现+浮动盈亏（负值=亏损）

        Returns
        -------
        CheckResult
            检查结果，limit_value 为 -max_daily_loss_ratio × balance
        """
        raise NotImplementedError("TODO: 实现单日亏损限额检查")

    def check_delta_exposure(self, net_delta_dollars: float) -> CheckResult:
        """
        Delta 敞口限额检查。

        Parameters
        ----------
        net_delta_dollars : float
            净 Delta 名义敞口（元）

        Returns
        -------
        CheckResult
            检查结果
        """
        raise NotImplementedError("TODO: 实现 Delta 敞口检查")

    def check_vega_exposure(self, net_vega_dollars: float) -> CheckResult:
        """
        Vega 敞口限额检查。

        Parameters
        ----------
        net_vega_dollars : float
            净 Vega 敞口（元/1% 波动率变动）

        Returns
        -------
        CheckResult
            检查结果
        """
        raise NotImplementedError("TODO: 实现 Vega 敞口检查")

    def check_liquidity(
        self,
        ts_code: str,
        volume: float,
        oi: float,
        order_volume: int,
        min_volume: float = 100,
        min_oi: float = 500,
    ) -> CheckResult:
        """
        合约流动性检查（防止成交量过低的合约导致大幅滑点）。

        Parameters
        ----------
        ts_code : str
            合约代码
        volume : float
            当日成交量（手）
        oi : float
            当日持仓量（手）
        order_volume : int
            拟下单数量（手）
        min_volume : float
            最低日成交量要求
        min_oi : float
            最低持仓量要求

        Returns
        -------
        CheckResult
            检查结果
        """
        raise NotImplementedError("TODO: 实现流动性检查")
