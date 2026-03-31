"""
order_manager.py
----------------
职责：订单管理，负责信号到订单的转换和生命周期管理。
- 将 VRPSignal 转换为具体的期权订单（买/卖 Call/Put 组合）
- 支持 dry_run 模式（只记录不实际下单，用于回测和验证）
- 维护订单状态（待提交 / 已提交 / 已成交 / 已撤销）
- 提供平仓和滚仓逻辑
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional
import uuid

from strategies.vol_arb.signal_types import VRPSignal

if TYPE_CHECKING:
    from data.sources.account_manager import AccountManager

logger = logging.getLogger(__name__)


class OrderDirection(str, Enum):
    """订单方向"""
    BUY = "buy"        # 买入开仓 / 买入平仓
    SELL = "sell"      # 卖出开仓 / 卖出平仓


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"        # 待提交
    SUBMITTED = "submitted"    # 已提交
    PARTIAL = "partial"        # 部分成交
    FILLED = "filled"          # 全部成交
    CANCELLED = "cancelled"    # 已撤销
    REJECTED = "rejected"      # 被拒绝（风控拦截）


class OrderType(str, Enum):
    """订单类型"""
    LIMIT = "limit"       # 限价单
    MARKET = "market"     # 市价单（期权流动性差，谨慎使用）


@dataclass
class Order:
    """单笔订单"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    ts_code: str = ""                    # 合约代码（Tushare 格式）
    direction: OrderDirection = OrderDirection.BUY
    volume: int = 0                      # 手数
    price: float = 0.0                   # 限价价格（0 表示市价）
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: int = 0               # 已成交手数
    avg_filled_price: float = 0.0        # 平均成交价
    signal_id: str = ""                  # 来源信号 ID（用于追踪）
    notes: str = ""                      # 备注

    @property
    def is_terminal(self) -> bool:
        """订单是否处于终态（已成交/已撤销/被拒绝）"""
        return self.status in (
            OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED
        )


@dataclass
class OrderGroup:
    """一组相关订单（如一个宽跨式 = 1个Call订单 + 1个Put订单）"""
    group_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy_type: str = ""             # 策略类型，如 strangle / straddle
    underlying: str = ""
    signal_date: str = ""
    orders: list[Order] = field(default_factory=list)

    @property
    def all_filled(self) -> bool:
        return all(o.status == OrderStatus.FILLED for o in self.orders)


class OrderManager:
    """
    订单管理器。

    Parameters
    ----------
    dry_run : bool
        干运行模式，True 时只记录订单不实际提交，用于策略验证
    account_manager : AccountManager, optional
        账户持仓管理器，注入后在下单前可查询实时可用资金，
        判断是否有足够余量承担新仓位的保证金需求。
        不注入时跳过可用资金预检（仅依赖风控层的事前检查）。
    """

    def __init__(
        self,
        dry_run: bool = True,
        account_manager: Optional["AccountManager"] = None,
    ) -> None:
        self.dry_run = dry_run
        self.account_manager = account_manager
        self._orders: dict[str, Order] = {}           # order_id -> Order
        self._groups: dict[str, OrderGroup] = {}      # group_id -> OrderGroup

    # ------------------------------------------------------------------
    # 信号转订单
    # ------------------------------------------------------------------

    def signal_to_orders(
        self,
        signal: VRPSignal,
        position_size: int,
        options_data: "pd.DataFrame",
        risk_free_rate: float,
    ) -> OrderGroup:
        """
        将 VRP 信号转换为期权组合订单。

        Parameters
        ----------
        signal : VRPSignal
            VRP 信号对象（含推荐到期日和行权价）
        position_size : int
            仓位大小（手数，每腿）
        options_data : pd.DataFrame
            当日期权行情（用于获取合约代码和最新价）
        risk_free_rate : float
            无风险利率（用于限价单定价参考）

        Returns
        -------
        OrderGroup
            包含 Call 空单和 Put 空单的订单组

        Notes
        -----
        - 做空波动率策略：卖出 OTM Call + 卖出 OTM Put（宽跨式/Strangle）
        - 限价单价格设为 bid_price1（保守卖出）或 mid_price
        - dry_run=True 时订单状态直接置为 FILLED（模拟成交）
        - 若 self.account_manager 已注入，实现时应先调用
          account_manager.get_account_summary() 确认 available 充足，
          不足时返回空 OrderGroup 并记录 WARNING 日志
        """
        raise NotImplementedError("TODO: 实现信号转订单")

    # ------------------------------------------------------------------
    # 平仓和滚仓
    # ------------------------------------------------------------------

    def close_group(
        self,
        group_id: str,
        options_data: "pd.DataFrame",
    ) -> list[Order]:
        """
        对指定订单组生成平仓订单（反向开仓）。

        Parameters
        ----------
        group_id : str
            OrderGroup ID
        options_data : pd.DataFrame
            当日期权行情（用于获取最新价）

        Returns
        -------
        list[Order]
            平仓订单列表

        Notes
        -----
        - 平仓价格设为 ask_price1（保守平仓）或 mid_price
        """
        raise NotImplementedError("TODO: 实现平仓订单生成")

    def roll_group(
        self,
        group_id: str,
        new_signal: VRPSignal,
        options_data: "pd.DataFrame",
    ) -> OrderGroup:
        """
        滚仓：平掉当前组合，同时开新的组合。

        Parameters
        ----------
        group_id : str
            当前持仓的 OrderGroup ID
        new_signal : VRPSignal
            新的 VRP 信号（含新的到期日和行权价）
        options_data : pd.DataFrame
            当日期权行情

        Returns
        -------
        OrderGroup
            新的订单组（平仓单 + 新开仓单）
        """
        raise NotImplementedError("TODO: 实现滚仓订单生成")

    # ------------------------------------------------------------------
    # 订单状态管理
    # ------------------------------------------------------------------

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_volume: int = 0,
        avg_price: float = 0.0,
    ) -> None:
        """更新订单状态（由 TqExecutor 回调调用）"""
        raise NotImplementedError("TODO: 实现订单状态更新")

    def get_open_groups(self) -> list[OrderGroup]:
        """获取所有未完全平仓的订单组"""
        raise NotImplementedError("TODO: 实现活跃订单组查询")

    def get_order_summary(self) -> "pd.DataFrame":
        """返回所有订单的汇总 DataFrame，用于日志和分析"""
        raise NotImplementedError("TODO: 实现订单汇总")
