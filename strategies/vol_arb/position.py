"""
position.py
-----------
职责：波动率套利持仓管理。

追踪已开仓的期权组合（strangle/straddle），
管理其生命周期（开仓→持仓→滚仓→平仓）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PositionStatus(str, Enum):
    """持仓状态"""
    OPEN = "open"           # 持仓中
    ROLLING = "rolling"     # 滚仓中（平旧仓+开新仓）
    CLOSING = "closing"     # 平仓中
    CLOSED = "closed"       # 已平仓


@dataclass
class OptionLeg:
    """期权单腿持仓"""
    ts_code: str           # 合约代码
    call_put: str          # 'C' 或 'P'
    strike_price: float    # 行权价
    expire_date: str       # 到期日
    direction: str         # 'short' 或 'long'
    volume: int            # 持仓手数
    open_price: float      # 开仓价格
    last_price: float = 0.0     # 最新价（用于计算浮盈亏）
    delta: float = 0.0          # 当前 Delta
    gamma: float = 0.0          # 当前 Gamma
    theta: float = 0.0          # 当前 Theta
    vega: float = 0.0           # 当前 Vega

    @property
    def float_pnl(self) -> float:
        """当前浮动盈亏（元）"""
        sign = -1 if self.direction == "short" else 1
        # IO 期权乘数为 100，MO 期权乘数为 200（这里返回点位，外部需乘以乘数）
        return sign * (self.last_price - self.open_price) * self.volume

    @property
    def net_delta(self) -> float:
        """净 Delta 敞口（考虑方向和手数）"""
        sign = -1 if self.direction == "short" else 1
        return sign * self.delta * self.volume


@dataclass
class VolArbPosition:
    """
    波动率套利持仓组合（一个 Strangle 或 Straddle）。

    一个组合通常包含 1 个 Call 腿 + 1 个 Put 腿，
    均为卖出方向（做空波动率）。
    """
    position_id: str
    underlying: str           # 标的品种，如 'IO'
    open_date: str            # 开仓日期
    strategy_type: str        # 'strangle' 或 'straddle'
    legs: list[OptionLeg] = field(default_factory=list)
    status: PositionStatus = PositionStatus.OPEN
    close_date: str = ""
    realized_pnl: float = 0.0

    @property
    def total_float_pnl(self) -> float:
        """组合总浮动盈亏"""
        return sum(leg.float_pnl for leg in self.legs)

    @property
    def net_delta(self) -> float:
        """组合净 Delta"""
        return sum(leg.net_delta for leg in self.legs)

    @property
    def net_vega(self) -> float:
        """组合净 Vega（考虑方向和手数）"""
        result = 0.0
        for leg in self.legs:
            sign = -1 if leg.direction == "short" else 1
            result += sign * leg.vega * leg.volume
        return result

    def days_to_nearest_expire(self, trade_date: str) -> Optional[int]:
        """返回距最近腿到期的交易日数（简单日历日差，实际应使用交易日历）"""
        if not self.legs:
            return None
        from datetime import date
        today = date.fromisoformat(
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        )
        expires = []
        for leg in self.legs:
            ed = leg.expire_date
            expires.append(
                date.fromisoformat(f"{ed[:4]}-{ed[4:6]}-{ed[6:]}")
            )
        nearest = min(expires)
        return (nearest - today).days


class VolArbPositionManager:
    """
    波动率套利持仓管理器。

    负责记录和查询所有 VolArbPosition，
    提供滚仓触发判断和 Greeks 汇总。
    """

    def __init__(self) -> None:
        self._positions: dict[str, VolArbPosition] = {}

    def add_position(self, position: VolArbPosition) -> None:
        """添加新持仓"""
        self._positions[position.position_id] = position

    def get_open_positions(self) -> list[VolArbPosition]:
        """获取所有开仓中的持仓"""
        return [p for p in self._positions.values() if p.status == PositionStatus.OPEN]

    def get_positions_to_roll(
        self,
        trade_date: str,
        roll_days_threshold: int = 7,
    ) -> list[VolArbPosition]:
        """
        返回需要滚仓的持仓列表。

        Parameters
        ----------
        trade_date : str
            当前交易日
        roll_days_threshold : int
            剩余天数低于此值时触发滚仓

        Returns
        -------
        list[VolArbPosition]
            需要滚仓的持仓列表
        """
        raise NotImplementedError("TODO: 实现滚仓触发判断")

    def close_position(
        self,
        position_id: str,
        close_date: str,
        realized_pnl: float,
    ) -> None:
        """标记持仓为已平仓"""
        if position_id in self._positions:
            pos = self._positions[position_id]
            pos.status = PositionStatus.CLOSED
            pos.close_date = close_date
            pos.realized_pnl = realized_pnl

    def portfolio_greeks_summary(self) -> dict[str, float]:
        """
        汇总所有开仓持仓的组合 Greeks。

        Returns
        -------
        dict
            包含 net_delta、net_vega 等键的字典
        """
        raise NotImplementedError("TODO: 实现 Greeks 汇总")
