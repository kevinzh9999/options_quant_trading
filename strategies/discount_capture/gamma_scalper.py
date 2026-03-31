"""
gamma_scalper.py
----------------
Gamma Scalping 执行引擎。

核心原理：
持有正Gamma的期权（买Put），标的每波动一次Delta就变化。
通过反向调整期货持仓来保持Delta中性，每次调整锁定一小段利润。

利润公式（单次调整）：Gamma P&L ≈ 0.5 × Gamma × ΔS²
成本：每天 Theta（时间价值衰减）
盈亏平衡：当日 sum(Gamma P&L) > |Theta|
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FUTURES_MULTIPLIER = 200  # IM期货合约乘数
OPTION_MULTIPLIER = 100   # MO期权合约乘数


@dataclass
class RehedgeRecord:
    """单次对冲记录"""
    timestamp: str
    price: float
    action: str           # "BUY" or "SELL"
    volume: int           # 期货手数调整量
    delta_before: float   # 对冲前组合delta
    delta_after: float    # 对冲后组合delta
    gamma_pnl: float      # 本次对冲锁定的Gamma利润
    commission: float     # 手续费


class GammaScalper:
    """
    Gamma Scalping 执行引擎。

    Parameters
    ----------
    rehedge_threshold_delta : float
        Delta偏移阈值（元/点），超过此值触发对冲
    rehedge_threshold_pct : float
        标的价格变动百分比阈值
    rehedge_method : str
        "delta" / "price" / "time"
    max_rehedge_per_day : int
        每天最多对冲次数
    commission_rate : float
        期货手续费率（单边）
    """

    def __init__(self, config: Dict = None):
        cfg = config or {}
        self.rehedge_threshold_delta = cfg.get("rehedge_threshold_delta", 50.0)
        self.rehedge_threshold_pct = cfg.get("rehedge_threshold_pct", 0.003)
        self.rehedge_method = cfg.get("rehedge_method", "delta")
        self.max_rehedge_per_day = cfg.get("max_rehedge_per_day", 10)
        self.commission_rate = cfg.get("commission_rate", 0.000023)
        self.time_interval_min = cfg.get("time_interval_min", 30)

        # 状态
        self._last_hedge_price: float = 0.0
        self._last_hedge_time: str = ""
        self._hedge_futures_delta: int = 0  # 对冲用期货净手数（正=多，负=空）
        self._today_rehedge_count: int = 0
        self._today_records: List[RehedgeRecord] = []
        self._cumulative_gamma_pnl: float = 0.0
        self._cumulative_commission: float = 0.0

    def reset_daily(self):
        """每日重置计数器"""
        self._today_rehedge_count = 0
        self._today_records = []
        self._cumulative_gamma_pnl = 0.0
        self._cumulative_commission = 0.0

    def initialize(self, initial_price: float, initial_time: str = ""):
        """设置初始对冲基准价"""
        self._last_hedge_price = initial_price
        self._last_hedge_time = initial_time
        self._hedge_futures_delta = 0

    def calc_portfolio_delta(
        self,
        base_futures_lots: int,
        put_lots: int,
        put_delta: float,
    ) -> float:
        """
        计算当前组合的净Delta（元/点）。

        组合Delta = 基础期货Delta + Put期权Delta + 对冲期货Delta

        Parameters
        ----------
        base_futures_lots : int
            基础多头期货手数（贴水仓位）
        put_lots : int
            买入Put手数
        put_delta : float
            单手Put的delta值（负数）

        Returns
        -------
        float
            组合净delta（元/点）
        """
        futures_delta = base_futures_lots * FUTURES_MULTIPLIER
        option_delta = put_lots * OPTION_MULTIPLIER * put_delta
        hedge_delta = self._hedge_futures_delta * FUTURES_MULTIPLIER
        return futures_delta + option_delta + hedge_delta

    def check_rehedge(
        self,
        current_price: float,
        current_time: str,
        portfolio_delta: float,
    ) -> Optional[Dict]:
        """
        检查是否需要重新对冲。

        Returns
        -------
        None or dict
            {"action": "BUY"/"SELL", "volume": int, "reason": str}
        """
        if self._today_rehedge_count >= self.max_rehedge_per_day:
            return None

        need_rehedge = False
        reason = ""

        if self.rehedge_method == "delta":
            if abs(portfolio_delta) > self.rehedge_threshold_delta:
                need_rehedge = True
                reason = f"delta偏移={portfolio_delta:.0f}"

        elif self.rehedge_method == "price":
            if self._last_hedge_price > 0:
                price_change_pct = abs(current_price - self._last_hedge_price) / self._last_hedge_price
                if price_change_pct >= self.rehedge_threshold_pct:
                    need_rehedge = True
                    reason = f"价格变动={price_change_pct*100:.2f}%"

        elif self.rehedge_method == "time":
            if current_time:
                try:
                    from datetime import datetime
                    fmt = "%Y-%m-%d %H:%M:%S"
                    t2 = datetime.strptime(current_time[:19], fmt)
                    if not self._last_hedge_time:
                        # 首次：立刻触发
                        need_rehedge = True
                        reason = "首次对冲"
                    else:
                        t1 = datetime.strptime(self._last_hedge_time[:19], fmt)
                        elapsed = (t2 - t1).total_seconds() / 60
                        if elapsed >= self.time_interval_min:
                            need_rehedge = True
                            reason = f"时间间隔={elapsed:.0f}min"
                except (ValueError, TypeError):
                    pass

        if not need_rehedge:
            return None

        # 计算需要调整的期货手数
        target_delta_adj = -portfolio_delta  # 想把delta调回零
        lots_adj = int(round(target_delta_adj / FUTURES_MULTIPLIER))
        if lots_adj == 0:
            return None

        action = "BUY" if lots_adj > 0 else "SELL"
        return {
            "action": action,
            "volume": abs(lots_adj),
            "reason": reason,
        }

    def execute_rehedge(
        self,
        current_price: float,
        current_time: str,
        action: str,
        volume: int,
        portfolio_delta_before: float,
    ) -> RehedgeRecord:
        """
        执行对冲交易，记录盈亏。

        Gamma利润 ≈ 0.5 × (portfolio_gamma * option_mult) × (ΔS)²
        这里简化计算：用价格变动产生的delta变化来计算

        Returns
        -------
        RehedgeRecord
        """
        # Gamma P&L from price movement since last hedge
        delta_s = current_price - self._last_hedge_price
        # 已持有对冲头寸产生的盈亏
        gamma_pnl = self._hedge_futures_delta * FUTURES_MULTIPLIER * delta_s

        # 手续费
        commission = volume * current_price * FUTURES_MULTIPLIER * self.commission_rate

        # 更新对冲头寸
        if action == "BUY":
            self._hedge_futures_delta += volume
        else:
            self._hedge_futures_delta -= volume

        delta_after = portfolio_delta_before
        if action == "BUY":
            delta_after += volume * FUTURES_MULTIPLIER
        else:
            delta_after -= volume * FUTURES_MULTIPLIER

        record = RehedgeRecord(
            timestamp=current_time,
            price=current_price,
            action=action,
            volume=volume,
            delta_before=portfolio_delta_before,
            delta_after=delta_after,
            gamma_pnl=gamma_pnl,
            commission=commission,
        )

        self._last_hedge_price = current_price
        self._last_hedge_time = current_time
        self._today_rehedge_count += 1
        self._today_records.append(record)
        self._cumulative_gamma_pnl += gamma_pnl
        self._cumulative_commission += commission

        return record

    def daily_summary(self, theta_cost: float) -> Dict:
        """
        每日Gamma Scalping汇总。

        Parameters
        ----------
        theta_cost : float
            当日Theta成本（正值，表示时间价值衰减）

        Returns
        -------
        dict
        """
        gamma_pnl = sum(r.gamma_pnl for r in self._today_records)
        commission_cost = sum(r.commission for r in self._today_records)
        net_scalp_pnl = gamma_pnl - theta_cost - commission_cost

        prices = [r.price for r in self._today_records]
        if len(prices) >= 2:
            returns = np.diff(np.log(prices))
            realized_vol = float(np.std(returns) * np.sqrt(252 * len(prices))) if len(returns) > 0 else 0.0
        else:
            realized_vol = 0.0

        return {
            "rehedge_count": self._today_rehedge_count,
            "gamma_pnl": gamma_pnl,
            "theta_cost": theta_cost,
            "commission_cost": commission_cost,
            "net_scalp_pnl": net_scalp_pnl,
            "hedge_futures_lots": self._hedge_futures_delta,
            "records": list(self._today_records),
        }

    @property
    def hedge_futures_delta(self) -> int:
        return self._hedge_futures_delta

    @property
    def today_rehedge_count(self) -> int:
        return self._today_rehedge_count
