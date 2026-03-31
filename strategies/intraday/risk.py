"""
risk.py
-------
日内风控管理器。

检查项：
  1. 当日最大亏损
  2. 当日最大交易笔数（每品种）
  3. 连续亏损冷静期
  4. 止损距离硬限（100基点）
  5. 总持仓上限
  6. 交易时间窗口
"""

from __future__ import annotations

from typing import Dict, Tuple

NO_TRADE_BEFORE = "01:35"           # 09:35 北京
NO_NEW_INTRADAY_AFTER = "06:50"     # 14:50 北京


class IntradayRiskManager:
    """日内风控。"""

    def __init__(self, config: Dict | None = None):
        cfg = config or {}
        self.max_daily_loss: float = cfg.get("max_daily_loss", 50_000)
        self.max_daily_trades: int = cfg.get("max_daily_trades_per_symbol", 5)
        self.max_consecutive_losses: int = cfg.get("max_consecutive_losses", 3)
        self.stop_loss_bps: int = cfg.get("stop_loss_bps", 100)

        self._daily_loss: float = 0.0
        self._daily_trade_counts: Dict[str, int] = {}
        self._consecutive_losses: int = 0
        self._cooldown_until: str = ""

    def check_pre_trade(self, signal, position_manager) -> Tuple[bool, str]:
        """开仓前风控检查。返回 (allowed, reason)。"""
        # 1. 当日亏损
        if self._daily_loss < 0 and abs(self._daily_loss) >= self.max_daily_loss:
            return False, f"当日亏损已达限额 ({self._daily_loss:,.0f}元)"

        # 2. 品种交易次数
        sym_count = self._daily_trade_counts.get(signal.symbol, 0)
        if sym_count >= self.max_daily_trades:
            return False, f"{signal.symbol} 当日交易已达上限 ({sym_count}笔)"

        # 3. 交易时间窗口
        time_str = signal.datetime.split(" ")[-1][:5]
        if time_str < NO_TRADE_BEFORE:
            return False, "开盘5分钟内不交易"
        if time_str >= NO_NEW_INTRADAY_AFTER:
            return False, "尾盘不开新仓"

        # 4. 连续亏损冷静
        if self._consecutive_losses >= self.max_consecutive_losses:
            return False, f"连续亏损 {self._consecutive_losses} 笔，需冷静"

        # 5. 止损距离
        if signal.entry_price > 0:
            sl_bps = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 10000
            if sl_bps > self.stop_loss_bps * 1.05:  # 5% tolerance
                return False, f"止损 {sl_bps:.0f}bps 超过 {self.stop_loss_bps}bps 限制"

        # 6. 总持仓
        total = sum(abs(v) for v in position_manager.get_net_positions().values())
        if total >= position_manager.max_total_lots:
            return False, f"总持仓 {total} 手已达上限"

        return True, "通过"

    def on_trade_complete(self, pnl: float, symbol: str = "") -> None:
        """记录交易完成。"""
        self._daily_loss += pnl
        self._daily_trade_counts[symbol] = \
            self._daily_trade_counts.get(symbol, 0) + 1
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset_daily(self) -> None:
        """每日开盘重置。"""
        self._daily_loss = 0.0
        self._daily_trade_counts = {}
        self._consecutive_losses = 0
        self._cooldown_until = ""
