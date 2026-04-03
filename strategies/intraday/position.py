"""
position.py
-----------
期货仓位管理 + 锁仓逻辑。

核心规则：
  - 每个品种最多持有1手净敞口
  - 三个品种合计最多2手净敞口
  - 止损硬限：100基点
  - 日内平仓通过锁仓实现（避免高额日内平仓手续费）
  - 锁仓对在第二天开盘后双平
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CONTRACT_MULTIPLIERS = {"IF": 300, "IH": 300, "IM": 200}


@dataclass
class FuturesPosition:
    """单个期货持仓"""
    symbol: str
    direction: str              # LONG / SHORT
    volume: int                 # 手数
    entry_price: float
    entry_time: str
    stop_loss: float
    hold_type: str              # INTRADAY / OVERNIGHT / LOCK
    signal_score: int
    is_lock: bool = False
    locked_by: Optional[str] = None
    position_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class LockPair:
    """锁仓对"""
    long_position: FuturesPosition
    short_position: FuturesPosition
    lock_time: str
    expected_unlock_date: str
    realized_pnl: float


class IntradayPositionManager:
    """仓位管理器。处理开仓、平仓、锁仓、解锁。"""

    def __init__(self, config: Dict | None = None):
        cfg = config or {}
        self.max_lots_per_symbol: int = cfg.get("max_lots_per_symbol", 1)
        self.max_total_lots: int = cfg.get("max_total_lots", 2)
        self.stop_loss_bps: int = cfg.get("stop_loss_bps", 100)
        self.use_lock: bool = cfg.get("use_lock", True)
        self.contract_multipliers: Dict[str, int] = cfg.get(
            "contract_multipliers", CONTRACT_MULTIPLIERS
        )

        self.positions: Dict[str, FuturesPosition] = {}
        self.lock_pairs: List[LockPair] = []
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0

    # ------------------------------------------------------------------
    # 开仓
    # ------------------------------------------------------------------

    def can_open(self, symbol: str, direction: str) -> bool:
        # 有待解锁的锁仓对 → 不在同品种开新仓
        for lp in self.lock_pairs:
            if lp.long_position.symbol == symbol:
                return False
        # 已有同方向持仓 → 不加仓
        net = self._net_position_for(symbol)
        if direction == "LONG" and net > 0:
            return False
        if direction == "SHORT" and net < 0:
            return False
        # 总净持仓已达上限且当前品种无持仓 → 不开新品种
        if net == 0 and self._total_net_lots() >= self.max_total_lots:
            return False
        return True

    def open_position(self, signal, fill_price: float) -> Optional[FuturesPosition]:
        if not self.can_open(signal.symbol, signal.direction):
            return None
        pos = FuturesPosition(
            symbol=signal.symbol,
            direction=signal.direction,
            volume=1,
            entry_price=fill_price,
            entry_time=signal.datetime,
            stop_loss=signal.stop_loss,
            hold_type=signal.signal_type,
            signal_score=signal.score,
        )
        self.positions[pos.position_id] = pos
        self.daily_trades += 1
        return pos

    # ------------------------------------------------------------------
    # 平仓 / 锁仓
    # ------------------------------------------------------------------

    def close_position(
        self,
        position_id: str,
        fill_price: float,
        reason: str,
        use_lock: bool = True,
        current_time: str = "",
        next_trade_date: str = "",
    ) -> Optional[Dict]:
        pos = self.positions.get(position_id)
        if pos is None or pos.is_lock:
            return None

        mult = self.contract_multipliers.get(pos.symbol, 300)
        if pos.direction == "LONG":
            pnl = (fill_price - pos.entry_price) * pos.volume * mult
        else:
            pnl = (pos.entry_price - fill_price) * pos.volume * mult

        if use_lock and self.use_lock:
            lock_dir = "SHORT" if pos.direction == "LONG" else "LONG"
            lock_pos = FuturesPosition(
                symbol=pos.symbol,
                direction=lock_dir,
                volume=pos.volume,
                entry_price=fill_price,
                entry_time=current_time,
                stop_loss=0.0,
                hold_type="LOCK",
                signal_score=0,
                is_lock=True,
                locked_by=pos.position_id,
            )
            self.positions[lock_pos.position_id] = lock_pos
            pos.is_lock = True

            lp = LockPair(
                long_position=pos if pos.direction == "LONG" else lock_pos,
                short_position=lock_pos if pos.direction == "LONG" else pos,
                lock_time=current_time,
                expected_unlock_date=next_trade_date,
                realized_pnl=pnl,
            )
            self.lock_pairs.append(lp)
            self.daily_pnl += pnl

            return {
                "action": "LOCK",
                "symbol": pos.symbol,
                "pnl": pnl,
                "reason": reason,
                "lock_position_id": lock_pos.position_id,
                "original_position_id": position_id,
            }
        else:
            del self.positions[position_id]
            self.daily_pnl += pnl
            return {
                "action": "CLOSE",
                "symbol": pos.symbol,
                "pnl": pnl,
                "reason": reason,
                "position_id": position_id,
            }

    # ------------------------------------------------------------------
    # 每日开盘：解锁锁仓对
    # ------------------------------------------------------------------

    def process_daily_open(
        self, current_date: str, prices: Dict[str, float]
    ) -> List[Dict]:
        actions: List[Dict] = []
        resolved: List[int] = []
        for i, lp in enumerate(self.lock_pairs):
            if lp.expected_unlock_date and lp.expected_unlock_date <= current_date:
                for pid in (lp.long_position.position_id,
                            lp.short_position.position_id):
                    self.positions.pop(pid, None)
                actions.append({
                    "action": "UNLOCK",
                    "symbol": lp.long_position.symbol,
                    "pnl": lp.realized_pnl,
                    "lock_time": lp.lock_time,
                    "unlock_date": current_date,
                })
                resolved.append(i)
        for i in reversed(resolved):
            self.lock_pairs.pop(i)
        return actions

    # ------------------------------------------------------------------
    # 止损检查
    # ------------------------------------------------------------------

    def check_stop_loss(
        self, prices: Dict[str, float]
    ) -> List[tuple[str, float]]:
        triggered: List[tuple[str, float]] = []
        for pid, pos in list(self.positions.items()):
            if pos.is_lock:
                continue
            price = prices.get(pos.symbol)
            if price is None:
                continue
            if pos.direction == "LONG" and price <= pos.stop_loss:
                triggered.append((pid, price))
            elif pos.direction == "SHORT" and price >= pos.stop_loss:
                triggered.append((pid, price))
        return triggered

    # ------------------------------------------------------------------
    # 收盘平仓检查
    # ------------------------------------------------------------------

    def check_eod_close(
        self,
        time_str: str,
        weekday: int,
        prices: Dict[str, float],
        next_trade_date: str = "",
    ) -> List[tuple[str, str, bool]]:
        """
        返回 [(position_id, reason, use_lock), ...]
        """
        to_close: List[tuple[str, str, bool]] = []
        is_friday = weekday >= 4

        for pid, pos in list(self.positions.items()):
            if pos.is_lock:
                continue
            price = prices.get(pos.symbol)
            if price is None:
                continue

            mult = self.contract_multipliers.get(pos.symbol, 300)
            if pos.direction == "LONG":
                floating_pnl = (price - pos.entry_price) * pos.volume * mult
            else:
                floating_pnl = (pos.entry_price - price) * pos.volume * mult

            if is_friday and time_str >= "06:30":
                to_close.append((pid, "FRIDAY_CLOSE", False))
            elif time_str >= "06:50":
                if pos.hold_type == "INTRADAY":
                    to_close.append((pid, "EOD_CLOSE", True))
                elif floating_pnl < 0:
                    to_close.append((pid, "EOD_LOSS_CLOSE", True))

        return to_close

    # ------------------------------------------------------------------
    # 移动止盈
    # ------------------------------------------------------------------

    def update_trailing_stop(
        self,
        prices: Dict[str, float],
        trailing_levels: List[Dict] | None = None,
    ) -> None:
        if trailing_levels is None:
            trailing_levels = [
                {"profit_bps": 50, "lock_bps": 0},
                {"profit_bps": 80, "lock_bps": 50},
                {"profit_bps": 120, "lock_bps": 80},
            ]
        sorted_levels = sorted(
            trailing_levels, key=lambda x: x["profit_bps"], reverse=True
        )
        for pos in self.positions.values():
            if pos.is_lock:
                continue
            price = prices.get(pos.symbol)
            if price is None:
                continue
            if pos.direction == "LONG":
                profit_bps = (price - pos.entry_price) / pos.entry_price * 10000
            else:
                profit_bps = (pos.entry_price - price) / pos.entry_price * 10000

            for level in sorted_levels:
                if profit_bps >= level["profit_bps"]:
                    lock_dist = pos.entry_price * level["lock_bps"] / 10000
                    if pos.direction == "LONG":
                        new_stop = pos.entry_price + lock_dist
                        if new_stop > pos.stop_loss:
                            pos.stop_loss = new_stop
                    else:
                        new_stop = pos.entry_price - lock_dist
                        if new_stop < pos.stop_loss:
                            pos.stop_loss = new_stop
                    break

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_net_positions(self) -> Dict[str, int]:
        nets: Dict[str, int] = {}
        for pos in self.positions.values():
            v = pos.volume if pos.direction == "LONG" else -pos.volume
            nets[pos.symbol] = nets.get(pos.symbol, 0) + v
        return {s: v for s, v in nets.items() if v != 0}

    def get_exposure_summary(
        self, prices: Dict[str, float] | None = None
    ) -> Dict:
        nets = self.get_net_positions()
        floating: Dict[str, float] = {}
        if prices:
            for pos in self.positions.values():
                if pos.is_lock:
                    continue
                p = prices.get(pos.symbol)
                if p is None:
                    continue
                mult = self.contract_multipliers.get(pos.symbol, 300)
                if pos.direction == "LONG":
                    pnl = (p - pos.entry_price) * pos.volume * mult
                else:
                    pnl = (pos.entry_price - p) * pos.volume * mult
                floating[pos.symbol] = floating.get(pos.symbol, 0) + pnl

        return {
            "net_positions": nets,
            "total_net_lots": sum(abs(v) for v in nets.values()),
            "floating_pnl": floating,
            "lock_pairs": len(self.lock_pairs),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _net_position_for(self, symbol: str) -> int:
        net = 0
        for pos in self.positions.values():
            if pos.symbol == symbol:
                net += pos.volume if pos.direction == "LONG" else -pos.volume
        return net

    def _total_net_lots(self) -> int:
        nets = self.get_net_positions()
        return sum(abs(v) for v in nets.values())

    def inject_position(
        self, symbol: str, direction: str, entry_price: float,
        entry_time: str = "", score: int = 0,
    ) -> str:
        """重启恢复：注入已知的活跃持仓占位（不增加daily_trades）。

        止损设为极端值，不会被 check_stop_loss 触发。
        返回 position_id 供后续 remove 使用。
        """
        stop = 1.0 if direction == "LONG" else entry_price * 10
        pos = FuturesPosition(
            symbol=symbol, direction=direction, volume=1,
            entry_price=entry_price, entry_time=entry_time,
            stop_loss=stop, hold_type="INTRADAY", signal_score=score,
        )
        self.positions[pos.position_id] = pos
        return pos.position_id

    def remove_by_symbol(self, symbol: str) -> None:
        """移除指定品种的所有持仓和锁仓对（shadow exit 时调用）。

        必须同时清理lock_pairs，否则can_open()会因残留lock_pair
        永久阻止该品种开新仓（2026-04-03发现的bug）。
        """
        to_del = [pid for pid, p in self.positions.items()
                  if p.symbol == symbol]
        for pid in to_del:
            del self.positions[pid]
        # 清理该品种的lock_pairs
        self.lock_pairs = [lp for lp in self.lock_pairs
                           if lp.long_position.symbol != symbol]

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.daily_trades = 0
