"""Position manager — tracks daily_xgb open trades, enforces hold + SL.

Authoritative source: daily_xgb_trades table (status='OPEN').
JSON cache: tmp/daily_xgb_positions.json (for executor consumption).

Functions:
  - register_entry(...)   下单成交后注册新持仓
  - check_exits(today)    EOD 跑: 找出今日应该平仓的 trades (TIME 或 SL)
  - apply_exit(...)       平仓成交后写 trade record
  - export_to_json(...)   把当前 open trades 写到 JSON
  - reload_from_db()      启动时从 DB 恢复持仓状态
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import DailyXGBConfig
from . import persist


def _bj_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=8)


def _trading_calendar(cfg: DailyXGBConfig) -> List[str]:
    """Trading dates from index_daily."""
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT DISTINCT trade_date FROM index_daily "
            "WHERE ts_code='000852.SH' ORDER BY trade_date"
        )
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def planned_exit_date(entry_date: str, hold_days: int, calendar: List[str]) -> str:
    """Compute exit date = entry_date + hold_days trading days."""
    if entry_date not in calendar:
        # Find next trading day if entry isn't in calendar
        candidates = [d for d in calendar if d >= entry_date]
        if not candidates:
            return entry_date
        entry_date = candidates[0]
    idx = calendar.index(entry_date)
    target_idx = idx + hold_days
    if target_idx >= len(calendar):
        return calendar[-1]
    return calendar[target_idx]


@dataclass
class OpenPosition:
    trade_id: int
    signal_date: str
    entry_date: str
    direction: str
    contract_code: str
    entry_price: float
    entry_lots: int
    sl_price: float
    sl_pct: float
    hold_days: int
    planned_exit_date: str
    enhancement_type: str

    @property
    def gross_lots(self) -> int:
        return self.entry_lots


# ─────────────────────────────────────────────────────────────────────
# Entry / exit registration
# ─────────────────────────────────────────────────────────────────────

def register_entry(cfg: DailyXGBConfig,
                    signal_date: str,
                    entry_date: str,
                    direction: str,
                    contract_code: str,
                    entry_price: float,
                    lots: int,
                    sl_pct: float,
                    atr_k: float,
                    hold_days: int,
                    enhancement_type: str,
                    atr20: float) -> int:
    """Insert new OPEN trade into daily_xgb_trades. Return trade_id."""
    calendar = _trading_calendar(cfg)
    pe_date = planned_exit_date(entry_date, hold_days, calendar)
    sl_price = entry_price * (1 - sl_pct) if direction == "LONG" else entry_price * (1 + sl_pct)
    record = {
        "strategy_id": cfg.strategy_id,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "direction": direction,
        "underlying": cfg.underlying,
        "contract_code": contract_code,
        "enhancement_type": enhancement_type,
        "hold_days": hold_days,
        "atr_k": atr_k,
        "sl_pct": sl_pct,
        "entry_price": entry_price,
        "entry_lots": lots,
        "entry_atr": atr20,
        "planned_exit_date": pe_date,
        "sl_price": sl_price,
        "exit_date": None, "exit_price": None, "exit_reason": None,
        "pnl_yuan": None, "net_ret": None,
        "status": "OPEN",
        "created_at": _bj_now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return persist.insert_trade(cfg, record)


def apply_exit(cfg: DailyXGBConfig, trade_id: int,
                exit_date: str, exit_price: float, exit_reason: str) -> None:
    """Apply exit to a trade: compute PnL, write to DB."""
    conn = sqlite3.connect(cfg.db_path)
    try:
        cur = conn.execute(
            "SELECT direction, entry_price, entry_lots FROM daily_xgb_trades "
            "WHERE id=?", (trade_id,))
        row = cur.fetchone()
        if not row:
            return
        direction, entry_price, entry_lots = row
        sign = 1 if direction == "LONG" else -1
        gross_ret = sign * (exit_price - entry_price) / entry_price
        net_ret = gross_ret - cfg.slippage_pct
        pnl_yuan = net_ret * entry_price * cfg.contract_mult * entry_lots
    finally:
        conn.close()
    persist.update_trade_exit(cfg, trade_id, exit_date, exit_price,
                                 exit_reason, pnl_yuan, net_ret)


# ─────────────────────────────────────────────────────────────────────
# EOD check
# ─────────────────────────────────────────────────────────────────────

def get_open_positions(cfg: DailyXGBConfig) -> List[OpenPosition]:
    """Load all OPEN positions from DB."""
    rows = persist.list_open_trades(cfg)
    return [
        OpenPosition(
            trade_id=r["id"],
            signal_date=r["signal_date"],
            entry_date=r["entry_date"],
            direction=r["direction"],
            contract_code=r["contract_code"],
            entry_price=r["entry_price"],
            entry_lots=r["entry_lots"],
            sl_price=r["sl_price"],
            sl_pct=r["sl_pct"],
            hold_days=r["hold_days"],
            planned_exit_date=r["planned_exit_date"],
            enhancement_type=r["enhancement_type"],
        )
        for r in rows
    ]


def check_exits(cfg: DailyXGBConfig, today: str,
                 today_close: float) -> List[Tuple[OpenPosition, str]]:
    """Find positions to exit today. Returns list of (position, reason).

    Reasons:
      'TIME' — hold_days reached
      'SL'   — close ≤ sl_price (LONG) or close ≥ sl_price (SHORT)
    """
    positions = get_open_positions(cfg)
    exits = []
    for p in positions:
        # Time exit: today >= planned_exit_date
        if today >= p.planned_exit_date:
            exits.append((p, "TIME"))
            continue
        # SL check via close
        if p.direction == "LONG":
            if today_close <= p.sl_price:
                exits.append((p, "SL"))
        else:  # SHORT
            if today_close >= p.sl_price:
                exits.append((p, "SL"))
    return exits


# ─────────────────────────────────────────────────────────────────────
# Concurrent capacity check
# ─────────────────────────────────────────────────────────────────────

def can_open(cfg: DailyXGBConfig, additional_lots: int) -> Tuple[bool, str]:
    """Check if opening additional_lots more would exceed cap."""
    positions = get_open_positions(cfg)
    cur = sum(p.gross_lots for p in positions)
    if cur + additional_lots > cfg.concurrent_cap:
        return False, f"cap reached: {cur} + {additional_lots} > {cfg.concurrent_cap}"
    return True, ""


def export_to_json(cfg: DailyXGBConfig) -> None:
    """Sync positions DB → JSON cache."""
    positions = get_open_positions(cfg)
    payload = [
        {
            "trade_id": p.trade_id,
            "signal_date": p.signal_date,
            "entry_date": p.entry_date,
            "direction": p.direction,
            "contract_code": p.contract_code,
            "entry_price": p.entry_price,
            "lots": p.entry_lots,
            "sl_price": p.sl_price,
            "sl_pct": p.sl_pct,
            "hold_days": p.hold_days,
            "planned_exit_date": p.planned_exit_date,
            "enhancement_type": p.enhancement_type,
        }
        for p in positions
    ]
    persist.write_positions(cfg, payload)


__all__ = [
    "OpenPosition",
    "register_entry", "apply_exit", "get_open_positions",
    "check_exits", "can_open", "export_to_json",
    "planned_exit_date",
]
