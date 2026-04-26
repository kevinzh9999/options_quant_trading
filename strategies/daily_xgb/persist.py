"""DB read/write + JSON IO helpers for Daily XGB.

DB tables (defined in data/storage/schemas.py):
  - daily_xgb_signals    每个 signal 的完整记录
  - daily_xgb_trades     完整生命周期 trade (entry → exit)
  - daily_xgb_orders     executor 提交的订单
  - daily_xgb_executor_log  executor 各阶段事件
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DailyXGBConfig


def _bj_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=8)


def _open_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


# ─────────────────────────────────────────────────────────────────────
# Signals table
# ─────────────────────────────────────────────────────────────────────

def insert_signal(cfg: DailyXGBConfig, record: Dict[str, Any]) -> None:
    """Insert (or replace) a signal record into daily_xgb_signals.

    Required keys:
      signal_date (YYYYMMDD), entry_date (YYYYMMDD or NULL),
      direction (LONG/SHORT/NONE), pred, top_thr, bot_thr,
      entry_intended_open (REAL or NULL), sl_pct, hold_days, atr_k,
      enhancement_type, lots_planned, status, reason, signal_json
    """
    conn = _open_conn(cfg.db_path)
    try:
        cols = list(record.keys())
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO daily_xgb_signals ({','.join(cols)}) VALUES ({placeholders})"
        conn.execute(sql, [record[c] for c in cols])
        conn.commit()
    finally:
        conn.close()


def update_signal_status(cfg: DailyXGBConfig, signal_date: str, status: str,
                           reason: Optional[str] = None) -> None:
    conn = _open_conn(cfg.db_path)
    try:
        conn.execute(
            "UPDATE daily_xgb_signals SET status=?, reason=COALESCE(?, reason) "
            "WHERE signal_date=?",
            (status, reason, signal_date),
        )
        conn.commit()
    finally:
        conn.close()


def get_signal(cfg: DailyXGBConfig, signal_date: str) -> Optional[Dict[str, Any]]:
    conn = _open_conn(cfg.db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM daily_xgb_signals WHERE signal_date=?",
            (signal_date,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# Trades table
# ─────────────────────────────────────────────────────────────────────

def insert_trade(cfg: DailyXGBConfig, record: Dict[str, Any]) -> int:
    conn = _open_conn(cfg.db_path)
    try:
        cols = list(record.keys())
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT INTO daily_xgb_trades ({','.join(cols)}) VALUES ({placeholders})"
        cur = conn.execute(sql, [record[c] for c in cols])
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_trade_exit(cfg: DailyXGBConfig, trade_id: int,
                       exit_date: str, exit_price: float,
                       exit_reason: str, pnl_yuan: float,
                       net_ret: float) -> None:
    conn = _open_conn(cfg.db_path)
    try:
        conn.execute(
            "UPDATE daily_xgb_trades "
            "SET exit_date=?, exit_price=?, exit_reason=?, pnl_yuan=?, net_ret=?, "
            "    status='CLOSED' "
            "WHERE id=?",
            (exit_date, exit_price, exit_reason, pnl_yuan, net_ret, trade_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_open_trades(cfg: DailyXGBConfig) -> List[Dict[str, Any]]:
    conn = _open_conn(cfg.db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM daily_xgb_trades WHERE status='OPEN' "
            "ORDER BY entry_date"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# Orders table
# ─────────────────────────────────────────────────────────────────────

def insert_order(cfg: DailyXGBConfig, record: Dict[str, Any]) -> None:
    conn = _open_conn(cfg.db_path)
    try:
        cols = list(record.keys())
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT INTO daily_xgb_orders ({','.join(cols)}) VALUES ({placeholders})"
        conn.execute(sql, [record[c] for c in cols])
        conn.commit()
    finally:
        conn.close()


def update_order(cfg: DailyXGBConfig, order_id: str,
                  status: str, filled_lots: int = 0,
                  filled_price: Optional[float] = None,
                  cancel_reason: Optional[str] = None) -> None:
    conn = _open_conn(cfg.db_path)
    try:
        conn.execute(
            "UPDATE daily_xgb_orders "
            "SET status=?, filled_lots=?, filled_price=?, cancel_reason=?, "
            "    update_time=? "
            "WHERE order_id=?",
            (status, filled_lots, filled_price, cancel_reason,
             _bj_now().strftime("%Y-%m-%d %H:%M:%S"), order_id),
        )
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# Executor log
# ─────────────────────────────────────────────────────────────────────

def insert_executor_event(cfg: DailyXGBConfig,
                            event_type: str,
                            order_id: Optional[str] = None,
                            details: Optional[str] = None) -> None:
    conn = _open_conn(cfg.db_path)
    try:
        conn.execute(
            "INSERT INTO daily_xgb_executor_log "
            "(event_time, event_type, order_id, details) VALUES (?,?,?,?)",
            (_bj_now().strftime("%Y-%m-%d %H:%M:%S"),
             event_type, order_id, details),
        )
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# JSON pending signal IO
# ─────────────────────────────────────────────────────────────────────

def write_pending_signal(cfg: DailyXGBConfig, signal: Dict[str, Any]) -> str:
    """Write pending signal JSON for executor consumption.

    Atomically write to tmp/daily_xgb_pending_{signal_date}.json.
    Single signal per day (the strategy only fires one signal/day max).
    """
    Path(cfg.pending_signal_path).parent.mkdir(parents=True, exist_ok=True)
    sd = signal.get("signal_date", _bj_now().strftime("%Y%m%d"))
    out_path = str(Path(cfg.pending_signal_path).parent / f"daily_xgb_pending_{sd}.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)
    return out_path


def read_pending_signals(cfg: DailyXGBConfig) -> List[Dict[str, Any]]:
    """Return all pending JSON signal files in tmp/."""
    pending = []
    parent = Path(cfg.pending_signal_path).parent
    if not parent.exists():
        return pending
    for f in parent.glob("daily_xgb_pending_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                pending.append(json.load(fh))
        except Exception:
            continue
    return sorted(pending, key=lambda s: s.get("signal_date", ""))


def remove_pending_signal(cfg: DailyXGBConfig, signal_date: str) -> None:
    parent = Path(cfg.pending_signal_path).parent
    f = parent / f"daily_xgb_pending_{signal_date}.json"
    if f.exists():
        f.unlink()


# ─────────────────────────────────────────────────────────────────────
# Positions JSON (executor state, not authoritative — DB is)
# ─────────────────────────────────────────────────────────────────────

def write_positions(cfg: DailyXGBConfig, positions: List[Dict[str, Any]]) -> None:
    Path(cfg.positions_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _bj_now().strftime("%Y-%m-%d %H:%M:%S"),
        "positions": positions,
    }
    tmp = cfg.positions_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cfg.positions_path)


def read_positions(cfg: DailyXGBConfig) -> List[Dict[str, Any]]:
    if not Path(cfg.positions_path).exists():
        return []
    try:
        with open(cfg.positions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("positions", [])
    except Exception:
        return []


__all__ = [
    "_bj_now",
    "insert_signal", "update_signal_status", "get_signal",
    "insert_trade", "update_trade_exit", "list_open_trades",
    "insert_order", "update_order", "insert_executor_event",
    "write_pending_signal", "read_pending_signals", "remove_pending_signal",
    "write_positions", "read_positions",
]
