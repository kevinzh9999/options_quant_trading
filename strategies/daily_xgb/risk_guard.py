"""Risk circuit breakers for Daily XGB.

Three layers (only suspend new opens; existing positions still managed):
  1. weekly_dd: 单周累计 PnL < -X% account → 暂停
  2. daily_loss: 当日累计 PnL < -X% account → 暂停
  3. margin_usage: 当前保证金占用 > X% account → 暂停
  4. kill_switch: 操作员放置 tmp/daily_xgb_kill.flag → 永久暂停 (until removed)

返回 (ok, reason)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from .config import DailyXGBConfig
from . import persist
from .position_manager import get_open_positions


def _bj_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=8)


def kill_switch_active(cfg: DailyXGBConfig) -> Tuple[bool, str]:
    if Path(cfg.kill_switch_file).exists():
        return True, f"Kill switch file exists: {cfg.kill_switch_file}"
    return False, ""


def weekly_dd(cfg: DailyXGBConfig) -> float:
    """Cumulative PnL of trades CLOSED in last 7 calendar days."""
    cutoff = (_bj_now() - timedelta(days=7)).strftime("%Y%m%d")
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT COALESCE(SUM(pnl_yuan), 0) FROM daily_xgb_trades "
            "WHERE status='CLOSED' AND exit_date >= ?",
            (cutoff,))
        return float(cur.fetchone()[0])
    finally:
        conn.close()


def daily_loss(cfg: DailyXGBConfig) -> float:
    """Today's cumulative PnL (closed trades on current BJ date)."""
    today = _bj_now().strftime("%Y%m%d")
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT COALESCE(SUM(pnl_yuan), 0) FROM daily_xgb_trades "
            "WHERE status='CLOSED' AND exit_date = ?",
            (today,))
        return float(cur.fetchone()[0])
    finally:
        conn.close()


def margin_usage(cfg: DailyXGBConfig) -> float:
    """Current margin used by open daily_xgb positions."""
    open_pos = get_open_positions(cfg)
    total_lots = sum(p.gross_lots for p in open_pos)
    return total_lots * cfg.margin_per_lot


def check_can_open(cfg: DailyXGBConfig,
                    additional_lots: int = 1) -> Tuple[bool, str]:
    """Comprehensive can-open check. Returns (ok, reason_if_blocked)."""
    # 1. Kill switch
    killed, reason = kill_switch_active(cfg)
    if killed:
        return False, f"KILL_SWITCH: {reason}"

    # 2. Daily loss
    dl = daily_loss(cfg)
    daily_thr = -cfg.daily_loss_threshold_pct * cfg.account_equity
    if dl < daily_thr:
        return False, f"DAILY_LOSS: {dl:+,.0f} < threshold {daily_thr:+,.0f}"

    # 3. Weekly DD
    wd = weekly_dd(cfg)
    weekly_thr = -cfg.weekly_dd_threshold_pct * cfg.account_equity
    if wd < weekly_thr:
        return False, f"WEEKLY_DD: {wd:+,.0f} < threshold {weekly_thr:+,.0f}"

    # 4. Margin usage with additional
    proj_margin = margin_usage(cfg) + additional_lots * cfg.margin_per_lot
    margin_thr = cfg.margin_usage_max_pct * cfg.account_equity
    if proj_margin > margin_thr:
        return False, f"MARGIN: {proj_margin:+,.0f} > threshold {margin_thr:+,.0f}"

    # 5. Concurrent cap (delegated to position_manager)
    from .position_manager import can_open as pm_can_open
    ok, msg = pm_can_open(cfg, additional_lots)
    if not ok:
        return False, f"CAP: {msg}"

    return True, "OK"


def status_report(cfg: DailyXGBConfig) -> str:
    """Multi-line status report for display."""
    killed, _ = kill_switch_active(cfg)
    dl = daily_loss(cfg)
    wd = weekly_dd(cfg)
    mu = margin_usage(cfg)
    open_pos = get_open_positions(cfg)
    total_lots = sum(p.gross_lots for p in open_pos)

    lines = [
        f"=== Daily XGB Risk Status ({_bj_now().strftime('%Y-%m-%d %H:%M:%S')} BJ) ===",
        f"  Kill switch:     {'⚠ ACTIVE' if killed else 'OFF'}",
        f"  Open positions:  {total_lots} 手 / cap {cfg.concurrent_cap}",
        f"  Margin used:     {mu:>12,.0f} / {cfg.margin_usage_max_pct*cfg.account_equity:>12,.0f} "
        f"({mu/cfg.account_equity*100:.1f}%)",
        f"  Daily PnL:       {dl:>+12,.0f}  threshold {-cfg.daily_loss_threshold_pct*cfg.account_equity:>+12,.0f}",
        f"  Weekly PnL:      {wd:>+12,.0f}  threshold {-cfg.weekly_dd_threshold_pct*cfg.account_equity:>+12,.0f}",
    ]
    ok, reason = check_can_open(cfg, 1)
    lines.append(f"  Can open new:    {'YES' if ok else 'NO — ' + reason}")
    return "\n".join(lines)


__all__ = [
    "kill_switch_active", "weekly_dd", "daily_loss", "margin_usage",
    "check_can_open", "status_report",
]
