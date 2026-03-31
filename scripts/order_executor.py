"""
order_executor.py
-----------------
信号驱动的半自动下单。监控 /tmp/signal_pending.json，
操作者确认后通过 TQ API 下限价单。

设计原则：
  - 限价单（做多ask1排队，做空bid1排队）
  - 手数 = monitor建议手数 ÷ 2（保守起步）
  - 必须手工确认，60秒超时自动放弃（止损除外）
  - 成交后记录到 order_log 表

用法：
    python scripts/order_executor.py           # 持续监控
    python scripts/order_executor.py --dry-run # 只显示不下单
"""

from __future__ import annotations

import json
import os
import select
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager

SIGNAL_FILE = "/tmp/signal_pending.json"
IM_MULT = 200
STOP_LOSS_PCT = 0.005
MAX_DAILY_ORDERS = 10
MAX_DAILY_LOSS_PCT = 0.01  # 1% of account
# 中金所股指期货品种（平今手续费10倍，用锁仓代替平今）
CFFEX_INDEX_FUTURES = {"IM", "IF", "IH", "IC"}

W = 60  # panel width


# ---------------------------------------------------------------------------
# 数据库
# ---------------------------------------------------------------------------

def _ensure_order_log(db_path: str):
    """确保 order_log 表存在。"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS order_log (
            datetime     TEXT,
            symbol       TEXT,
            direction    TEXT,
            action       TEXT,
            limit_price  REAL,
            lots         INT,
            filled_lots  INT,
            filled_price REAL,
            status       TEXT,
            signal_score INT,
            reason       TEXT,
            PRIMARY KEY (datetime, symbol, action)
        )
    """)
    conn.commit()
    conn.close()


def _record_order(db_path: str, record: dict):
    """记录下单结果。"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        "INSERT OR REPLACE INTO order_log "
        "(datetime, symbol, direction, action, limit_price, lots, "
        "filled_lots, filled_price, status, signal_score, reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.get("datetime", ""),
            record.get("symbol", ""),
            record.get("direction", ""),
            record.get("action", "OPEN"),
            record.get("limit_price", 0),
            record.get("lots", 0),
            record.get("filled_lots", 0),
            record.get("filled_price", 0),
            record.get("status", ""),
            record.get("signal_score", 0),
            record.get("reason", ""),
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 合约解析
# ---------------------------------------------------------------------------

def _get_main_contract(symbol: str) -> str:
    """获取当月主力合约。"""
    try:
        from utils.cffex_calendar import active_im_months
        today = datetime.now().strftime("%Y%m%d")
        months = active_im_months(today)
        if months:
            return f"CFFEX.{symbol}{months[0]}"
    except Exception:
        pass
    # fallback: 用当前年月估算
    now = datetime.now()
    ym = f"{now.year % 100:02d}{now.month:02d}"
    return f"CFFEX.{symbol}{ym}"


# ---------------------------------------------------------------------------
# 信号读取
# ---------------------------------------------------------------------------

def _read_signal() -> Optional[dict]:
    """读取并删除 pending signal 文件。"""
    if not os.path.exists(SIGNAL_FILE):
        return None
    try:
        with open(SIGNAL_FILE, "r") as f:
            data = json.load(f)
        os.remove(SIGNAL_FILE)
        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 确认交互
# ---------------------------------------------------------------------------

def _confirm(prompt: str, timeout: float = 60.0) -> str:
    """等待用户输入，超时返回空字符串。"""
    print(prompt)
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except Exception:
        ready = []
    if not ready:
        return ""
    try:
        return sys.stdin.readline().strip().lower()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 下单执行
# ---------------------------------------------------------------------------

def _execute_order(
    signal: dict, dry_run: bool, db_path: str,
    daily_orders: int, daily_loss: float, account: float,
) -> dict:
    """处理一个信号。返回 {status, filled_lots, ...}。"""
    ts = signal.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sym = signal.get("symbol", "IM")
    direction = signal.get("direction", "")
    action = signal.get("action", "OPEN")
    score = signal.get("score", 0)
    bid1 = signal.get("bid1", 0)
    ask1 = signal.get("ask1", 0)
    last = signal.get("last", 0)
    limit_price = signal.get("limit_price", 0)
    suggested = signal.get("suggested_lots", 1)
    reason = signal.get("reason", "")
    pnl_pts = signal.get("pnl_pts", 0)

    # 计算下单手数
    if action == "CLOSE":
        lots = signal.get("lots", 1)
    else:
        lots = max(1, suggested // 2)

    # 安全检查
    if action == "OPEN":
        if daily_orders >= MAX_DAILY_ORDERS:
            print(f"  ⚠ 已达单日最大下单次数({MAX_DAILY_ORDERS})，拒绝开仓")
            return {"status": "REJECTED", "reason": "MAX_ORDERS"}
        if account > 0 and daily_loss > account * MAX_DAILY_LOSS_PCT:
            print(f"  ⚠ 单日亏损已超{MAX_DAILY_LOSS_PCT*100:.0f}%，拒绝开仓")
            return {"status": "REJECTED", "reason": "MAX_LOSS"}

    # 止损金额
    stop_loss_price = limit_price * (1 + STOP_LOSS_PCT) if direction == "SHORT" \
        else limit_price * (1 - STOP_LOSS_PCT)
    max_loss = abs(limit_price - stop_loss_price) * IM_MULT * lots

    # 判断是否用锁仓（股指期货平今手续费10倍）
    is_close = action == "CLOSE"
    use_lock = is_close and sym in CFFEX_INDEX_FUTURES

    # 展示订单
    if not is_close:
        dir_cn = "卖出开仓" if direction == "SHORT" else "买入开仓"
    elif use_lock:
        dir_cn = "卖出开仓(锁仓)" if direction == "LONG" else "买入开仓(锁仓)"
    else:
        dir_cn = "买入平仓" if direction == "SHORT" else "卖出平仓"

    print(f"\n{'═' * W}")
    print(f" {'平仓信号' if is_close else '新信号'} | {ts}")
    print(f"{'═' * W}")
    print(f" 品种: {sym}          方向: {dir_cn}")
    print(f" 限价: {limit_price:.1f}        手数: {lots}手")
    if not is_close:
        print(f" 信号分数: {score}        建议手数: {suggested}手(已÷2)")
        print(f" 止损价: {stop_loss_price:.1f}      最大亏损: ¥{max_loss:,.0f}")
    else:
        print(f" 平仓原因: {reason}")
        if pnl_pts:
            print(f" 预计盈亏: {pnl_pts:+.0f}pt = ¥{pnl_pts * IM_MULT * lots:+,.0f}")
        if use_lock:
            print(f" 说明: 股指期货平今手续费10倍，改用锁仓。明日开盘后双向平仓。")

    # 记录用的 action 标签
    record_action = "LOCK" if use_lock else action
    is_urgent_close = is_close  # 所有平仓信号超时都持续提醒

    if dry_run:
        print(f"\n [DRY RUN] 不实际下单")
        print(f"{'═' * W}")
        return {"status": "DRY_RUN"}

    # 确认
    timeout = 60.0
    resp = _confirm(
        f" 确认下单？[y/n] (60s timeout, 默认放弃)", timeout)

    if resp == "y":
        pass  # continue to execution
    elif resp == "n":
        print(f"  -> SKIPPED")
        _record_order(db_path, {
            "datetime": ts, "symbol": sym, "direction": direction,
            "action": record_action, "limit_price": limit_price, "lots": lots,
            "filled_lots": 0, "filled_price": 0, "status": "SKIPPED",
            "signal_score": score, "reason": reason,
        })
        return {"status": "SKIPPED"}
    else:
        # 超时
        if is_urgent_close:
            # 平仓超时：持续警告直到确认
            print(f"\n  ⚠ 平仓信号({reason})未确认！请手动处理持仓")
            while True:
                resp2 = _confirm(
                    f"  ⚠ {sym} {reason} 未执行！确认处理？[y/n] (30s)", 30.0)
                if resp2 == "y":
                    break
                elif resp2 == "n":
                    print(f"  -> 平仓放弃（手动处理）")
                    _record_order(db_path, {
                        "datetime": ts, "symbol": sym, "direction": direction,
                        "action": action, "limit_price": limit_price,
                        "lots": lots, "filled_lots": 0, "filled_price": 0,
                        "status": "CLOSE_SKIP", "signal_score": score,
                        "reason": reason,
                    })
                    return {"status": "CLOSE_SKIP"}
                else:
                    print(f"  ⚠ {sym} {reason} 未执行！请手动处理")
        else:
            print(f"  -> TIMEOUT_SKIP")
            _record_order(db_path, {
                "datetime": ts, "symbol": sym, "direction": direction,
                "action": record_action, "limit_price": limit_price, "lots": lots,
                "filled_lots": 0, "filled_price": 0, "status": "TIMEOUT_SKIP",
                "signal_score": score, "reason": reason,
            })
            return {"status": "TIMEOUT_SKIP"}

    # === 实际下单 ===
    print(f"\n  下单中...")
    try:
        from data.sources.tq_client import TqClient

        creds = {
            "auth_account": os.getenv("TQ_ACCOUNT", ""),
            "auth_password": os.getenv("TQ_PASSWORD", ""),
            "broker_id": os.getenv("TQ_BROKER", ""),
            "account_id": os.getenv("TQ_ACCOUNT_ID", ""),
            "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
        }
        contract = _get_main_contract(sym)

        client = TqClient(**creds)
        client.connect()
        api = client._api

        try:
            # TQ 方向和开平
            if action == "OPEN":
                tq_dir = "SELL" if direction == "SHORT" else "BUY"
                tq_offset = "OPEN"
            elif use_lock:
                # 股指期货锁仓：反向开仓（避免平今10倍手续费）
                tq_dir = "SELL" if direction == "LONG" else "BUY"
                tq_offset = "OPEN"
            else:
                # 普通平仓（期权或平昨仓）
                tq_dir = "BUY" if direction == "SHORT" else "SELL"
                tq_offset = "CLOSE"

            order = api.insert_order(
                contract, direction=tq_dir, offset=tq_offset,
                volume=lots, limit_price=limit_price,
            )
            print(f"  订单已提交: {contract} {tq_dir} {tq_offset}"
                  f" {lots}手 @ {limit_price:.1f}")

            # 等待成交（最多60秒）
            deadline = time.time() + 60
            while not order.is_dead:
                api.wait_update()
                if time.time() > deadline:
                    api.cancel_order(order)
                    print(f"  超时未成交，已撤单")
                    break

            filled = order.volume_orign - order.volume_left
            trade_price = getattr(order, "trade_price", limit_price)

            if filled == 0:
                status = "TIMEOUT"
                print(f"  未成交，限价{limit_price:.1f}未触及")
                if is_urgent_close:
                    resp3 = _confirm(
                        f"  ⚠ {reason}未成交！改市价单？[y/n]", 30.0)
                    if resp3 == "y":
                        order2 = api.insert_order(
                            contract, direction=tq_dir, offset=tq_offset,
                            volume=lots,
                        )
                        deadline2 = time.time() + 10
                        while not order2.is_dead:
                            api.wait_update()
                            if time.time() > deadline2:
                                break
                        filled = order2.volume_orign - order2.volume_left
                        trade_price = getattr(order2, "trade_price", 0)
                        status = "FILLED_MARKET" if filled > 0 else "FAILED"
                        print(f"  市价单: {filled}手成交"
                              f" @ {trade_price:.1f}" if filled > 0
                              else "  市价单也未成交")
            elif filled == lots:
                status = "FILLED"
                print(f"  全部成交 {filled}手 @ {trade_price:.1f}")
            else:
                status = "PARTIAL"
                print(f"  部分成交 {filled}/{lots}手 @ {trade_price:.1f}")

        finally:
            client.disconnect()

    except Exception as e:
        status = "ERROR"
        filled = 0
        trade_price = 0
        print(f"  下单失败: {e}")

    # 记录
    _record_order(db_path, {
        "datetime": ts, "symbol": sym, "direction": direction,
        "action": record_action, "limit_price": limit_price, "lots": lots,
        "filled_lots": filled, "filled_price": trade_price,
        "status": status, "signal_score": score, "reason": reason,
    })

    print(f"{'═' * W}")
    return {
        "status": status,
        "filled_lots": filled,
        "pnl_yuan": (trade_price - limit_price) * IM_MULT * filled
        if direction == "LONG" else
        (limit_price - trade_price) * IM_MULT * filled,
    }


# ---------------------------------------------------------------------------
# 锁仓检查
# ---------------------------------------------------------------------------

def _check_locked_positions(db_path: str, dry_run: bool):
    """启动时检查是否有前日锁仓持仓需要平仓。"""
    try:
        db = DBManager(db_path)
        locks = db.query_df(
            "SELECT * FROM order_log WHERE action='LOCK' AND status='FILLED' "
            "AND datetime >= date('now', '-1 day') "
            "ORDER BY datetime"
        )
        if locks is None or locks.empty:
            return

        print(f"\n ⚠ 发现锁仓持仓，需要平仓（平昨仓，手续费低）：")
        for _, row in locks.iterrows():
            sym = row.get("symbol", "")
            direction = row.get("direction", "")
            lots = int(row.get("filled_lots", 0))
            price = float(row.get("filled_price", 0))
            orig_dir = "多头" if direction == "SHORT" else "空头"
            lock_dir = "空头" if direction == "SHORT" else "多头"
            print(f"   {sym}: 原{orig_dir}{lots}手 + 锁{lock_dir}{lots}手"
                  f" @ {price:.1f} → 建议双向平仓")

        if not dry_run:
            resp = _confirm(
                f" 需要现在平仓吗？（开盘后执行）[y/n]", 30.0)
            if resp == "y":
                print(f"  → 请在盘中手动平仓（双向都用CLOSE，平昨仓手续费低）")
            else:
                print(f"  → 稍后处理")
        print()
    except Exception as e:
        print(f"  [WARN] 锁仓检查失败: {e}")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="半自动下单")
    parser.add_argument("--dry-run", action="store_true",
                        help="只显示不下单")
    args = parser.parse_args()

    db_path = ConfigLoader().get_db_path()
    _ensure_order_log(db_path)

    # 读取账户权益
    account = 0.0
    try:
        db = DBManager(db_path)
        df = db.query_df(
            "SELECT balance FROM account_snapshots "
            "ORDER BY trade_date DESC LIMIT 1"
        )
        if df is not None and not df.empty:
            account = float(df.iloc[0]["balance"])
    except Exception:
        pass

    print(f"{'═' * W}")
    print(f" Order Executor | {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f" 账户: {account:,.0f}  信号文件: {SIGNAL_FILE}")
    print(f" 安全: 单日最多{MAX_DAILY_ORDERS}单"
          f"  亏损上限{MAX_DAILY_LOSS_PCT*100:.0f}%")
    print(f"{'═' * W}")

    # 检查昨日锁仓持仓
    _check_locked_positions(db_path, args.dry_run)

    print(f" 等待信号...\n")

    daily_orders = 0
    daily_loss = 0.0

    try:
        while True:
            signal = _read_signal()
            if signal:
                result = _execute_order(
                    signal, args.dry_run, db_path,
                    daily_orders, daily_loss, account,
                )
                if result.get("status") in ("FILLED", "PARTIAL", "FILLED_MARKET"):
                    daily_orders += 1
                    pnl = result.get("pnl_yuan", 0)
                    if pnl < 0:
                        daily_loss += abs(pnl)

            # 检查时间
            now = datetime.now()
            if now.hour >= 15 and now.minute >= 5:
                print(f"\n  收盘，退出")
                break

            time.sleep(1)  # 每秒检查一次信号文件

    except KeyboardInterrupt:
        print(f"\n  退出")


if __name__ == "__main__":
    main()
