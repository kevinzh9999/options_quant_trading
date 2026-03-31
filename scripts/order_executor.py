"""
order_executor.py
-----------------
信号驱动的半自动下单。监控 /tmp/signal_pending.json，
操作者确认后通过 TQ API 下限价单。

设计原则：
  - 限价单（做多ask1排队，做空bid1排队）
  - 手数 = monitor建议手数 ÷ 2（保守起步）
  - 开仓：60秒超时自动放弃
  - 平仓：超时持续提醒直到确认
  - 所有信号无论是否下单都记录到 executor_log 表
  - 下单后60秒未成交自动撤单（止损30秒后提示改市价）

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
CONTRACT_MULT = {"IF": 300, "IH": 300, "IM": 200, "IC": 200}
STOP_LOSS_PCT = 0.005
MAX_DAILY_ORDERS = 10
MAX_DAILY_LOSS_PCT = 0.01  # 1% of account
SIGNAL_EXPIRY_SECS = 300   # 5分钟过期
OPEN_FILL_TIMEOUT = 60     # 开仓等成交60秒
CLOSE_FILL_TIMEOUT = 30    # 平仓等成交30秒（更紧急）
# 中金所股指期货品种（平今手续费10倍，用锁仓代替平今）
CFFEX_INDEX_FUTURES = {"IM", "IF", "IH", "IC"}

W = 60  # panel width


# ---------------------------------------------------------------------------
# 数据库
# ---------------------------------------------------------------------------

def _open_db_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_tables(db_path: str):
    """确保 order_log 和 executor_log 表存在。"""
    conn = _open_db_conn(db_path)
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS executor_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time       TEXT,
            receive_time      TEXT,
            symbol            TEXT,
            direction         TEXT,
            action            TEXT,
            score             INT,
            reason            TEXT,
            limit_price       REAL,
            suggested_lots    INT,
            actual_lots       INT,
            operator_response TEXT,
            response_reason   TEXT,
            order_submitted   INT DEFAULT 0,
            order_id          TEXT,
            filled_lots       INT DEFAULT 0,
            filled_price      REAL,
            order_status      TEXT,
            cancel_time       TEXT,
            cancel_reason     TEXT,
            signal_json       TEXT
        )
    """)
    conn.commit()
    conn.close()


# Keep old function names for backward compatibility (tests import these)
_ensure_order_log = _ensure_tables


def _record_order(db_path: str, record: dict):
    """记录下单结果到 order_log（向后兼容）。"""
    conn = _open_db_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO order_log "
        "(datetime, symbol, direction, action, limit_price, lots, "
        "filled_lots, filled_price, status, signal_score, reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (record.get("datetime", ""), record.get("symbol", ""),
         record.get("direction", ""), record.get("action", "OPEN"),
         record.get("limit_price", 0), record.get("lots", 0),
         record.get("filled_lots", 0), record.get("filled_price", 0),
         record.get("status", ""), record.get("signal_score", 0),
         record.get("reason", "")),
    )
    conn.commit()
    conn.close()


def _log_signal(db_path: str, signal: dict, receive_time: str,
                actual_lots: int) -> int:
    """写入 executor_log，返回 row id。"""
    conn = _open_db_conn(db_path)
    cur = conn.execute(
        "INSERT INTO executor_log "
        "(signal_time, receive_time, symbol, direction, action, score, "
        "reason, limit_price, suggested_lots, actual_lots, signal_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (signal.get("timestamp", ""), receive_time,
         signal.get("symbol", ""), signal.get("direction", ""),
         signal.get("action", "OPEN"), signal.get("score", 0),
         signal.get("reason", ""), signal.get("limit_price", 0),
         signal.get("suggested_lots", 0), actual_lots,
         json.dumps(signal, ensure_ascii=False)),
    )
    log_id = cur.lastrowid
    conn.commit()
    conn.close()
    return log_id


def _update_log(db_path: str, log_id: int, **fields):
    """更新 executor_log 的指定字段。"""
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [log_id]
    conn = _open_db_conn(db_path)
    conn.execute(f"UPDATE executor_log SET {sets} WHERE id = ?", vals)
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
    """等待用户输入，超时返回空字符串。按c撤单。"""
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
    positions: dict,
) -> dict:
    """处理一个信号。返回 {status, filled_lots, ...}。"""
    receive_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = signal.get("timestamp", receive_time)
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
    mult = CONTRACT_MULT.get(sym, 200)

    # 计算下单手数
    if action == "CLOSE":
        # 用实际持仓手数（如果有）
        pos = positions.get(sym)
        lots = pos["lots"] if pos else signal.get("lots", max(1, suggested // 2))
    else:
        lots = max(1, suggested // 2)

    # 记录到 executor_log（无论后续是否下单）
    log_id = _log_signal(db_path, signal, receive_time, lots)

    # 信号过期检查
    try:
        sig_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - sig_dt).total_seconds()
        # UTC时间可能差8小时，用utcnow比较
        age_utc = (datetime.utcnow() - sig_dt).total_seconds()
        age = min(abs(age), abs(age_utc))
    except Exception:
        age = 0
    if age > SIGNAL_EXPIRY_SECS:
        print(f"  信号已过期（{age:.0f}秒前），跳过")
        _update_log(db_path, log_id,
                    operator_response="EXPIRED",
                    response_reason=f"信号{age:.0f}秒前产生，超过{SIGNAL_EXPIRY_SECS}秒",
                    order_status="EXPIRED")
        return {"status": "EXPIRED"}

    # 安全检查
    if action == "OPEN":
        if daily_orders >= MAX_DAILY_ORDERS:
            print(f"  ⚠ 已达单日最大下单次数({MAX_DAILY_ORDERS})，拒绝开仓")
            _update_log(db_path, log_id,
                        operator_response="REJECTED",
                        response_reason="MAX_ORDERS",
                        order_status="NOT_SUBMITTED")
            return {"status": "REJECTED", "reason": "MAX_ORDERS"}
        if account > 0 and daily_loss > account * MAX_DAILY_LOSS_PCT:
            print(f"  ⚠ 单日亏损已超{MAX_DAILY_LOSS_PCT*100:.0f}%，拒绝开仓")
            _update_log(db_path, log_id,
                        operator_response="REJECTED",
                        response_reason="MAX_LOSS",
                        order_status="NOT_SUBMITTED")
            return {"status": "REJECTED", "reason": "MAX_LOSS"}
        # 检查是否已有同品种持仓
        if sym in positions:
            pos = positions[sym]
            if pos["direction"] == direction:
                print(f"  {sym} 已有{pos['direction']}持仓{pos['lots']}手，跳过重复开仓")
                _update_log(db_path, log_id,
                            operator_response="REJECTED",
                            response_reason=f"已有{pos['direction']}持仓",
                            order_status="NOT_SUBMITTED")
                return {"status": "REJECTED", "reason": "DUPLICATE"}

    if action == "CLOSE" and sym not in positions:
        print(f"  {sym} 无实际持仓，忽略平仓信号")
        _update_log(db_path, log_id,
                    operator_response="REJECTED",
                    response_reason="无实际持仓",
                    order_status="NOT_SUBMITTED")
        return {"status": "REJECTED", "reason": "NO_POSITION"}

    # 止损金额
    stop_loss_price = limit_price * (1 + STOP_LOSS_PCT) if direction == "SHORT" \
        else limit_price * (1 - STOP_LOSS_PCT)
    max_loss = abs(limit_price - stop_loss_price) * mult * lots

    # 判断是否用锁仓（股指期货平今手续费10倍）
    is_close = action == "CLOSE"
    use_lock = is_close and sym in CFFEX_INDEX_FUTURES
    is_urgent_close = is_close

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
        print(f" 信号分数: {score}        建议手数: {suggested}手(已÷2={lots})")
        print(f" 止损价: {stop_loss_price:.1f}      最大亏损: ¥{max_loss:,.0f}")
    else:
        print(f" 平仓原因: {reason}")
        if pnl_pts:
            print(f" 预计盈亏: {pnl_pts:+.0f}pt = ¥{pnl_pts * mult * lots:+,.0f}")
        if use_lock:
            print(f" 说明: 股指期货平今手续费10倍，改用锁仓。明日开盘后双向平仓。")

    record_action = "LOCK" if use_lock else action

    if dry_run:
        print(f"\n [DRY RUN] 不实际下单")
        print(f"{'═' * W}")
        _update_log(db_path, log_id,
                    operator_response="DRY_RUN", order_status="NOT_SUBMITTED")
        return {"status": "DRY_RUN"}

    # 确认
    resp = _confirm(f" 确认下单？[y/n] (60s timeout)", 60.0)

    if resp == "y":
        _update_log(db_path, log_id, operator_response="Y",
                    response_reason="手工确认")
    elif resp == "n":
        print(f"  -> SKIPPED")
        _update_log(db_path, log_id, operator_response="N",
                    response_reason="手工拒绝", order_status="NOT_SUBMITTED")
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
            _update_log(db_path, log_id, operator_response="TIMEOUT",
                        response_reason="60秒超时，平仓持续提醒")
            print(f"\n  ⚠ 平仓信号({reason})未确认！请处理持仓")
            while True:
                resp2 = _confirm(
                    f"  ⚠ {sym} {reason} 未执行！确认？[y/n] (30s)", 30.0)
                if resp2 == "y":
                    _update_log(db_path, log_id, operator_response="Y",
                                response_reason="超时后手工确认")
                    break
                elif resp2 == "n":
                    print(f"  -> 平仓放弃（手动处理）")
                    _update_log(db_path, log_id, operator_response="N",
                                response_reason="超时后手工拒绝",
                                order_status="NOT_SUBMITTED")
                    _record_order(db_path, {
                        "datetime": ts, "symbol": sym, "direction": direction,
                        "action": action, "limit_price": limit_price,
                        "lots": lots, "filled_lots": 0, "filled_price": 0,
                        "status": "CLOSE_SKIP", "signal_score": score,
                        "reason": reason,
                    })
                    return {"status": "CLOSE_SKIP"}
                else:
                    print(f"  ⚠ {sym} {reason} 未执行！请处理")
        else:
            print(f"  -> TIMEOUT_SKIP")
            _update_log(db_path, log_id, operator_response="TIMEOUT",
                        response_reason="60秒超时自动放弃",
                        order_status="NOT_SUBMITTED")
            _record_order(db_path, {
                "datetime": ts, "symbol": sym, "direction": direction,
                "action": record_action, "limit_price": limit_price, "lots": lots,
                "filled_lots": 0, "filled_price": 0, "status": "TIMEOUT_SKIP",
                "signal_score": score, "reason": reason,
            })
            return {"status": "TIMEOUT_SKIP"}

    # === 实际下单 ===
    print(f"\n  下单中...")
    _update_log(db_path, log_id, order_submitted=1)
    filled = 0
    trade_price = 0.0
    status = "ERROR"

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
                tq_dir = "SELL" if direction == "LONG" else "BUY"
                tq_offset = "OPEN"
            else:
                tq_dir = "BUY" if direction == "SHORT" else "SELL"
                tq_offset = "CLOSE"

            order = api.insert_order(
                contract, direction=tq_dir, offset=tq_offset,
                volume=lots, limit_price=limit_price,
            )
            order_id = getattr(order, "order_id", "")
            _update_log(db_path, log_id, order_id=str(order_id))
            print(f"  订单已提交: {contract} {tq_dir} {tq_offset}"
                  f" {lots}手 @ {limit_price:.1f}")

            # 等待成交（带倒计时和手动撤单支持）
            fill_timeout = CLOSE_FILL_TIMEOUT if is_close else OPEN_FILL_TIMEOUT
            deadline = time.time() + fill_timeout
            last_print = 0
            while not order.is_dead:
                api.wait_update()
                remaining = int(deadline - time.time())
                cur_filled = order.volume_orign - order.volume_left
                # 每10秒打印一次
                if remaining != last_print and remaining % 10 == 0 and remaining > 0:
                    last_print = remaining
                    print(f"  等待成交... {remaining}秒"
                          f" 已成交{cur_filled}/{order.volume_orign}手")
                if time.time() > deadline:
                    break

            filled = order.volume_orign - order.volume_left
            trade_price = getattr(order, "trade_price", limit_price)

            # 未完全成交 → 撤单
            if order.volume_left > 0 and not order.is_dead:
                api.cancel_order(order)
                cancel_time = datetime.now().strftime("%H:%M:%S")
                print(f"  撤单: {order.volume_left}手未成交，已撤销")
                # 等撤单确认
                cancel_deadline = time.time() + 5
                while not order.is_dead:
                    api.wait_update()
                    if time.time() > cancel_deadline:
                        break
                filled = order.volume_orign - order.volume_left
                _update_log(db_path, log_id, cancel_time=cancel_time,
                            cancel_reason=f"{fill_timeout}秒未全部成交")

            if filled == 0:
                status = "TIMEOUT_CANCEL"
                print(f"  未成交，限价{limit_price:.1f}未触及")
                # 平仓未成交：提示改激进价
                if is_urgent_close:
                    quote = api.get_quote(contract)
                    api.wait_update()
                    if direction == "LONG":
                        aggr_price = float(quote.bid_price1) - 2.0
                    else:
                        aggr_price = float(quote.ask_price1) + 2.0
                    resp3 = _confirm(
                        f"  ⚠ {reason}未成交！改激进价{aggr_price:.1f}？[y/n]",
                        30.0)
                    if resp3 == "y":
                        order2 = api.insert_order(
                            contract, direction=tq_dir, offset=tq_offset,
                            volume=lots, limit_price=aggr_price,
                        )
                        deadline2 = time.time() + 15
                        while not order2.is_dead:
                            api.wait_update()
                            if time.time() > deadline2:
                                if not order2.is_dead:
                                    api.cancel_order(order2)
                                break
                        filled = order2.volume_orign - order2.volume_left
                        trade_price = getattr(order2, "trade_price", aggr_price)
                        status = "FILLED" if filled == lots else (
                            "PARTIAL" if filled > 0 else "FAILED")
                        if filled > 0:
                            print(f"  激进价成交 {filled}手 @ {trade_price:.1f}")
                        else:
                            print(f"  激进价也未成交")
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

    # 更新 executor_log
    _update_log(db_path, log_id,
                filled_lots=filled, filled_price=trade_price,
                order_status=status)

    # 记录到 order_log（向后兼容）
    _record_order(db_path, {
        "datetime": ts, "symbol": sym, "direction": direction,
        "action": record_action, "limit_price": limit_price, "lots": lots,
        "filled_lots": filled, "filled_price": trade_price,
        "status": status, "signal_score": score, "reason": reason,
    })

    # 更新内部持仓追踪
    if filled > 0:
        if action == "OPEN":
            positions[sym] = {
                "direction": direction, "lots": filled,
                "entry_price": trade_price,
                "entry_time": datetime.now().strftime("%H:%M"),
            }
        elif action == "CLOSE":
            positions.pop(sym, None)

    print(f"{'═' * W}")
    pnl_yuan = 0.0
    if filled > 0 and action == "OPEN":
        pnl_yuan = 0  # 开仓无PnL
    elif filled > 0 and action == "CLOSE" and pnl_pts:
        pnl_yuan = pnl_pts * mult * filled

    return {"status": status, "filled_lots": filled, "pnl_yuan": pnl_yuan}


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
            lots_val = int(row.get("filled_lots", 0))
            price = float(row.get("filled_price", 0))
            orig_dir = "多头" if direction == "SHORT" else "空头"
            lock_dir = "空头" if direction == "SHORT" else "多头"
            print(f"   {sym}: 原{orig_dir}{lots_val}手 + 锁{lock_dir}{lots_val}手"
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
    _ensure_tables(db_path)

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

    # 内部持仓追踪
    positions: dict = {}

    print(f"{'═' * W}")
    print(f" Order Executor | {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f" 账户: {account:,.0f}  信号文件: {SIGNAL_FILE}")
    print(f" 安全: 单日最多{MAX_DAILY_ORDERS}单"
          f"  亏损上限{MAX_DAILY_LOSS_PCT*100:.0f}%")
    print(f"{'═' * W}")

    # 检查昨日锁仓持仓
    _check_locked_positions(db_path, args.dry_run)

    daily_orders = 0
    daily_loss = 0.0
    signals_received = 0
    signals_executed = 0

    print(f" 等待信号...\n")

    try:
        while True:
            signal = _read_signal()
            if signal:
                signals_received += 1
                result = _execute_order(
                    signal, args.dry_run, db_path,
                    daily_orders, daily_loss, account,
                    positions,
                )
                if result.get("status") in ("FILLED", "PARTIAL"):
                    daily_orders += 1
                    signals_executed += 1
                    pnl = result.get("pnl_yuan", 0)
                    if pnl < 0:
                        daily_loss += abs(pnl)

                # 显示状态
                print(f"\n 持仓: ", end="")
                if positions:
                    for s, p in positions.items():
                        d = "L" if p["direction"] == "LONG" else "S"
                        print(f"{s} {d}{p['lots']}手@{p['entry_price']:.0f}", end="  ")
                    print()
                else:
                    print("无")
                print(f" 今日: 收到{signals_received}信号  "
                      f"成交{signals_executed}单  亏损¥{daily_loss:,.0f}")
                print(f" 等待信号...\n")

            # 检查时间
            now = datetime.now()
            if now.hour >= 15 and now.minute >= 5:
                print(f"\n  收盘，退出")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n  退出")


if __name__ == "__main__":
    main()
