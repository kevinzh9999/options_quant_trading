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

_TMP_DIR = ConfigLoader().get_tmp_dir()
SIGNAL_FILE = os.path.join(_TMP_DIR, "signal_pending.json")
CONTRACT_MULT = {"IF": 300, "IH": 300, "IM": 200, "IC": 200}
STOP_LOSS_PCT = 0.005
MAX_DAILY_ORDERS = 10
MAX_DAILY_LOSS_PCT = 0.01  # 1% of account
SIGNAL_EXPIRY_SECS = 300   # 5分钟过期
OPEN_FILL_TIMEOUT = 60     # 开仓等成交60秒
CLOSE_FILL_TIMEOUT = 30    # 平仓等成交30秒（更紧急）
EXEC_MAX_LOTS = 1          # 实盘验证期，限定最大1手（稳定后改回None）
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
    # 新增列（已有表兼容）
    try:
        conn.execute("ALTER TABLE order_log ADD COLUMN lock_resolved TEXT")
    except sqlite3.OperationalError:
        pass  # 列已存在
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

# ---------------------------------------------------------------------------
# 平仓否决记录
# ---------------------------------------------------------------------------

_DENIED_FILE = os.path.join(_TMP_DIR, "denied_positions.json")


def _record_denied_position(symbol: str, contract: str, direction: str,
                            lots: int, reason: str, entry_price: float):
    """将被否决的平仓持仓写入 denied_positions.json（追加）。"""
    record = {
        "date": datetime.now().strftime("%Y%m%d"),
        "symbol": symbol,
        "contract": contract,
        "direction": direction,
        "lots": lots,
        "deny_time": datetime.now().strftime("%H:%M:%S"),
        "deny_reason": reason,
        "entry_price": entry_price,
    }
    existing = []
    if os.path.exists(_DENIED_FILE):
        try:
            with open(_DENIED_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(record)
    try:
        with open(_DENIED_FILE, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [WARN] 写入 denied_positions.json 失败: {e}")


# ---------------------------------------------------------------------------
# 信号读取
# ---------------------------------------------------------------------------

def _read_signals() -> list:
    """读取所有待处理信号（列表），读后删除文件。兼容旧单对象格式。"""
    if not os.path.exists(SIGNAL_FILE):
        return []
    try:
        with open(SIGNAL_FILE, "r") as f:
            data = json.load(f)
        os.remove(SIGNAL_FILE)
        if isinstance(data, list):
            return data
        return [data]  # 兼容旧格式
    except Exception:
        try:
            os.remove(SIGNAL_FILE)
        except OSError:
            pass
        return []


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
    positions: dict, tq_api=None,
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

    # 计算下单手数（三级：建议 → 减半 → 实际执行上限）
    if action == "CLOSE":
        pos = positions.get(sym)
        lots = pos["lots"] if pos else signal.get("lots", max(1, suggested // 2))
    else:
        half_lots = max(1, suggested // 2)
        lots = min(half_lots, EXEC_MAX_LOTS) if EXEC_MAX_LOTS else half_lots

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
    # 紧急平仓(opt-out: 超时自动执行)：STOP_LOSS / EOD_CLOSE / LUNCH_CLOSE
    _URGENT_REASONS = {"STOP_LOSS", "EOD_CLOSE", "LUNCH_CLOSE"}
    is_urgent_close = is_close and reason in _URGENT_REASONS

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
    print(f" 限价: {limit_price:.1f}")
    if not is_close:
        half = max(1, suggested // 2)
        print(f" 手数: 建议{suggested}手 → 减半{half}手 → 实际执行{lots}手")
        print(f" 信号分数: {score}")
        print(f" 止损价: {stop_loss_price:.1f}      最大亏损: ¥{max_loss:,.0f}")
    else:
        print(f" 手数: {lots}手")
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

    # 确认逻辑分三级：
    #   紧急平仓(STOP_LOSS/EOD/LUNCH): opt-out，60s无响应自动执行
    #   非紧急平仓(其他CLOSE原因):     opt-in，60s无响应自动放弃
    #   开仓:                          opt-in，60s无响应自动放弃
    if is_urgent_close:
        resp = _confirm(
            f" 确认平仓？[y/n] (60s无响应将自动执行)", 60.0)
    elif is_close:
        resp = _confirm(f" 确认平仓？[y/n] (60s timeout)", 60.0)
    else:
        resp = _confirm(f" 确认下单？[y/n] (60s timeout)", 60.0)

    if resp == "y":
        _update_log(db_path, log_id, operator_response="Y",
                    response_reason="手工确认")
    elif resp == "n":
        if is_close:
            # 所有平仓被否决 → CLOSE_DENIED + 写入denied_positions.json
            print(f"  -> CLOSE_DENIED（操作者否决平仓）")
            _update_log(db_path, log_id, operator_response="N",
                        response_reason="手工否决平仓",
                        order_status="NOT_SUBMITTED")
            _record_order(db_path, {
                "datetime": ts, "symbol": sym, "direction": direction,
                "action": "CLOSE_DENIED", "limit_price": limit_price,
                "lots": lots, "filled_lots": 0, "filled_price": 0,
                "status": "CLOSE_DENIED", "signal_score": score,
                "reason": reason,
            })
            _record_denied_position(
                sym, signal.get("contract", sym),
                direction, lots,
                reason, positions.get(sym, {}).get("entry_price", 0))
            return {"status": "CLOSE_DENIED"}
        else:
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
        # 超时：只有紧急平仓自动执行，其余全部放弃
        if is_urgent_close:
            print(f"\n  ⚠ 60s无响应，自动执行平仓（{reason}）")
            _update_log(db_path, log_id, operator_response="AUTO_TIMEOUT",
                        response_reason="60秒超时自动执行平仓")
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
        api = tq_api
        if api is None:
            print(f"  下单失败: 无TQ连接，请重启executor")
            _update_log(db_path, log_id, order_status="ERROR",
                        response_reason="无TQ连接")
            return {"status": "ERROR", "reason": "NO_TQ"}

        # 合约代码必须由monitor提供（已按OI选好），不自行猜测
        contract = signal.get("contract", "")
        if not contract:
            print(f"  下单失败: 信号缺少contract字段，请重启monitor")
            _update_log(db_path, log_id, order_status="ERROR",
                        response_reason="信号缺少contract字段")
            return {"status": "ERROR", "reason": "NO_CONTRACT"}

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
                # 平仓未成交：自动以激进价重新下单
                if is_urgent_close:
                    quote = api.get_quote(contract)
                    api.wait_update()
                    if direction == "LONG":
                        aggr_price = float(quote.bid_price1) - 2.0
                    else:
                        aggr_price = float(quote.ask_price1) + 2.0
                    print(f"  限价未成交，自动以激进价 {aggr_price:.1f} 重新下单")
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
                        print(f"  ⚠ 激进价也未成交！请手动处理持仓")
            elif filled == lots:
                status = "FILLED"
                print(f"  全部成交 {filled}手 @ {trade_price:.1f}")
            else:
                status = "PARTIAL"
                print(f"  部分成交 {filled}/{lots}手 @ {trade_price:.1f}")

        finally:
            pass

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

def _check_locked_positions(db_path: str, dry_run: bool, tq_api=None):
    """启动时检查是否有前日锁仓持仓需要平仓。跟TQ实盘对账验证。"""
    try:
        db = DBManager(db_path)
        locks = db.query_df(
            "SELECT * FROM order_log WHERE action='LOCK' AND status='FILLED' "
            "AND lock_resolved IS NULL "
            "AND datetime >= date('now', '-2 day') "
            "ORDER BY datetime"
        )
        if locks is None or locks.empty:
            return

        # 用TQ实盘持仓验证锁仓是否还存在
        tq_positions = {}
        if tq_api is not None:
            try:
                import time as _time
                all_pos = tq_api.get_position()
                tq_api.wait_update(deadline=_time.time() + 3)
                for symbol, pos in all_pos.items():
                    vol_long = int(getattr(pos, "volume_long", 0) or 0)
                    vol_short = int(getattr(pos, "volume_short", 0) or 0)
                    if vol_long > 0 or vol_short > 0:
                        tq_positions[symbol] = {"long": vol_long, "short": vol_short}
            except Exception as e:
                print(f"  [WARN] TQ持仓查询失败，仅用DB记录判断: {e}")

        confirmed_locks = []
        for _, row in locks.iterrows():
            sym = row.get("symbol", "")
            dt = row.get("datetime", "")

            # 在TQ持仓中查找对应品种（order_log存的是 "IC"，TQ是 "CFFEX.IC2604"）
            if tq_positions:
                has_lock = False
                for tq_sym, vol in tq_positions.items():
                    if sym in tq_sym and vol["long"] > 0 and vol["short"] > 0:
                        has_lock = True
                        break
                if not has_lock:
                    # TQ没有锁仓 → 已处理，标记resolved
                    _mark_lock_resolved(db_path, dt, sym, "TQ无锁仓持仓")
                    continue

            confirmed_locks.append(row)

        if not confirmed_locks:
            return

        print(f"\n ⚠ 发现锁仓持仓，需要平仓（平昨仓，手续费低）：")
        for row in confirmed_locks:
            sym = row.get("symbol", "")
            direction = row.get("direction", "")
            lots_val = int(row.get("filled_lots", 0))
            price = float(row.get("filled_price", 0))
            orig_dir = "多头" if direction == "SHORT" else "空头"
            lock_dir = "空头" if direction == "SHORT" else "多头"
            src = "TQ确认" if tq_positions else "DB记录"
            print(f"   {sym}: 原{orig_dir}{lots_val}手 + 锁{lock_dir}{lots_val}手"
                  f" @ {price:.1f} → 建议双向平仓 ({src})")

        if not dry_run:
            resp = _confirm(
                f" 需要现在平仓吗？（开盘后执行）[y/n]", 30.0)
            if resp == "y":
                print(f"  → 请在盘中手动平仓（双向都用CLOSE，平昨仓手续费低）")
                for row in confirmed_locks:
                    _mark_lock_resolved(db_path, row.get("datetime", ""),
                                        row.get("symbol", ""), "操作者确认平仓")
            else:
                print(f"  → 稍后处理")
        print()
    except Exception as e:
        print(f"  [WARN] 锁仓检查失败: {e}")


def _mark_lock_resolved(db_path: str, dt: str, symbol: str, reason: str):
    """标记LOCK记录为已处理，下次启动不再提示。"""
    try:
        conn = _open_db_conn(db_path)
        conn.execute(
            "UPDATE order_log SET lock_resolved = ? "
            "WHERE datetime = ? AND symbol = ? AND action = 'LOCK'",
            (reason, dt, symbol),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _check_denied_positions():
    """启动时检查 denied_positions.json 中是否有前日否决的平仓持仓。"""
    if not os.path.exists(_DENIED_FILE):
        return
    try:
        with open(_DENIED_FILE, "r") as f:
            records = json.load(f)
    except Exception:
        return
    if not records:
        return

    today = datetime.now().strftime("%Y%m%d")
    old_records = [r for r in records if r.get("date", "") != today]
    today_records = [r for r in records if r.get("date", "") == today]

    if old_records:
        print(f"\n{'═' * W}")
        print(f" ⚠ 前日否决的平仓持仓")
        print(f"{'═' * W}")
        for r in old_records:
            d = r.get("date", "")
            d_fmt = f"{d[4:6]}/{d[6:8]}" if len(d) == 8 else d
            d_cn = "多头" if r.get("direction") == "LONG" else "空头"
            print(f" {r.get('symbol','')} {d_cn} {r.get('lots',0)}手"
                  f" | 否决时间: {d_fmt} {r.get('deny_time','')}")
            print(f" 原平仓原因: {r.get('deny_reason','')}"
                  f" | 入场价: {r.get('entry_price',0):.1f}")
        print(f" 请确认此持仓是否已手动处理！")
        print(f"{'═' * W}\n")

    # 只保留今天的记录
    if today_records:
        with open(_DENIED_FILE, "w") as f:
            json.dump(today_records, f, indent=2, ensure_ascii=False)
    else:
        os.remove(_DENIED_FILE)


# ---------------------------------------------------------------------------
# 持仓对账
# ---------------------------------------------------------------------------

_POSITIONS_FILE = os.path.join(_TMP_DIR, "futures_positions.json")
_last_reconcile_msg = ""  # 去重：上次打印的消息


def _reconcile_positions(positions: dict):
    """读取 monitor 写出的期货持仓，与 executor 内部 positions 对账。"""
    global _last_reconcile_msg
    if not os.path.exists(_POSITIONS_FILE):
        return
    try:
        with open(_POSITIONS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        return

    ts = data.get("timestamp", "")
    tq_positions = data.get("positions", {})

    # 检查数据新鲜度：超过 10 分钟的数据不用
    try:
        ts_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - ts_dt).total_seconds() > 600:
            return
    except Exception:
        return

    # 构建本次对账摘要（用于去重打印）
    msgs = []

    # 对账：TQ 有持仓的品种
    for sym, info in tq_positions.items():
        net_tq = info.get("long", 0) - info.get("short", 0)
        if sym in positions:
            d = 1 if positions[sym]["direction"] == "LONG" else -1
            net_exec = positions[sym]["lots"] * d
            if net_tq != net_exec:
                msgs.append(f"不一致 {sym}: TQ={net_tq} exec={net_exec}")
                if net_tq == 0:
                    del positions[sym]
                else:
                    positions[sym]["lots"] = abs(net_tq)
                    positions[sym]["direction"] = "LONG" if net_tq > 0 else "SHORT"
        elif net_tq != 0:
            msgs.append(f"TQ有 {sym}: 净{net_tq}手 executor无记录")

    # 反向检查：executor 有记录但 TQ 没有
    for sym in list(positions.keys()):
        if sym not in tq_positions or (
                tq_positions[sym].get("long", 0) == 0
                and tq_positions[sym].get("short", 0) == 0):
            # 安全检查：如果持仓是本次会话成交的，且快照比成交更旧，不删
            # （monitor 快照可能还没更新到 executor 刚开的仓）
            entry_t = positions[sym].get("entry_time", "")
            if entry_t and entry_t != "restored" and ts:
                try:
                    snap_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    # entry_time 格式是 "HH:MM"
                    now = datetime.now()
                    entry_dt = now.replace(
                        hour=int(entry_t[:2]), minute=int(entry_t[3:5]),
                        second=0, microsecond=0)
                    if snap_dt < entry_dt:
                        msgs.append(f"exec有 {sym} TQ快照较旧，保留")
                        continue
                except Exception:
                    pass
            msgs.append(f"exec有 {sym} TQ无 → 清除")
            del positions[sym]

    # 只在状态变化时打印
    msg_key = "|".join(sorted(msgs))
    if msgs and msg_key != _last_reconcile_msg:
        for m in msgs:
            print(f" ⚠ 持仓对账: {m}")
    _last_reconcile_msg = msg_key


# ---------------------------------------------------------------------------
# 持仓恢复（重启后从 order_log 推断）
# ---------------------------------------------------------------------------

def _restore_positions_from_log(db_path: str, positions: dict):
    """从当天 order_log 推断应有持仓，用于 executor 重启后恢复。

    逻辑：当天 OPEN(FILLED) - CLOSE/LOCK(FILLED) = 净持仓。
    恢复后再由 _reconcile_positions 用 TQ 实盘数据校正。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        db = DBManager(db_path)
        rows = db.query_df(
            "SELECT symbol, direction, action, filled_lots, filled_price "
            "FROM order_log "
            "WHERE datetime >= ? AND status = 'FILLED'",
            params=(today,),
        )
    except Exception as e:
        print(f"  [WARN] 持仓恢复查询失败: {e}")
        return

    if rows is None or rows.empty:
        return

    # 按品种汇总：先累计所有OPEN，再减去所有CLOSE/LOCK
    # 两遍扫描，避免因timestamp时区不一致导致先减后加的错误
    net: dict = {}  # sym -> {"long": lots, "short": lots, "last_price": float}

    # 第1遍：只处理OPEN
    for _, r in rows.iterrows():
        sym = str(r["symbol"])
        d = str(r["direction"])
        act = str(r["action"])
        lots = int(r.get("filled_lots", 0) or 0)
        price = float(r.get("filled_price", 0) or 0)
        if lots <= 0 or act != "OPEN":
            continue
        if sym not in net:
            net[sym] = {"long": 0, "short": 0, "last_price": 0}
        if d == "LONG":
            net[sym]["long"] += lots
        else:
            net[sym]["short"] += lots
        net[sym]["last_price"] = price

    # 第2遍：处理CLOSE/LOCK（减去对应方向）
    for _, r in rows.iterrows():
        sym = str(r["symbol"])
        d = str(r["direction"])
        act = str(r["action"])
        lots = int(r.get("filled_lots", 0) or 0)
        if lots <= 0 or act not in ("CLOSE", "LOCK", "CLOSE_DENIED"):
            continue
        if sym not in net:
            continue
        if d == "LONG":
            net[sym]["long"] = max(0, net[sym]["long"] - lots)
        else:
            net[sym]["short"] = max(0, net[sym]["short"] - lots)

    # 写入 positions
    restored = []
    for sym, info in net.items():
        net_long = info["long"] - info["short"]
        if net_long == 0:
            continue
        direction = "LONG" if net_long > 0 else "SHORT"
        lots = abs(net_long)
        positions[sym] = {
            "direction": direction,
            "lots": lots,
            "entry_price": info["last_price"],
            "entry_time": "restored",
        }
        d_cn = "多" if direction == "LONG" else "空"
        restored.append(f"{sym} {d_cn}{lots}手@{info['last_price']:.0f}")

    if restored:
        print(f" 从order_log恢复持仓: {', '.join(restored)}")


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

    # 从 order_log 恢复当天持仓（重启后内存清空的恢复手段）
    _restore_positions_from_log(db_path, positions)

    # 启动时用 TQ 实盘持仓对账（覆盖 order_log 推断的结果）
    _reconcile_positions(positions)

    # 建立持久TQ连接 + 预热合约信息（非dry-run模式）
    tq_client = None
    tq_api = None
    if not args.dry_run:
        try:
            from data.sources.tq_client import TqClient
            creds = {
                "auth_account": os.getenv("TQ_ACCOUNT", ""),
                "auth_password": os.getenv("TQ_PASSWORD", ""),
                "broker_id": os.getenv("TQ_BROKER", ""),
                "account_id": os.getenv("TQ_ACCOUNT_ID", ""),
                "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
            }
            tq_client = TqClient(**creds)
            tq_client.connect()
            tq_api = tq_client._api
            # 预热：订阅所有品种的主力合约，使合约信息进入本地缓存
            # 下单时 insert_order 内部 _ensure_symbol 会直接命中缓存，零延迟
            from utils.cffex_calendar import active_im_months
            import time as _time
            today_str = datetime.now().strftime("%Y%m%d")
            _active = active_im_months(today_str)
            _warmup_contracts = []
            for _sym in CFFEX_INDEX_FUTURES:
                for _m in _active:
                    _c = f"CFFEX.{_sym}{_m}"
                    try:
                        tq_api.get_quote(_c)
                        _warmup_contracts.append(_c)
                    except Exception:
                        pass
            if _warmup_contracts:
                tq_api.wait_update(deadline=_time.time() + 5)
            print(f" TQ连接已建立（持久模式，预热{len(_warmup_contracts)}个合约）")
        except Exception as e:
            print(f" [ERROR] TQ连接失败: {e}")
            print(f" 实盘模式需要TQ连接，请检查网络和账户配置")
            return

    # 检查昨日锁仓持仓 + 否决平仓提醒（TQ连接后，可跟实盘对账）
    _check_locked_positions(db_path, args.dry_run, tq_api=tq_api)
    _check_denied_positions()

    daily_orders = 0
    daily_loss = 0.0
    signals_received = 0
    signals_executed = 0
    last_reconcile = 0.0

    print(f" 等待信号...\n")

    try:
        while True:
            # 定期对账（每 60 秒）
            now_ts = time.time()
            if now_ts - last_reconcile > 60:
                _reconcile_positions(positions)
                last_reconcile = now_ts

            pending = _read_signals()
            for signal in pending:
                signals_received += 1
                result = _execute_order(
                    signal, args.dry_run, db_path,
                    daily_orders, daily_loss, account,
                    positions, tq_api=tq_api,
                )
                if result.get("status") in ("FILLED", "PARTIAL"):
                    daily_orders += 1
                    signals_executed += 1
                    pnl = result.get("pnl_yuan", 0)
                    if pnl < 0:
                        daily_loss += abs(pnl)

            if pending:
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
    finally:
        if tq_client is not None:
            try:
                tq_client.disconnect()
                print(f"  TQ连接已断开")
            except Exception:
                pass


if __name__ == "__main__":
    main()
