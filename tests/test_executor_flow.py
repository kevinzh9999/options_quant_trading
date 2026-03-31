#!/usr/bin/env python3
"""测试order_executor完整流程：信号JSON→手数计算→TqSim下单→锁仓→SQLite WAL。

用法：
    python tests/test_executor_flow.py
"""
import json
import os
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(Path(ROOT) / ".env")

from config.config_loader import ConfigLoader

DB_PATH = ConfigLoader().get_db_path()
passed = 0
failed = 0


def _result(ok: bool, msg: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ 通过{('  ' + msg) if msg else ''}")
    else:
        failed += 1
        print(f"  ❌ 失败{('  ' + msg) if msg else ''}")


# ── TEST 1: 手数计算逻辑 ──────────────────────────────────────────────
print("[TEST 1] 手数计算（monitor → executor ÷2）")
# monitor._calc_suggested_lots: risk / (price * mult * 0.005)
# executor: suggested_lots // 2
from strategies.intraday.monitor import IntradayMonitor

m = IntradayMonitor.__new__(IntradayMonitor)
m._account_equity = 6_400_000

# IM: mult=200
lots_im = m._calc_suggested_lots(7600, "IM")
exec_im = max(1, lots_im // 2)
print(f"  IM@7600: monitor={lots_im}手 → executor={exec_im}手")
_result(1 <= exec_im <= 5, f"IM建议{exec_im}手")

# IH: mult=300
lots_ih = m._calc_suggested_lots(2815, "IH")
exec_ih = max(1, lots_ih // 2)
print(f"  IH@2815: monitor={lots_ih}手 → executor={exec_ih}手")
_result(2 <= exec_ih <= 5, f"IH建议{exec_ih}手（修复后用mult=300）")

# IF: mult=300
lots_if = m._calc_suggested_lots(4500, "IF")
exec_if = max(1, lots_if // 2)
print(f"  IF@4500: monitor={lots_if}手 → executor={exec_if}手")
_result(1 <= exec_if <= 5, f"IF建议{exec_if}手")

# ── TEST 2: 信号JSON解析 ──────────────────────────────────────────────
print()
print("[TEST 2] 信号JSON写入和解析")
test_signal = {
    "timestamp": "2026-03-31 10:00:00",
    "symbol": "IM", "direction": "SHORT", "action": "OPEN",
    "score": 75, "bid1": 7600.0, "ask1": 7602.0,
    "suggested_lots": 4, "limit_price": 7600.0, "reason": "test",
}
test_path = "/tmp/signal_pending_test.json"
with open(test_path, "w") as f:
    json.dump(test_signal, f)
with open(test_path) as f:
    loaded = json.load(f)
print(f"  {loaded['symbol']} {loaded['direction']} @{loaded['limit_price']}")
_result(loaded["symbol"] == "IM" and loaded["direction"] == "SHORT")
os.remove(test_path)

# 平仓信号格式
close_signal = {
    "timestamp": "2026-03-31 10:30:00",
    "symbol": "IM", "direction": "SHORT", "action": "CLOSE",
    "reason": "TRAILING_STOP", "bid1": 7580.0, "ask1": 7582.0,
    "last": 7581.0, "suggested_lots": 4,
    "limit_price": 7582.0, "pnl_pts": 19.0,
}
with open(test_path, "w") as f:
    json.dump(close_signal, f)
with open(test_path) as f:
    loaded = json.load(f)
print(f"  CLOSE信号: {loaded['action']} {loaded['reason']} PnL={loaded['pnl_pts']}")
_result(loaded["action"] == "CLOSE" and loaded["reason"] == "TRAILING_STOP")
os.remove(test_path)

# ── TEST 3: SQLite WAL模式 ────────────────────────────────────────────
print()
print("[TEST 3] SQLite WAL模式")
conn = sqlite3.connect(DB_PATH, timeout=30)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=30000")
mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
print(f"  journal_mode = {mode}")
_result(mode == "wal")
conn.close()

# ── TEST 4: order_log表读写 ───────────────────────────────────────────
print()
print("[TEST 4] order_log表读写")
from scripts.order_executor import _ensure_order_log, _record_order
_ensure_order_log(DB_PATH)
test_record = {
    "datetime": "2026-03-31T99:99:99",
    "symbol": "TEST_EXECUTOR",
    "direction": "BUY",
    "action": "OPEN",
    "limit_price": 7600.0,
    "lots": 2,
    "filled_lots": 2,
    "filled_price": 7600.0,
    "status": "TEST",
    "signal_score": 75,
    "reason": "test_executor_flow",
}
_record_order(DB_PATH, test_record)
conn = sqlite3.connect(DB_PATH, timeout=30)
row = conn.execute(
    "SELECT * FROM order_log WHERE symbol='TEST_EXECUTOR'"
).fetchone()
print(f"  写入并读回: symbol={row[1]} status={row[8]}")
_result(row is not None and row[1] == "TEST_EXECUTOR")
conn.execute("DELETE FROM order_log WHERE symbol='TEST_EXECUTOR'")
conn.commit()
conn.close()

# ── TEST 5: TqSim下单+锁仓 ──────────────────────────────────────────
print()
print("[TEST 5] TqSim开仓+锁仓")
TQ_ACCOUNT = os.getenv("TQ_ACCOUNT")
TQ_PASSWORD = os.getenv("TQ_PASSWORD")
if not TQ_ACCOUNT or not TQ_PASSWORD:
    print("  ⚠ TQ_ACCOUNT/TQ_PASSWORD未设置，跳过TqSim测试")
else:
    from tqsdk import TqApi, TqAuth, TqSim, TqBacktest, BacktestFinished
    try:
        sim = TqSim(init_balance=6_400_000)
        api = TqApi(
            sim,
            backtest=TqBacktest(start_dt=date(2026, 3, 27), end_dt=date(2026, 3, 27)),
            auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD),
        )
        CONTRACT = "CFFEX.IM2604"
        quote = api.get_quote(CONTRACT)

        step = "wait"
        open_order = lock_order = None
        tick = 0

        while True:
            api.wait_update()
            tick += 1
            bid = float(quote.bid_price1) if quote.bid_price1 == quote.bid_price1 else 0
            ask = float(quote.ask_price1) if quote.ask_price1 == quote.ask_price1 else 0

            if step == "wait" and bid > 0 and ask > 0:
                print(f"  行情就绪 bid={bid} ask={ask}")
                open_order = api.insert_order(
                    symbol=CONTRACT, direction="SELL", offset="OPEN",
                    volume=2, limit_price=bid)
                print(f"  开仓: SELL OPEN 2手 @{bid}")
                step = "wait_fill"

            elif step == "wait_fill" and open_order.status == "FINISHED":
                filled = open_order.volume_orign - open_order.volume_left
                print(f"  成交: {filled}手")
                lock_order = api.insert_order(
                    symbol=CONTRACT, direction="BUY", offset="OPEN",
                    volume=2, limit_price=ask)
                print(f"  锁仓: BUY OPEN 2手 @{ask}")
                step = "wait_lock"

            elif step == "wait_lock" and lock_order.status == "FINISHED":
                pos = api.get_position(CONTRACT)
                print(f"  锁仓后: 多头={pos.pos_long} 空头={pos.pos_short}")
                _result(pos.pos_long >= 2 and pos.pos_short >= 2,
                        f"L={pos.pos_long} S={pos.pos_short}")
                break

            if tick > 500:
                print("  超时")
                _result(False, "500 ticks超时")
                break

    except BacktestFinished:
        print("  回测提前结束")
        _result(False, "BacktestFinished")
    except Exception as e:
        print(f"  异常: {e}")
        _result(False, str(e))
    finally:
        api.close()

# ── 汇总 ─────────────────────────────────────────────────────────────
print()
print(f"{'=' * 40}")
print(f"  {passed} passed / {failed} failed")
print(f"{'=' * 40}")
sys.exit(1 if failed > 0 else 0)
