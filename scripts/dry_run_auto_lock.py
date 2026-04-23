"""Dry-run test for bar_low 自动锁仓 (executor guardian + monitor sync).

Exercises both halves of the flow WITHOUT connecting to TQ:
  1. executor._check_auto_stop_lock with mock tq_api + fake positions dict
     → should detect stop breach, write tmp/auto_lock_IM.json, pop position
  2. monitor._check_auto_locks with fake IntradayMonitor state
     → should consume JSON, write shadow_trades row, pop shadow, delete file

Run:
    python scripts/dry_run_auto_lock.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config_loader import ConfigLoader

TMP = ConfigLoader().get_tmp_dir()
DB = ConfigLoader().get_db_path()


# ---------------------------------------------------------------------------
# Mock tq_api: 仅返回 get_quote 带 last_price/bid/ask
# ---------------------------------------------------------------------------
class _FakeQuote:
    def __init__(self, last, bid, ask):
        self.last_price = last
        self.bid_price1 = bid
        self.ask_price1 = ask


class _FakeTqApi:
    def __init__(self, quotes: dict):
        self._quotes = quotes

    def get_quote(self, contract):
        return self._quotes.get(contract, _FakeQuote(0, 0, 0))

    def wait_update(self, deadline=None):
        pass

    def insert_order(self, *args, **kwargs):
        raise RuntimeError("dry-run不应到达insert_order")

    def cancel_order(self, order):
        pass


def test_executor_guardian():
    print("=" * 70)
    print(" TEST 1: executor._check_auto_stop_lock (dry-run)")
    print("=" * 70)

    from scripts import order_executor as oe
    # 测试时 bypass 14:30 BJ 的停用关断（否则盘后无法跑）
    oe.AUTO_LOCK_DISABLE_AFTER_UTC = "23:59"

    # 构造: IM LONG @ 8250，stop=8225，当前价 8220（已破 stop）
    positions = {
        "IM": {
            "direction": "LONG",
            "lots": 1,
            "entry_price": 8250.0,
            "entry_time": "10:00",
            "contract": "CFFEX.IM2505",
            "stop_price": 8225.0,
            "auto_lock_sent": False,
        }
    }
    api = _FakeTqApi({
        "CFFEX.IM2505": _FakeQuote(last=8220.0, bid=8219.0, ask=8221.0),
    })

    # 清理任何残留的文件
    lock_file = os.path.join(TMP, "auto_lock_IM.json")
    if os.path.exists(lock_file):
        os.remove(lock_file)

    # 调用守护（dry_run=True 不触发真实下单）
    oe._check_auto_stop_lock(positions, api, DB, dry_run=True)

    # 断言 1: positions 已移除 IM
    assert "IM" not in positions, f"FAIL: IM 未从 positions 移除: {positions}"
    print(" ✓ positions['IM'] 已 pop")

    # 断言 2: auto_lock_IM.json 已写出
    assert os.path.exists(lock_file), "FAIL: auto_lock_IM.json 未写出"
    with open(lock_file) as f:
        info = json.load(f)
    assert info["sym"] == "IM"
    assert info["direction"] == "LONG"
    assert info["lots"] == 1
    assert info.get("dry_run") is True
    assert abs(info["exit_price"] - 8220.0) < 0.01
    assert abs(info["stop_price"] - 8225.0) < 0.01
    print(f" ✓ auto_lock_IM.json 格式正确: {info}")

    # 断言 3: 再次调用不会重复触发（positions 已空）
    oe._check_auto_stop_lock(positions, api, DB, dry_run=True)
    print(" ✓ 重入测试通过（positions 空时安全 no-op）")

    return lock_file  # 留给 test_monitor_consume 消费


def test_monitor_consume(lock_file_path: str):
    print()
    print("=" * 70)
    print(" TEST 2: monitor._check_auto_locks (consume auto_lock file)")
    print("=" * 70)

    # 文件应已存在（test 1 写的）
    assert os.path.exists(lock_file_path), (
        f"FAIL: {lock_file_path} 不存在，无法测 consume")

    # 构造最小 fake monitor
    from strategies.intraday.monitor import IntradayRecorder, IntradayMonitor

    class _FakePosMgr:
        def __init__(self):
            self.removed = []

        def remove_by_symbol(self, sym):
            self.removed.append(sym)

    class _FakeStrategy:
        def __init__(self):
            self.position_mgr = _FakePosMgr()

    # 用隔离的测试 db（避免污染实盘 trading.db）
    test_db = os.path.join(TMP, "dry_run_auto_lock_test.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    recorder = IntradayRecorder(db_path=test_db)

    # 绕过 IntradayMonitor.__init__（需要 TQ creds 等），手工组装实例
    mon = IntradayMonitor.__new__(IntradayMonitor)
    mon._tmp_dir = TMP
    mon.symbols = ["IM"]
    mon.recorder = recorder
    mon.strategy = _FakeStrategy()
    mon._shadow_positions = {
        "IM": {
            "direction": "LONG",
            "entry_price": 8250.0,
            "entry_time_bj": "10:00",
            "entry_time_utc": "02:00",
            "entry_score": 72,
            "entry_dm": 1.1,
            "entry_f": 1.0,
            "entry_t": 1.0,
            "entry_s": 1.0,
            "entry_m": 40,
            "entry_v": 15,
            "entry_q": 12,
        }
    }
    mon._shadow_closed_pnl = 0.0
    mon._shadow_closed_count = 0
    # _save_shadow_state 需要 _prompted_bars
    mon._prompted_bars = set()

    # 调用被测方法
    mon._check_auto_locks()

    # 断言 1: shadow_positions 已清除 IM
    assert "IM" not in mon._shadow_positions, (
        f"FAIL: shadow_positions 未清除: {mon._shadow_positions}")
    print(" ✓ _shadow_positions['IM'] 已 del")

    # 断言 2: position_mgr.remove_by_symbol 被调用
    assert "IM" in mon.strategy.position_mgr.removed, (
        "FAIL: position_mgr.remove_by_symbol('IM') 未调用")
    print(" ✓ position_mgr.remove_by_symbol('IM') 已调")

    # 断言 3: auto_lock_IM.json 已删除
    assert not os.path.exists(lock_file_path), (
        f"FAIL: {lock_file_path} 未删除")
    print(" ✓ auto_lock_IM.json 已删除")

    # 断言 4: shadow_trades 表写入了一行
    conn = sqlite3.connect(test_db)
    rows = conn.execute(
        "SELECT symbol, direction, entry_price, exit_price, "
        "exit_reason, pnl_pts, operator_action, is_executed "
        "FROM shadow_trades"
    ).fetchall()
    conn.close()
    assert len(rows) == 1, f"FAIL: 期望1行 shadow_trades, 实际 {len(rows)}"
    r = rows[0]
    assert r[0] == "IM"
    assert r[1] == "LONG"
    assert abs(r[2] - 8250.0) < 0.01
    assert abs(r[3] - 8220.0) < 0.01
    assert r[4] == "AUTO_STOP_LOCK"
    # PnL: LONG, exit<entry → 负
    assert r[5] < 0, f"FAIL: PnL 应为负 (LONG 止损)，实际 {r[5]}"
    assert r[6] == "AUTO"
    assert r[7] == 1
    print(f" ✓ shadow_trades 记录正确: sym={r[0]} dir={r[1]} "
          f"entry={r[2]:.0f} exit={r[3]:.0f} reason={r[4]} pnl={r[5]:+.1f}")

    # 断言 5: 重入安全 — 再次调用 shadow 已空
    mon._check_auto_locks()
    print(" ✓ 重入测试通过（文件已删，shadow 已空）")

    # 清理测试 db
    os.remove(test_db)


def test_no_trigger():
    print()
    print("=" * 70)
    print(" TEST 3: stop 未触发时 guardian no-op")
    print("=" * 70)

    from scripts import order_executor as oe
    oe.AUTO_LOCK_DISABLE_AFTER_UTC = "23:59"

    positions = {
        "IM": {
            "direction": "LONG",
            "lots": 1,
            "entry_price": 8250.0,
            "entry_time": "10:00",
            "contract": "CFFEX.IM2505",
            "stop_price": 8225.0,
            "auto_lock_sent": False,
        }
    }
    # 当前价 8240 > stop 8225，不应触发
    api = _FakeTqApi({
        "CFFEX.IM2505": _FakeQuote(last=8240.0, bid=8239.0, ask=8241.0),
    })

    lock_file = os.path.join(TMP, "auto_lock_IM.json")
    if os.path.exists(lock_file):
        os.remove(lock_file)

    oe._check_auto_stop_lock(positions, api, DB, dry_run=True)

    assert "IM" in positions, "FAIL: stop 未触达却 pop 了 IM"
    assert not positions["IM"]["auto_lock_sent"], (
        "FAIL: stop 未触达却标记 auto_lock_sent=True")
    assert not os.path.exists(lock_file), (
        "FAIL: stop 未触达却写了 auto_lock 文件")
    print(" ✓ last=8240 > stop=8225，positions 保留，无文件写出")


def test_short_direction():
    print()
    print("=" * 70)
    print(" TEST 4: SHORT 方向 stop 触发（last >= stop）")
    print("=" * 70)

    from scripts import order_executor as oe
    oe.AUTO_LOCK_DISABLE_AFTER_UTC = "23:59"

    positions = {
        "IC": {
            "direction": "SHORT",
            "lots": 1,
            "entry_price": 6000.0,
            "entry_time": "10:00",
            "contract": "CFFEX.IC2505",
            "stop_price": 6018.0,
            "auto_lock_sent": False,
        }
    }
    # 当前价 6020 > stop 6018，SHORT 止损触发
    api = _FakeTqApi({
        "CFFEX.IC2505": _FakeQuote(last=6020.0, bid=6019.0, ask=6021.0),
    })

    lock_file = os.path.join(TMP, "auto_lock_IC.json")
    if os.path.exists(lock_file):
        os.remove(lock_file)

    oe._check_auto_stop_lock(positions, api, DB, dry_run=True)

    assert "IC" not in positions, "FAIL: SHORT 止损触发但 IC 未 pop"
    assert os.path.exists(lock_file), "FAIL: SHORT 止损触发但无文件"
    with open(lock_file) as f:
        info = json.load(f)
    assert info["direction"] == "SHORT"
    print(f" ✓ SHORT 触发: exit={info['exit_price']:.0f} > stop={info['stop_price']:.0f}")

    os.remove(lock_file)


def test_auto_lock_sent_blocks_reentry():
    print()
    print("=" * 70)
    print(" TEST 5: auto_lock_sent=True 阻止重入")
    print("=" * 70)

    from scripts import order_executor as oe
    oe.AUTO_LOCK_DISABLE_AFTER_UTC = "23:59"

    positions = {
        "IM": {
            "direction": "LONG", "lots": 1, "entry_price": 8250.0,
            "entry_time": "10:00", "contract": "CFFEX.IM2505",
            "stop_price": 8225.0,
            "auto_lock_sent": True,   # 已标记
        }
    }
    api = _FakeTqApi({
        "CFFEX.IM2505": _FakeQuote(last=8200.0, bid=8199.0, ask=8201.0),
    })

    lock_file = os.path.join(TMP, "auto_lock_IM.json")
    if os.path.exists(lock_file):
        os.remove(lock_file)

    oe._check_auto_stop_lock(positions, api, DB, dry_run=True)

    assert "IM" in positions, "FAIL: auto_lock_sent=True 却被移除"
    assert not os.path.exists(lock_file), (
        "FAIL: auto_lock_sent=True 却写了文件")
    print(" ✓ auto_lock_sent=True 时守护不重入")


def test_disable_after_cutoff():
    print()
    print("=" * 70)
    print(" TEST 6: UTC > 06:30 时停用守护（monitor EOD 接管）")
    print("=" * 70)

    from scripts import order_executor as oe
    # 设置为极早时间，模拟"现在已经超过关断时刻"
    oe.AUTO_LOCK_DISABLE_AFTER_UTC = "00:00"

    positions = {
        "IM": {
            "direction": "LONG", "lots": 1, "entry_price": 8250.0,
            "entry_time": "10:00", "contract": "CFFEX.IM2505",
            "stop_price": 8225.0, "auto_lock_sent": False,
        }
    }
    api = _FakeTqApi({
        "CFFEX.IM2505": _FakeQuote(last=8200.0, bid=8199.0, ask=8201.0),
    })

    lock_file = os.path.join(TMP, "auto_lock_IM.json")
    if os.path.exists(lock_file):
        os.remove(lock_file)

    oe._check_auto_stop_lock(positions, api, DB, dry_run=True)

    assert "IM" in positions, "FAIL: 已过关断但仍触发"
    assert not positions["IM"]["auto_lock_sent"], "FAIL: 已过关断但标记 sent"
    assert not os.path.exists(lock_file), "FAIL: 已过关断但写了文件"
    print(" ✓ UTC >= AUTO_LOCK_DISABLE_AFTER_UTC 时守护 short-circuit")


def main():
    print()
    print(" bar_low 自动锁仓 — 端到端 dry-run 测试")
    print()
    lock_path = test_executor_guardian()
    test_monitor_consume(lock_path)
    test_no_trigger()
    test_short_direction()
    test_auto_lock_sent_blocks_reentry()
    test_disable_after_cutoff()
    print()
    print("=" * 70)
    print(" 全部测试通过 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
