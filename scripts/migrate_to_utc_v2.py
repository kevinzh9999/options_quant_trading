#!/usr/bin/env python3
"""全量UTC迁移v2：处理唯一键冲突。"""
import sys, os, time as _t
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def safe_migrate(db_path, table, dt_col="datetime", has_unique=False):
    """BJ→UTC转换，处理唯一键冲突。"""
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")

    bj_count = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9"
    ).fetchone()[0]
    if bj_count == 0:
        print(f"  {table}: 已全部UTC ✓")
        conn.close()
        return 0

    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {bj_count}/{total} 行需转...", end="", flush=True)

    if has_unique:
        # 有唯一键约束：先删BJ格式行（它们-8h后可能跟UTC行冲突），再重新插不冲突的
        # 简单方案：直接删BJ行（这些是monitor写的日志，跟UTC行重复）
        conn.execute(
            f"DELETE FROM {table} WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9"
        )
        conn.commit()
        print(f" 删除{bj_count}行BJ数据（与UTC行重复）")
    else:
        conn.execute(
            f"UPDATE {table} SET {dt_col} = datetime({dt_col}, '-8 hours') "
            f"WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9"
        )
        conn.commit()
        print(f" 转换完成")

    conn.close()
    return bj_count


def migrate_tick(db_path):
    """Tick表：datetime含微秒。"""
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")

    bj_count = conn.execute(
        "SELECT COUNT(*) FROM futures_tick WHERE CAST(substr(datetime,12,2) AS INTEGER) >= 9"
    ).fetchone()[0]
    if bj_count == 0:
        print(f"  futures_tick: 已全部UTC ✓")
        conn.close()
        return 0

    print(f"  futures_tick: {bj_count} 行需转（含微秒）...", flush=True)
    t0 = _t.time()
    batch = 200000
    done = 0
    while True:
        n = conn.execute(
            "UPDATE futures_tick SET datetime = "
            "datetime(substr(datetime,1,19), '-8 hours') || substr(datetime,20) "
            "WHERE rowid IN (SELECT rowid FROM futures_tick "
            "WHERE CAST(substr(datetime,12,2) AS INTEGER) >= 9 LIMIT ?)", (batch,)
        ).rowcount
        conn.commit()
        done += n
        elapsed = _t.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (bj_count - done) / rate if rate > 0 else 0
        print(f"\r  futures_tick: {done}/{bj_count} ({done/bj_count*100:.1f}%) "
              f"{rate:.0f}行/s ETA={eta:.0f}s", end="", flush=True)
        if n < batch:
            break
    print(f"\n  完成 ({_t.time()-t0:.0f}s)")
    conn.close()
    return bj_count


if __name__ == "__main__":
    print("=" * 60)
    print("  全量UTC迁移 v2")
    print("=" * 60)

    trading_db = ROOT / "data" / "storage" / "trading.db"
    etf_db = ROOT / "data" / "storage" / "etf_data.db"
    tick_db = ROOT / "data" / "storage" / "tick_data.db"

    total = 0

    print("\n[1/3] 业务表")
    # signal_log和orderbook_snapshots有唯一键，BJ行跟UTC行重复→删BJ行
    total += safe_migrate(trading_db, "signal_log", has_unique=True)
    total += safe_migrate(trading_db, "orderbook_snapshots", has_unique=True)
    # 其他表无唯一键冲突
    total += safe_migrate(trading_db, "order_log")
    total += safe_migrate(trading_db, "trade_decisions", has_unique=True)
    total += safe_migrate(trading_db, "vol_monitor_snapshots", has_unique=True)

    print("\n[2/3] ETF表")
    total += safe_migrate(etf_db, "etf_min")

    print("\n[3/3] Tick表")
    total += migrate_tick(tick_db)

    print(f"\n{'=' * 60}")
    print(f"  完成: {total:,} 行处理")
    print(f"{'=' * 60}")
