#!/usr/bin/env python3
"""全量数据库时间戳统一为UTC。

BJ时间（hour 9-15）减8小时转为UTC（hour 1-7）。
只转换hour>=9的行（已经是UTC的不动）。
"""
import sys, os, time as _t
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def migrate_table(db_path, table, dt_col="datetime", batch_size=100000):
    """将表中BJ时间的行转为UTC（-8小时）。"""
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")

    # 统计BJ格式行数
    bj_count = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9"
    ).fetchone()[0]

    if bj_count == 0:
        print(f"  {table}.{dt_col}: 已全部UTC，跳过")
        conn.close()
        return 0

    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}.{dt_col}: {bj_count}/{total} 行需转UTC...", end="", flush=True)

    t0 = _t.time()
    # 对于小表直接UPDATE，大表分批
    if bj_count < 500000:
        conn.execute(
            f"UPDATE {table} SET {dt_col} = datetime({dt_col}, '-8 hours') "
            f"WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9"
        )
        conn.commit()
    else:
        # 大表分批更新（避免长事务锁）
        done = 0
        while True:
            n = conn.execute(
                f"UPDATE {table} SET {dt_col} = datetime({dt_col}, '-8 hours') "
                f"WHERE rowid IN (SELECT rowid FROM {table} "
                f"WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9 LIMIT {batch_size})"
            ).rowcount
            conn.commit()
            done += n
            print(f"\r  {table}.{dt_col}: {done}/{bj_count} 行已转", end="", flush=True)
            if n < batch_size:
                break

    elapsed = _t.time() - t0
    print(f"  完成 ({elapsed:.0f}s)")
    conn.close()
    return bj_count


def migrate_tick_subsecond(db_path, table="futures_tick", dt_col="datetime"):
    """Tick表特殊处理：datetime含微秒(.ffffff)，sqlite的datetime()函数不支持。"""
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")

    bj_count = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9"
    ).fetchone()[0]

    if bj_count == 0:
        print(f"  {table}: 已全部UTC，跳过")
        conn.close()
        return 0

    print(f"  {table}: {bj_count} 行需转UTC（含微秒）...", flush=True)
    t0 = _t.time()

    # tick的datetime格式: "2025-07-15 09:30:00.500000"
    # datetime()不支持微秒，用字符串拼接：取前19位减8小时，再拼微秒部分
    batch = 200000
    done = 0
    while True:
        n = conn.execute(
            f"UPDATE {table} SET {dt_col} = "
            f"datetime(substr({dt_col},1,19), '-8 hours') || substr({dt_col},20) "
            f"WHERE rowid IN (SELECT rowid FROM {table} "
            f"WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9 LIMIT {batch})"
        ).rowcount
        conn.commit()
        done += n
        elapsed = _t.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (bj_count - done) / rate if rate > 0 else 0
        print(f"\r  {table}: {done}/{bj_count} ({done/bj_count*100:.1f}%) "
              f"{rate:.0f}行/s ETA={eta:.0f}s", end="", flush=True)
        if n < batch:
            break

    print(f"\n  完成 ({_t.time()-t0:.0f}s)")
    conn.close()
    return bj_count


if __name__ == "__main__":
    print("=" * 60)
    print("  全量UTC迁移")
    print("=" * 60)

    trading_db = ROOT / "data" / "storage" / "trading.db"
    options_db = ROOT / "data" / "storage" / "options_data.db"
    tick_db = ROOT / "data" / "storage" / "tick_data.db"
    etf_db = ROOT / "data" / "storage" / "etf_data.db"

    total_converted = 0

    # 小表（秒级）
    print("\n[1/3] 业务表（trading.db）")
    for table, col in [
        ("signal_log", "datetime"),
        ("orderbook_snapshots", "datetime"),
        ("order_log", "datetime"),
        ("trade_decisions", "datetime"),
        ("vol_monitor_snapshots", "datetime"),
    ]:
        total_converted += migrate_table(trading_db, table, col)

    # ETF表
    print("\n[2/3] ETF表（etf_data.db）")
    total_converted += migrate_table(etf_db, "etf_min")

    # Tick表（最慢，1.12亿行）
    print("\n[3/3] Tick表（tick_data.db）— 预计10-20分钟")
    total_converted += migrate_tick_subsecond(tick_db)

    print(f"\n{'=' * 60}")
    print(f"  迁移完成: 共转换 {total_converted:,} 行")
    print(f"{'=' * 60}")
