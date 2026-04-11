#!/usr/bin/env python3
"""数据库完整性检查：去重、时间戳格式、数据连续性。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DBS = {
    "trading": ROOT / "data" / "storage" / "trading.db",
    "options": ROOT / "data" / "storage" / "options_data.db",
    "tick":    ROOT / "data" / "storage" / "tick_data.db",
    "etf":    ROOT / "data" / "storage" / "etf_data.db",
}

issues = []


def check_table(conn, db_name, table, dt_col="datetime", sym_col="symbol",
                period_col=None, expected_per_day=None, date_range=("2025-05-16", "2026-04-10")):
    """检查单表：去重、时间戳格式、每日bar数。"""
    print(f"\n  [{db_name}] {table}")

    # 1. 总行数
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"    总行数: {n:,}")
    if n == 0:
        print(f"    ⚠ 空表")
        issues.append(f"{db_name}.{table}: 空表")
        return

    # 2. 时间范围
    r = conn.execute(f"SELECT MIN({dt_col}), MAX({dt_col}) FROM {table}").fetchone()
    print(f"    时间范围: {r[0]} ~ {r[1]}")

    # 3. 重复检查
    if period_col:
        dup_sql = (f"SELECT {sym_col}, {dt_col}, {period_col}, COUNT(*) as c "
                   f"FROM {table} GROUP BY {sym_col}, {dt_col}, {period_col} HAVING c > 1")
    else:
        dup_sql = (f"SELECT {sym_col}, {dt_col}, COUNT(*) as c "
                   f"FROM {table} GROUP BY {sym_col}, {dt_col} HAVING c > 1")
    dups = conn.execute(dup_sql).fetchall()
    if dups:
        print(f"    ✗ 重复: {len(dups)} 组")
        for d in dups[:3]:
            print(f"      {dict(d)}")
        issues.append(f"{db_name}.{table}: {len(dups)}组重复")
    else:
        print(f"    ✓ 无重复")

    # 4. 时间戳格式检查（UTC应该是 HH:00-06:59，BJ格式是09:00-15:00）
    if dt_col == "datetime":
        bj_count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 9 "
            f"AND CAST(substr({dt_col},12,2) AS INTEGER) <= 15"
        ).fetchone()[0]
        utc_count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE CAST(substr({dt_col},12,2) AS INTEGER) >= 0 "
            f"AND CAST(substr({dt_col},12,2) AS INTEGER) <= 7"
        ).fetchone()[0]
        if bj_count > 0 and utc_count > 0:
            pct_bj = bj_count / n * 100
            pct_utc = utc_count / n * 100
            print(f"    ⚠ 混合时间: UTC={utc_count}({pct_utc:.0f}%) BJ={bj_count}({pct_bj:.0f}%)")
            issues.append(f"{db_name}.{table}: UTC/BJ混合时间")
        elif bj_count > utc_count:
            print(f"    时间格式: BJ时间")
        else:
            print(f"    时间格式: UTC ✓")

    # 5. 每日bar数检查（仅5m K线表）
    if expected_per_day and period_col and sym_col:
        # 检查2025-05-16之后的数据
        bad_days = conn.execute(
            f"SELECT substr({dt_col},1,10) as d, {sym_col}, COUNT(*) as c "
            f"FROM {table} WHERE {period_col}=300 AND d >= '{date_range[0]}' AND d <= '{date_range[1]}' "
            f"GROUP BY d, {sym_col} HAVING c != {expected_per_day}"
        ).fetchall()
        if bad_days:
            print(f"    ✗ bar数异常天: {len(bad_days)} (期望{expected_per_day}根/天)")
            for d in bad_days[:5]:
                print(f"      {dict(d)}")
            issues.append(f"{db_name}.{table}: {len(bad_days)}天bar数异常")
        else:
            print(f"    ✓ 每日{expected_per_day}根一致")


if __name__ == "__main__":
    print("=" * 60)
    print("  数据库完整性检查")
    print("=" * 60)

    # trading.db
    conn = sqlite3.connect(DBS["trading"])
    conn.row_factory = sqlite3.Row
    check_table(conn, "trading", "futures_min", period_col="period", expected_per_day=48)
    check_table(conn, "trading", "index_min", period_col="period", expected_per_day=48)
    check_table(conn, "trading", "futures_daily", dt_col="trade_date", sym_col="ts_code", period_col=None)
    check_table(conn, "trading", "index_daily", dt_col="trade_date", sym_col="ts_code", period_col=None)
    check_table(conn, "trading", "signal_log")
    check_table(conn, "trading", "shadow_trades", dt_col="trade_date")
    conn.close()

    # options_data.db
    conn = sqlite3.connect(DBS["options"])
    conn.row_factory = sqlite3.Row
    check_table(conn, "options", "options_daily", dt_col="trade_date", sym_col="ts_code")
    check_table(conn, "options", "options_min", sym_col="ts_code", period_col="period")
    conn.close()

    # tick_data.db
    conn = sqlite3.connect(DBS["tick"])
    conn.row_factory = sqlite3.Row
    check_table(conn, "tick", "futures_tick")
    conn.close()

    # etf_data.db
    conn = sqlite3.connect(DBS["etf"])
    conn.row_factory = sqlite3.Row
    check_table(conn, "etf", "etf_min", period_col="period", expected_per_day=48,
                date_range=("2025-05-16", "2026-04-10"))
    conn.close()

    # 汇总
    print(f"\n{'=' * 60}")
    if issues:
        print(f"  发现 {len(issues)} 个问题:")
        for i, iss in enumerate(issues):
            print(f"    {i+1}. {iss}")
    else:
        print("  ✓ 全部通过")
    print(f"{'=' * 60}")
