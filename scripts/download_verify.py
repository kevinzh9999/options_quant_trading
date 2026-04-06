#!/usr/bin/env python3
"""
download_verify.py
------------------
校验下载后的数据完整性，输出汇总报告。
"""

from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRADING_DB = ROOT / "data" / "storage" / "trading.db"
TICK_DB = ROOT / "data" / "storage" / "tick_data.db"


def verify_trading_db():
    print("=" * 90)
    print("  trading.db 校验")
    print("=" * 90)

    conn = sqlite3.connect(str(TRADING_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")

    # integrity
    r = conn.execute("PRAGMA integrity_check").fetchone()
    print(f"\n  integrity_check: {r[0]}")

    # ── index_min ──
    print(f"\n{'─'*90}")
    print("  index_min（现货指数分钟线）")
    print(f"{'─'*90}")
    print(f"  {'品种':<10s} {'周期':>4s} {'行数':>12s} {'天数':>6s} {'最早':>20s} {'最晚':>20s} {'状态'}")
    rows = conn.execute("""
        SELECT symbol, period, COUNT(*) as cnt,
               COUNT(DISTINCT substr(datetime,1,10)) as days,
               MIN(datetime) as mn, MAX(datetime) as mx
        FROM index_min GROUP BY symbol, period ORDER BY symbol, period
    """).fetchall()
    for r in rows:
        period_label = f"{r[1]//60}m"
        # 校验：最新日期应到 2026-04-03
        status = "✓" if "2026-04-03" in str(r[5]) else "⚠ 不到04-03"
        print(f"  {r[0]:<10s} {period_label:>4s} {r[2]:>12,} {r[3]:>6d} {r[4]:>20s} {r[5]:>20s} {status}")

    # ── futures_min ──
    print(f"\n{'─'*90}")
    print("  futures_min（期货主连分钟线）")
    print(f"{'─'*90}")
    print(f"  {'品种':<10s} {'周期':>4s} {'行数':>12s} {'天数':>6s} {'最早':>20s} {'最晚':>20s} {'状态'}")
    rows = conn.execute("""
        SELECT symbol, period, COUNT(*) as cnt,
               COUNT(DISTINCT substr(datetime,1,10)) as days,
               MIN(datetime) as mn, MAX(datetime) as mx
        FROM futures_min GROUP BY symbol, period ORDER BY symbol, period
    """).fetchall()
    for r in rows:
        period_label = f"{r[1]//60}m"
        status = "✓" if "2026-04-03" in str(r[5]) else "⚠ 不到04-03"
        print(f"  {r[0]:<10s} {period_label:>4s} {r[2]:>12,} {r[3]:>6d} {r[4]:>20s} {r[5]:>20s} {status}")

    # ── options_min ──
    print(f"\n{'─'*90}")
    print("  options_min（期权5分钟线）")
    print(f"{'─'*90}")
    # 按品种前缀汇总
    rows = conn.execute("""
        SELECT
            CASE
                WHEN ts_code LIKE 'MO%' THEN 'MO'
                WHEN ts_code LIKE 'IO%' THEN 'IO'
                WHEN ts_code LIKE 'HO%' THEN 'HO'
                ELSE 'OTHER'
            END as product,
            COUNT(*) as cnt,
            COUNT(DISTINCT ts_code) as n_contracts,
            COUNT(DISTINCT substr(datetime,1,10)) as days,
            MIN(datetime) as mn, MAX(datetime) as mx
        FROM options_min GROUP BY product ORDER BY product
    """).fetchall()
    if rows:
        print(f"  {'品种':<8s} {'行数':>12s} {'合约数':>8s} {'天数':>6s} {'最早':>20s} {'最晚':>20s}")
        for r in rows:
            print(f"  {r[0]:<8s} {r[1]:>12,} {r[2]:>8,} {r[3]:>6d} {r[4]:>20s} {r[5]:>20s}")
    else:
        print("  (空)")

    # ── 去重检查 ──
    print(f"\n{'─'*90}")
    print("  去重检查")
    print(f"{'─'*90}")
    for table, pk in [("index_min", "symbol, datetime, period"),
                       ("futures_min", "symbol, datetime, period"),
                       ("options_min", "ts_code, datetime, period")]:
        try:
            dup = conn.execute(f"""
                SELECT {pk}, COUNT(*) as c FROM {table}
                GROUP BY {pk} HAVING c > 1 LIMIT 1
            """).fetchone()
            status = "✓ 无重复" if dup is None else f"⚠ 有重复! 例: {dup}"
            print(f"  {table:<20s} {status}")
        except Exception:
            print(f"  {table:<20s} (表不存在)")

    # ── 数据连续性抽查（000852 1m） ──
    print(f"\n{'─'*90}")
    print("  数据连续性抽查（000852 1m，每月bar数）")
    print(f"{'─'*90}")
    rows = conn.execute("""
        SELECT substr(datetime,1,7) as month, COUNT(*) as cnt
        FROM index_min WHERE symbol='000852' AND period=60
        GROUP BY month ORDER BY month
    """).fetchall()
    if rows:
        for r in rows[-12:]:  # 最近12个月
            expected = 240 * 22  # 大约
            pct = r[1] / expected * 100
            flag = "" if pct > 80 else " ⚠ 偏少"
            print(f"  {r[0]}  {r[1]:>6,} 根{flag}")

    # ── DB大小 ──
    db_size = TRADING_DB.stat().st_size / (1024**2)
    print(f"\n  trading.db 文件大小: {db_size:.1f} MB")

    conn.close()


def verify_tick_db():
    print(f"\n\n{'='*90}")
    print("  tick_data.db 校验")
    print("=" * 90)

    if not TICK_DB.exists():
        print("  (文件不存在)")
        return

    conn = sqlite3.connect(str(TICK_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")

    r = conn.execute("PRAGMA integrity_check").fetchone()
    print(f"\n  integrity_check: {r[0]}")

    rows = conn.execute("""
        SELECT symbol, COUNT(*) as cnt,
               COUNT(DISTINCT substr(datetime,1,10)) as days,
               MIN(datetime) as mn, MAX(datetime) as mx
        FROM futures_tick GROUP BY symbol ORDER BY symbol
    """).fetchall()

    if rows:
        print(f"\n  {'品种':<8s} {'Tick数':>14s} {'天数':>6s} {'最早':>12s} {'最晚':>12s}")
        for r in rows:
            print(f"  {r[0]:<8s} {r[1]:>14,} {r[2]:>6d} {r[3][:10]:>12s} {r[4][:10]:>12s}")

        total = conn.execute("SELECT COUNT(*) FROM futures_tick").fetchone()[0]
        db_size = TICK_DB.stat().st_size / (1024**3)
        print(f"\n  总计: {total:,} ticks")
        print(f"  tick_data.db 文件大小: {db_size:.2f} GB")

        # 去重检查
        dup = conn.execute("""
            SELECT symbol, datetime, COUNT(*) as c FROM futures_tick
            GROUP BY symbol, datetime HAVING c > 1 LIMIT 1
        """).fetchone()
        print(f"  去重检查: {'✓ 无重复' if dup is None else f'⚠ 有重复!'}")

        # 每日tick数抽查
        print(f"\n  每日tick数抽查（最近5天）:")
        daily = conn.execute("""
            SELECT substr(datetime,1,10) as dt, symbol, COUNT(*) as cnt
            FROM futures_tick
            GROUP BY dt, symbol ORDER BY dt DESC, symbol LIMIT 10
        """).fetchall()
        for r in daily:
            print(f"    {r[0]}  {r[1]}  {r[2]:>8,}")
    else:
        print("  (空)")

    conn.close()


def print_grand_summary():
    print(f"\n\n{'='*90}")
    print("  ═══ 数据库内容大汇总 ═══")
    print("=" * 90)

    conn = sqlite3.connect(str(TRADING_DB), timeout=30)

    # 全部表统计
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence' ORDER BY name"
    ).fetchall()

    print(f"\n  trading.db: {len(tables)} 张表")
    print(f"  {'表名':<35s} {'行数':>12s} {'最早':>12s} {'最晚':>12s}")
    print(f"  {'─'*75}")

    for (t,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info([{t}])").fetchall()]

        date_col = None
        for c in ['trade_date', 'datetime', 'signal_time', 'snapshot_time',
                   'briefing_date', 'receive_time']:
            if c in cols:
                date_col = c
                break

        if count > 0 and date_col:
            mn, mx = conn.execute(
                f"SELECT MIN([{date_col}]), MAX([{date_col}]) FROM [{t}]"
            ).fetchone()
            mn_s = str(mn)[:10] if mn else "—"
            mx_s = str(mx)[:10] if mx else "—"
        else:
            mn_s = mx_s = "—"

        print(f"  {t:<35s} {count:>12,} {mn_s:>12s} {mx_s:>12s}")

    db_size = TRADING_DB.stat().st_size / (1024**2)
    print(f"\n  trading.db 总大小: {db_size:.0f} MB")

    conn.close()

    # tick db
    if TICK_DB.exists():
        conn2 = sqlite3.connect(str(TICK_DB), timeout=30)
        total = conn2.execute("SELECT COUNT(*) FROM futures_tick").fetchone()[0]
        conn2.close()
        tick_size = TICK_DB.stat().st_size / (1024**3)
        print(f"  tick_data.db: {total:,} ticks, {tick_size:.2f} GB")


if __name__ == "__main__":
    verify_trading_db()
    verify_tick_db()
    print_grand_summary()
