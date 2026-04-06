#!/usr/bin/env python3
"""
download_tick_pro.py
--------------------
用 TQ 专业版 get_tick_data_series 批量下载期货主连 Tick 数据。

数据写入单独的 tick_data.db（与 trading.db 分离，避免主库膨胀）。

用法:
    # 全量下载（2024-01-01 ~ 今天，4品种）
    python scripts/download_tick_pro.py

    # 指定日期范围
    python scripts/download_tick_pro.py --start 20240101 --end 20260403

    # 指定品种
    python scripts/download_tick_pro.py --symbols IM,IC

    # 增量更新（从DB已有数据末尾继续）
    python scripts/download_tick_pro.py --update

    # 查看当前数据统计
    python scripts/download_tick_pro.py --stats

    # 测试模式（只下1天看数据量）
    python scripts/download_tick_pro.py --test

注意:
  - 按天请求，每天间隔 2 秒，避免触发 TQ 限流
  - INSERT OR IGNORE 去重，可安全重复运行
  - executemany 批量写入（5万行一批），比逐行快 50 倍
  - Tick 数据量大（~2亿行），预计占用 25-35 GB
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

# ── 配置 ──────────────────────────────────────────────────────────────────

TICK_DB_PATH = Path(ROOT) / "data" / "storage" / "tick_data.db"

FUTURES_SYMBOLS = {
    "IM": "KQ.m@CFFEX.IM",
    "IC": "KQ.m@CFFEX.IC",
    "IF": "KQ.m@CFFEX.IF",
    "IH": "KQ.m@CFFEX.IH",
}

DEFAULT_START = date(2024, 1, 1)

# 请求间隔（秒）— TQ Pro 可以更快
REQUEST_INTERVAL = 0.5

# 批量提交行数
BATCH_SIZE = 50_000

_INSERT_SQL = (
    "INSERT OR IGNORE INTO futures_tick "
    "(symbol, datetime, last_price, average, highest, lowest, "
    "bid_price1, bid_volume1, ask_price1, ask_volume1, "
    "volume, amount, open_interest) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _init_tick_db() -> sqlite3.Connection:
    """初始化 tick_data.db。"""
    TICK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TICK_DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

    from data.storage.schemas import FUTURES_TICK_SQL
    conn.executescript(FUTURES_TICK_SQL)
    conn.commit()
    return conn


def _get_latest_date(conn: sqlite3.Connection, symbol: str) -> date | None:
    """查询某品种在 tick DB 中的最新日期。"""
    row = conn.execute(
        "SELECT MAX(datetime) FROM futures_tick WHERE symbol=?", (symbol,)
    ).fetchone()
    if row and row[0]:
        return datetime.strptime(row[0][:10], "%Y-%m-%d").date()
    return None


def _get_trading_days(start: date, end: date) -> list[date]:
    """从 trading.db 获取交易日列表。"""
    from config.config_loader import ConfigLoader
    trading_db = sqlite3.connect(ConfigLoader().get_db_path(), timeout=30)
    rows = trading_db.execute(
        "SELECT DISTINCT trade_date FROM futures_daily "
        "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    ).fetchall()
    trading_db.close()
    return sorted(set(datetime.strptime(r[0], "%Y%m%d").date() for r in rows))


def _connect_tq():
    """建立 TQ 连接（行情模式）。"""
    from tqsdk import TqApi, TqAuth
    auth = TqAuth(
        os.getenv("TQ_ACCOUNT", ""),
        os.getenv("TQ_PASSWORD", ""),
    )
    api = TqApi(auth=auth)
    return api


def _write_tick_batch(conn: sqlite3.Connection, symbol: str,
                      df: pd.DataFrame) -> int:
    """批量写入 tick 数据（全向量化）。返回写入行数。"""
    if df.empty:
        return 0

    import numpy as np

    # 全向量化：不用 iterrows
    dt_strs = pd.to_datetime(df["datetime"], unit="ns").dt.strftime("%Y-%m-%d %H:%M:%S.%f").tolist()
    n = len(df)

    def _col(name, dtype=float, default=0):
        if name in df.columns:
            return df[name].fillna(default).astype(dtype).tolist()
        return [default] * n

    symbols = [symbol] * n
    rows = list(zip(
        symbols, dt_strs,
        _col("last_price"), _col("average"), _col("highest"), _col("lowest"),
        _col("bid_price1"), _col("bid_volume1", int), _col("ask_price1"), _col("ask_volume1", int),
        _col("volume", int), _col("amount"), _col("open_interest", int),
    ))

    # 分批 executemany
    total = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        conn.executemany(_INSERT_SQL, batch)
        total += len(batch)
        conn.commit()

    return total


def download_tick_for_symbol(api, conn: sqlite3.Connection,
                             symbol: str, tq_symbol: str,
                             trading_days: list[date]) -> int:
    """下载一个品种的全部 tick 数据。按天请求。"""
    total_new = 0
    n_days = len(trading_days)

    for i, td in enumerate(trading_days):
        pct = (i + 1) / n_days * 100
        print(f"\r  {symbol}: {td} [{i+1}/{n_days} {pct:5.1f}%]  +{total_new:,} ticks",
              end="", flush=True)

        try:
            df = api.get_tick_data_series(
                tq_symbol,
                start_dt=td,
                end_dt=td,
            )
        except Exception as e:
            print(f"\n    ⚠ {td} 请求失败: {e}")
            time.sleep(REQUEST_INTERVAL * 3)
            continue

        if df is None or df.empty:
            time.sleep(REQUEST_INTERVAL)
            continue

        df = df[df["last_price"] > 0].copy()
        if df.empty:
            time.sleep(REQUEST_INTERVAL)
            continue

        n = _write_tick_batch(conn, symbol, df)
        total_new += n
        time.sleep(REQUEST_INTERVAL)

    print(f"\r  {symbol}: 完成  共新增 {total_new:,} ticks" + " " * 30)
    return total_new


def print_stats(conn: sqlite3.Connection):
    """打印 tick_data.db 统计信息。"""
    print("\n" + "=" * 70)
    print("  tick_data.db 数据统计")
    print("=" * 70)

    try:
        rows = conn.execute(
            "SELECT symbol, COUNT(*) as cnt, "
            "MIN(datetime) as mn, MAX(datetime) as mx, "
            "COUNT(DISTINCT substr(datetime,1,10)) as days "
            "FROM futures_tick GROUP BY symbol ORDER BY symbol"
        ).fetchall()

        if not rows:
            print("  (空)")
            return

        for r in rows:
            print(f"  {r[0]:<6s}  {r[1]:>12,} ticks  {r[4]:>4d} 天  "
                  f"{r[2][:10]} ~ {r[3][:10]}")

        total = conn.execute("SELECT COUNT(*) FROM futures_tick").fetchone()[0]
        db_size = TICK_DB_PATH.stat().st_size / (1024 ** 3) if TICK_DB_PATH.exists() else 0
        print(f"\n  总计: {total:,} ticks  文件大小: {db_size:.2f} GB")

    except Exception as e:
        print(f"  查询失败: {e}")


def run_test(api, conn):
    """测试模式：下载 1 天数据看数据量。"""
    test_date = date(2026, 4, 3)
    print(f"\n=== Tick 数据测试 (日期: {test_date}) ===\n")

    for sym, tq_sym in FUTURES_SYMBOLS.items():
        try:
            df = api.get_tick_data_series(tq_sym, start_dt=test_date, end_dt=test_date)
            n = len(df) if df is not None else 0
            valid = len(df[df["last_price"] > 0]) if n > 0 else 0
            print(f"  {sym}: {n:,} 条原始, {valid:,} 条有效")
        except Exception as e:
            print(f"  {sym}: 请求失败 - {e}")
        time.sleep(REQUEST_INTERVAL)

    # 估算
    print(f"\n  估算全量 (544天×4品种): 约 {544 * 4 * 100_000:,} ticks")
    print(f"  预计存储: ~25-35 GB")


def main():
    parser = argparse.ArgumentParser(description="TQ Pro 批量下载期货 Tick 数据")
    parser.add_argument("--symbols", type=str, default="",
                        help="逗号分隔品种代码 (如 IM,IC)，默认全部4品种")
    parser.add_argument("--start", type=str, default="",
                        help="起始日期 YYYYMMDD (默认 20240101)")
    parser.add_argument("--end", type=str, default="",
                        help="结束日期 YYYYMMDD (默认今天)")
    parser.add_argument("--update", action="store_true",
                        help="增量模式：从DB已有数据末尾继续")
    parser.add_argument("--stats", action="store_true",
                        help="只打印统计信息，不下载")
    parser.add_argument("--test", action="store_true",
                        help="测试模式：下1天数据看数据量")
    args = parser.parse_args()

    conn = _init_tick_db()

    if args.stats:
        print_stats(conn)
        conn.close()
        return

    symbols_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols_list:
        symbols_list = list(FUTURES_SYMBOLS.keys())
    default_start = (datetime.strptime(args.start, "%Y%m%d").date()
                     if args.start else DEFAULT_START)
    end = datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()

    api = _connect_tq()

    try:
        if args.test:
            run_test(api, conn)
            return

        print("=" * 70)
        print("  TQ Pro Tick 批量下载")
        print("=" * 70)
        print(f"  品种: {symbols_list}")
        print(f"  范围: {default_start} ~ {end}")
        print(f"  模式: {'增量' if args.update else '全量'}")
        print(f"  DB路径: {TICK_DB_PATH}")
        print(f"  请求间隔: {REQUEST_INTERVAL}s")

        all_trading_days = _get_trading_days(default_start, end)
        print(f"  交易日: {len(all_trading_days)} 天")

        if not all_trading_days:
            print("  ⚠ 无交易日数据")
            return

        grand_total = 0
        for sym in symbols_list:
            if sym not in FUTURES_SYMBOLS:
                print(f"  ⚠ 未知品种: {sym}，跳过")
                continue

            tq_sym = FUTURES_SYMBOLS[sym]

            if args.update:
                latest = _get_latest_date(conn, sym)
                if latest:
                    start = latest + timedelta(days=1)
                else:
                    start = default_start
            else:
                start = default_start

            days = [d for d in all_trading_days if start <= d <= end]
            if not days:
                print(f"\n  {sym}: 已是最新")
                continue

            est_hours = len(days) * REQUEST_INTERVAL / 3600
            print(f"\n  {sym}: {len(days)} 天待下载 ({days[0]} ~ {days[-1]})"
                  f"  预计耗时 ~{est_hours:.1f}h")
            n = download_tick_for_symbol(api, conn, sym, tq_sym, days)
            grand_total += n

        print(f"\n  全部完成，共新增 {grand_total:,} ticks")
        print_stats(conn)

    finally:
        api.close()
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
