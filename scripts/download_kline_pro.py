#!/usr/bin/env python3
"""
download_kline_pro.py
---------------------
用 TQ 专业版 get_kline_data_series 批量下载历史K线数据。

数据写入 trading.db：
  - 现货指数 1m/5m → index_min (period=60/300)
  - 期货主连 1m/5m → futures_min (period=60/300)
  - MO期权   5m   → options_min (period=300)

用法:
    # 全量下载（各品种从上市日起）
    python scripts/download_kline_pro.py

    # 只下指定类别
    python scripts/download_kline_pro.py --category spot
    python scripts/download_kline_pro.py --category futures
    python scripts/download_kline_pro.py --category options

    # 指定日期范围
    python scripts/download_kline_pro.py --start 20260101 --end 20260403

    # 指定品种
    python scripts/download_kline_pro.py --category spot --symbols 000852

    # 增量更新（从DB已有数据的最后日期继续）
    python scripts/download_kline_pro.py --update

注意:
  - 按月分段请求，每段间隔 2 秒，避免触发 TQ 限流
  - INSERT OR IGNORE 去重，可安全重复运行
  - MO 期权按月合约逐个下载，深度虚值无成交合约自动跳过
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

# 现货指数
SPOT_SYMBOLS = {
    "000852": ("SSE.000852",   date(2022, 7, 22)),   # 中证1000（IM上市日）
    "000300": ("SSE.000300",   date(2015, 1, 5)),     # 沪深300
    "000016": ("SSE.000016",   date(2015, 1, 5)),     # 上证50
    "000905": ("SSE.000905",   date(2015, 1, 5)),     # 中证500
}

# 期货主连
FUTURES_SYMBOLS = {
    "IM": ("KQ.m@CFFEX.IM",  date(2022, 7, 22)),
    "IC": ("KQ.m@CFFEX.IC",  date(2015, 4, 16)),
    "IF": ("KQ.m@CFFEX.IF",  date(2015, 1, 5)),
    "IH": ("KQ.m@CFFEX.IH",  date(2015, 4, 16)),
}

# 期权品种配置 (prefix, TQ exchange prefix, 上市日)
OPTION_PRODUCTS = {
    "MO": ("CFFEX", date(2022, 7, 22)),   # 中证1000期权
    "IO": ("CFFEX", date(2019, 12, 23)),   # 沪深300期权
    "HO": ("CFFEX", date(2022, 12, 19)),   # 上证50期权
}

# 请求间隔（秒）— TQ Pro 可以更快
REQUEST_INTERVAL = 0.5

# 每次请求的最大月跨度
CHUNK_MONTHS = 1


def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """把日期范围切分为按月的段。"""
    ranges = []
    cur = start
    while cur <= end:
        month_end = (cur.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        seg_end = min(month_end, end)
        ranges.append((cur, seg_end))
        cur = seg_end + timedelta(days=1)
    return ranges


def _get_db():
    from config.config_loader import ConfigLoader
    db_path = ConfigLoader().get_db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _get_options_db():
    from config.config_loader import ConfigLoader
    db_path = ConfigLoader().get_options_db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    from data.storage.schemas import OPTIONS_MIN_SQL
    conn.executescript(OPTIONS_MIN_SQL)
    conn.commit()
    return conn


def _get_latest_date(conn: sqlite3.Connection, table: str, symbol_col: str,
                     symbol: str, period: int) -> date | None:
    """查询某品种在DB中的最新日期。"""
    row = conn.execute(
        f"SELECT MAX(datetime) FROM {table} WHERE {symbol_col}=? AND period=?",
        (symbol, period)
    ).fetchone()
    if row and row[0]:
        return datetime.strptime(row[0][:10], "%Y-%m-%d").date()
    return None


def _connect_tq():
    """建立 TQ 连接（行情模式，不连实盘）。"""
    from tqsdk import TqApi, TqAuth
    auth = TqAuth(
        os.getenv("TQ_ACCOUNT", ""),
        os.getenv("TQ_PASSWORD", ""),
    )
    api = TqApi(auth=auth)
    return api


def _write_kline_rows(conn: sqlite3.Connection, table: str, symbol_col: str,
                      symbol: str, period: int, df: pd.DataFrame) -> int:
    """写入K线数据（向量化），INSERT OR IGNORE 去重。返回写入行数。"""
    if df.empty:
        return 0

    # 向量化时间戳转换
    dt_col = df["datetime"]
    if dt_col.dtype in ("int64", "float64"):
        dt_strs = pd.to_datetime(dt_col, unit="ns").dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    else:
        dt_strs = pd.to_datetime(dt_col).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    oi_col = "open_interest" if "open_interest" in df.columns else "close_oi" if "close_oi" in df.columns else None

    n = len(df)
    symbols = [symbol] * n
    periods = [period] * n
    oi_vals = df[oi_col].fillna(0).tolist() if oi_col else [None] * n

    rows = list(zip(
        symbols, dt_strs, periods,
        df["open"].tolist(), df["high"].tolist(),
        df["low"].tolist(), df["close"].tolist(),
        df["volume"].tolist(), oi_vals,
    ))

    sql = (f"INSERT OR IGNORE INTO {table} "
           f"({symbol_col}, datetime, period, open, high, low, close, volume, open_interest) "
           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def download_kline(api, conn: sqlite3.Connection,
                   tq_symbol: str, db_symbol: str,
                   table: str, symbol_col: str,
                   period_sec: int, start: date, end: date,
                   label: str = "") -> int:
    """下载一个品种一个周期的全部K线。按月分段请求。"""
    chunks = _month_ranges(start, end)
    total_new = 0
    desc = label or f"{db_symbol} {period_sec}s"

    for i, (seg_start, seg_end) in enumerate(chunks):
        pct = (i + 1) / len(chunks) * 100
        print(f"\r  {desc}: {seg_start}~{seg_end}  ({pct:5.1f}%)  +{total_new} rows", end="", flush=True)

        try:
            df = api.get_kline_data_series(
                tq_symbol, period_sec,
                start_dt=seg_start, end_dt=seg_end,
            )
        except Exception as e:
            print(f"\n    ⚠ 请求失败: {e}")
            time.sleep(REQUEST_INTERVAL * 2)
            continue

        if df is None or df.empty:
            time.sleep(REQUEST_INTERVAL)
            continue

        # 过滤无效bar
        df = df[df["close"] > 0].copy()
        if df.empty:
            time.sleep(REQUEST_INTERVAL)
            continue

        n = _write_kline_rows(conn, table, symbol_col, db_symbol, period_sec, df)
        total_new += n
        time.sleep(REQUEST_INTERVAL)

    print(f"\r  {desc}: 完成  共新增 {total_new:,} 行" + " " * 30)
    return total_new


def download_spot(api, conn, symbols: list[str] | None, start_override: date | None,
                  end: date, update: bool):
    """下载现货指数 1m + 5m。"""
    print("\n" + "=" * 70)
    print("  现货指数 K线")
    print("=" * 70)

    for sym, (tq_sym, default_start) in SPOT_SYMBOLS.items():
        if symbols and sym not in symbols:
            continue

        for period in [60, 300]:
            period_label = "1m" if period == 60 else "5m"
            if update:
                latest = _get_latest_date(conn, "index_min", "symbol", sym, period)
                seg_start = (latest + timedelta(days=1)) if latest else default_start
            else:
                seg_start = start_override or default_start

            if seg_start > end:
                print(f"  {sym} {period_label}: 已是最新")
                continue

            download_kline(api, conn, tq_sym, sym, "index_min", "symbol",
                           period, seg_start, end,
                           label=f"{sym} {period_label}")


def download_futures(api, conn, symbols: list[str] | None, start_override: date | None,
                     end: date, update: bool):
    """下载期货主连 1m + 5m。"""
    print("\n" + "=" * 70)
    print("  期货主连 K线")
    print("=" * 70)

    for sym, (tq_sym, default_start) in FUTURES_SYMBOLS.items():
        if symbols and sym not in symbols:
            continue

        for period in [60, 300]:
            period_label = "1m" if period == 60 else "5m"
            if update:
                latest = _get_latest_date(conn, "futures_min", "symbol", sym, period)
                seg_start = (latest + timedelta(days=1)) if latest else default_start
            else:
                seg_start = start_override or default_start

            if seg_start > end:
                print(f"  {sym} {period_label}: 已是最新")
                continue

            download_kline(api, conn, tq_sym, sym, "futures_min", "symbol",
                           period, seg_start, end,
                           label=f"{sym} {period_label}")


def _get_option_contracts_by_month(conn: sqlite3.Connection,
                                    product: str) -> dict[str, list[str]]:
    """按到期月份分组，获取某期权品种的所有历史合约。
    返回 {YYMM: [ts_code, ...]}
    product: "MO", "IO", "HO"
    """
    # MO ts_code 长度=2, IO/HO 也是2字符前缀
    prefix_len = len(product)
    rows = conn.execute(
        "SELECT DISTINCT ts_code FROM options_daily "
        "WHERE ts_code LIKE ? ORDER BY ts_code",
        (f"{product}%",)
    ).fetchall()
    contracts: dict[str, list[str]] = {}
    for (ts_code,) in rows:
        # MO2605-C-7000.CFX → month = 2605
        month = ts_code[prefix_len:prefix_len + 4]
        contracts.setdefault(month, []).append(ts_code)
    return contracts


def _ts_to_tq_option(ts_code: str, exchange: str = "CFFEX") -> str:
    """Tushare 期权代码 → TQ 代码。MO2605-C-7000.CFX → CFFEX.MO2605-C-7000"""
    return f"{exchange}." + ts_code.replace(".CFX", "")


def download_options_product(api, conn, product: str, exchange: str,
                             start_override: date | None, end: date, update: bool) -> int:
    """下载一个期权品种的 5m K线。逐合约下载。"""
    by_month = _get_option_contracts_by_month(conn, product)
    if not by_month:
        print(f"  ⚠ options_daily 中无 {product} 合约数据，跳过")
        return 0

    total_contracts = sum(len(v) for v in by_month.values())
    print(f"  共 {len(by_month)} 个到期月, {total_contracts} 个合约")

    grand_total = 0
    done_count = 0

    for month_prefix in sorted(by_month.keys()):
        contracts = by_month[month_prefix]
        # 该月份合约的交易日期范围（取第一个合约查）
        row = conn.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM options_daily "
            "WHERE ts_code = ?", (contracts[0],)
        ).fetchone()
        if not row or not row[0]:
            done_count += len(contracts)
            continue

        month_start = datetime.strptime(row[0], "%Y%m%d").date()
        month_end = datetime.strptime(row[1], "%Y%m%d").date()

        if start_override:
            month_start = max(month_start, start_override)
        if update:
            latest = _get_latest_date(conn, "options_min", "ts_code", contracts[0], 300)
            if latest:
                month_start = max(month_start, latest + timedelta(days=1))
        month_end = min(month_end, end)

        if month_start > month_end:
            done_count += len(contracts)
            continue

        print(f"\n  ── {product}{month_prefix} ({len(contracts)} 合约, "
              f"{month_start}~{month_end}) ──")

        for ts_code in contracts:
            done_count += 1
            tq_sym = _ts_to_tq_option(ts_code, exchange)
            pct = done_count / total_contracts * 100
            print(f"\r    [{done_count}/{total_contracts} {pct:5.1f}%] {ts_code}",
                  end="", flush=True)

            try:
                df = api.get_kline_data_series(
                    tq_sym, 300,
                    start_dt=month_start, end_dt=month_end,
                )
            except Exception as e:
                time.sleep(REQUEST_INTERVAL)
                continue

            if df is None or df.empty:
                time.sleep(REQUEST_INTERVAL)
                continue

            df = df[df["close"] > 0].copy()
            if df.empty:
                time.sleep(REQUEST_INTERVAL)
                continue

            n = _write_kline_rows(conn, "options_min", "ts_code", ts_code, 300, df)
            grand_total += n
            time.sleep(REQUEST_INTERVAL)

    print(f"\n  {product} 期权 5m: 完成  共新增 {grand_total:,} 行")
    return grand_total


def download_options(api, conn, products: list[str] | None,
                     start_override: date | None, end: date, update: bool):
    """下载期权 5m K线。支持 MO/IO/HO。"""
    print("\n" + "=" * 70)
    print("  期权 5分钟K线")
    print("=" * 70)

    for product, (exchange, _default_start) in OPTION_PRODUCTS.items():
        if products and product not in products:
            continue
        print(f"\n  === {product} ===")
        download_options_product(api, conn, product, exchange,
                                start_override, end, update)


def print_summary(conn: sqlite3.Connection):
    """打印下载后的数据库统计。"""
    print("\n" + "=" * 70)
    print("  下载后数据统计")
    print("=" * 70)

    for table, sym_col in [("index_min", "symbol"), ("futures_min", "symbol"), ("options_min", "ts_code")]:
        try:
            rows = conn.execute(
                f"SELECT {sym_col}, period, COUNT(*) as cnt, "
                f"MIN(datetime) as mn, MAX(datetime) as mx "
                f"FROM {table} GROUP BY {sym_col}, period ORDER BY {sym_col}, period"
            ).fetchall()
            if rows:
                print(f"\n  {table}:")
                for r in rows:
                    period_label = f"{r[1]//60}m" if r[1] < 86400 else "d"
                    print(f"    {r[0]:<20s} {period_label:>3s}  {r[2]:>10,} 行  {r[3]} ~ {r[4]}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="TQ Pro 批量下载历史K线")
    parser.add_argument("--category", choices=["spot", "futures", "options", "all"],
                        default="all", help="下载类别")
    parser.add_argument("--symbols", type=str, default="",
                        help="逗号分隔的品种代码 (如 IM,IC 或 000852)")
    parser.add_argument("--start", type=str, default="",
                        help="起始日期 YYYYMMDD (默认各品种上市日)")
    parser.add_argument("--end", type=str, default="",
                        help="结束日期 YYYYMMDD (默认今天)")
    parser.add_argument("--update", action="store_true",
                        help="增量模式：从DB已有数据末尾继续")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    start = datetime.strptime(args.start, "%Y%m%d").date() if args.start else None
    end = datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()

    print("=" * 70)
    print("  TQ Pro K线批量下载")
    print("=" * 70)
    print(f"  类别: {args.category}")
    print(f"  品种: {symbols or '全部'}")
    print(f"  范围: {start or '各品种上市日'} ~ {end}")
    print(f"  模式: {'增量' if args.update else '全量'}")
    print(f"  请求间隔: {REQUEST_INTERVAL}s")

    conn = _get_db()
    opt_conn = _get_options_db()
    api = _connect_tq()

    try:
        if args.category in ("all", "spot"):
            download_spot(api, conn, symbols, start, end, args.update)

        if args.category in ("all", "futures"):
            download_futures(api, conn, symbols, start, end, args.update)

        if args.category in ("all", "options"):
            # --symbols 用于过滤期权品种 (MO/IO/HO)
            opt_products = [s for s in (symbols or []) if s in OPTION_PRODUCTS] or None
            download_options(api, opt_conn, opt_products, start, end, args.update)

        print_summary(conn)
        print_summary(opt_conn)
    finally:
        api.close()
        conn.close()
        opt_conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
