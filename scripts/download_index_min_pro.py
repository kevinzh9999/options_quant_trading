#!/usr/bin/env python3
"""
download_index_min_pro.py
-------------------------
用TQ专业版 get_kline_data_series 下载现货指数分钟K线。
分批下载，每批之间等待，避免触发访问限制。

用法：
    python scripts/download_index_min_pro.py --period 300   # 5分钟线
    python scripts/download_index_min_pro.py --period 60    # 1分钟线
    python scripts/download_index_min_pro.py --period 300 --symbol 000852  # 单品种
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
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

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

# 品种映射
SYMBOLS = {
    "000852": "SSE.000852",  # 中证1000 (IM)
    "000905": "SSE.000905",  # 中证500 (IC)
    "000300": "SSE.000300",  # 沪深300 (IF)
    "000016": "SSE.000016",  # 上证50 (IH)
}

# 安全参数
BATCH_MONTHS = 6        # 每批下载6个月
SLEEP_BETWEEN = 3       # 每批之间等3秒
SLEEP_BETWEEN_SYM = 5   # 每个品种之间等5秒


def download_one_batch(api, tq_sym: str, db_sym: str, period: int,
                       start: date, end: date, db: DBManager) -> int:
    """下载一个时间段的数据并写入DB。返回新增行数。"""
    try:
        df = api.get_kline_data_series(tq_sym, period,
                                       start_dt=start, end_dt=end)
    except Exception as e:
        print(f"    API error: {e}")
        return 0

    if df is None or len(df) == 0:
        return 0

    # 过滤有效数据
    df = df[df['close'] > 0].copy()
    if df.empty:
        return 0

    # 转换datetime
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['symbol'] = db_sym
    df['period'] = period

    # 写入DB（INSERT OR IGNORE避免重复）
    conn = db._conn
    n_new = 0
    for _, row in df.iterrows():
        try:
            conn.execute(
                "INSERT OR IGNORE INTO index_min "
                "(symbol, datetime, period, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row['symbol'], row['datetime'], row['period'],
                 float(row['open']), float(row['high']),
                 float(row['low']), float(row['close']),
                 float(row['volume']))
            )
            n_new += conn.total_changes  # approximate
        except Exception:
            pass
    conn.commit()

    return len(df)


def main():
    parser = argparse.ArgumentParser(description="TQ专业版下载现货指数分钟K线")
    parser.add_argument("--period", type=int, default=300,
                        help="周期秒数: 300=5min, 60=1min")
    parser.add_argument("--symbol", default="all",
                        help="品种: 000852/000905/000300/000016/all")
    parser.add_argument("--start", default="",
                        help="起始日期YYYYMMDD（默认按周期自动选）")
    args = parser.parse_args()

    db = get_db()

    # 确定起止日期
    if args.start:
        start_date = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:]))
    elif args.period == 300:
        start_date = date(2022, 7, 22)  # IM上市日，TQ 5分钟最远可到此
    else:
        start_date = date(2024, 1, 1)   # 1分钟从2024年起

    end_date = date(2026, 4, 4)

    symbols = {args.symbol: SYMBOLS[args.symbol]} if args.symbol != "all" else SYMBOLS

    period_name = f"{args.period//60}min"
    print(f"下载计划: {len(symbols)}个品种 × {period_name}")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"分批: 每{BATCH_MONTHS}个月一批, 批间等待{SLEEP_BETWEEN}s")

    # 连接TQ
    from tqsdk import TqApi, TqAuth
    account = os.getenv("TQ_ACCOUNT", "")
    password = os.getenv("TQ_PASSWORD", "")
    print(f"\n连接TQ...")
    api = TqApi(auth=TqAuth(account, password))

    total_rows = 0

    try:
        for db_sym, tq_sym in symbols.items():
            print(f"\n{'='*60}")
            print(f"  {db_sym} ({tq_sym}) {period_name}")
            print(f"{'='*60}")

            # 检查已有数据
            existing = db.query_df(
                f"SELECT MIN(datetime) mn, MAX(datetime) mx, COUNT(*) c "
                f"FROM index_min WHERE symbol='{db_sym}' AND period={args.period}"
            )
            if existing is not None and existing.iloc[0]['c'] > 0:
                print(f"  已有: {int(existing.iloc[0]['c'])}根 "
                      f"({existing.iloc[0]['mn']} ~ {existing.iloc[0]['mx']})")

            # 分批下载
            batch_start = start_date
            sym_total = 0
            batch_num = 0

            while batch_start < end_date:
                batch_end = min(
                    batch_start + timedelta(days=BATCH_MONTHS * 30),
                    end_date
                )
                batch_num += 1

                print(f"  [{batch_num}] {batch_start} ~ {batch_end} ...", end="", flush=True)

                n = download_one_batch(api, tq_sym, db_sym, args.period,
                                      batch_start, batch_end, db)
                print(f" {n}根")
                sym_total += n

                batch_start = batch_end
                if batch_start < end_date:
                    time.sleep(SLEEP_BETWEEN)

            # 验证
            after = db.query_df(
                f"SELECT MIN(datetime) mn, MAX(datetime) mx, COUNT(*) c "
                f"FROM index_min WHERE symbol='{db_sym}' AND period={args.period}"
            )
            if after is not None and after.iloc[0]['c'] > 0:
                print(f"  完成: {int(after.iloc[0]['c'])}根 "
                      f"({after.iloc[0]['mn']} ~ {after.iloc[0]['mx']})")

            total_rows += sym_total

            # 品种间等待
            if db_sym != list(symbols.keys())[-1]:
                print(f"  等待{SLEEP_BETWEEN_SYM}s...")
                time.sleep(SLEEP_BETWEEN_SYM)

    finally:
        api.close()

    print(f"\n总计下载: {total_rows}根")


if __name__ == "__main__":
    main()
