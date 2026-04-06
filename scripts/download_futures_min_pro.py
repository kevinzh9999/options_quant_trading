#!/usr/bin/env python3
"""
download_futures_min_pro.py
---------------------------
用TQ专业版下载期货主连分钟K线。

用法：
    python scripts/download_futures_min_pro.py --period 300   # 5分钟
    python scripts/download_futures_min_pro.py --period 60    # 1分钟
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

SYMBOLS = {
    "IM": "KQ.m@CFFEX.IM",
    "IC": "KQ.m@CFFEX.IC",
    "IF": "KQ.m@CFFEX.IF",
    "IH": "KQ.m@CFFEX.IH",
}

BATCH_MONTHS = 6
SLEEP_BETWEEN = 3
SLEEP_BETWEEN_SYM = 5


def download_batch(api, tq_sym, db_sym, period, start, end, db):
    try:
        df = api.get_kline_data_series(tq_sym, period, start_dt=start, end_dt=end)
    except Exception as e:
        print("    API error: %s" % e)
        return 0
    if df is None or len(df) == 0:
        return 0
    df = df[df['close'] > 0].copy()
    if df.empty:
        return 0
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns').dt.strftime('%Y-%m-%d %H:%M:%S')
    df['symbol'] = db_sym
    df['period'] = period
    conn = db._conn
    for _, row in df.iterrows():
        conn.execute(
            "INSERT OR IGNORE INTO futures_min "
            "(symbol, datetime, period, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row['symbol'], row['datetime'], row['period'],
             float(row['open']), float(row['high']),
             float(row['low']), float(row['close']),
             float(row['volume']))
        )
    conn.commit()
    return len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=300)
    parser.add_argument("--symbol", default="all")
    parser.add_argument("--start", default="")
    args = parser.parse_args()

    db = get_db()

    if args.start:
        start_date = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:]))
    elif args.period == 300:
        start_date = date(2022, 7, 22)
    else:
        start_date = date(2024, 1, 1)
    end_date = date(2026, 4, 4)

    symbols = {args.symbol: SYMBOLS[args.symbol]} if args.symbol != "all" else SYMBOLS
    period_name = "%dmin" % (args.period // 60)

    print("下载: %d品种 × %s, %s ~ %s" % (len(symbols), period_name, start_date, end_date))

    from tqsdk import TqApi, TqAuth
    api = TqApi(auth=TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", "")))

    total = 0
    try:
        for db_sym, tq_sym in symbols.items():
            print("\n  %s (%s) %s" % (db_sym, tq_sym, period_name))
            existing = db.query_df(
                "SELECT COUNT(*) c, MIN(datetime) mn, MAX(datetime) mx "
                "FROM futures_min WHERE symbol='%s' AND period=%d" % (db_sym, args.period))
            if existing is not None and existing.iloc[0]['c'] > 0:
                print("    已有: %d根 (%s ~ %s)" % (
                    int(existing.iloc[0]['c']), existing.iloc[0]['mn'], existing.iloc[0]['mx']))

            batch_start = start_date
            sym_total = 0
            while batch_start < end_date:
                batch_end = min(batch_start + timedelta(days=BATCH_MONTHS * 30), end_date)
                print("    %s ~ %s ..." % (batch_start, batch_end), end="", flush=True)
                n = download_batch(api, tq_sym, db_sym, args.period, batch_start, batch_end, db)
                print(" %d根" % n)
                sym_total += n
                batch_start = batch_end
                if batch_start < end_date:
                    time.sleep(SLEEP_BETWEEN)

            after = db.query_df(
                "SELECT COUNT(*) c, MIN(datetime) mn, MAX(datetime) mx "
                "FROM futures_min WHERE symbol='%s' AND period=%d" % (db_sym, args.period))
            if after is not None and after.iloc[0]['c'] > 0:
                print("    完成: %d根 (%s ~ %s)" % (
                    int(after.iloc[0]['c']), after.iloc[0]['mn'], after.iloc[0]['mx']))
            total += sym_total
            if db_sym != list(symbols.keys())[-1]:
                time.sleep(SLEEP_BETWEEN_SYM)
    finally:
        api.close()

    print("\n总计: %d根" % total)


if __name__ == "__main__":
    main()
