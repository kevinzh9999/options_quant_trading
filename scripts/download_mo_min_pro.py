#!/usr/bin/env python3
"""
download_mo_min_pro.py
----------------------
用TQ专业版下载MO期权5分钟K线（所有有成交的合约）。

用法：
    python scripts/download_mo_min_pro.py
    python scripts/download_mo_min_pro.py --expire 2506   # 只下某月
"""
from __future__ import annotations

import argparse
import os
import re
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

# 安全参数
SLEEP_BETWEEN = 1       # 每合约间等1秒
SLEEP_EVERY_50 = 10     # 每50个合约多等10秒
PERIOD = 300            # 5分钟

# TQ MO代码格式：CFFEX.MO2509-C-7400（无.CFX后缀）
_MO_RE = re.compile(r'^MO(\d{4})-([CP])-(\d+)\.CFX$')


def tushare_to_tq(ts_code: str) -> str:
    """MO2509-C-7400.CFX → CFFEX.MO2509-C-7400"""
    return "CFFEX." + ts_code.replace(".CFX", "")


def ensure_options_min_table(db):
    """确保options_min表存在。"""
    conn = db._conn_for_table("options_min")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS options_min (
            symbol    TEXT NOT NULL,
            datetime  TEXT NOT NULL,
            period    INTEGER NOT NULL,
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            volume    REAL,
            open_oi   REAL,
            close_oi  REAL,
            PRIMARY KEY (symbol, datetime, period)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_options_min_sym_dt "
        "ON options_min(symbol, datetime)"
    )
    conn.commit()


def download_one_contract(api, ts_code, tq_code, start, end, db):
    """下载单个合约的5分钟K线。"""
    try:
        df = api.get_kline_data_series(tq_code, PERIOD,
                                       start_dt=start, end_dt=end)
    except Exception as e:
        return 0, str(e)[:60]

    if df is None or len(df) == 0:
        return 0, ""

    df = df[df['close'] > 0].copy()
    if df.empty:
        return 0, ""

    df['datetime'] = pd.to_datetime(df['datetime'], unit='ns').dt.strftime('%Y-%m-%d %H:%M:%S')
    df['symbol'] = ts_code  # 用Tushare格式存储（与options_daily一致）
    df['period'] = PERIOD

    conn = db._conn_for_table("options_min")
    for _, row in df.iterrows():
        conn.execute(
            "INSERT OR IGNORE INTO options_min "
            "(symbol, datetime, period, open, high, low, close, volume, open_oi, close_oi) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row['symbol'], row['datetime'], row['period'],
             float(row['open']), float(row['high']),
             float(row['low']), float(row['close']),
             float(row['volume']),
             float(row.get('open_oi', 0) or 0),
             float(row.get('close_oi', 0) or 0))
        )
    conn.commit()
    return len(df), ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expire", default="", help="只下某到期月(如2506)")
    args = parser.parse_args()

    db = get_db()
    ensure_options_min_table(db)

    # 获取有成交的MO合约列表
    where = ""
    if args.expire:
        where = f" AND ts_code LIKE 'MO{args.expire}%%'"
    contracts = db.query_df(
        f"SELECT ts_code, MIN(trade_date) first_date, MAX(trade_date) last_date "
        f"FROM options_daily "
        f"WHERE ts_code LIKE 'MO%%' AND volume > 0 {where} "
        f"GROUP BY ts_code ORDER BY ts_code"
    )
    print("MO合约: %d个" % len(contracts))

    # 检查已下载的
    existing = set()
    try:
        r = db.query_df("SELECT DISTINCT symbol FROM options_min WHERE period=%d" % PERIOD)
        if r is not None:
            existing = set(r['symbol'].tolist())
    except Exception:
        pass
    print("已下载: %d个" % len(existing))

    to_download = contracts[~contracts['ts_code'].isin(existing)]
    print("待下载: %d个" % len(to_download))

    if len(to_download) == 0:
        print("全部已下载完成。")
        return

    from tqsdk import TqApi, TqAuth
    api = TqApi(auth=TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", "")))

    total_rows = 0
    errors = 0

    try:
        for i, (_, row) in enumerate(to_download.iterrows()):
            ts_code = row['ts_code']
            tq_code = tushare_to_tq(ts_code)
            first = row['first_date']
            last = row['last_date']

            start = date(int(first[:4]), int(first[4:6]), int(first[6:]))
            end = date(int(last[:4]), int(last[4:6]), int(last[6:])) + timedelta(days=1)

            n, err = download_one_contract(api, ts_code, tq_code, start, end, db)
            total_rows += n

            if err:
                errors += 1
                if (i + 1) % 100 == 0 or i < 5:
                    print("  [%d/%d] %s: ERROR %s" % (i+1, len(to_download), ts_code, err))
            elif (i + 1) % 50 == 0 or i < 3:
                print("  [%d/%d] %s: %d根 (%s~%s)" % (i+1, len(to_download), ts_code, n, first, last))

            time.sleep(SLEEP_BETWEEN)
            if (i + 1) % 50 == 0:
                time.sleep(SLEEP_EVERY_50)

    finally:
        api.close()

    # 统计
    r = db.query_df("SELECT COUNT(*) c, COUNT(DISTINCT symbol) n FROM options_min WHERE period=%d" % PERIOD)
    print("\nDone: 新增%d根, 错误%d个" % (total_rows, errors))
    print("options_min总计: %d根, %d个合约" % (int(r.iloc[0]['c']), int(r.iloc[0]['n'])))


if __name__ == "__main__":
    main()
