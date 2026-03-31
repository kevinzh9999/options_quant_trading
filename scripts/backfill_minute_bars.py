#!/usr/bin/env python3
"""
backfill_minute_bars.py
-----------------------
从TQ拉取历史分钟K线并写入数据库。

支持期货和现货指数：
    python scripts/backfill_minute_bars.py --symbol IC --bars 8000
    python scripts/backfill_minute_bars.py --symbol 000852 --exchange SSE --bars 10000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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

# Spot index mapping
_SPOT_MAP = {
    "000852": "SSE.000852",
    "000300": "SSE.000300",
    "000016": "SSE.000016",
    "000905": "SSE.000905",
}


def main():
    parser = argparse.ArgumentParser(description="从TQ拉取历史分钟K线")
    parser.add_argument("--symbol", required=True, help="品种: IF/IH/IM/IC 或 000852/000300/000016/000905")
    parser.add_argument("--exchange", default="", help="交易所: SSE/SZSE (指数用), CFFEX (期货默认)")
    parser.add_argument("--bars", type=int, default=8000, help="K线数量")
    parser.add_argument("--period", type=int, default=300, help="周期秒数 (300=5m)")
    args = parser.parse_args()

    sym = args.symbol.upper()
    exchange = args.exchange.upper()

    # Determine if this is a spot index or futures
    is_index = sym in _SPOT_MAP or exchange in ("SSE", "SZSE")
    if is_index:
        tq_sym = _SPOT_MAP.get(sym, f"{exchange}.{sym}")
        table = "index_min"
        db_symbol = sym  # e.g. "000852"
    else:
        tq_sym = f"KQ.m@CFFEX.{sym}"
        table = "futures_min"
        db_symbol = sym

    print(f"Backfill {db_symbol} {args.period}s bars → {table}")
    print(f"  TQ symbol: {tq_sym}, requesting {args.bars} bars")

    from data.storage.db_manager import DBManager
    from config.config_loader import ConfigLoader
    db = DBManager(ConfigLoader().get_db_path())

    from data.sources.tq_client import TqClient
    creds = {
        "auth_account": os.getenv("TQ_ACCOUNT", ""),
        "auth_password": os.getenv("TQ_PASSWORD", ""),
        "broker_id": os.getenv("TQ_BROKER", ""),
        "account_id": os.getenv("TQ_ACCOUNT_ID", ""),
        "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
    }

    client = TqClient(**creds)
    client.connect()
    api = client._api

    try:
        print(f"  Subscribing to {tq_sym}...")
        klines = api.get_kline_serial(tq_sym, args.period, args.bars)

        print("  Waiting for data...")
        deadline = time.time() + 30
        while True:
            api.wait_update(deadline=min(time.time() + 2, deadline))
            if len(klines) > 0 and klines.iloc[-1]["close"] > 0:
                break
            if time.time() > deadline:
                print("  Timeout")
                return

        df = klines[["open", "high", "low", "close", "volume"]].copy()
        df["datetime"] = pd.to_datetime(klines["datetime"], unit="ns")
        df = df[df["close"] > 0].copy()
        df = df[df["open"] > 0].copy()

        if df.empty:
            print("  No valid data")
            return

        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["symbol"] = db_symbol
        df["period"] = args.period

        print(f"  Received {len(df)} bars")
        print(f"  Range: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

        conn = db._conn
        for _, row in df.iterrows():
            try:
                conn.execute(
                    f"INSERT OR IGNORE INTO {table} "
                    "(symbol, datetime, period, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["symbol"], row["datetime"], row["period"],
                     float(row["open"]), float(row["high"]),
                     float(row["low"]), float(row["close"]),
                     float(row["volume"])),
                )
            except Exception:
                pass
        conn.commit()

        r = db.query_df(
            f"SELECT COUNT(*) as cnt, MIN(datetime) as mn, MAX(datetime) as mx "
            f"FROM {table} WHERE symbol='{db_symbol}' AND period={args.period}"
        )
        if r is not None and not r.empty:
            print(f"\n  DB total: {r.iloc[0]['cnt']} bars")
            print(f"  Range: {r.iloc[0]['mn']} ~ {r.iloc[0]['mx']}")

    finally:
        client.disconnect()

    print("Done.")


if __name__ == "__main__":
    main()
