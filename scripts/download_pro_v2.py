#!/usr/bin/env python3
"""
download_pro_v2.py
------------------
用 TQ DataDownloader 批量下载历史数据（比 get_kline_data_series 快 10-50 倍）。

DataDownloader 优势：
  - 服务端流式推送，无逐段往返延迟
  - 多任务并行（同一 API 连接内同时下载多个品种）
  - 内置进度跟踪

流程：DataDownloader → CSV → SQLite（本地导入极快）

用法:
    # 下载指定任务
    python scripts/download_pro_v2.py --task spot_1m
    python scripts/download_pro_v2.py --task futures_5m
    python scripts/download_pro_v2.py --task tick_im_ic

    # 自定义下载
    python scripts/download_pro_v2.py --symbols KQ.m@CFFEX.IM --dur 60 --start 20240101 --end 20260403 --csv im_1m.csv --table futures_min --db-symbol IM

    # 列出预置任务
    python scripts/download_pro_v2.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
import sqlite3
import time
from contextlib import closing
from datetime import date, datetime
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

CSV_DIR = Path(ROOT) / "tmp" / "downloads"


# ── 预置下载任务 ──────────────────────────────────────────────────────────

PRESET_TASKS = {
    # 现货指数 1m
    "spot_1m": {
        "desc": "现货指数 1分钟K线 (000852/000300/000016/000905)",
        "symbols": {
            "SSE.000852": ("index_min", "000852", date(2022, 7, 22)),
            "SSE.000300": ("index_min", "000300", date(2018, 1, 1)),
            "SSE.000016": ("index_min", "000016", date(2018, 1, 1)),
            "SSE.000905": ("index_min", "000905", date(2018, 1, 1)),
        },
        "dur_sec": 60,
        "period": 60,
    },
    "spot_5m": {
        "desc": "现货指数 5分钟K线",
        "symbols": {
            "SSE.000852": ("index_min", "000852", date(2022, 7, 22)),
            "SSE.000300": ("index_min", "000300", date(2018, 1, 1)),
            "SSE.000016": ("index_min", "000016", date(2018, 1, 1)),
            "SSE.000905": ("index_min", "000905", date(2018, 1, 1)),
        },
        "dur_sec": 300,
        "period": 300,
    },
    # 期货主连 1m
    "futures_1m": {
        "desc": "期货主连 1分钟K线 (IM/IC/IF/IH)",
        "symbols": {
            "KQ.m@CFFEX.IM": ("futures_min", "IM", date(2022, 7, 22)),
            "KQ.m@CFFEX.IC": ("futures_min", "IC", date(2016, 1, 5)),
            "KQ.m@CFFEX.IF": ("futures_min", "IF", date(2016, 1, 5)),
            "KQ.m@CFFEX.IH": ("futures_min", "IH", date(2016, 1, 5)),
        },
        "dur_sec": 60,
        "period": 60,
    },
    "futures_5m": {
        "desc": "期货主连 5分钟K线",
        "symbols": {
            "KQ.m@CFFEX.IM": ("futures_min", "IM", date(2022, 7, 22)),
            "KQ.m@CFFEX.IC": ("futures_min", "IC", date(2016, 1, 5)),
            "KQ.m@CFFEX.IF": ("futures_min", "IF", date(2016, 1, 5)),
            "KQ.m@CFFEX.IH": ("futures_min", "IH", date(2016, 1, 5)),
        },
        "dur_sec": 300,
        "period": 300,
    },
    # Tick
    "tick_im_ic": {
        "desc": "IM/IC 主连 Tick (2024-01~)",
        "symbols": {
            "KQ.m@CFFEX.IM": ("futures_tick", "IM", date(2024, 1, 1)),
            "KQ.m@CFFEX.IC": ("futures_tick", "IC", date(2024, 1, 1)),
        },
        "dur_sec": 0,  # 0 = tick
        "period": 0,
    },
    "tick_all": {
        "desc": "IM/IC/IF/IH 主连 Tick (2024-01~)",
        "symbols": {
            "KQ.m@CFFEX.IM": ("futures_tick", "IM", date(2024, 1, 1)),
            "KQ.m@CFFEX.IC": ("futures_tick", "IC", date(2024, 1, 1)),
            "KQ.m@CFFEX.IF": ("futures_tick", "IF", date(2024, 1, 1)),
            "KQ.m@CFFEX.IH": ("futures_tick", "IH", date(2024, 1, 1)),
        },
        "dur_sec": 0,
        "period": 0,
    },
}


def _connect_tq():
    from tqsdk import TqApi, TqAuth
    auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))
    return TqApi(auth=auth)


def _get_db_conn(table: str) -> tuple[sqlite3.Connection, str]:
    """根据目标表返回对应的 DB 连接和路径。"""
    from config.config_loader import ConfigLoader
    cfg = ConfigLoader()

    if table in ("options_daily", "options_contracts", "options_min"):
        path = cfg.get_options_db_path()
    elif table == "futures_tick":
        path = str(Path(ROOT) / "data" / "storage" / "tick_data.db")
    else:
        path = cfg.get_db_path()

    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn, path


def _import_kline_csv(csv_path: str, conn: sqlite3.Connection,
                      table: str, symbol_col: str, db_symbol: str,
                      period: int) -> int:
    """将 K线 CSV 导入 SQLite。返回新增行数。"""
    df = pd.read_csv(csv_path)
    if df.empty:
        return 0

    # DataDownloader CSV 列名带合约前缀（如 SSE.000852.close），去掉
    df.columns = [c.split(".")[-1] if "." in c else c for c in df.columns]

    df = df[df["close"] > 0].copy()
    if df.empty:
        return 0

    # 时间格式：DataDownloader 输出 "2024-01-02 09:31:00.000000" 格式
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    oi_col = "close_oi" if "close_oi" in df.columns else None
    n = len(df)
    symbols = [db_symbol] * n
    periods = [period] * n
    oi_vals = df[oi_col].fillna(0).tolist() if oi_col else [None] * n

    rows = list(zip(
        symbols, df["datetime"].tolist(), periods,
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


def _import_tick_csv(csv_path: str, conn: sqlite3.Connection,
                     db_symbol: str) -> int:
    """将 Tick CSV 导入 SQLite。返回新增行数。"""
    df = pd.read_csv(csv_path)
    if df.empty:
        return 0

    # DataDownloader CSV 列名带合约前缀（如 KQ.m@CFFEX.IM.last_price），去掉
    df.columns = [c.split(".")[-1] if "." in c else c for c in df.columns]

    df = df[df["last_price"] > 0].copy()
    if df.empty:
        return 0

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    n = len(df)

    def _col(name, dtype=float, default=0):
        if name in df.columns:
            return df[name].fillna(default).astype(dtype).tolist()
        return [default] * n

    rows = list(zip(
        [db_symbol] * n, df["datetime"].tolist(),
        _col("last_price"), _col("average"), _col("highest"), _col("lowest"),
        _col("bid_price1"), _col("bid_volume1", int), _col("ask_price1"), _col("ask_volume1", int),
        _col("volume", int), _col("amount"), _col("open_interest", int),
    ))

    sql = ("INSERT OR IGNORE INTO futures_tick "
           "(symbol, datetime, last_price, average, highest, lowest, "
           "bid_price1, bid_volume1, ask_price1, ask_volume1, "
           "volume, amount, open_interest) "
           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")

    batch_size = 100_000
    total = 0
    for start in range(0, len(rows), batch_size):
        conn.executemany(sql, rows[start:start + batch_size])
        total += min(batch_size, len(rows) - start)
        conn.commit()
    return total


def run_preset(task_name: str, end_date: date):
    """运行预置下载任务。"""
    task = PRESET_TASKS[task_name]
    dur_sec = task["dur_sec"]
    period = task["period"]
    is_tick = dur_sec == 0

    print(f"\n{'='*70}")
    print(f"  {task['desc']}")
    print(f"  周期: {'tick' if is_tick else f'{dur_sec}s'}, 截止: {end_date}")
    print(f"{'='*70}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 启动所有 DataDownloader 任务
    api = _connect_tq()
    from tqsdk.tools import DataDownloader

    downloads = {}
    csv_files = {}

    for tq_sym, (table, db_sym, start) in task["symbols"].items():
        csv_name = f"{db_sym}_{task_name}.csv"
        csv_path = str(CSV_DIR / csv_name)
        csv_files[tq_sym] = (csv_path, table, db_sym)

        print(f"  创建任务: {tq_sym} ({start} ~ {end_date})")
        downloads[tq_sym] = DataDownloader(
            api, symbol_list=tq_sym, dur_sec=dur_sec,
            start_dt=start, end_dt=end_date,
            csv_file_name=csv_path,
        )

    # 2. 等待所有任务完成
    print(f"\n  并行下载中...")
    with closing(api):
        last_print = 0
        while not all(d.is_finished() for d in downloads.values()):
            api.wait_update()
            now = time.time()
            if now - last_print > 10:
                progress = "  ".join(
                    f"{sym.split('.')[-1]}:{d.get_progress():.1f}%"
                    for sym, d in downloads.items()
                )
                print(f"  {progress}")
                last_print = now

    print(f"  下载完成!")

    # 3. CSV → SQLite
    print(f"\n  导入数据库...")
    for tq_sym, (csv_path, table, db_sym) in csv_files.items():
        conn, db_path = _get_db_conn(table)

        # 确保表存在
        if table == "futures_tick":
            from data.storage.schemas import FUTURES_TICK_SQL
            conn.executescript(FUTURES_TICK_SQL)
        elif table == "options_min":
            from data.storage.schemas import OPTIONS_MIN_SQL
            conn.executescript(OPTIONS_MIN_SQL)

        csv_size = Path(csv_path).stat().st_size / 1e6
        print(f"    {db_sym}: {csv_path} ({csv_size:.1f} MB) → {table}")

        t0 = time.time()
        if is_tick:
            n = _import_tick_csv(csv_path, conn, db_sym)
        else:
            sym_col = "ts_code" if table == "options_min" else "symbol"
            n = _import_kline_csv(csv_path, conn, table, sym_col, db_sym, period)
        elapsed = time.time() - t0
        print(f"      {n:,} 行导入 ({elapsed:.1f}s)")
        conn.close()

    # 4. 清理 CSV（可选保留）
    print(f"\n  CSV 文件保留在: {CSV_DIR}")
    print("Done.")


def list_presets():
    print("\n预置下载任务:")
    print(f"  {'任务名':<20s} {'说明'}")
    print(f"  {'─'*60}")
    for name, task in PRESET_TASKS.items():
        n_sym = len(task["symbols"])
        print(f"  {name:<20s} {task['desc']} ({n_sym}品种)")


def main():
    parser = argparse.ArgumentParser(description="TQ DataDownloader 批量下载")
    parser.add_argument("--task", type=str, help="预置任务名 (用 --list 查看)")
    parser.add_argument("--list", action="store_true", help="列出预置任务")
    parser.add_argument("--end", type=str, default="", help="结束日期 YYYYMMDD")
    # 自定义模式
    parser.add_argument("--symbols", type=str, default="", help="TQ合约代码(逗号分隔)")
    parser.add_argument("--dur", type=int, default=0, help="周期秒数 (0=tick, 60=1m, 300=5m)")
    parser.add_argument("--start", type=str, default="", help="起始日期 YYYYMMDD")
    parser.add_argument("--csv", type=str, default="", help="CSV输出文件名")
    parser.add_argument("--table", type=str, default="", help="目标DB表名")
    parser.add_argument("--db-symbol", type=str, default="", help="DB中的品种代码")
    args = parser.parse_args()

    if args.list:
        list_presets()
        return

    end = datetime.strptime(args.end, "%Y%m%d").date() if args.end else date.today()

    if args.task:
        if args.task not in PRESET_TASKS:
            print(f"未知任务: {args.task}")
            list_presets()
            return
        run_preset(args.task, end)
    elif args.symbols:
        # 自定义模式 — 简单的单品种下载
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = str(CSV_DIR / (args.csv or "custom.csv"))
        start = datetime.strptime(args.start, "%Y%m%d").date()
        sym = args.symbols

        api = _connect_tq()
        from tqsdk.tools import DataDownloader
        dl = DataDownloader(api, symbol_list=sym, dur_sec=args.dur,
                            start_dt=start, end_dt=end, csv_file_name=csv_path)
        with closing(api):
            last_print = 0
            while not dl.is_finished():
                api.wait_update()
                now = time.time()
                if now - last_print > 10:
                    short_sym = sym.split(".")[-1]
                    print(f"  {short_sym}: {dl.get_progress():.1f}%")
                    last_print = now
        print(f"  完成: {csv_path}")

        if args.table and args.db_symbol:
            conn, _ = _get_db_conn(args.table)
            if args.dur == 0:
                n = _import_tick_csv(csv_path, conn, args.db_symbol)
            else:
                sym_col = "ts_code" if args.table == "options_min" else "symbol"
                n = _import_kline_csv(csv_path, conn, args.table, sym_col,
                                      args.db_symbol, args.dur)
            print(f"  导入 {n:,} 行")
            conn.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
