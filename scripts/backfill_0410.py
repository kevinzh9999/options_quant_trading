#!/usr/bin/env python3
"""用DataDownloader补4/10的futures_tick和options_min。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from datetime import datetime
from tqsdk import TqApi, TqAuth
from tqsdk.tools import DataDownloader
import sqlite3, pandas as pd

auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))
api = TqApi(auth=auth)

start_dt = datetime(2026, 4, 10, 0, 0, 0)
end_dt = datetime(2026, 4, 10, 23, 59, 59)
date_dash = "2026-04-10"
tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")

tasks = []

# 1. Tick: IM/IC/IF主连
for sym, contract in [("IM", "CFFEX.IM2604"), ("IC", "CFFEX.IC2604"), ("IF", "CFFEX.IF2604")]:
    csv = os.path.join(tmp_dir, f"tick_{sym}_20260410.csv")
    print(f"下载 {sym} tick...")
    dl = DataDownloader(api, symbol_list=contract, dur_sec=0,
                        start_dt=start_dt, end_dt=end_dt, csv_file_name=csv)
    tasks.append(("tick", sym, csv, dl))

# 2. Options 5m: 从options_daily获取4/10有交易的合约
db_opt = sqlite3.connect("data/storage/options_data.db", timeout=30)
rows = db_opt.execute(
    "SELECT DISTINCT ts_code FROM options_daily WHERE trade_date='20260410' AND volume > 0"
).fetchall()
db_opt.close()

opt_contracts = [r[0].replace(".CFX", "") for r in rows]  # 去掉Tushare后缀
print(f"4/10有交易的期权合约: {len(opt_contracts)} 个")

# 先完成tick下载
print(f"\n等待 {len(tasks)} 个tick任务完成...")
while not all(dl.is_finished() for _, _, _, dl in tasks):
    api.wait_update()

api.close()
print("Tick下载完成")

# Options分批下载（每批50个）
BATCH = 50
for i in range(0, len(opt_contracts), BATCH):
    batch = opt_contracts[i:i+BATCH]
    print(f"\n期权批次 {i//BATCH+1}/{(len(opt_contracts)+BATCH-1)//BATCH} ({len(batch)}个)...")
    api2 = TqApi(auth=TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", "")))
    batch_tasks = []
    for ts_code in batch:
        tq_sym = f"CFFEX.{ts_code}"
        csv = os.path.join(tmp_dir, f"opt_{ts_code}_20260410.csv")
        try:
            dl = DataDownloader(api2, symbol_list=tq_sym, dur_sec=300,
                                start_dt=start_dt, end_dt=end_dt, csv_file_name=csv)
            batch_tasks.append(("opt", ts_code, csv, dl))
            tasks.append(("opt", ts_code, csv, dl))
        except Exception:
            pass
    while batch_tasks and not all(dl.is_finished() for _, _, _, dl in batch_tasks):
        api2.wait_update()
        done = sum(1 for _, _, _, dl in batch_tasks if dl.is_finished())
        print(f"\r  {done}/{len(batch_tasks)}", end="", flush=True)
    api2.close()
    print()
print(f"\n下载完成，导入数据库...")

# 导入tick
db_tick = sqlite3.connect("data/storage/tick_data.db", timeout=30)
db_tick.execute("PRAGMA journal_mode=WAL")
tick_total = 0

for task_type, sym, csv, dl in tasks:
    if task_type != "tick" or not os.path.exists(csv):
        continue
    try:
        df = pd.read_csv(csv)
        if df.empty:
            os.remove(csv); continue
        # DataDownloader列名带合约前缀，去掉
        df.columns = [c.split(".")[-1] if "." in c else c for c in df.columns]
        # datetime是BJ时间，转UTC
        df["datetime"] = pd.to_datetime(df["datetime"]) - pd.Timedelta(hours=8)
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        df = df[df["datetime"].str.startswith(date_dash)]
        if df.empty:
            os.remove(csv); continue
        n = len(df)
        rows = list(zip(
            [sym]*n, df["datetime"].tolist(),
            df["last_price"].tolist(),
            df.get("average", pd.Series([0]*n)).tolist(),
            df.get("highest", pd.Series([0]*n)).tolist(),
            df.get("lowest", pd.Series([0]*n)).tolist(),
            df.get("bid_price1", pd.Series([0]*n)).tolist(),
            df.get("bid_volume1", pd.Series([0]*n)).tolist(),
            df.get("ask_price1", pd.Series([0]*n)).tolist(),
            df.get("ask_volume1", pd.Series([0]*n)).tolist(),
            df.get("volume", pd.Series([0]*n)).tolist(),
            df.get("amount", pd.Series([0]*n)).tolist(),
            df.get("open_interest", pd.Series([0]*n)).tolist(),
        ))
        db_tick.executemany("INSERT OR IGNORE INTO futures_tick VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        db_tick.commit()
        tick_total += n
        print(f"  tick {sym}: {n} 条")
    except Exception as e:
        print(f"  tick {sym} 失败: {e}")
    if os.path.exists(csv): os.remove(csv)

db_tick.close()

# 导入options_min
db_opt = sqlite3.connect("data/storage/options_data.db", timeout=30)
db_opt.execute("PRAGMA journal_mode=WAL")
opt_total = 0

for task_type, ts, csv, dl in tasks:
    if task_type != "opt" or not os.path.exists(csv):
        continue
    try:
        df = pd.read_csv(csv)
        if df.empty:
            os.remove(csv); continue
        # 去掉合约前缀
        df.columns = [c.split(".")[-1] if "." in c else c for c in df.columns]
        df["datetime"] = pd.to_datetime(df["datetime"]) - pd.Timedelta(hours=8)
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df = df[df["datetime"].str.startswith(date_dash)]
        if df.empty:
            os.remove(csv); continue
        n = len(df)
        oi = df["close_oi"].tolist() if "close_oi" in df.columns else [0]*n
        rows = list(zip(
            [ts]*n, df["datetime"].tolist(), [300]*n,
            df["open"].tolist(), df["high"].tolist(),
            df["low"].tolist(), df["close"].tolist(),
            df["volume"].tolist(), oi,
        ))
        db_opt.executemany(
            "INSERT OR IGNORE INTO options_min (ts_code,datetime,period,open,high,low,close,volume,open_interest) "
            "VALUES (?,?,?,?,?,?,?,?,?)", rows)
        db_opt.commit()
        opt_total += n
    except Exception:
        pass
    if os.path.exists(csv): os.remove(csv)

db_opt.close()
print(f"\n导入完成: tick={tick_total} 条, options_min={opt_total} 条")
