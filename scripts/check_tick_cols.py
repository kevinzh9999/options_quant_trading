#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
from datetime import datetime
from tqsdk import TqApi, TqAuth
from tqsdk.tools import DataDownloader
import pandas as pd

api = TqApi(auth=TqAuth(os.getenv("TQ_ACCOUNT",""), os.getenv("TQ_PASSWORD","")))

# tick
csv_t = "tmp/test_tick.csv"
dl = DataDownloader(api, symbol_list="CFFEX.IM2604", dur_sec=0,
                    start_dt=datetime(2026,4,10,9,30,0),
                    end_dt=datetime(2026,4,10,10,0,0),
                    csv_file_name=csv_t)
# 5m kline
csv_k = "tmp/test_opt5m.csv"
dl2 = DataDownloader(api, symbol_list="CFFEX.MO2605-C-7000", dur_sec=300,
                     start_dt=datetime(2026,4,10,0,0,0),
                     end_dt=datetime(2026,4,10,23,59,59),
                     csv_file_name=csv_k)

while not dl.is_finished() or not dl2.is_finished():
    api.wait_update()
api.close()

print("=== Tick CSV ===")
df = pd.read_csv(csv_t)
print(f"列名: {list(df.columns)}")
print(df.head(2).to_string())

print("\n=== Opt 5m CSV ===")
df2 = pd.read_csv(csv_k)
print(f"列名: {list(df2.columns)}")
print(df2.head(2).to_string())
