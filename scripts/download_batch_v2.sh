#!/bin/bash
# 串联下载：IC tick补足 + ETF 5m
set -e
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python
cd "$(dirname "$0")/.."

echo "=== [1/3] IC tick 补足 (2015-04-16 ~ 2024-01-01) ==="
$PYTHON scripts/download_pro_v2.py --symbols KQ.m@CFFEX.IC --dur 0 \
    --start 20150416 --end 20240101 \
    --csv ic_tick_early.csv \
    --table futures_tick --db-symbol IC

echo ""
echo "=== [2/3] ETF 5m K线 (2023-01 ~ 今天) → etf_data.db ==="
$PYTHON -c "
import sqlite3
from pathlib import Path
from data.storage.schemas import ETF_MIN_SQL
db_path = Path('data/storage/etf_data.db')
conn = sqlite3.connect(str(db_path), timeout=30)
conn.execute('PRAGMA journal_mode=WAL')
conn.executescript(ETF_MIN_SQL)
conn.commit()
conn.close()
print('etf_data.db 已创建')
"

# 逐个下载 ETF 5m（DataDownloader 不支持不同起始日期的并行，但 ETF 起始日相同可以并行）
$PYTHON scripts/download_pro_v2.py --symbols SSE.512100 --dur 300 \
    --start 20230101 --end 20260403 --csv etf_512100_5m.csv
$PYTHON -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/storage/etf_data.db', timeout=30)
df = pd.read_csv('tmp/downloads/etf_512100_5m.csv')
df = df[df['close']>0].copy()
df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
rows = list(zip(['512100']*len(df), df['datetime'].tolist(), [300]*len(df),
    df['open'].tolist(), df['high'].tolist(), df['low'].tolist(), df['close'].tolist(),
    df['volume'].tolist(), [None]*len(df)))
conn.executemany('INSERT OR IGNORE INTO etf_min (symbol,datetime,period,open,high,low,close,volume,open_interest) VALUES (?,?,?,?,?,?,?,?,?)', rows)
conn.commit()
print(f'512100: {len(rows):,} 行')
conn.close()
"

$PYTHON scripts/download_pro_v2.py --symbols SSE.510500 --dur 300 \
    --start 20230101 --end 20260403 --csv etf_510500_5m.csv
$PYTHON -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/storage/etf_data.db', timeout=30)
df = pd.read_csv('tmp/downloads/etf_510500_5m.csv')
df = df[df['close']>0].copy()
df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
rows = list(zip(['510500']*len(df), df['datetime'].tolist(), [300]*len(df),
    df['open'].tolist(), df['high'].tolist(), df['low'].tolist(), df['close'].tolist(),
    df['volume'].tolist(), [None]*len(df)))
conn.executemany('INSERT OR IGNORE INTO etf_min (symbol,datetime,period,open,high,low,close,volume,open_interest) VALUES (?,?,?,?,?,?,?,?,?)', rows)
conn.commit()
print(f'510500: {len(rows):,} 行')
conn.close()
"

$PYTHON scripts/download_pro_v2.py --symbols SSE.510300 --dur 300 \
    --start 20230101 --end 20260403 --csv etf_510300_5m.csv
$PYTHON -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/storage/etf_data.db', timeout=30)
df = pd.read_csv('tmp/downloads/etf_510300_5m.csv')
df = df[df['close']>0].copy()
df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
rows = list(zip(['510300']*len(df), df['datetime'].tolist(), [300]*len(df),
    df['open'].tolist(), df['high'].tolist(), df['low'].tolist(), df['close'].tolist(),
    df['volume'].tolist(), [None]*len(df)))
conn.executemany('INSERT OR IGNORE INTO etf_min (symbol,datetime,period,open,high,low,close,volume,open_interest) VALUES (?,?,?,?,?,?,?,?,?)', rows)
conn.commit()
print(f'510300: {len(rows):,} 行')
conn.close()
"

$PYTHON scripts/download_pro_v2.py --symbols SSE.510050 --dur 300 \
    --start 20230101 --end 20260403 --csv etf_510050_5m.csv
$PYTHON -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/storage/etf_data.db', timeout=30)
df = pd.read_csv('tmp/downloads/etf_510050_5m.csv')
df = df[df['close']>0].copy()
df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
rows = list(zip(['510050']*len(df), df['datetime'].tolist(), [300]*len(df),
    df['open'].tolist(), df['high'].tolist(), df['low'].tolist(), df['close'].tolist(),
    df['volume'].tolist(), [None]*len(df)))
conn.executemany('INSERT OR IGNORE INTO etf_min (symbol,datetime,period,open,high,low,close,volume,open_interest) VALUES (?,?,?,?,?,?,?,?,?)', rows)
conn.commit()
print(f'510050: {len(rows):,} 行')
conn.close()
"

echo ""
echo "=== [3/3] 校验 ==="
$PYTHON -c "
import sqlite3
from pathlib import Path

# tick
tick_db = sqlite3.connect('data/storage/tick_data.db', timeout=30)
print('=== tick_data.db ===')
for r in tick_db.execute('SELECT symbol, COUNT(*), MIN(substr(datetime,1,10)), MAX(substr(datetime,1,10)) FROM futures_tick GROUP BY symbol ORDER BY symbol').fetchall():
    print(f'  {r[0]}: {r[1]:>12,} ticks  {r[2]} ~ {r[3]}')
tick_db.close()

# etf
etf_path = Path('data/storage/etf_data.db')
if etf_path.exists():
    etf_db = sqlite3.connect(str(etf_path), timeout=30)
    print('\n=== etf_data.db ===')
    for r in etf_db.execute('SELECT symbol, period, COUNT(*), MIN(datetime), MAX(datetime) FROM etf_min GROUP BY symbol, period ORDER BY symbol').fetchall():
        print(f'  {r[0]} {r[1]//60}m: {r[2]:>8,} 行  {r[3][:10]} ~ {r[4][:10]}')
    print(f'  文件大小: {etf_path.stat().st_size/1e6:.1f} MB')
    etf_db.close()
"

echo ""
echo "全部完成: $(date)"
