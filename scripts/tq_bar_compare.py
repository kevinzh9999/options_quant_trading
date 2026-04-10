#!/usr/bin/env python3
"""TqBacktest逐bar数据对比：打印TQ replay和归档数据的close差异。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONUNBUFFERED'] = '1'

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from tqsdk import TqApi, TqBacktest, BacktestFinished, TqAuth
from datetime import datetime
import pandas as pd

dates = ['20260407', '20260408', '20260409', '20260410']

for td in dates:
    y, m, d = int(td[:4]), int(td[4:6]), int(td[6:8])
    date_dash = f"{y}-{m:02d}-{d:02d}"

    auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))
    api = TqApi(backtest=TqBacktest(
        start_dt=datetime(y, m, d, 0, 0, 0),
        end_dt=datetime(y, m, d, 23, 59, 59),
    ), auth=auth)

    k5_im = api.get_kline_serial("SSE.000852", 300, 200)
    k5_ic = api.get_kline_serial("SSE.000905", 300, 200)

    last_dt = {}
    tq_bars = {'IM': [], 'IC': []}

    try:
        while True:
            api.wait_update()
            for sym, k5 in [('IM', k5_im), ('IC', k5_ic)]:
                if api.is_changing(k5) and len(k5) >= 2:
                    cdt = int(k5.iloc[-2]['datetime'])
                    if last_dt.get(sym) != cdt:
                        last_dt[sym] = cdt
                        completed = k5.iloc[-2]
                        ts = pd.Timestamp(cdt, unit='ns')
                        bj = ts + pd.Timedelta(hours=8)
                        if bj.strftime('%Y-%m-%d') == date_dash:
                            tq_bars[sym].append({
                                'bj': bj.strftime('%H:%M'),
                                'close': float(completed['close']),
                                'volume': float(completed['volume']),
                            })
    except BacktestFinished:
        pass
    finally:
        api.close()

    from data.storage.db_manager import get_db
    db = get_db()

    print(f"\n{'='*50}")
    print(f"  {td} bar close对比")
    print(f"{'='*50}")

    for sym, spot in [('IM', '000852'), ('IC', '000905')]:
        archived = db.query_df(
            f"SELECT datetime, close, volume FROM index_min "
            f"WHERE symbol='{spot}' AND period=300 "
            f"AND datetime LIKE '{date_dash}%' ORDER BY datetime"
        )
        if archived is None or archived.empty:
            print(f"  {sym}: 归档无数据")
            continue

        arch_dict = {}
        for _, r in archived.iterrows():
            dt_s = str(r['datetime'])
            h = int(dt_s[11:13]) + 8
            bj_key = f"{h:02d}:{dt_s[14:16]}"
            arch_dict[bj_key] = float(r['close'])

        tq_dict = {b['bj']: b['close'] for b in tq_bars[sym]}

        diffs = []
        for t in sorted(arch_dict.keys()):
            a = arch_dict.get(t)
            q = tq_dict.get(t)
            if a and q and abs(q - a) > 0.01:
                diffs.append((t, a, q, q - a))

        if diffs:
            print(f"  {sym}: {len(diffs)}/{len(arch_dict)} bar有差异")
            for t, a, q, dc in diffs:
                print(f"    {t} 归档={a:.1f} TQ={q:.1f} Δ={dc:+.1f}")
        else:
            print(f"  {sym}: {len(arch_dict)}根bar全部一致 ✓")
