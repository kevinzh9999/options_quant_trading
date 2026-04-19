#!/usr/bin/env python3
"""对比实盘monitor记录的bar值和归档index_min，找出差异。
用法：python scripts/verify_bar_audit.py [--date YYYYMMDD]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from data.storage.db_manager import get_db

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="日期YYYYMMDD，默认今天")
    args = parser.parse_args()

    td = args.date or pd.Timestamp.now().strftime("%Y%m%d")
    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"

    # 读实盘记录
    # 实盘monitor的_tmp_dir默认是项目根目录下的tmp/
    audit_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "bar_audit.csv")
    if not os.path.exists(audit_path):
        print(f"  {audit_path} 不存在"); return

    audit = pd.read_csv(audit_path)
    audit = audit[audit["datetime"].str.startswith(date_dash)]
    if audit.empty:
        print(f"  {td} 无实盘记录"); return

    # 读归档
    db = get_db()
    SPOT = {"IM": "000852", "IC": "000905", "IF": "000300", "IH": "000016"}

    for sym in audit["symbol"].unique():
        spot = SPOT.get(sym, sym)
        archived = db.query_df(
            f"SELECT datetime, open, high, low, close, volume FROM index_min "
            f"WHERE symbol='{spot}' AND period=300 AND datetime >= '{date_dash}' "
            f"AND datetime < '{date_dash[:8]}{int(date_dash[8:])+1:02d}' ORDER BY datetime"
        )
        if archived is None or archived.empty:
            print(f"  {sym}: 归档无数据"); continue
        for c in ["open","high","low","close","volume"]:
            archived[c] = archived[c].astype(float)

        sym_audit = audit[audit["symbol"] == sym].copy()
        sym_audit["datetime"] = sym_audit["datetime"].str.strip()

        print(f"\n{'='*70}")
        print(f" {sym} | {td} | 实盘 vs 归档对比")
        print(f"{'='*70}")

        diffs = []
        for _, a in sym_audit.iterrows():
            dt = a["datetime"]
            arc = archived[archived["datetime"] == dt]
            if arc.empty:
                print(f"  {dt}: 实盘有，归档无")
                continue
            ar = arc.iloc[0]
            d_close = abs(a["close"] - ar["close"])
            d_high = abs(a["high"] - ar["high"])
            d_low = abs(a["low"] - ar["low"])
            if d_close > 0.01 or d_high > 0.01 or d_low > 0.01:
                diffs.append((dt, a["close"], ar["close"], d_close))
                print(f"  ⚠ {dt}: close 实盘={a['close']:.4f} 归档={ar['close']:.4f} Δ={d_close:.4f}")

        if not diffs:
            print(f"  ✓ 全部 {len(sym_audit)} 根bar完全一致（精度0.01）")
        else:
            print(f"\n  差异: {len(diffs)}/{len(sym_audit)} 根bar")
            avg_d = sum(d[3] for d in diffs) / len(diffs)
            max_d = max(d[3] for d in diffs)
            print(f"  平均差异: {avg_d:.4f}  最大差异: {max_d:.4f}")

if __name__ == "__main__":
    main()
