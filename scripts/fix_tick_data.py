"""
修复 tick_data.db 中不完整和时间标准不一致的数据。

问题：
1. 4/13, 4/14, 4/16: 只有10000条（get_tick_serial上限），缺失上午
2. 4/15: DataDownloader下载但存入了BJ时间（应为UTC）

方案：删掉这些天的数据，用DataDownloader重新下载，转UTC后入库。
"""

import os, sys, time, sqlite3, tempfile
from pathlib import Path
from datetime import date, datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
from tqsdk import TqApi, TqAuth
from tqsdk.tools import DataDownloader

DB_PATH = str(ROOT / "data" / "storage" / "tick_data.db")

# 需要修复的日期
DATES_TO_FIX = [
    date(2026, 4, 13),
    date(2026, 4, 14),
    date(2026, 4, 15),
    date(2026, 4, 16),
]

SYMBOLS = [
    ("IM", "KQ.m@CFFEX.IM"),
    ("IC", "KQ.m@CFFEX.IC"),
    ("IF", "KQ.m@CFFEX.IF"),
]


def delete_bad_data(conn, symbols, dates):
    """删掉指定日期的tick数据（UTC和BJ两种前缀都删）。"""
    for sym in symbols:
        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            # UTC时间的数据前缀
            n1 = conn.execute(
                "DELETE FROM futures_tick WHERE symbol=? AND datetime >= ? AND datetime < ?",
                (sym, d_str, (d + timedelta(days=1)).strftime("%Y-%m-%d"))
            ).rowcount
            conn.commit()
            print(f"  删除 {sym} {d_str}: {n1} rows")


def download_and_insert(api, sym, tq_sym, d, conn):
    """用DataDownloader下载一天的tick，转UTC后写入DB。"""
    csv_path = os.path.join(tempfile.gettempdir(), f"tick_{sym}_{d.strftime('%Y%m%d')}.csv")

    dl = DataDownloader(api, symbol_list=tq_sym, dur_sec=0,
                        start_dt=d, end_dt=d, csv_file_name=csv_path)

    deadline = time.time() + 120
    while not dl.is_finished():
        api.wait_update(deadline=deadline)
        if time.time() > deadline:
            print(f"    TIMEOUT downloading {sym} {d}")
            return 0

    if not dl.is_finished():
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"    {sym} {d}: 空数据（非交易日？）")
        return 0

    # DataDownloader输出的datetime列是北京时间纳秒字符串，转为UTC
    dt_col = df.columns[0]  # 第一列是datetime
    df["dt_parsed"] = pd.to_datetime(df[dt_col])
    # BJ -> UTC: 减8小时
    df["dt_utc"] = df["dt_parsed"] - timedelta(hours=8)
    df["dt_str"] = df["dt_utc"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    # 过滤只保留当天UTC数据 (d-1 21:00 UTC到d 07:00 UTC范围，CFFEX只有日盘01:30-07:00)
    df = df[df["dt_utc"].dt.date == d]

    n = len(df)
    if n == 0:
        print(f"    {sym} {d}: 转UTC后无数据")
        return 0

    # 构造插入行
    def _c(name, dtype=float, default=0):
        if name in df.columns:
            return df[name].fillna(default).astype(dtype).tolist()
        return [default] * n

    # 列名映射（DataDownloader CSV列名可能带合约前缀）
    cols = df.columns.tolist()
    def _find(keywords):
        for c in cols:
            if all(k in c.lower() for k in keywords):
                return c
        return None

    last_price_col = _find(["last_price"]) or "last_price"
    average_col = _find(["average"]) or "average"
    highest_col = _find(["highest"]) or "highest"
    lowest_col = _find(["lowest"]) or "lowest"
    bid_price_col = _find(["bid_price1"]) or "bid_price1"
    bid_vol_col = _find(["bid_volume1"]) or "bid_volume1"
    ask_price_col = _find(["ask_price1"]) or "ask_price1"
    ask_vol_col = _find(["ask_volume1"]) or "ask_volume1"
    volume_col = _find(["volume"]) or "volume"
    amount_col = _find(["amount"]) or "amount"
    oi_col = _find(["open_interest"]) or "open_interest"

    rows = list(zip(
        [sym] * n,
        df["dt_str"].tolist(),
        _c(last_price_col),
        _c(average_col),
        _c(highest_col),
        _c(lowest_col),
        _c(bid_price_col),
        _c(bid_vol_col, int),
        _c(ask_price_col),
        _c(ask_vol_col, int),
        _c(volume_col, int),
        _c(amount_col),
        _c(oi_col, int),
    ))

    conn.executemany(
        "INSERT OR IGNORE INTO futures_tick VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()

    # 清理临时文件
    os.remove(csv_path)
    return n


def verify_all_dates(conn, symbols, n_days=30):
    """校验最近n_days的数据完整性。"""
    print("\n" + "=" * 70)
    print("数据完整性校验")
    print("=" * 70)

    issues = []
    for sym in symbols:
        rows = conn.execute("""
            SELECT substr(datetime, 1, 10) as d, COUNT(*) as cnt,
                   MIN(substr(datetime, 12, 8)) as mn,
                   MAX(substr(datetime, 12, 8)) as mx
            FROM futures_tick
            WHERE symbol = ? AND datetime >= date('now', ? || ' days')
            GROUP BY d ORDER BY d
        """, (sym, str(-n_days))).fetchall()

        print(f"\n{sym}:")
        for r in rows:
            d, cnt, mn, mx = r
            # 完整日应有 ~22K-28K rows，起始01:29-01:30 UTC
            is_ok = cnt >= 20000 and mn < "02:00"
            flag = "OK" if is_ok else "ISSUE"
            if not is_ok:
                issues.append((sym, d, cnt, mn, mx))
            print(f"  {d}: {cnt:>6} rows  ({mn} ~ {mx})  [{flag}]")

    if issues:
        print(f"\n发现 {len(issues)} 个问题:")
        for sym, d, cnt, mn, mx in issues:
            print(f"  {sym} {d}: {cnt} rows ({mn}~{mx})")
    else:
        print("\n全部通过！")

    return issues


def main():
    print("=" * 70)
    print("Tick数据修复脚本")
    print(f"修复日期: {', '.join(d.strftime('%Y-%m-%d') for d in DATES_TO_FIX)}")
    print(f"品种: {', '.join(s for s,_ in SYMBOLS)}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # Step 1: 删除不完整数据
    print("\n[1/3] 删除不完整数据...")
    delete_bad_data(conn, [s for s, _ in SYMBOLS], DATES_TO_FIX)

    # Step 2: 用DataDownloader重新下载
    print("\n[2/3] DataDownloader下载完整tick数据...")
    api = TqApi(auth=TqAuth(os.getenv("TQ_ACCOUNT"), os.getenv("TQ_PASSWORD")))

    for sym, tq_sym in SYMBOLS:
        for d in DATES_TO_FIX:
            print(f"  下载 {sym} {d}...", end=" ", flush=True)
            n = download_and_insert(api, sym, tq_sym, d, conn)
            print(f"{n} rows")

    api.close()

    # Step 3: 校验
    print("\n[3/3] 校验数据完整性...")
    issues = verify_all_dates(conn, [s for s, _ in SYMBOLS], n_days=30)

    conn.close()
    print("\n修复完成。" if not issues else f"\n修复后仍有 {len(issues)} 个问题需检查。")


if __name__ == "__main__":
    main()
