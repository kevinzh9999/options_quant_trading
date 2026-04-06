"""
download_briefing_history.py
----------------------------
批量下载 Morning Briefing 所需的全部历史数据。

用法：
    python scripts/download_briefing_history.py              # 增量更新（默认）
    python scripts/download_briefing_history.py --full        # 全量下载
    python scripts/download_briefing_history.py --start 20260320  # 从指定日期
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager, get_db


def _get_pro():
    import tushare as ts
    ts.set_token(os.getenv("TUSHARE_TOKEN", ""))
    return ts.pro_api()


def _safe_call(fn, retries=3, sleep_s=7, **kwargs):
    """带重试的Tushare调用。"""
    for attempt in range(retries):
        try:
            time.sleep(sleep_s)
            df = fn(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            msg = str(e)
            if "每分钟" in msg or "频率" in msg or "limit" in msg.lower():
                print(f"    频率限制，等待60秒...")
                time.sleep(60)
            elif attempt < retries - 1:
                print(f"    重试({attempt+1}): {e}")
                time.sleep(10)
            else:
                print(f"    失败: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def _get_max_date(db: DBManager, table: str, date_col: str = "trade_date") -> str:
    """获取表中最大日期。"""
    df = db.query_df(f"SELECT MAX({date_col}) as md FROM {table}")
    if df is not None and not df.empty and df.iloc[0]["md"]:
        return str(df.iloc[0]["md"])
    return ""


def _next_date(date_str: str) -> str:
    """日期+1天。"""
    dt = datetime.strptime(date_str, "%Y%m%d")
    return (dt + timedelta(days=1)).strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# 下载函数
# ---------------------------------------------------------------------------

def download_global_indices(pro, db: DBManager, start: str, end: str):
    """全球指数日线。"""
    print(f"\n=== 全球指数 ({start}~{end}) ===")
    indices = ["XIN9", "SPX", "IXIC", "DJI", "HSI"]
    total = 0
    for ts_code in indices:
        # 按年下载
        sy = int(start[:4])
        ey = int(end[:4])
        code_total = 0
        for year in range(sy, ey + 1):
            s = f"{year}0101" if year > sy else start
            e = f"{year}1231" if year < ey else end
            df = _safe_call(pro.index_global, ts_code=ts_code,
                            start_date=s, end_date=e,
                            fields="trade_date,ts_code,close,pct_chg")
            if not df.empty:
                db.upsert_dataframe("global_index_daily", df)
                code_total += len(df)
                print(f"  {ts_code} {year}: {len(df)}条")
        total += code_total
        print(f"  [{ts_code}] 共 {code_total} 条")
    print(f"  [完成] 全球指数共 {total} 条")


def download_market_breadth(pro, db: DBManager, start: str, end: str):
    """市场宽度（涨跌家数）。"""
    print(f"\n=== 市场宽度 ({start}~{end}) ===")
    # 获取交易日列表
    cal = _safe_call(pro.trade_cal, exchange="SSE",
                     start_date=start, end_date=end,
                     fields="cal_date,is_open", sleep_s=1)
    if cal.empty:
        print("  无法获取交易日历")
        return
    trade_dates = sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())
    print(f"  共 {len(trade_dates)} 个交易日")

    total = 0
    for i, td in enumerate(trade_dates):
        df = _safe_call(pro.daily, trade_date=td,
                        fields="ts_code,pct_chg", sleep_s=0.5)
        if df.empty:
            continue
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
        row = {
            "trade_date": td,
            "advance_count": int((df["pct_chg"] > 0).sum()),
            "decline_count": int((df["pct_chg"] < 0).sum()),
            "limit_up": int((df["pct_chg"] >= 9.9).sum()),
            "limit_down": int((df["pct_chg"] <= -9.9).sum()),
            "total_stocks": len(df),
        }
        row["ad_ratio"] = row["advance_count"] / max(row["decline_count"], 1)
        db.upsert_dataframe("market_breadth", pd.DataFrame([row]))
        total += 1
        if (i + 1) % 50 == 0:
            print(f"  {td}: {i+1}/{len(trade_dates)} 天完成")
    print(f"  [完成] 市场宽度共 {total} 天")


def download_market_turnover(pro, db: DBManager, start: str, end: str):
    """沪深成交额。"""
    print(f"\n=== 市场成交额 ({start}~{end}) ===")
    combined = {}
    for idx_code, col in [("000001.SH", "sh_amount"), ("399001.SZ", "sz_amount")]:
        df = _safe_call(pro.index_daily, ts_code=idx_code,
                        start_date=start, end_date=end,
                        fields="trade_date,amount", sleep_s=2)
        if df.empty:
            continue
        for _, row in df.iterrows():
            td = row["trade_date"]
            amt = float(row["amount"]) / 1e5  # 千元→亿元
            if td not in combined:
                combined[td] = {"trade_date": td, "sh_amount": 0, "sz_amount": 0}
            combined[td][col] = amt

    rows = []
    for td in sorted(combined.keys()):
        r = combined[td]
        r["total_amount"] = r["sh_amount"] + r["sz_amount"]
        rows.append(r)

    if rows:
        db.upsert_dataframe("market_turnover", pd.DataFrame(rows))
    print(f"  [完成] 成交额共 {len(rows)} 天")


def download_northbound(pro, db: DBManager, start: str, end: str):
    """北向资金。"""
    print(f"\n=== 北向资金 ({start}~{end}) ===")
    # 沪港通2014-11-17开始
    if start < "20141117":
        start = "20141117"
    total = 0
    sy, ey = int(start[:4]), int(end[:4])
    for year in range(sy, ey + 1):
        s = f"{year}0101" if year > sy else start
        e = f"{year}1231" if year < ey else end
        df = _safe_call(pro.moneyflow_hsgt, start_date=s, end_date=e,
                        fields="trade_date,north_money,south_money")
        if not df.empty:
            df["north_money"] = pd.to_numeric(df["north_money"], errors="coerce") / 10000  # 万→亿
            df["south_money"] = pd.to_numeric(df["south_money"], errors="coerce") / 10000
            db.upsert_dataframe("northbound_flow", df[["trade_date", "north_money", "south_money"]])
            total += len(df)
            print(f"  {year}: {len(df)}条")
    print(f"  [完成] 北向资金共 {total} 条")


def download_margin(pro, db: DBManager, start: str, end: str):
    """融资融券（按月循环，SSE+SZSE合计）。"""
    print(f"\n=== 融资融券 ({start}~{end}) ===", flush=True)
    if start < "20120104":
        start = "20120104"

    # 生成月份列表
    dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    all_data: dict[str, float] = {}
    month_count = 0
    cur_year = ""

    while dt <= end_dt:
        month_s = dt.strftime("%Y%m%d")
        # 月末
        if dt.month == 12:
            next_m = dt.replace(year=dt.year + 1, month=1, day=1)
        else:
            next_m = dt.replace(month=dt.month + 1, day=1)
        month_e = min(next_m - timedelta(days=1), end_dt).strftime("%Y%m%d")

        try:
            for exch in ["SSE", "SZSE"]:
                df = _safe_call(pro.margin, start_date=month_s, end_date=month_e,
                                exchange_id=exch, fields="trade_date,rzye",
                                sleep_s=1)
                if not df.empty:
                    df["rzye"] = pd.to_numeric(df["rzye"], errors="coerce") / 1e8
                    for _, row in df.iterrows():
                        td = row["trade_date"]
                        all_data[td] = all_data.get(td, 0) + float(row["rzye"])
        except Exception as e:
            print(f"  [WARN] {month_s}: {e}", flush=True)

        month_count += 1
        yr = month_s[:4]
        if yr != cur_year:
            cur_year = yr
            print(f"  {yr}: {len(all_data)}天已收集", flush=True)

        dt = next_m

    # 计算日变化并批量写入
    rows = []
    sorted_dates = sorted(all_data.keys())
    for i, td in enumerate(sorted_dates):
        rzye = all_data[td]
        rzye_chg = rzye - all_data[sorted_dates[i - 1]] if i > 0 else 0
        rows.append({"trade_date": td, "rzye": rzye, "rzye_chg": rzye_chg})

    if rows:
        batch = 500
        for j in range(0, len(rows), batch):
            db.upsert_dataframe("margin_data", pd.DataFrame(rows[j:j + batch]))
    print(f"  [完成] 融资融券共 {len(rows)} 天 ({month_count}个月)", flush=True)


def download_etf_share(pro, db: DBManager, start: str, end: str):
    """ETF份额（按年循环，逐ETF下载）。"""
    print(f"\n=== ETF份额 ({start}~{end}) ===", flush=True)
    etfs = [("510300.SH", "20150101"), ("560010.SH", "20240101")]
    total = 0
    for ts_code, earliest in etfs:
        s = max(start, earliest)
        if s > end:
            continue
        sy, ey = int(s[:4]), int(end[:4])
        code_rows = []
        for year in range(sy, ey + 1):
            ys = f"{year}0101" if year > sy else s
            ye = f"{year}1231" if year < ey else end
            try:
                df = _safe_call(pro.fund_share, ts_code=ts_code,
                                start_date=ys, end_date=ye)
                if not df.empty:
                    df["fd_share"] = pd.to_numeric(df["fd_share"], errors="coerce")
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    code_rows.extend(df[["trade_date", "ts_code", "fd_share"]].to_dict("records"))
                    print(f"  {ts_code} {year}: {len(df)}条", flush=True)
            except Exception as e:
                print(f"  [WARN] {ts_code} {year}: {e}", flush=True)

        if code_rows:
            cdf = pd.DataFrame(code_rows).sort_values("trade_date").reset_index(drop=True)
            cdf["fd_share_chg"] = cdf["fd_share"].diff().fillna(0)
            db.upsert_dataframe("etf_share", cdf)
            total += len(cdf)
    print(f"  [完成] ETF份额共 {total} 条", flush=True)


def download_fut_holding(pro, db: DBManager, start: str, end: str):
    """期货持仓排名汇总（按日循环，每天聚合IM/IF/IH/IC）。"""
    print(f"\n=== 期货持仓 ({start}~{end}) ===", flush=True)
    if start < "20150105":
        start = "20150105"
    cal = _safe_call(pro.trade_cal, exchange="CFFEX",
                     start_date=start, end_date=end,
                     fields="cal_date,is_open", sleep_s=1)
    if cal.empty:
        print("  无法获取CFFEX交易日历", flush=True)
        return
    trade_dates = sorted(cal[cal["is_open"] == 1]["cal_date"].tolist())
    print(f"  共 {len(trade_dates)} 个交易日", flush=True)

    total = 0
    batch_rows = []
    for i, td in enumerate(trade_dates):
        try:
            df = _safe_call(pro.fut_holding, trade_date=td, exchange="CFFEX",
                            sleep_s=0.5)
        except Exception as e:
            print(f"  [WARN] {td}: {e}", flush=True)
            continue
        if df.empty:
            continue
        # 按品种前缀汇总（用.copy()避免SettingWithCopyWarning）
        for prefix in ["IM", "IF", "IH", "IC"]:
            mask = df["symbol"].astype(str).str.startswith(prefix)
            sub = df[mask].copy()
            if sub.empty:
                continue
            for col in ["long_hld", "short_hld", "long_chg", "short_chg"]:
                if col in sub.columns:
                    sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
            row = {
                "trade_date": td,
                "symbol": prefix,
                "total_long": float(sub["long_hld"].sum()),
                "total_short": float(sub["short_hld"].sum()),
                "long_chg": float(sub["long_chg"].sum()) if "long_chg" in sub.columns else 0,
                "short_chg": float(sub["short_chg"].sum()) if "short_chg" in sub.columns else 0,
            }
            row["net_position"] = row["total_long"] - row["total_short"]
            row["net_chg"] = row["long_chg"] - row["short_chg"]
            batch_rows.append(row)
            total += 1

        # 每50天批量写入+打印进度
        if (i + 1) % 50 == 0:
            if batch_rows:
                db.upsert_dataframe("fut_holding_summary", pd.DataFrame(batch_rows))
                batch_rows = []
            print(f"  {td}: {i+1}/{len(trade_dates)} 天完成 ({total}条)", flush=True)

    # 写入剩余
    if batch_rows:
        db.upsert_dataframe("fut_holding_summary", pd.DataFrame(batch_rows))
    print(f"  [完成] 期货持仓共 {total} 条", flush=True)


def download_option_pcr(db: DBManager, start: str, end: str):
    """从本地 options_daily 计算期权PCR（不调Tushare）。"""
    print(f"\n=== 期权PCR ({start}~{end}) ===", flush=True)
    # 一次性读取范围内全量数据
    df = db.query_df(
        "SELECT trade_date, ts_code, volume, oi FROM options_daily "
        "WHERE trade_date >= ? AND trade_date <= ? ",
        (start, end),
    )
    if df is None or df.empty:
        print("  options_daily 无数据", flush=True)
        return

    # 解析品种和方向
    df["product"] = df["ts_code"].apply(
        lambda x: "MO" if str(x).startswith("MO") else
        ("IO" if str(x).startswith("IO") else
         ("HO" if str(x).startswith("HO") else None)))
    df["cp"] = df["ts_code"].apply(
        lambda x: "C" if "-C-" in str(x) else ("P" if "-P-" in str(x) else None))
    df = df.dropna(subset=["product", "cp"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)

    rows = []
    for (td, prod), grp in df.groupby(["trade_date", "product"]):
        c = grp[grp["cp"] == "C"]
        p = grp[grp["cp"] == "P"]
        cv = float(c["volume"].sum())
        pv = float(p["volume"].sum())
        co = float(c["oi"].sum())
        po = float(p["oi"].sum())
        rows.append({
            "trade_date": td,
            "product": prod,
            "call_volume": cv,
            "put_volume": pv,
            "total_volume": cv + pv,
            "volume_pcr": pv / cv if cv > 0 else None,
            "call_oi": co,
            "put_oi": po,
            "total_oi": co + po,
            "oi_pcr": po / co if co > 0 else None,
        })

    if rows:
        batch = 1000
        rdf = pd.DataFrame(rows)
        for j in range(0, len(rdf), batch):
            db.upsert_dataframe("option_pcr_daily", rdf.iloc[j:j + batch])
    print(f"  [完成] 期权PCR共 {len(rows)} 条", flush=True)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="下载Briefing历史数据")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--full", action="store_true", help="全量下载（从最早可用）")
    parser.add_argument("--update", action="store_true",
                        help="增量更新（从各表MAX日期+1开始，默认行为）")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="跳过的数据集（global breadth turnover north margin etf fut）")
    args = parser.parse_args()

    db = get_db()
    db.initialize_tables()
    pro = _get_pro()
    today = datetime.now().strftime("%Y%m%d")

    # 确定起始日期
    if args.start:
        default_start = args.start
    elif args.full:
        default_start = "20100101"
    else:
        default_start = None  # 增量模式，每个表单独判断

    t0 = time.time()

    datasets = [
        ("global", "global_index_daily", "20100101",
         lambda s, e: download_global_indices(pro, db, s, e)),
        ("breadth", "market_breadth", "20100104",
         lambda s, e: download_market_breadth(pro, db, s, e)),
        ("turnover", "market_turnover", "20100104",
         lambda s, e: download_market_turnover(pro, db, s, e)),
        ("north", "northbound_flow", "20141117",
         lambda s, e: download_northbound(pro, db, s, e)),
        ("margin", "margin_data", "20120104",
         lambda s, e: download_margin(pro, db, s, e)),
        ("etf", "etf_share", "20150101",
         lambda s, e: download_etf_share(pro, db, s, e)),
        ("fut", "fut_holding_summary", "20150105",
         lambda s, e: download_fut_holding(pro, db, s, e)),
        ("pcr", "option_pcr_daily", "20190101",
         lambda s, e: download_option_pcr(db, s, e)),
    ]

    for name, table, earliest, fn in datasets:
        if name in args.skip:
            print(f"\n  [跳过] {name}")
            continue
        if default_start:
            start = max(default_start, earliest)
        else:
            # 增量模式
            max_d = _get_max_date(db, table)
            start = _next_date(max_d) if max_d else earliest
            if start > today:
                print(f"\n  [{name}] 已是最新 ({max_d})")
                continue
        try:
            fn(start, today)
        except Exception as e:
            print(f"\n  [{name}] 错误: {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f" 下载完成 | 耗时 {elapsed/60:.1f} 分钟")

    # 显示各表状态
    for _, table, _, _ in datasets:
        try:
            df = db.query_df(
                f"SELECT COUNT(*) as n, MIN(trade_date) as mn, "
                f"MAX(trade_date) as mx FROM {table}"
            )
            if df is not None and not df.empty:
                r = df.iloc[0]
                print(f"  {table:25s}: {r['n']:>6} 条  {r['mn']}~{r['mx']}")
        except Exception:
            print(f"  {table:25s}: 空")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
