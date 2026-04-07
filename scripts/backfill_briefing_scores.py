#!/usr/bin/env python3
"""
backfill_briefing_scores.py
----------------------------
回算历史 Morning Briefing 评分和 d_override，写入 morning_briefing 表。

用于验证 d_override 对日内策略的实际贡献。
所有数据严格用 T-1 日（或更早）截断，无前瞻偏差。

用法:
    python scripts/backfill_briefing_scores.py --start 20250516 --end 20260403
    python scripts/backfill_briefing_scores.py --start 20250516 --end 20260403 --dry-run
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import get_db

# 复用 briefing 的评分函数（不调用 Tushare，纯本地 DB）
from scripts.morning_briefing import (
    _get_recent_trade_dates,
    _get_global_indices,
    _get_market_breadth,
    _get_market_volume,
    _get_vol_environment,
    _get_price_position,
    _get_hurst,
    _get_north_money,
    _get_margin,
    _get_pcr,
    _get_etf_share,
    _get_fut_holding,
    _calc_score,
    _score_to_direction,
    _calc_d_override,
)


class _NullPro:
    """Tushare pro 的空替身，所有调用返回空 DataFrame 让函数 fallback 到 DB。"""
    def __getattr__(self, name):
        def _noop(**kwargs):
            return pd.DataFrame()
        return _noop


def backfill_one_day(db, td: str, prev_td: str, verbose: bool = False) -> dict | None:
    """回算一天的 briefing 评分。严格用 prev_td（T-1）的数据。"""
    try:
        # 所有数据用 prev_td（上一交易日），和实盘 briefing 一致
        _pro = _NullPro()
        global_idx = _get_global_indices(_pro, prev_td, db=db)
        breadth = _get_market_breadth(_pro, prev_td, db=db)
        volume = _get_market_volume(_pro, prev_td, db=db)
        vol_env = _get_vol_environment(db)
        price_pos = _get_price_position(db, prev_td)
        north = _get_north_money(_pro, prev_td, db=db)
        margin = _get_margin(_pro, prev_td, db=db)
        pcr_data = _get_pcr(db, prev_td)
        etf = _get_etf_share(_pro, prev_td)
        fut_hold = _get_fut_holding(_pro, prev_td)

        score, reasons = _calc_score(
            global_idx, breadth, volume, vol_env, price_pos,
            north=north, margin=margin, pcr_data=pcr_data,
            etf=etf, fut_hold=fut_hold, target_date=td,
        )
        direction, confidence = _score_to_direction(score)
        d_override = _calc_d_override(direction, confidence)

        result = {
            "trade_date": td,
            "direction": direction,
            "confidence": confidence,
            "score": score,
            "a50_pct": global_idx.get("a50_pct"),
            "spx_pct": global_idx.get("spx_pct"),
            "ixic_pct": global_idx.get("ixic_pct"),
            "hsi_pct": global_idx.get("hsi_pct"),
            "ad_ratio": breadth.get("ad_ratio"),
            "limit_up": breadth.get("limit_up"),
            "limit_down": breadth.get("limit_down"),
            "market_amount": volume.get("total_amount"),
            "amount_ratio": volume.get("amount_ratio"),
            "iv_percentile": vol_env.get("iv_percentile"),
            "vrp": vol_env.get("vrp"),
            "daily_5d_mom": price_pos.get("mom_5d"),
            "range_position": price_pos.get("range_pos"),
            "streak": price_pos.get("streak"),
            "d_override_long": d_override.get("LONG") if d_override else None,
            "d_override_short": d_override.get("SHORT") if d_override else None,
            "reasons": " | ".join(reasons[:5]),
        }

        if verbose:
            d_str = f"L={d_override['LONG']}/S={d_override['SHORT']}" if d_override else "None"
            print(f"  {td}  score={score:>+4d}  {direction}(★{confidence})  dm={d_str}")

        return result
    except Exception as e:
        if verbose:
            print(f"  {td}  ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="回算历史 Briefing 评分")
    parser.add_argument("--start", default="20250516", help="起始日期")
    parser.add_argument("--end", default="20260403", help="结束日期")
    parser.add_argument("--dry-run", action="store_true", help="只计算不写DB")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示每天详情")
    args = parser.parse_args()

    db = get_db()

    # 获取交易日列表
    trade_dates = db.query(
        "SELECT DISTINCT trade_date FROM futures_daily "
        "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (args.start, args.end),
    )["trade_date"].tolist()

    print(f"回算 Briefing: {args.start} ~ {args.end} ({len(trade_dates)} 交易日)")
    if args.dry_run:
        print("  (dry-run 模式，不写 DB)")

    results = []
    for i, td in enumerate(trade_dates):
        # 获取 T-1 交易日
        recent = _get_recent_trade_dates(db, td, 2)
        prev_td = recent[0]
        if prev_td == td and len(recent) >= 2:
            prev_td = recent[1]

        r = backfill_one_day(db, td, prev_td, verbose=args.verbose)
        if r:
            results.append(r)

        if not args.verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(trade_dates)}...")

    print(f"\n成功回算: {len(results)} / {len(trade_dates)} 天")

    if not results:
        return

    df = pd.DataFrame(results)

    # 统计
    print(f"\n=== 评分分布 ===")
    print(f"  均值: {df['score'].mean():+.1f}  中位数: {df['score'].median():+.1f}")
    print(f"  偏多: {(df['direction']=='偏多').sum()} 天")
    print(f"  中性: {(df['direction']=='中性').sum()} 天")
    print(f"  偏空: {(df['direction']=='偏空').sum()} 天")

    has_override = df['d_override_long'].notna()
    print(f"\n=== d_override 分布 ===")
    print(f"  有 d_override: {has_override.sum()} 天 ({has_override.mean()*100:.0f}%)")
    print(f"  无 d_override: {(~has_override).sum()} 天")
    if has_override.any():
        ov = df[has_override]
        print(f"  d_long 分布: {ov['d_override_long'].value_counts().to_dict()}")
        print(f"  d_short 分布: {ov['d_override_short'].value_counts().to_dict()}")

    if not args.dry_run:
        # 写入 DB（不覆盖已有的实盘 briefing）
        conn = db._conn
        inserted = 0
        for _, row in df.iterrows():
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO morning_briefing "
                    "(trade_date, direction, confidence, score, "
                    "a50_pct, spx_pct, ixic_pct, hsi_pct, "
                    "ad_ratio, limit_up, limit_down, "
                    "market_amount, amount_ratio, "
                    "iv_percentile, vrp, daily_5d_mom, range_position, streak, "
                    "d_override_long, d_override_short, reasons) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["trade_date"], row["direction"], row["confidence"], row["score"],
                     row["a50_pct"], row["spx_pct"], row["ixic_pct"], row["hsi_pct"],
                     row["ad_ratio"], row["limit_up"], row["limit_down"],
                     row["market_amount"], row["amount_ratio"],
                     row["iv_percentile"], row["vrp"],
                     row["daily_5d_mom"], row["range_position"], row["streak"],
                     row["d_override_long"], row["d_override_short"], row["reasons"]),
                )
                inserted += conn.total_changes
            except Exception:
                pass
        conn.commit()
        print(f"\n写入 morning_briefing: {inserted} 条 (INSERT OR IGNORE)")
    else:
        print("\n  [dry-run] 未写入 DB")


if __name__ == "__main__":
    main()
