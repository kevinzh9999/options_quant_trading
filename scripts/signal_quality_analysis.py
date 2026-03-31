"""
signal_quality_analysis.py
--------------------------
分析回测信号的质量维度：V=0过滤、边界信号、日线方向乘数、
跳空追势、反弹/回调距离、趋势反转模式。

用法：
    python scripts/signal_quality_analysis.py --symbol IM
    python scripts/signal_quality_analysis.py --symbol IM --days 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager
from scripts.backtest_signals_day import run_day


def _get_dates(db: DBManager, sym: str, n: int) -> list[str]:
    """获取有现货5分钟数据的最近n个交易日。"""
    _SPOT = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
    spot = _SPOT.get(sym, "000852")
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime, 1, 10) as d "
        f"FROM index_min WHERE symbol='{spot}' AND period=300 "
        f"ORDER BY d DESC LIMIT {n}"
    )
    if df is None or df.empty:
        return []
    dates = [d.replace("-", "") for d in df["d"].tolist()]
    dates.reverse()
    return dates


def _group_stats(trades: list[dict], label: str) -> dict:
    """计算一组交易的统计。"""
    if not trades:
        return {"label": label, "n": 0, "wr": 0, "avg": 0, "total": 0}
    pts = [t["pnl_pts"] for t in trades]
    wins = len([p for p in pts if p > 0])
    return {
        "label": label,
        "n": len(trades),
        "wr": wins / len(trades) * 100,
        "avg": float(np.mean(pts)),
        "total": float(np.sum(pts)),
    }


def _print_group(g: dict):
    """打印一行统计。"""
    if g["n"] == 0:
        print(f"  {g['label']:12s}:   0 trades")
        return
    tag = "✅" if g["total"] > 0 else "❌"
    print(f"  {g['label']:12s}: {g['n']:3d} trades  "
          f"WR={g['wr']:.0f}%  avg={g['avg']:+.1f}pt  "
          f"total={g['total']:+.0f}pt  {tag}")


def _analyze_reversals(db: DBManager, sym: str, dates: list[str]):
    """分析6：趋势反转模式。从5分钟K线检测EMA方向反转。"""
    _SPOT = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
    spot_sym = _SPOT.get(sym, "000852")

    total_reversals = 0
    s_to_l_moves: list[float] = []
    l_to_s_moves: list[float] = []
    # Track: if we entered at reversal point and held to close, what's the PnL?
    reversal_trades_pnl: list[float] = []

    for td in dates:
        date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"
        bars = db.query_df(
            f"SELECT datetime, open, high, low, close, volume "
            f"FROM index_min WHERE symbol='{spot_sym}' AND period=300 "
            f"AND datetime LIKE '{date_dash}%' ORDER BY datetime"
        )
        if bars is None or len(bars) < 20:
            continue
        for c in ["open", "high", "low", "close"]:
            bars[c] = bars[c].astype(float)

        closes = bars["close"].values
        # Compute EMA5, EMA10 for direction
        ema5 = pd.Series(closes).ewm(span=5, adjust=False).mean().values
        ema10 = pd.Series(closes).ewm(span=10, adjust=False).mean().values
        # Direction: L if EMA5 > EMA10 else S
        dirs = ["L" if e5 > e10 else "S" for e5, e10 in zip(ema5, ema10)]

        day_close = closes[-1]
        MIN_STREAK = 3

        for i in range(MIN_STREAK, len(dirs)):
            # Check if dirs[i] != dirs[i-1] (direction change)
            if dirs[i] == dirs[i - 1]:
                continue
            # Check streak: previous MIN_STREAK bars all same direction
            streak_dir = dirs[i - 1]
            streak_ok = all(dirs[i - 1 - k] == streak_dir for k in range(MIN_STREAK))
            if not streak_ok:
                continue

            total_reversals += 1
            rev_price = closes[i]
            new_dir = dirs[i]

            # Measure move over next 6 bars (30 min) or to end of day
            future_end = min(i + 6, len(closes) - 1)
            future_price = closes[future_end]
            move = future_price - rev_price

            if new_dir == "L":  # S→L reversal
                s_to_l_moves.append(move)
            else:  # L→S reversal
                l_to_s_moves.append(move)

            # Theoretical trade: enter at reversal, hold to close
            if new_dir == "L":
                pnl = day_close - rev_price
            else:
                pnl = rev_price - day_close
            reversal_trades_pnl.append(pnl)

    print(f"\n{'=' * 60}")
    print(f" Reversal Pattern Analysis ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    avg_per_day = total_reversals / len(dates) if dates else 0
    print(f"  Total reversals detected: {total_reversals} (avg {avg_per_day:.1f}/day)")

    if s_to_l_moves:
        avg_s2l = float(np.mean(s_to_l_moves))
        print(f"  S→L reversals: {len(s_to_l_moves)}, avg subsequent move(30min): {avg_s2l:+.1f}pt")
    if l_to_s_moves:
        avg_l2s = float(np.mean(l_to_s_moves))
        print(f"  L→S reversals: {len(l_to_s_moves)}, avg subsequent move(30min): {avg_l2s:+.1f}pt")

    if reversal_trades_pnl:
        wins = len([p for p in reversal_trades_pnl if p > 0])
        wr = wins / len(reversal_trades_pnl) * 100
        avg_pnl = float(np.mean(reversal_trades_pnl))
        print(f"  If traded at reversal→close: WR={wr:.0f}%  avg={avg_pnl:+.1f}pt")

    # Conclusion
    if s_to_l_moves and l_to_s_moves and reversal_trades_pnl:
        avg_abs = float(np.mean([abs(m) for m in s_to_l_moves + l_to_s_moves]))
        if avg_abs > 5 and wr > 50:
            print(f"  结论：反转后有正期望（WR={wr:.0f}%），30min平均幅度{avg_abs:.1f}pt")
        else:
            print(f"  结论：反转信号需严格过滤（WR={wr:.0f}%，幅度{avg_abs:.1f}pt）")


def analyze(sym: str, days: int = 30):
    """运行分析。"""
    db = DBManager(ConfigLoader().get_db_path())
    dates = _get_dates(db, sym, days)
    if not dates:
        print("没有可用数据")
        return

    print(f"  回测 {sym} | {dates[0]}~{dates[-1]} | {len(dates)} 天")
    print(f"  加载中...")

    # 收集所有交易 + 被压制的信号
    all_trades: list[dict] = []
    all_suppressed: list[dict] = []
    for td in dates:
        trades = run_day(sym, td, db, verbose=False)
        completed = [t for t in trades if not t.get("partial")]
        all_trades.extend(completed)
        all_suppressed.extend(getattr(trades, "_suppressed", []))

    if not all_trades:
        print("  无交易")
        return

    total_pnl = sum(t["pnl_pts"] for t in all_trades)
    print(f"  总交易: {len(all_trades)} 笔  总PnL: {total_pnl:+.0f}pt\n")

    # === 分析1：V-Score ===
    print(f"{'=' * 60}")
    print(f" V-Score Filter Analysis ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    v0 = [t for t in all_trades if t.get("entry_v_score", 0) == 0]
    v5 = [t for t in all_trades if 0 < t.get("entry_v_score", 0) < 15]
    v15 = [t for t in all_trades if 15 <= t.get("entry_v_score", 0) < 25]
    v25 = [t for t in all_trades if t.get("entry_v_score", 0) >= 25]

    _print_group(_group_stats(v0, "V=0"))
    _print_group(_group_stats(v5, "V=5-10"))
    _print_group(_group_stats(v15, "V=15-20"))
    _print_group(_group_stats(v25, "V>=25"))

    v0_s = _group_stats(v0, "")
    vge = _group_stats([t for t in all_trades if t.get("entry_v_score", 0) >= 15], "")
    if v0_s["n"] > 0 and vge["n"] > 0:
        delta = vge["avg"] - v0_s["avg"]
        print(f"\n  V=0 vs V>=15: 均盈差 {delta:+.1f}pt/笔")
        if v0_s["avg"] < 0 and vge["avg"] > 0:
            print(f"  → V=0 信号整体亏损，建议过滤或惩罚")
        elif v0_s["avg"] < vge["avg"]:
            print(f"  → V=0 信号质量较差，但不一定需要完全过滤")

    # === 分析2：Score Band ===
    print(f"\n{'=' * 60}")
    print(f" Score Band Analysis ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    band1 = [t for t in all_trades if 60 <= t.get("entry_score", 0) <= 62]
    band2 = [t for t in all_trades if 63 <= t.get("entry_score", 0) <= 69]
    band3 = [t for t in all_trades if t.get("entry_score", 0) >= 70]

    _print_group(_group_stats(band1, "60-62(边界)"))
    _print_group(_group_stats(band2, "63-69(中等)"))
    _print_group(_group_stats(band3, "70+(高分)"))

    b1s = _group_stats(band1, "")
    b3s = _group_stats(band3, "")
    if b1s["n"] > 0 and b3s["n"] > 0:
        print(f"\n  边界 vs 高分: WR差 {b3s['wr'] - b1s['wr']:+.0f}pp  "
              f"均盈差 {b3s['avg'] - b1s['avg']:+.1f}pt")

    # === 分析3：Daily Mult ===
    print(f"\n{'=' * 60}")
    print(f" Daily Mult Analysis ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    d07 = [t for t in all_trades if abs(t.get("entry_daily_mult", 1.0) - 0.7) < 0.05]
    d10 = [t for t in all_trades if abs(t.get("entry_daily_mult", 1.0) - 1.0) < 0.05]
    d12 = [t for t in all_trades if abs(t.get("entry_daily_mult", 1.0) - 1.2) < 0.05]

    _print_group(_group_stats(d07, "d=0.7(逆势)"))
    _print_group(_group_stats(d10, "d=1.0(中性)"))
    _print_group(_group_stats(d12, "d=1.2(顺势)"))

    d07s = _group_stats(d07, "")
    d12s = _group_stats(d12, "")
    if d07s["n"] > 0 and d12s["n"] > 0:
        print(f"\n  逆势 vs 顺势: WR差 {d12s['wr'] - d07s['wr']:+.0f}pp  "
              f"均盈差 {d12s['avg'] - d07s['avg']:+.1f}pt")

    # === 分析3b：Direction × Daily Mult ===
    print(f"\n{'=' * 60}")
    print(f" Direction × Daily Mult Analysis ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    _dm_groups = [
        ("顺势做多(d≥1.1)", lambda t: t.get("entry_daily_mult", 1.0) >= 1.1
                                       and t.get("direction") == "LONG"),
        ("顺势做空(d≥1.1)", lambda t: t.get("entry_daily_mult", 1.0) >= 1.1
                                       and t.get("direction") == "SHORT"),
        ("中性做多",        lambda t: 0.6 < t.get("entry_daily_mult", 1.0) < 1.1
                                       and t.get("direction") == "LONG"),
        ("中性做空",        lambda t: 0.6 < t.get("entry_daily_mult", 1.0) < 1.1
                                       and t.get("direction") == "SHORT"),
        ("逆势做多(d≤0.6)", lambda t: t.get("entry_daily_mult", 1.0) <= 0.6
                                       and t.get("direction") == "LONG"),
        ("逆势做空(d≤0.6)", lambda t: t.get("entry_daily_mult", 1.0) <= 0.6
                                       and t.get("direction") == "SHORT"),
    ]
    for label, pred in _dm_groups:
        grp = [t for t in all_trades if pred(t)]
        _print_group(_group_stats(grp, label))

    # 被 daily_mult 压制的信号（raw 够分但 d<0.8 压到阈值以下）
    dm_suppressed = [s for s in all_suppressed if s.get("suppressor") == "daily_mult"]
    sup_long = [s for s in dm_suppressed if s["direction"] == "LONG"]
    sup_short = [s for s in dm_suppressed if s["direction"] == "SHORT"]

    print()
    for label, sigs in [("逆势做多", sup_long), ("逆势做空", sup_short)]:
        if not sigs:
            print(f"  被d压制的{label}信号: 0 个")
            continue
        pts = [s["pnl_pts"] for s in sigs]
        wins = len([p for p in pts if p > 0])
        wr = wins / len(pts) * 100
        avg = float(np.mean(pts))
        total = float(np.sum(pts))
        tag = "✅" if total > 0 else "❌"
        print(f"  被d压制的{label}信号（raw≥60, filtered<60）:")
        print(f"    {len(sigs)} 个信号，如果执行→收盘: "
              f"WR={wr:.0f}%  avg={avg:+.1f}pt  total={total:+.0f}pt  {tag}")

    # === 分析3c：逆势做多 × 跌幅分层 ===
    print(f"\n{'=' * 60}")
    print(f" 逆势做多 × 跌幅分层 ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    # 已触发的逆势做多
    contrarian_long = [t for t in all_trades
                       if t.get("entry_daily_mult", 1.0) <= 0.6
                       and t.get("direction") == "LONG"]
    _drop_groups = [
        ("大跌后(drop<-1%)",    lambda x: x < -0.01),
        ("小跌后(-1%~-0.5%)",   lambda x: -0.01 <= x < -0.005),
        ("未跌(>-0.5%)",        lambda x: x >= -0.005),
    ]

    print("  已触发的逆势做多信号:")
    for label, pred in _drop_groups:
        grp = [t for t in contrarian_long if pred(t.get("entry_total_drop", 0))]
        _print_group(_group_stats(grp, label))

    # 被压制的逆势做多
    print("  被d=0.5压制的逆势做多信号（raw≥60但filtered<60）:")
    for label, pred in _drop_groups:
        grp = [s for s in sup_long if pred(s.get("entry_total_drop", 0))]
        if not grp:
            print(f"  {label:14s}:   0 signals")
            continue
        pts = [s["pnl_pts"] for s in grp]
        wins = len([p for p in pts if p > 0])
        wr = wins / len(pts) * 100
        avg = float(np.mean(pts))
        total_p = float(np.sum(pts))
        tag = "✅" if total_p > 0 else "❌"
        print(f"  {label:14s}: {len(grp):3d} signals  如果执行→收盘: "
              f"WR={wr:.0f}%  avg={avg:+.1f}pt  total={total_p:+.0f}pt  {tag}")

    # === 分析3d：Intraday Filter 分支验证 ===
    print(f"\n{'=' * 60}")
    print(f" Intraday Filter Analysis ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    def _chg(t):
        return t.get("entry_total_drop", 0)

    _idf_groups = [
        ("±1%内(baseline)",   lambda t: abs(_chg(t)) <= 0.01,           "1.0"),
        ("涨1-2% 做多",      lambda t: 0.01 < _chg(t) <= 0.02 and t["direction"] == "LONG",  "1.0"),
        ("涨1-2% 做空",      lambda t: 0.01 < _chg(t) <= 0.02 and t["direction"] == "SHORT", "0.7"),
        ("涨2-3% 做多",      lambda t: 0.02 < _chg(t) <= 0.03 and t["direction"] == "LONG",  "0.9"),
        ("涨2-3% 做空",      lambda t: 0.02 < _chg(t) <= 0.03 and t["direction"] == "SHORT", "0.5"),
        ("涨>3%  做多",      lambda t: _chg(t) > 0.03 and t["direction"] == "LONG",           "0.8"),
        ("涨>3%  做空",      lambda t: _chg(t) > 0.03 and t["direction"] == "SHORT",          "0.3"),
        ("跌1-2% 做空",      lambda t: -0.02 <= _chg(t) < -0.01 and t["direction"] == "SHORT","1.0"),
        ("跌1-2% 做多",      lambda t: -0.02 <= _chg(t) < -0.01 and t["direction"] == "LONG", "0.7"),
        ("跌2-3% 做空",      lambda t: -0.03 <= _chg(t) < -0.02 and t["direction"] == "SHORT","0.9"),
        ("跌2-3% 做多",      lambda t: -0.03 <= _chg(t) < -0.02 and t["direction"] == "LONG", "0.5"),
        ("跌>3%  做空",      lambda t: _chg(t) < -0.03 and t["direction"] == "SHORT",         "0.8"),
        ("跌>3%  做多",      lambda t: _chg(t) < -0.03 and t["direction"] == "LONG",          "0.3"),
    ]

    print(f"  {'区间':<16s} | {'f值':>4s} | {'交易数':>4s} | {'胜率':>4s} | {'avg PnL':>8s} | {'total':>8s}")
    print(f"  {'-'*16}-+------+------+------+----------+----------")
    for label, pred, f_val in _idf_groups:
        grp = [t for t in all_trades if pred(t)]
        g = _group_stats(grp, label)
        if g["n"] == 0:
            print(f"  {label:<16s} | {f_val:>4s} |    0 |   -- |       -- |       --")
        else:
            tag = "✅" if g["total"] > 0 else "❌"
            print(f"  {label:<16s} | {f_val:>4s} | {g['n']:4d} | {g['wr']:3.0f}% | {g['avg']:+7.1f}pt | {g['total']:+7.0f}pt {tag}")

    # 被 intraday_filter 过滤掉的信号
    idf_suppressed = [s for s in all_suppressed if s.get("suppressor") == "intraday_filter"]
    if idf_suppressed:
        print(f"\n  被intraday_filter过滤的信号（f<1.0导致score<60）:")
        # Group by change_pct range × direction
        _idf_sup_groups = [
            ("涨1-2%做空(f=0.7)", lambda s: 0.01 < s.get("entry_total_drop",0) <= 0.02 and s["direction"]=="SHORT"),
            ("涨2-3%做空(f=0.5)", lambda s: 0.02 < s.get("entry_total_drop",0) <= 0.03 and s["direction"]=="SHORT"),
            ("涨>3% 做空(f=0.3)", lambda s: s.get("entry_total_drop",0) > 0.03 and s["direction"]=="SHORT"),
            ("跌1-2%做多(f=0.7)", lambda s: -0.02 <= s.get("entry_total_drop",0) < -0.01 and s["direction"]=="LONG"),
            ("跌2-3%做多(f=0.5)", lambda s: -0.03 <= s.get("entry_total_drop",0) < -0.02 and s["direction"]=="LONG"),
            ("跌>3% 做多(f=0.3)", lambda s: s.get("entry_total_drop",0) < -0.03 and s["direction"]=="LONG"),
            ("其他",              lambda s: True),  # catch-all
        ]
        seen = set()
        for label, pred in _idf_sup_groups:
            grp = [s for s in idf_suppressed if id(s) not in seen and pred(s)]
            for s in grp:
                seen.add(id(s))
            if not grp:
                continue
            pts = [s["pnl_pts"] for s in grp]
            wins = len([p for p in pts if p > 0])
            wr = wins / len(pts) * 100
            avg = float(np.mean(pts))
            tot = float(np.sum(pts))
            tag = "✅" if tot > 0 else "❌"
            print(f"    {label:<18s}: {len(grp):3d}个  如果执行→收盘: "
                  f"WR={wr:.0f}%  avg={avg:+.1f}pt  total={tot:+.0f}pt  {tag}")
    else:
        print(f"\n  被intraday_filter过滤的信号: 0 个")

    # === 交叉分析：V=0 + 边界 ===
    print(f"\n{'=' * 60}")
    print(f" Cross Analysis: V=0 + Score Band")
    print(f"{'=' * 60}")

    v0_border = [t for t in all_trades
                 if t.get("entry_v_score", 0) == 0
                 and 60 <= t.get("entry_score", 0) <= 62]
    v0_high = [t for t in all_trades
               if t.get("entry_v_score", 0) == 0
               and t.get("entry_score", 0) >= 70]
    vge_border = [t for t in all_trades
                  if t.get("entry_v_score", 0) >= 15
                  and 60 <= t.get("entry_score", 0) <= 62]

    _print_group(_group_stats(v0_border, "V=0+边界"))
    _print_group(_group_stats(v0_high, "V=0+高分"))
    _print_group(_group_stats(vge_border, "V>=15+边界"))

    # === 方向分拆 ===
    print(f"\n{'=' * 60}")
    print(f" Direction Breakdown")
    print(f"{'=' * 60}")

    longs = [t for t in all_trades if t.get("direction") == "LONG"]
    shorts = [t for t in all_trades if t.get("direction") == "SHORT"]
    _print_group(_group_stats(longs, "LONG"))
    _print_group(_group_stats(shorts, "SHORT"))

    # LONG with d=0.7
    long_d07 = [t for t in longs if abs(t.get("entry_daily_mult", 1.0) - 0.7) < 0.05]
    short_d07 = [t for t in shorts if abs(t.get("entry_daily_mult", 1.0) - 0.7) < 0.05]
    if long_d07:
        _print_group(_group_stats(long_d07, "L+d=0.7"))
    if short_d07:
        _print_group(_group_stats(short_d07, "S+d=0.7"))

    # === 分析4：Gap-Chase ===
    print(f"\n{'=' * 60}")
    print(f" Gap-Chase Analysis ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    GAP_THRESHOLD = 0.008  # 0.8%
    big_gap_chase = [t for t in all_trades
                     if abs(t.get("entry_gap_pct", 0)) > GAP_THRESHOLD
                     and t.get("entry_gap_aligned", False)]
    big_gap_fade = [t for t in all_trades
                    if abs(t.get("entry_gap_pct", 0)) > GAP_THRESHOLD
                    and not t.get("entry_gap_aligned", False)]
    small_gap = [t for t in all_trades
                 if abs(t.get("entry_gap_pct", 0)) <= GAP_THRESHOLD]

    _print_group(_group_stats(big_gap_chase, "追跳空"))
    _print_group(_group_stats(big_gap_fade, "逆跳空"))
    _print_group(_group_stats(small_gap, "小跳空"))

    gc = _group_stats(big_gap_chase, "")
    gf = _group_stats(big_gap_fade, "")
    sg = _group_stats(small_gap, "")
    if gc["n"] > 0 and sg["n"] > 0:
        print(f"\n  追跳空 vs baseline: WR差 {gc['wr'] - sg['wr']:+.0f}pp  "
              f"均盈差 {gc['avg'] - sg['avg']:+.1f}pt")
    if gf["n"] > 0 and sg["n"] > 0:
        print(f"  逆跳空 vs baseline: WR差 {gf['wr'] - sg['wr']:+.0f}pp  "
              f"均盈差 {gf['avg'] - sg['avg']:+.1f}pt")

    # === 分析5：Rebound/Pullback ===
    print(f"\n{'=' * 60}")
    print(f" Rebound/Pullback Analysis ({len(dates)} days, {sym})")
    print(f"{'=' * 60}")

    near = [t for t in all_trades if t.get("entry_rebound_pct", 0) < 0.003]
    mid = [t for t in all_trades if 0.003 <= t.get("entry_rebound_pct", 0) < 0.007]
    far = [t for t in all_trades if t.get("entry_rebound_pct", 0) >= 0.007]

    _print_group(_group_stats(near, "极值附近<0.3%"))
    _print_group(_group_stats(mid, "中等0.3-0.7%"))
    _print_group(_group_stats(far, "远离>0.7%"))

    ns = _group_stats(near, "")
    fs = _group_stats(far, "")
    if ns["n"] > 0 and fs["n"] > 0:
        print(f"\n  近极值 vs 远极值: WR差 {ns['wr'] - fs['wr']:+.0f}pp  "
              f"均盈差 {ns['avg'] - fs['avg']:+.1f}pt")

    # === 分析6：Reversal Pattern ===
    _analyze_reversals(db, sym, dates)

    print()


def main():
    parser = argparse.ArgumentParser(description="信号质量分析")
    parser.add_argument("--symbol", default="IM")
    parser.add_argument("--days", type=int, default=30, help="回测天数")
    args = parser.parse_args()
    analyze(args.symbol, args.days)


if __name__ == "__main__":
    main()
