#!/usr/bin/env python3
"""
adx_macd_backtest.py
--------------------
ADX 和 MACD 指标对日内交易信号质量的影响分析。

复用 backtest_signals_day.py 的回测框架，在每笔开仓时额外记录 ADX(14)
和 MACD histogram，然后分析这两个指标对胜率/PnL的预测能力，并模拟加入
乘数后的净效果。

Usage:
    python scripts/adx_macd_backtest.py
    python scripts/adx_macd_backtest.py --symbol IM
    python scripts/adx_macd_backtest.py --symbol IC
    python scripts/adx_macd_backtest.py --symbol IM IC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# Import the backtest framework
from scripts.backtest_signals_day import run_day, _patch_threshold

# ─── 30天回测日期 ────────────────────────────────────────────────────────────

DATES_30D = (
    "20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,"
    "20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,"
    "20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,"
    "20260319,20260320,20260323,20260324,20260325,20260326"
)

SYMBOL_THR = {"IM": 60, "IC": 65}


# ─── 指标计算 ────────────────────────────────────────────────────────────────

def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    """标准 Wilder EMA ADX(14)。"""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # 只保留较大的那个，另一个归零
    mask_plus = plus_dm >= minus_dm
    plus_dm = plus_dm.where(mask_plus, 0.0)
    minus_dm = minus_dm.where(~mask_plus, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    # Wilder EMA (alpha = 1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx


def calc_macd_hist(close: pd.Series,
                   fast: int = 12, slow: int = 26, signal: int = 9
                   ) -> pd.Series:
    """MACD histogram = MACD line - Signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


# ─── 核心：带 ADX/MACD 注入的单日回测 ─────────────────────────────────────

def run_day_with_indicators(sym: str, td: str, db: DBManager,
                             thr: int) -> List[Dict]:
    """
    调用 run_day() 获取基准交易，然后将 ADX/MACD 注入每笔交易记录。

    run_day 返回的每条交易记录里有 entry_time（北京时间 HH:MM），
    我们用它在当天 5 分钟 bar 中找到对应索引，取信号 bar（当前 bar 的
    上一根，与回测框架一致），计算 ADX / MACD。
    """
    _patch_threshold(thr)
    trades_raw = run_day(sym, td, db, verbose=False, slippage=0, version="auto")
    if not trades_raw:
        return []

    # ── 加载当天现货 5min bar（与 run_day 一致）─────────────────────────
    _SPOT_SYM = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
    spot_sym = _SPOT_SYM[sym]
    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"

    all_bars = db.query_df(
        f"SELECT datetime, open, high, low, close, volume "
        f"FROM index_min WHERE symbol='{spot_sym}' AND period=300 "
        f"ORDER BY datetime"
    )
    if all_bars is None or all_bars.empty:
        return list(trades_raw)

    for c in ["open", "high", "low", "close", "volume"]:
        all_bars[c] = all_bars[c].astype(float)

    today_mask = all_bars["datetime"].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()
    if not today_indices:
        return list(trades_raw)

    # ── 用全历史 bar 计算 ADX / MACD（尽量多历史，结果更稳定）───────────
    all_bars.reset_index(drop=True, inplace=True)
    # 确保今天数据结束位置
    today_end_idx = today_indices[-1]

    # 取到今天最后一根（足够历史），计算两个指标
    hist = all_bars.loc[:today_end_idx].copy()
    adx_series = calc_adx(hist["high"], hist["low"], hist["close"])
    macd_hist_series = calc_macd_hist(hist["close"])

    # 建立 datetime → idx 映射
    dt_to_idx: Dict[str, int] = {
        str(all_bars.loc[i, "datetime"]): i for i in today_indices
    }

    # BJ time → UTC 转换（run_day 用 _utc_to_bj 转换了入口时间）
    def bj_to_utc(bj_hm: str) -> str:
        h = int(bj_hm[:2]) - 8
        if h < 0:
            h += 24
        return f"{h:02d}:{bj_hm[3:5]}"

    # 建立 UTC 时间 → bar 在 today_indices 中的索引位置
    utc_to_ti: Dict[str, int] = {}
    for ti in today_indices:
        utc_hm = str(all_bars.loc[ti, "datetime"])[11:16]
        utc_to_ti[utc_hm] = ti

    enriched = []
    for t in trades_raw:
        t = dict(t)  # 复制
        entry_bj = t.get("entry_time", "")
        entry_utc = bj_to_utc(entry_bj) if entry_bj else ""

        # 找到入场 bar 的 global index
        global_idx = utc_to_ti.get(entry_utc)

        adx_val = float("nan")
        macd_val = float("nan")

        if global_idx is not None and global_idx > 0:
            # 信号 bar = 入场 bar 的前一根（lookahead fix 与回测框架一致）
            signal_idx = global_idx - 1
            if signal_idx in adx_series.index:
                adx_val = float(adx_series.loc[signal_idx])
            if signal_idx in macd_hist_series.index:
                macd_val = float(macd_hist_series.loc[signal_idx])

        t["adx"] = adx_val
        t["macd_hist"] = macd_val
        t["trade_date"] = td
        t["symbol"] = sym
        enriched.append(t)

    return enriched


# ─── 分析函数 ────────────────────────────────────────────────────────────────

def _adx_bucket(v: float) -> str:
    if np.isnan(v):
        return "NA"
    if v < 15:
        return "<15"
    elif v < 25:
        return "15-25"
    elif v < 35:
        return "25-35"
    else:
        return ">35"


def _macd_direction(v: float) -> str:
    if np.isnan(v):
        return "NA"
    return "MACD>0" if v > 0 else "MACD<0"


def print_analysis_1(df: pd.DataFrame, sym: str):
    """分析1：ADX分组统计。"""
    print(f"\n{'─'*65}")
    print(f"  分析1：ADX 分组  [{sym}]")
    print(f"{'─'*65}")
    print(f"  {'ADX区间':<10} {'笔数':>6} {'WR%':>8} {'Avg PnL':>10} {'总PnL':>10}")
    print(f"  {'─'*10} {'─'*6} {'─'*8} {'─'*10} {'─'*10}")

    df2 = df[~df.get("partial", pd.Series(False, index=df.index)).fillna(False)]
    for bucket in ["<15", "15-25", "25-35", ">35", "NA"]:
        sub = df2[df2["adx_bucket"] == bucket]
        n = len(sub)
        if n == 0:
            continue
        wr = (sub["pnl_pts"] > 0).mean() * 100
        avg_pnl = sub["pnl_pts"].mean()
        total_pnl = sub["pnl_pts"].sum()
        print(f"  {bucket:<10} {n:>6} {wr:>7.1f}% {avg_pnl:>+10.1f} {total_pnl:>+10.1f}")

    n_all = len(df2)
    wr_all = (df2["pnl_pts"] > 0).mean() * 100 if n_all > 0 else 0
    print(f"  {'合计':<10} {n_all:>6} {wr_all:>7.1f}% {df2['pnl_pts'].mean():>+10.1f} {df2['pnl_pts'].sum():>+10.1f}")


def print_analysis_2(df: pd.DataFrame, sym: str):
    """分析2：MACD顺逆势统计。"""
    print(f"\n{'─'*75}")
    print(f"  分析2：MACD 方向 vs 交易方向  [{sym}]")
    print(f"{'─'*75}")
    print(f"  {'MACD方向':<12} {'交易方向':<10} {'笔数':>6} {'WR%':>8} {'Avg PnL':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*6} {'─'*8} {'─'*10}")

    df2 = df[~df.get("partial", pd.Series(False, index=df.index)).fillna(False)]
    for macd_dir in ["MACD>0", "MACD<0"]:
        for trade_dir in ["LONG", "SHORT"]:
            sub = df2[(df2["macd_dir"] == macd_dir) & (df2["direction"] == trade_dir)]
            n = len(sub)
            if n == 0:
                continue
            wr = (sub["pnl_pts"] > 0).mean() * 100
            avg_pnl = sub["pnl_pts"].mean()
            aligned = (
                (macd_dir == "MACD>0" and trade_dir == "LONG") or
                (macd_dir == "MACD<0" and trade_dir == "SHORT")
            )
            tag = " ← 顺势" if aligned else " ← 逆势"
            print(f"  {macd_dir:<12} {trade_dir:<10} {n:>6} {wr:>7.1f}% {avg_pnl:>+10.1f}{tag}")


def _sim_with_multipliers(
    df: pd.DataFrame, thr: int,
    adx_mult: bool = False, macd_mult: bool = False,
) -> Tuple[int, float, float]:
    """
    模拟加入 ADX/MACD 乘数后的效果。

    规则：
    - ADX乘数: <15→×0.85, 15-25→×1.0, >25→×1.10
    - MACD乘数: 顺势→×1.05, 逆势→×0.90

    返回 (n_trades, total_pnl, win_rate)
    """
    # 排除 partial trades（半仓平仓）
    df2 = df[~df.get("partial", pd.Series(False, index=df.index)).fillna(False)].copy()

    accepted_pnls = []
    for _, row in df2.iterrows():
        score = row.get("entry_score", 60)
        raw_score = row.get("entry_raw_total", score)  # pre-filter raw score

        mult = 1.0
        if adx_mult:
            adx = row["adx"]
            if not np.isnan(adx):
                if adx < 15:
                    mult *= 0.85
                elif adx >= 25:
                    mult *= 1.10
                # 15-25: ×1.0 (no change)

        if macd_mult:
            macd = row["macd_hist"]
            direction = row["direction"]
            if not np.isnan(macd):
                is_aligned = (
                    (macd > 0 and direction == "LONG") or
                    (macd < 0 and direction == "SHORT")
                )
                if is_aligned:
                    mult *= 1.05
                else:
                    mult *= 0.90

        # 模拟：新 score = entry_score（已经是最终score）× mult
        # 如果新 score < thr，这笔交易被过滤掉
        new_score = score * mult
        if new_score >= thr:
            accepted_pnls.append(row["pnl_pts"])

    n = len(accepted_pnls)
    if n == 0:
        return 0, 0.0, 0.0
    total = sum(accepted_pnls)
    wr = sum(1 for p in accepted_pnls if p > 0) / n * 100
    return n, total, wr


def print_analysis_3(df: pd.DataFrame, sym: str, thr: int):
    """分析3：ADX乘数过滤效果。"""
    print(f"\n{'─'*75}")
    print(f"  分析3：ADX 乘数模拟  [{sym}]  (thr={thr})")
    print(f"{'─'*75}")
    print(f"  ADX<15 × 0.85, ADX 15-25 × 1.0, ADX>25 × 1.10")
    print()

    df2 = df[~df.get("partial", pd.Series(False, index=df.index)).fillna(False)]
    n_base = len(df2)
    wr_base = (df2["pnl_pts"] > 0).mean() * 100 if n_base > 0 else 0
    pnl_base = df2["pnl_pts"].sum()

    n_adx, pnl_adx, wr_adx = _sim_with_multipliers(df, thr, adx_mult=True)
    delta = pnl_adx - pnl_base
    delta_n = n_adx - n_base

    print(f"  {'方案':<20} {'笔数':>6} {'WR%':>8} {'总PnL':>10} {'增量PnL':>10}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*10}")
    print(f"  {'基准':20} {n_base:>6} {wr_base:>7.1f}% {pnl_base:>+10.1f} {'─':>10}")
    print(f"  {'+ADX乘数':20} {n_adx:>6} {wr_adx:>7.1f}% {pnl_adx:>+10.1f} {delta:>+10.1f}  (Δn={delta_n:+d})")


def print_analysis_4(df: pd.DataFrame, sym: str, thr: int):
    """分析4：MACD确认乘数效果。"""
    print(f"\n{'─'*75}")
    print(f"  分析4：MACD 乘数模拟  [{sym}]  (thr={thr})")
    print(f"{'─'*75}")
    print(f"  MACD顺势 × 1.05, MACD逆势 × 0.90")
    print()

    df2 = df[~df.get("partial", pd.Series(False, index=df.index)).fillna(False)]
    n_base = len(df2)
    wr_base = (df2["pnl_pts"] > 0).mean() * 100 if n_base > 0 else 0
    pnl_base = df2["pnl_pts"].sum()

    n_macd, pnl_macd, wr_macd = _sim_with_multipliers(df, thr, macd_mult=True)
    delta = pnl_macd - pnl_base
    delta_n = n_macd - n_base

    print(f"  {'方案':<20} {'笔数':>6} {'WR%':>8} {'总PnL':>10} {'增量PnL':>10}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*10} {'─'*10}")
    print(f"  {'基准':20} {n_base:>6} {wr_base:>7.1f}% {pnl_base:>+10.1f} {'─':>10}")
    print(f"  {'+MACD乘数':20} {n_macd:>6} {wr_macd:>7.1f}% {pnl_macd:>+10.1f} {delta:>+10.1f}  (Δn={delta_n:+d})")


def print_final_comparison(results: Dict):
    """最终对比表。"""
    print(f"\n{'═'*85}")
    print(f"  最终对比表")
    print(f"{'═'*85}")

    header = f"  {'方案':<20}"
    for sym in results:
        header += f" {sym+' PnL':>10} {sym+' WR':>8}"
    header += f"  {'合计PnL':>10}  {'合计Δ':>10}"
    print(header)
    sep = f"  {'─'*20}"
    for sym in results:
        sep += f" {'─'*10} {'─'*8}"
    sep += f"  {'─'*10}  {'─'*10}"
    print(sep)

    scenarios = ["基准", "+ADX乘数", "+MACD乘数", "+ADX+MACD"]
    for scen in scenarios:
        line = f"  {scen:<20}"
        total_pnl = 0.0
        base_total = 0.0
        for sym, data in results.items():
            pnl = data[scen]["pnl"]
            wr = data[scen]["wr"]
            line += f" {pnl:>+10.1f} {wr:>7.1f}%"
            total_pnl += pnl
            if scen == "基准":
                pass
            base_total += data["基准"]["pnl"]

        if scen == "基准":
            _base_total = total_pnl
            line += f"  {total_pnl:>+10.1f}  {'─':>10}"
        else:
            # compute base fresh
            base = sum(data["基准"]["pnl"] for data in results.values())
            line += f"  {total_pnl:>+10.1f}  {total_pnl - base:>+10.1f}"
        print(line)

    print()


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def run_symbol(sym: str, dates: List[str], db: DBManager) -> Tuple[pd.DataFrame, Dict]:
    """运行单个品种的全部分析，返回 (enriched_df, scenario_results)。"""
    thr = SYMBOL_THR.get(sym, 60)
    print(f"\n{'━'*70}")
    print(f"  {sym}  ({len(dates)} 天回测, thr={thr})")
    print(f"{'━'*70}")

    all_trades: List[Dict] = []
    for i, td in enumerate(dates):
        trades = run_day_with_indicators(sym, td, db, thr)
        all_trades.extend(trades)
        # 进度
        if (i + 1) % 5 == 0 or (i + 1) == len(dates):
            print(f"  [{i+1:2d}/{len(dates)}] {td}  累计 {len(all_trades)} 笔")

    if not all_trades:
        print("  !! 无交易记录")
        return pd.DataFrame(), {}

    df = pd.DataFrame(all_trades)
    df["adx_bucket"] = df["adx"].apply(_adx_bucket)
    df["macd_dir"] = df["macd_hist"].apply(_macd_direction)

    # 过滤 partial trades 计算基准
    df_full = df[~df.get("partial", pd.Series(False, index=df.index)).fillna(False)]
    n_base = len(df_full)
    pnl_base = df_full["pnl_pts"].sum()
    wr_base = (df_full["pnl_pts"] > 0).mean() * 100 if n_base > 0 else 0.0

    print(f"\n  基准: {n_base} 笔, PnL={pnl_base:+.1f}pt, WR={wr_base:.1f}%")

    # 运行各分析
    print_analysis_1(df, sym)
    print_analysis_2(df, sym)
    print_analysis_3(df, sym, thr)
    print_analysis_4(df, sym, thr)

    # 计算所有场景（用于最终对比表）
    n_adx, pnl_adx, wr_adx = _sim_with_multipliers(df, thr, adx_mult=True)
    n_macd, pnl_macd, wr_macd = _sim_with_multipliers(df, thr, macd_mult=True)
    n_both, pnl_both, wr_both = _sim_with_multipliers(df, thr, adx_mult=True, macd_mult=True)

    scenario_results = {
        "基准":    {"n": n_base,  "pnl": pnl_base,  "wr": wr_base},
        "+ADX乘数": {"n": n_adx,   "pnl": pnl_adx,   "wr": wr_adx},
        "+MACD乘数": {"n": n_macd,  "pnl": pnl_macd,  "wr": wr_macd},
        "+ADX+MACD": {"n": n_both,  "pnl": pnl_both,  "wr": wr_both},
    }

    return df, scenario_results


def main():
    parser = argparse.ArgumentParser(description="ADX/MACD 回测分析")
    parser.add_argument("--symbol", nargs="+", default=["IM", "IC"],
                        help="品种列表，默认 IM IC")
    parser.add_argument("--dates", default=DATES_30D,
                        help="逗号分隔的回测日期，默认30天")
    args = parser.parse_args()

    db = DBManager(ConfigLoader().get_db_path())
    dates = [d.strip() for d in args.dates.split(",")]

    print(f"\n{'═'*70}")
    print(f"  ADX/MACD 回测分析  ({len(dates)} 天)")
    print(f"  日期: {dates[0]} ~ {dates[-1]}")
    print(f"  品种: {', '.join(args.symbol)}")
    print(f"{'═'*70}")

    all_results: Dict = {}
    all_dfs: Dict[str, pd.DataFrame] = {}

    for sym in args.symbol:
        df, scenario_results = run_symbol(sym, dates, db)
        all_results[sym] = scenario_results
        all_dfs[sym] = df

    if len(all_results) >= 1:
        print_final_comparison(all_results)

    # 详细数据导出（可选）
    for sym, df in all_dfs.items():
        if not df.empty:
            out = f"/tmp/adx_macd_{sym}.csv"
            df.to_csv(out, index=False)
            print(f"  详细数据已保存: {out}")


if __name__ == "__main__":
    main()
