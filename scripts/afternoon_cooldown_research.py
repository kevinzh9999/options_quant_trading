#!/usr/bin/env python3
"""
afternoon_cooldown_research.py
-------------------------------
研究两个参数变更对30天回测的影响：

测试1：午后时间权重敏感分析
  _TIME_WEIGHTS 中 13:00-13:30 BJ (UTC 05:00-05:30) 的权重
  当前 0.8，测试 0.8 / 0.9 / 1.0

测试2：分级冷却
  当前：所有exit_reason统一15分钟冷却
  方案A: TREND_COMPLETE=15, MOMENTUM_EXHAUSTED=20, STOP_LOSS=25, 其他=15
  方案B: TREND_COMPLETE=10, MOMENTUM_EXHAUSTED=25, STOP_LOSS=30, 其他=15
  方案C: 全部20min

**纯分析，不改正式代码。**

Usage:
    python scripts/afternoon_cooldown_research.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# ---------------------------------------------------------------------------
# 30天回测日期
# ---------------------------------------------------------------------------
DATES_30D = [
    "20260204","20260205","20260206","20260209","20260210",
    "20260211","20260212","20260213","20260225","20260226",
    "20260227","20260302","20260303","20260304","20260305",
    "20260306","20260309","20260310","20260311","20260312",
    "20260313","20260316","20260317","20260318","20260319",
    "20260320","20260323","20260324","20260325","20260326",
]

# Per-symbol threshold
THRESHOLDS = {"IM": 60, "IC": 65}
MULT = {"IM": 200, "IC": 200}


# ---------------------------------------------------------------------------
# 辅助：从 backtest_signals_day 借用的工具函数（直接 import 以复用）
# ---------------------------------------------------------------------------
from scripts.backtest_signals_day import (
    _utc_to_bj,
    _build_15m_from_5m,
    _calc_minutes,
)
from strategies.intraday.A_share_momentum_signal_v2 import (
    STOP_LOSS_PCT, TRAILING_STOP_HIVOL, TRAILING_STOP_NORMAL,
    SYMBOL_PROFILES, _DEFAULT_PROFILE,
)


# ---------------------------------------------------------------------------
# 核心回测函数（带 cooldown_map 和 afternoon_weight 参数）
# ---------------------------------------------------------------------------

def run_day_patched(
    sym: str,
    td: str,
    db: DBManager,
    afternoon_weight: float = 0.8,   # UTC 05:00-05:30 权重
    cooldown_map: Dict[str, int] = None,  # exit_reason → cooldown分钟数
) -> List[Dict]:
    """
    与 backtest_signals_day.run_day 逻辑完全相同，但允许：
    1. 覆盖 _TIME_WEIGHTS 中午后时间段权重
    2. 根据 exit_reason 设定不同冷却时长

    Lookahead 防护：
    - bar_5m_signal = bar_5m.iloc[:-1]
    - daily_df 截断到 trade_date < td
    - Z-Score 截断到 trade_date < td
    - sentiment/GARCH 截断到 trade_date < td
    - prevClose 用 trade_date < td 过滤
    """
    # 默认冷却
    if cooldown_map is None:
        cooldown_map = {}
    DEFAULT_COOLDOWN = 15  # 未在 map 中的 reason 用此值

    # ---- Monkey-patch 时间权重（只对本次调用有效） ----
    import strategies.intraday.A_share_momentum_signal_v2 as sig_mod
    original_tw = list(sig_mod._TIME_WEIGHTS)  # 备份
    new_tw = list(sig_mod._TIME_WEIGHTS)
    # index 2 = (time(5,0), time(5,30), 0.8) = 13:00-13:30 BJ
    if len(new_tw) > 2:
        start, end, _ = new_tw[2]
        new_tw[2] = (start, end, afternoon_weight)
    sig_mod._TIME_WEIGHTS = new_tw

    try:
        result = _run_day_inner(sym, td, db, cooldown_map, DEFAULT_COOLDOWN)
    finally:
        sig_mod._TIME_WEIGHTS = original_tw  # 还原

    return result


def _run_day_inner(
    sym: str,
    td: str,
    db: DBManager,
    cooldown_map: Dict[str, int],
    default_cooldown: int,
) -> List[Dict]:
    """实际回测逻辑，几乎与 run_day 完全相同。"""
    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"

    _SPOT_SYM = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
    spot_sym = _SPOT_SYM.get(sym)
    all_bars = None
    if spot_sym:
        all_bars = db.query_df(
            f"SELECT datetime, open, high, low, close, volume "
            f"FROM index_min WHERE symbol='{spot_sym}' AND period=300 "
            f"ORDER BY datetime"
        )
    if all_bars is None or all_bars.empty:
        all_bars = db.query_df(
            f"SELECT datetime, open, high, low, close, volume "
            f"FROM futures_min WHERE symbol='{sym}' AND period=300 "
            f"ORDER BY datetime"
        )
    if all_bars is None or all_bars.empty:
        return []

    for c in ["open", "high", "low", "close", "volume"]:
        all_bars[c] = all_bars[c].astype(float)

    today_mask = all_bars["datetime"].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()
    if not today_indices:
        return []

    _SPOT_IDX = {"IM": "000852.SH", "IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH"}
    idx_code = _SPOT_IDX.get(sym, f"{sym}.CFX")
    daily_all = db.query_df(
        f"SELECT trade_date, close as open, close as high, close as low, close, 0 as volume "
        f"FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if daily_all is not None and not daily_all.empty:
        daily_all["close"] = daily_all["close"].astype(float)
    else:
        daily_all = None

    spot_all = db.query_df(
        f"SELECT trade_date, close FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if spot_all is not None:
        spot_all["close"] = spot_all["close"].astype(float)

    def _zscore_for_date(target_date: str):
        if spot_all is None or spot_all.empty:
            return 0.0, 0.0
        sub = spot_all[spot_all["trade_date"] < target_date].tail(30)
        if len(sub) < 20:
            return 0.0, 0.0
        import pandas as pd
        closes = sub["close"].values
        ema = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])
        std = float(pd.Series(closes).rolling(20).std().iloc[-1])
        return ema, std

    ema20, std20 = _zscore_for_date(td)

    is_high_vol = True
    dmo = db.query_df(
        "SELECT garch_forecast_vol FROM daily_model_output "
        "WHERE underlying='IM' AND garch_forecast_vol > 0 "
        f"AND trade_date < '{td}' "
        "ORDER BY trade_date DESC LIMIT 1"
    )
    if dmo is not None and not dmo.empty:
        is_high_vol = (float(dmo.iloc[0].iloc[0]) * 100 / 24.9) > 1.2

    from strategies.intraday.A_share_momentum_signal_v2 import (
        SignalGeneratorV2, SignalGeneratorV3, SentimentData, check_exit,
        is_open_allowed, SIGNAL_ROUTING,
    )

    sentiment = None
    try:
        sdf = db.query_df(
            "SELECT atm_iv, atm_iv_market, vrp, term_structure_shape, rr_25d "
            "FROM daily_model_output WHERE underlying='IM' "
            f"AND trade_date < '{td}' "
            "ORDER BY trade_date DESC LIMIT 2"
        )
        if sdf is not None and len(sdf) >= 1:
            cur, prev = sdf.iloc[0], (sdf.iloc[1] if len(sdf) >= 2 else sdf.iloc[0])
            sentiment = SentimentData(
                atm_iv=float(cur.get("atm_iv_market") or cur.get("atm_iv") or 0),
                atm_iv_prev=float(prev.get("atm_iv_market") or prev.get("atm_iv") or 0),
                rr_25d=float(cur.get("rr_25d") or 0),
                rr_25d_prev=float(prev.get("rr_25d") or 0),
                vrp=float(cur.get("vrp") or 0),
                term_structure=str(cur.get("term_structure_shape") or ""),
            )
    except Exception:
        pass

    _ver = SIGNAL_ROUTING.get(sym, "v2")
    gen = SignalGeneratorV2({"min_signal_score": 60}) if _ver == "v2" else SignalGeneratorV3({"min_signal_score": 60})

    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    effective_threshold = _sym_prof.get("signal_threshold", 60)

    d_override = None
    try:
        briefing_row = db.query_df(
            "SELECT d_override_long, d_override_short FROM morning_briefing "
            f"WHERE trade_date = '{td}' LIMIT 1"
        )
        if briefing_row is not None and len(briefing_row) > 0:
            d_long = briefing_row.iloc[0].get("d_override_long")
            d_short = briefing_row.iloc[0].get("d_override_short")
            if d_long is not None and d_short is not None:
                d_override = {"LONG": float(d_long), "SHORT": float(d_short)}
    except Exception:
        pass

    position: Optional[Dict] = None
    completed_trades: List[Dict] = []
    last_exit_utc: str = ""
    last_exit_dir: str = ""
    last_exit_reason: str = ""   # ← 新增：记录上次平仓原因

    daily_df = None
    if daily_all is not None:
        daily_df = daily_all[daily_all["trade_date"] < td].tail(30).reset_index(drop=True)
        if daily_df.empty:
            daily_df = None

    _gap_pct = 0.0
    prev_c = 0.0
    if daily_df is not None and len(daily_df) >= 2:
        prev_rows = daily_df[daily_df["trade_date"] < td]
        if len(prev_rows) > 0:
            prev_c = float(prev_rows.iloc[-1]["close"])

    _today_open = 0.0
    if prev_c > 0 and today_indices:
        _today_open = float(all_bars.loc[today_indices[0], "open"])
        _gap_pct = (_today_open - prev_c) / prev_c

    for idx in today_indices:
        bar_5m = all_bars.loc[:idx].tail(200).copy()
        if len(bar_5m) < 15:
            continue

        bar_5m_signal = bar_5m.iloc[:-1]
        if len(bar_5m_signal) < 15:
            continue

        price = float(bar_5m.iloc[-1]["close"])
        high = float(bar_5m.iloc[-1]["high"])
        low = float(bar_5m.iloc[-1]["low"])
        signal_price = float(bar_5m_signal.iloc[-1]["close"])
        dt_str = str(all_bars.loc[idx, "datetime"])
        utc_hm = dt_str[11:16]

        z_val = (signal_price - ema20) / std20 if std20 > 0 else None
        bar_15m = _build_15m_from_5m(bar_5m_signal)

        result = gen.score_all(
            sym, bar_5m_signal, bar_15m, daily_df, None, sentiment,
            zscore=z_val, is_high_vol=is_high_vol, d_override=d_override,
        )

        score = result["total"] if result else 0
        direction = result["direction"] if result else ""

        action_str = ""

        if position is not None:
            stop_price = position.get("stop_loss", 0)
            bar_stopped = False
            if stop_price > 0:
                if position["direction"] == "LONG" and low <= stop_price:
                    bar_stopped = True
                elif position["direction"] == "SHORT" and high >= stop_price:
                    bar_stopped = True

            if bar_stopped:
                entry_p = position["entry_price"]
                exit_p = stop_price if position["direction"] == "LONG" else stop_price
                if position["direction"] == "LONG":
                    pnl_pts = exit_p - entry_p
                else:
                    pnl_pts = entry_p - exit_p
                elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)
                reason = "STOP_LOSS"
                completed_trades.append({
                    "entry_time": _utc_to_bj(position["entry_time_utc"]),
                    "entry_price": entry_p,
                    "exit_time": _utc_to_bj(utc_hm),
                    "exit_price": exit_p,
                    "direction": position["direction"],
                    "pnl_pts": pnl_pts,
                    "reason": reason,
                    "minutes": elapsed,
                    "entry_score": position.get("entry_score", 0),
                    "entry_daily_mult": position.get("entry_daily_mult", 1.0),
                })
                last_exit_utc = utc_hm
                last_exit_dir = position["direction"]
                last_exit_reason = reason
                position = None
                action_str = f"EXIT {reason}"
            else:
                if position["direction"] == "LONG":
                    position["highest_since"] = max(position["highest_since"], high)
                else:
                    position["lowest_since"] = min(position["lowest_since"], low)

                reverse_score = 0
                if result and direction and direction != position["direction"]:
                    reverse_score = score

                exit_info = check_exit(
                    position, price, bar_5m_signal,
                    bar_15m if not bar_15m.empty else None,
                    utc_hm, reverse_score, is_high_vol=is_high_vol,
                    symbol=sym,
                )

                if exit_info["should_exit"]:
                    exit_vol = exit_info["exit_volume"]
                    reason = exit_info["exit_reason"]
                    entry_p = position["entry_price"]
                    exit_p = price
                    if position["direction"] == "LONG":
                        pnl_pts = exit_p - entry_p
                    else:
                        pnl_pts = entry_p - exit_p
                    elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)

                    is_full = (exit_vol >= position["volume"])
                    completed_trades.append({
                        "entry_time": _utc_to_bj(position["entry_time_utc"]),
                        "entry_price": entry_p,
                        "exit_time": _utc_to_bj(utc_hm),
                        "exit_price": exit_p,
                        "direction": position["direction"],
                        "pnl_pts": pnl_pts,
                        "reason": reason + ("" if is_full else "(半仓)"),
                        "minutes": elapsed,
                        "partial": not is_full,
                        "entry_score": position.get("entry_score", 0),
                        "entry_daily_mult": position.get("entry_daily_mult", 1.0),
                    })
                    if is_full:
                        last_exit_utc = utc_hm
                        last_exit_dir = position["direction"]
                        last_exit_reason = reason
                        position = None
                    else:
                        position["volume"] -= exit_vol
                        position["half_closed"] = True
                    action_str = f"EXIT {reason}"

        # ---- 分级冷却检查 ----
        in_cooldown = False
        if last_exit_utc and direction == last_exit_dir:
            cd_elapsed = _calc_minutes(last_exit_utc, utc_hm)
            # 根据上次平仓原因决定冷却时长
            cd_minutes = cooldown_map.get(last_exit_reason, default_cooldown)
            if 0 < cd_elapsed < cd_minutes:
                in_cooldown = True

        if (position is None and not action_str and result and not in_cooldown
                and score >= effective_threshold and direction and is_open_allowed(utc_hm)):
            entry_p = price
            stop = entry_p * (1 - STOP_LOSS_PCT) if direction == "LONG" else entry_p * (1 + STOP_LOSS_PCT)
            position = {
                "entry_price": entry_p,
                "direction": direction,
                "entry_time_utc": utc_hm,
                "highest_since": high,
                "lowest_since": low,
                "stop_loss": stop,
                "volume": 1,
                "half_closed": False,
                "bars_below_mid": 0,
                "entry_score": score,
                "entry_daily_mult": result.get("daily_mult", 1.0),
                "entry_raw_total": result.get("raw_total", 0),
                "entry_gap_pct": _gap_pct,
                "entry_filter_mult": result.get("intraday_filter", 1.0),
                "entry_zscore": z_val,
            }

    # Force close
    if position is not None:
        last_price = float(all_bars.loc[today_indices[-1], "close"])
        entry_p = position["entry_price"]
        exit_p = last_price
        pnl = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
        elapsed = _calc_minutes(position["entry_time_utc"], str(all_bars.loc[today_indices[-1], "datetime"])[11:16])
        completed_trades.append({
            "entry_time": _utc_to_bj(position["entry_time_utc"]),
            "entry_price": entry_p,
            "exit_time": _utc_to_bj(str(all_bars.loc[today_indices[-1], "datetime"])[11:16]),
            "exit_price": exit_p,
            "direction": position["direction"],
            "pnl_pts": pnl,
            "reason": "EOD_FORCE",
            "minutes": elapsed,
            "entry_score": position.get("entry_score", 0),
            "entry_daily_mult": position.get("entry_daily_mult", 1.0),
        })

    return completed_trades


# ---------------------------------------------------------------------------
# 汇总统计
# ---------------------------------------------------------------------------

def summarize(sym: str, all_trades: List[Tuple[str, Dict]], dates: List[str]) -> Dict:
    """从 (td, trade_dict) 列表中计算汇总统计。"""
    full_trades = [t for _, t in all_trades if not t.get("partial")]
    pnl = sum(t["pnl_pts"] for t in full_trades)
    n = len(full_trades)
    wins = len([t for t in full_trades if t["pnl_pts"] > 0])
    wr = wins / n * 100 if n > 0 else 0.0
    avg_pnl = pnl / n if n > 0 else 0.0
    mult = MULT.get(sym, 200)
    be_slip = avg_pnl / 2 if n > 0 else 0.0  # breakeven slippage per side
    return {
        "pnl": pnl,
        "yuan": pnl * mult,
        "n": n,
        "wr": wr,
        "avg_pnl": avg_pnl,
        "be_slip": be_slip,
    }


def run_scenario(sym: str, dates: List[str], db: DBManager,
                 afternoon_weight: float, cooldown_map: Dict[str, int]) -> Dict:
    all_trades = []
    for td in dates:
        trades = run_day_patched(sym, td, db, afternoon_weight, cooldown_map)
        all_trades.extend([(td, t) for t in trades])
    return summarize(sym, all_trades, dates)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    db = DBManager(ConfigLoader().get_db_path())
    dates = DATES_30D
    symbols = ["IM", "IC"]

    print(f"\n{'='*80}")
    print(f" 午后时间权重 + 分级冷却 回测研究")
    print(f" 日期范围: {dates[0]}~{dates[-1]} ({len(dates)}天)")
    print(f" 品种: {symbols}")
    print(f"{'='*80}\n")

    # =========================================================================
    # 测试1：午后时间权重敏感分析
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f" 测试1：午后时间权重（13:00-13:30 BJ）敏感分析")
    print(f"{'─'*80}")

    afternoon_weights = [0.8, 0.9, 1.0]
    baseline_cooldown = {}  # 默认15min（不分级）

    t1_results = {}  # (sym, aw) -> stats
    for sym in symbols:
        for aw in afternoon_weights:
            print(f"  Running {sym} afternoon_weight={aw}...", flush=True)
            stats = run_scenario(sym, dates, db, aw, baseline_cooldown)
            t1_results[(sym, aw)] = stats

    # 输出表格
    print(f"\n{'='*80}")
    print(f" 测试1 结果表")
    print(f"{'='*80}")

    header = f"{'t_afternoon':>12} | {'IM PnL':>8} | {'IM T':>5} | {'IM WR':>6} | {'IC PnL':>8} | {'IC T':>5} | {'IC WR':>6} | {'合计':>8} | {'vs base':>8}"
    print(header)
    print(f"{'─'*len(header)}")

    baseline_total = (t1_results[("IM", 0.8)]["pnl"] + t1_results[("IC", 0.8)]["pnl"])
    for aw in afternoon_weights:
        im = t1_results[("IM", aw)]
        ic = t1_results[("IC", aw)]
        total = im["pnl"] + ic["pnl"]
        vs_base = total - baseline_total
        tag = "  ← baseline" if aw == 0.8 else ""
        print(
            f"  {aw:>10.1f} | {im['pnl']:>+8.0f} | {im['n']:>5} | {im['wr']:>5.1f}%"
            f" | {ic['pnl']:>+8.0f} | {ic['n']:>5} | {ic['wr']:>5.1f}%"
            f" | {total:>+8.0f} | {vs_base:>+8.0f}{tag}"
        )

    print(f"\n  详细（avg PnL/trade, breakeven slip）：")
    for sym in symbols:
        print(f"  {sym}:")
        for aw in afternoon_weights:
            s = t1_results[(sym, aw)]
            tag = " ← baseline" if aw == 0.8 else ""
            print(f"    aw={aw}: avg={s['avg_pnl']:+.1f}pt  BE_slip={s['be_slip']:.1f}pt  "
                  f"yuan={s['yuan']:+,.0f}{tag}")

    # 找最优 aw
    best_aw_by_total = max(afternoon_weights, key=lambda aw: t1_results[("IM", aw)]["pnl"] + t1_results[("IC", aw)]["pnl"])
    print(f"\n  → 综合最优 afternoon_weight = {best_aw_by_total}")

    # =========================================================================
    # 测试2：分级冷却
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f" 测试2：分级冷却方案")
    print(f"{'─'*80}")

    cooldown_schemes = {
        "Baseline(15min)": {},
        "方案A(SL=25,ME=20,TC=15)": {"STOP_LOSS": 25, "MOMENTUM_EXHAUSTED": 20, "TREND_COMPLETE": 15},
        "方案B(SL=30,ME=25,TC=10)": {"STOP_LOSS": 30, "MOMENTUM_EXHAUSTED": 25, "TREND_COMPLETE": 10},
        "方案C(全部20min)": {"STOP_LOSS": 20, "MOMENTUM_EXHAUSTED": 20, "TREND_COMPLETE": 20,
                           "TRAILING_STOP": 20, "TIME_STOP": 20, "EOD_CLOSE": 20, "LUNCH_CLOSE": 20},
    }

    t2_results = {}  # (sym, scheme_name) -> stats
    for sym in symbols:
        for scheme_name, cooldown_map in cooldown_schemes.items():
            print(f"  Running {sym} {scheme_name}...", flush=True)
            stats = run_scenario(sym, dates, db, 0.8, cooldown_map)  # 用 baseline aw=0.8
            t2_results[(sym, scheme_name)] = stats

    print(f"\n{'='*80}")
    print(f" 测试2 结果表")
    print(f"{'='*80}")

    header2 = f"{'冷却方案':>28} | {'IM PnL':>8} | {'IM T':>5} | {'IM WR':>6} | {'IC PnL':>8} | {'IC T':>5} | {'IC WR':>6} | {'合计':>8} | {'vs base':>8}"
    print(header2)
    print(f"{'─'*len(header2)}")

    baseline_scheme = "Baseline(15min)"
    base_total2 = (t2_results[("IM", baseline_scheme)]["pnl"] + t2_results[("IC", baseline_scheme)]["pnl"])
    for scheme_name in cooldown_schemes.keys():
        im = t2_results[("IM", scheme_name)]
        ic = t2_results[("IC", scheme_name)]
        total = im["pnl"] + ic["pnl"]
        vs_base = total - base_total2
        tag = "  ← baseline" if scheme_name == baseline_scheme else ""
        print(
            f"  {scheme_name:>26} | {im['pnl']:>+8.0f} | {im['n']:>5} | {im['wr']:>5.1f}%"
            f" | {ic['pnl']:>+8.0f} | {ic['n']:>5} | {ic['wr']:>5.1f}%"
            f" | {total:>+8.0f} | {vs_base:>+8.0f}{tag}"
        )

    print(f"\n  详细（avg PnL/trade, breakeven slip）：")
    for sym in symbols:
        print(f"  {sym}:")
        for scheme_name in cooldown_schemes.keys():
            s = t2_results[(sym, scheme_name)]
            tag = " ← baseline" if scheme_name == baseline_scheme else ""
            print(f"    {scheme_name}: avg={s['avg_pnl']:+.1f}pt  BE_slip={s['be_slip']:.1f}pt  "
                  f"yuan={s['yuan']:+,.0f}{tag}")

    best_scheme = max(cooldown_schemes.keys(),
                      key=lambda k: t2_results[("IM", k)]["pnl"] + t2_results[("IC", k)]["pnl"])
    print(f"\n  → 综合最优冷却方案 = {best_scheme}")

    # =========================================================================
    # 交叉测试：最优 aw × 最优冷却方案
    # =========================================================================
    best_scheme_map = cooldown_schemes[best_scheme]
    run_cross = (best_aw_by_total != 0.8) or (best_scheme != baseline_scheme)

    if run_cross:
        print(f"\n{'─'*80}")
        print(f" 交叉测试：最优参数组合")
        print(f"  afternoon_weight = {best_aw_by_total}")
        print(f"  冷却方案 = {best_scheme}")
        print(f"{'─'*80}")

        cross_results = {}
        for sym in symbols:
            print(f"  Running {sym} cross...", flush=True)
            stats = run_scenario(sym, dates, db, best_aw_by_total, best_scheme_map)
            cross_results[sym] = stats

        im_c = cross_results["IM"]
        ic_c = cross_results["IC"]
        total_c = im_c["pnl"] + ic_c["pnl"]
        vs_t1_base = total_c - baseline_total
        vs_t2_base = total_c - base_total2

        print(f"\n  组合结果：")
        print(f"  IM: {im_c['pnl']:+.0f}pt  {im_c['n']}笔  WR={im_c['wr']:.1f}%  avg={im_c['avg_pnl']:+.1f}pt  BE={im_c['be_slip']:.1f}pt")
        print(f"  IC: {ic_c['pnl']:+.0f}pt  {ic_c['n']}笔  WR={ic_c['wr']:.1f}%  avg={ic_c['avg_pnl']:+.1f}pt  BE={ic_c['be_slip']:.1f}pt")
        print(f"  合计: {total_c:+.0f}pt  vs Baseline: {vs_t1_base:+.0f}pt")
    else:
        print(f"\n  → 最优参数均为 baseline，无需交叉测试")

    # =========================================================================
    # 最终汇总
    # =========================================================================
    print(f"\n{'='*80}")
    print(f" 最终结论")
    print(f"{'='*80}")
    print(f"  测试1 最优 afternoon_weight: {best_aw_by_total}")
    print(f"  测试2 最优冷却方案: {best_scheme}")

    # 对比各指标
    print(f"\n  IM 各方案对比 (合计 IM+IC PnL):")
    print(f"  {'参数':>40} | {'PnL':>8} | {'增量':>8}")
    print(f"  {'─'*60}")
    # T1
    for aw in afternoon_weights:
        im = t1_results[("IM", aw)]
        ic = t1_results[("IC", aw)]
        total = im["pnl"] + ic["pnl"]
        delta = total - baseline_total
        tag = " ← baseline" if aw == 0.8 else ""
        print(f"  {'T1: aw='+str(aw):>40} | {total:>+8.0f} | {delta:>+8.0f}{tag}")
    # T2
    for scheme_name in cooldown_schemes.keys():
        im = t2_results[("IM", scheme_name)]
        ic = t2_results[("IC", scheme_name)]
        total = im["pnl"] + ic["pnl"]
        delta = total - baseline_total
        tag = " ← baseline" if scheme_name == baseline_scheme else ""
        label = f"T2: {scheme_name}"
        print(f"  {label:>40} | {total:>+8.0f} | {delta:>+8.0f}{tag}")

    print()


if __name__ == "__main__":
    main()
