#!/usr/bin/env python3
"""
body_ratio_research.py
----------------------
K线实体占比（Body Ratio）作为M分调节器的可行性分析。

Usage:
    python scripts/body_ratio_research.py --symbol IM
    python scripts/body_ratio_research.py --symbol IC
    python scripts/body_ratio_research.py --symbol IM,IC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# ────────────────────────────────────────────────────────────────────────────
# 常量
# ────────────────────────────────────────────────────────────────────────────

_SPOT_SYM = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
_SPOT_IDX = {"IM": "000852.SH", "IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH"}
_MULT = {"IM": 200, "IF": 300, "IH": 300, "IC": 200}

DATES = (
    "20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,"
    "20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,"
    "20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,"
    "20260319,20260320,20260323,20260324,20260325,20260326,20260327,20260328,"
    "20260401,20260402"
).split(",")

# ────────────────────────────────────────────────────────────────────────────
# Body Ratio 计算
# ────────────────────────────────────────────────────────────────────────────

def body_ratio(o, h, l, c) -> float:
    """(close - open) / (high - low + 1e-5).  +1=大阳线, -1=大阴线, 0=十字星."""
    return (c - o) / (h - l + 1e-5)


def upper_shadow(o, h, l, c) -> float:
    """上影线占比."""
    return (h - max(o, c)) / (h - l + 1e-5)


def lower_shadow(o, h, l, c) -> float:
    """下影线占比."""
    return (min(o, c) - l) / (h - l + 1e-5)


# ────────────────────────────────────────────────────────────────────────────
# 辅助：复用 backtest_signals_day.py 的核心逻辑
# ────────────────────────────────────────────────────────────────────────────

def _utc_to_bj(utc_str: str) -> str:
    h = int(utc_str[:2]) + 8
    if h >= 24:
        h -= 24
    return f"{h:02d}:{utc_str[3:5]}"


def _build_15m_from_5m(bar_5m: pd.DataFrame) -> pd.DataFrame:
    if len(bar_5m) < 3:
        return pd.DataFrame()
    df = bar_5m.copy()
    df["dt"] = pd.to_datetime(df["datetime"] if "datetime" in df.columns else df.index)
    df = df.set_index("dt")
    resampled = df.resample("15min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()
    return resampled.reset_index(drop=True)


def _calc_minutes(t1: str, t2: str) -> int:
    try:
        h1, m1 = int(t1[:2]), int(t1[3:5])
        h2, m2 = int(t2[:2]), int(t2[3:5])
        return (h2 * 60 + m2) - (h1 * 60 + m1)
    except Exception:
        return 0


# ────────────────────────────────────────────────────────────────────────────
# Step 1: 全量Bar相关性分析
# ────────────────────────────────────────────────────────────────────────────

def step1_correlation(sym: str, db: DBManager) -> pd.DataFrame:
    """
    对所有5分钟bar，计算body_ratio与M/V/Q proxy的Pearson相关。
    使用全量历史数据（不限于回测窗口），样本更充分。
    """
    spot_sym = _SPOT_SYM[sym]
    bars = db.query_df(
        f"SELECT datetime, open, high, low, close, volume "
        f"FROM index_min WHERE symbol='{spot_sym}' AND period=300 "
        f"ORDER BY datetime"
    )
    if bars is None or bars.empty:
        print(f"  [WARN] No data for {sym}")
        return pd.DataFrame()

    for c in ["open", "high", "low", "close", "volume"]:
        bars[c] = bars[c].astype(float)

    # 计算 body_ratio
    bars["br"] = [
        body_ratio(r["open"], r["high"], r["low"], r["close"])
        for _, r in bars.iterrows()
    ]

    # M_proxy: 最近12根bar的close变化率（当前收盘相对12根前）
    bars["M_proxy"] = bars["close"].pct_change(12)

    # V_proxy: ATR(5)/ATR(40)
    def rolling_atr(df: pd.DataFrame, n: int) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    atr5  = rolling_atr(bars, 5)
    atr40 = rolling_atr(bars, 40)
    bars["V_proxy"] = atr5 / (atr40 + 1e-10)

    # Q_proxy: volume/MA(volume, 20)
    bars["Q_proxy"] = bars["volume"] / (bars["volume"].rolling(20).mean() + 1e-10)

    # 过滤无效行
    valid = bars.dropna(subset=["br", "M_proxy", "V_proxy", "Q_proxy"])
    valid = valid[valid["M_proxy"].abs() < 0.1]   # 去除异常跳变

    n = len(valid)
    results = []
    for proxy, label in [("M_proxy", "M(12根变化率)"),
                          ("V_proxy", "V(ATR5/ATR40)"),
                          ("Q_proxy", "Q(量/MA20)")]:
        r, pval = scipy_stats.pearsonr(valid["br"], valid[proxy])
        results.append({
            "品种": sym,
            "Proxy": label,
            "Pearson_r": round(r, 4),
            "p_value": f"{pval:.2e}",
            "|r|<0.5?": "✓ 独立" if abs(r) < 0.5 else "✗ 相关",
            "样本数": n,
        })
    return pd.DataFrame(results)


# ────────────────────────────────────────────────────────────────────────────
# Step 2 + Step 3 核心回测：带body_ratio记录的完整回测
# ────────────────────────────────────────────────────────────────────────────

def run_day_with_br(sym: str, td: str, db: DBManager,
                    scheme: str = "baseline") -> List[Dict]:
    """
    在 backtest_signals_day.py 的 run_day 基础上，
    额外记录 entry_body_ratio 和影线信息。
    scheme: "baseline" / "scheme_A" / "scheme_B"

    注意：scheme_A/B 是事后模拟（对baseline的score做乘数修正），
    与真实集成会有微小差异（因为score调整不影响check_exit中的reverse_score）。
    标注：事后模拟，与真实集成可能有偏差（<2%）。
    """
    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"
    spot_sym = _SPOT_SYM.get(sym)
    all_bars = None
    if spot_sym:
        all_bars = db.query_df(
            f"SELECT datetime, open, high, low, close, volume "
            f"FROM index_min WHERE symbol='{spot_sym}' AND period=300 "
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

    # 日线数据
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
        SignalGeneratorV2, SentimentData, check_exit,
        is_open_allowed, SYMBOL_PROFILES, _DEFAULT_PROFILE,
        STOP_LOSS_PCT, TRAILING_STOP_HIVOL, TRAILING_STOP_NORMAL,
    )

    sentiment = None
    try:
        sdf = db.query_df(
            "SELECT atm_iv, atm_iv_market, vrp, term_structure_shape, rr_25d "
            "FROM daily_model_output WHERE underlying='IM' "
            f"AND trade_date < '{td}' ORDER BY trade_date DESC LIMIT 2"
        )
        if sdf is not None and len(sdf) >= 1:
            cur = sdf.iloc[0]
            prev = sdf.iloc[1] if len(sdf) >= 2 else sdf.iloc[0]
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

    gen = SignalGeneratorV2({"min_signal_score": 60})
    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    effective_threshold = _sym_prof.get("signal_threshold", 60)

    # 截取日线
    daily_df = None
    if daily_all is not None:
        daily_df = daily_all[daily_all["trade_date"] < td].tail(30).reset_index(drop=True)
        if daily_df.empty:
            daily_df = None

    # prevClose
    prev_c = 0.0
    if daily_df is not None:
        prev_rows = daily_df[daily_df["trade_date"] < td]
        if len(prev_rows) > 0:
            prev_c = float(prev_rows.iloc[-1]["close"])

    _gap_pct = 0.0
    _today_open = 0.0
    if prev_c > 0 and today_indices:
        _today_open = float(all_bars.loc[today_indices[0], "open"])
        _gap_pct = (_today_open - prev_c) / prev_c

    position: Optional[Dict] = None
    completed_trades: List[Dict] = []
    last_exit_utc: str = ""
    last_exit_dir: str = ""
    COOLDOWN_MINUTES = 15

    for idx in today_indices:
        bar_5m = all_bars.loc[:idx].tail(200).copy()
        if len(bar_5m) < 15:
            continue

        bar_5m_signal = bar_5m.iloc[:-1]
        if len(bar_5m_signal) < 15:
            continue

        price = float(bar_5m.iloc[-1]["close"])
        high  = float(bar_5m.iloc[-1]["high"])
        low   = float(bar_5m.iloc[-1]["low"])
        signal_price = float(bar_5m_signal.iloc[-1]["close"])
        dt_str = str(all_bars.loc[idx, "datetime"])
        utc_hm = dt_str[11:16]
        bj_time = _utc_to_bj(utc_hm)

        z_val = (signal_price - ema20) / std20 if std20 > 0 else None
        bar_15m = _build_15m_from_5m(bar_5m_signal)

        result = gen.score_all(
            sym, bar_5m_signal, bar_15m, daily_df, None, sentiment,
            zscore=z_val, is_high_vol=is_high_vol, d_override=d_override,
        )

        score_orig = result["total"] if result else 0
        direction  = result["direction"] if result else ""

        # ── 计算开仓时的 body_ratio（T-1 bar，即 bar_5m_signal 的最后一根）
        # 这就是实盘monitor中「已完成的最后一根bar」
        last_sig_bar = bar_5m_signal.iloc[-1]
        br_entry = body_ratio(
            last_sig_bar["open"], last_sig_bar["high"],
            last_sig_bar["low"], last_sig_bar["close"]
        )
        us_entry = upper_shadow(
            last_sig_bar["open"], last_sig_bar["high"],
            last_sig_bar["low"], last_sig_bar["close"]
        )
        ls_entry = lower_shadow(
            last_sig_bar["open"], last_sig_bar["high"],
            last_sig_bar["low"], last_sig_bar["close"]
        )

        # ── 连续3根body_ratio（用于Scheme B）
        br3 = []
        for i in [-3, -2, -1]:
            try:
                r = bar_5m_signal.iloc[i]
                br3.append(body_ratio(r["open"], r["high"], r["low"], r["close"]))
            except Exception:
                br3.append(0.0)

        # ── Scheme 修正 score（事后模拟）
        def _apply_scheme(score: int, direction: str, br: float, br_list: List[float]) -> int:
            if scheme == "baseline" or not direction:
                return score
            if scheme == "scheme_A":
                # 方向一致性
                if direction == "LONG":
                    if br < -0.2:            # 阴线矛盾
                        mult = 0.85
                    elif abs(br) < 0.2:      # 十字星
                        mult = 0.90
                    else:
                        mult = 1.0           # 一致或弱阳
                else:  # SHORT
                    if br > 0.2:             # 阳线矛盾
                        mult = 0.85
                    elif abs(br) < 0.2:
                        mult = 0.90
                    else:
                        mult = 1.0
                return int(round(min(100, max(0, score * mult))))
            elif scheme == "scheme_B":
                # 连续3根一致性
                if direction == "LONG":
                    all_pos = all(b > 0 for b in br_list)
                    last_strong = br_list[-1] > 0.5 if br_list else False
                    all_same = (sum(1 for b in br_list if b > 0) >= 2)  # 至少2/3同向
                    inconsistent = sum(1 for b in br_list if b < 0) >= 2
                elif direction == "SHORT":
                    all_pos = all(b < 0 for b in br_list)
                    last_strong = br_list[-1] < -0.5 if br_list else False
                    all_same = (sum(1 for b in br_list if b < 0) >= 2)
                    inconsistent = sum(1 for b in br_list if b > 0) >= 2
                else:
                    return score
                if all_pos and last_strong:
                    return int(round(min(100, score * 1.05)))
                elif inconsistent:
                    return int(round(max(0, score * 0.90)))
                else:
                    return score
            return score

        score = _apply_scheme(score_orig, direction, br_entry, br3)

        # ── 止损检查
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
                exit_p = stop_price
                pnl_pts = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)
                completed_trades.append({**_trade_dict(position, exit_p, pnl_pts, bj_time, elapsed, "STOP_LOSS")})
                last_exit_utc = utc_hm
                last_exit_dir = position["direction"]
                position = None
            else:
                if position["direction"] == "LONG":
                    position["highest_since"] = max(position["highest_since"], high)
                else:
                    position["lowest_since"] = min(position["lowest_since"], low)

                reverse_score = score if (direction and direction != position["direction"]) else 0
                exit_info = check_exit(
                    position, price, bar_5m_signal,
                    bar_15m if not bar_15m.empty else None,
                    utc_hm, reverse_score, is_high_vol=is_high_vol, symbol=sym,
                )
                if exit_info["should_exit"]:
                    exit_vol = exit_info["exit_volume"]
                    reason = exit_info["exit_reason"]
                    entry_p = position["entry_price"]
                    exit_p = price
                    pnl_pts = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                    elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)
                    if exit_vol >= position["volume"]:
                        completed_trades.append({**_trade_dict(position, exit_p, pnl_pts, bj_time, elapsed, reason)})
                        last_exit_utc = utc_hm
                        last_exit_dir = position["direction"]
                        position = None
                    else:
                        position["volume"] -= exit_vol
                        position["half_closed"] = True
                        completed_trades.append({
                            **_trade_dict(position, exit_p, pnl_pts, bj_time, elapsed, reason + "(半仓)"),
                            "partial": True
                        })

        # ── 开仓检查
        in_cooldown = (last_exit_utc and direction == last_exit_dir
                       and 0 < _calc_minutes(last_exit_utc, utc_hm) < COOLDOWN_MINUTES)

        if (position is None and not action_str and result and not in_cooldown
                and score >= effective_threshold and direction and is_open_allowed(utc_hm)):
            stop = price * (1 - STOP_LOSS_PCT) if direction == "LONG" else price * (1 + STOP_LOSS_PCT)
            position = {
                "entry_price": price,
                "direction": direction,
                "entry_time_utc": utc_hm,
                "highest_since": high,
                "lowest_since": low,
                "stop_loss": stop,
                "volume": 1,
                "half_closed": False,
                "bars_below_mid": 0,
                "entry_score": score,
                "entry_score_orig": score_orig,
                "entry_br": br_entry,
                "entry_us": us_entry,
                "entry_ls": ls_entry,
                "entry_br3": br3,
                "entry_z": z_val,
                "entry_daily_mult": result.get("daily_mult", 1.0),
                "entry_raw_total": result.get("raw_total", 0),
                "entry_s_momentum": result.get("s_momentum", 0),
                "entry_gap_pct": _gap_pct,
            }

    # 强制平仓
    if position is not None:
        last_price = float(all_bars.loc[today_indices[-1], "close"])
        entry_p = position["entry_price"]
        pnl_pts = (last_price - entry_p) if position["direction"] == "LONG" else (entry_p - last_price)
        elapsed = _calc_minutes(
            position["entry_time_utc"],
            str(all_bars.loc[today_indices[-1], "datetime"])[11:16]
        )
        completed_trades.append({
            **_trade_dict(position, last_price, pnl_pts,
                          _utc_to_bj(str(all_bars.loc[today_indices[-1], "datetime"])[11:16]),
                          elapsed, "EOD_FORCE")
        })

    return completed_trades


def _trade_dict(position: Dict, exit_p: float, pnl_pts: float,
                exit_time: str, elapsed: int, reason: str) -> Dict:
    return {
        "direction": position["direction"],
        "entry_price": position["entry_price"],
        "exit_price": exit_p,
        "entry_time": _utc_to_bj(position["entry_time_utc"]),
        "exit_time": exit_time,
        "pnl_pts": pnl_pts,
        "reason": reason,
        "minutes": elapsed,
        "entry_score": position.get("entry_score", 0),
        "entry_score_orig": position.get("entry_score_orig", 0),
        "entry_br": position.get("entry_br", 0.0),
        "entry_us": position.get("entry_us", 0.0),
        "entry_ls": position.get("entry_ls", 0.0),
        "entry_br3": position.get("entry_br3", []),
        "entry_z": position.get("entry_z"),
        "entry_daily_mult": position.get("entry_daily_mult", 1.0),
        "entry_raw_total": position.get("entry_raw_total", 0),
        "entry_s_momentum": position.get("entry_s_momentum", 0),
        "entry_gap_pct": position.get("entry_gap_pct", 0.0),
    }


# ────────────────────────────────────────────────────────────────────────────
# Step 2: 交易预测力分析
# ────────────────────────────────────────────────────────────────────────────

def step2_predictive_power(trades_df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """
    按 body_ratio 分组，分析各组的 WR 和平均 PnL。
    分 LONG/SHORT 两个方向分别计算。
    """
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for direction in ["LONG", "SHORT"]:
        sub = trades_df[trades_df["direction"] == direction].copy()
        if sub.empty:
            continue
        n_total = len(sub)

        # 分组定义
        if direction == "LONG":
            bins = [
                (">+0.6 强阳确认",   sub["entry_br"] > 0.6),
                ("+0.2~+0.6 阳线",   (sub["entry_br"] >= 0.2) & (sub["entry_br"] <= 0.6)),
                ("-0.2~+0.2 十字星", (sub["entry_br"] > -0.2)  & (sub["entry_br"] < 0.2)),
                ("<-0.2 阴线矛盾",   sub["entry_br"] < -0.2),
            ]
        else:
            bins = [
                ("<-0.6 强阴确认",   sub["entry_br"] < -0.6),
                ("-0.6~-0.2 阴线",   (sub["entry_br"] <= -0.2) & (sub["entry_br"] >= -0.6)),
                ("-0.2~+0.2 十字星", (sub["entry_br"] > -0.2)  & (sub["entry_br"] < 0.2)),
                (">+0.2 阳线矛盾",   sub["entry_br"] > 0.2),
            ]

        for label, mask in bins:
            grp = sub[mask]
            if len(grp) == 0:
                rows.append({
                    "品种": sym, "方向": direction, "分组": label,
                    "笔数": 0, "占比%": 0, "WR%": "-", "均PnL(pt)": "-",
                    "总PnL(pt)": "-",
                })
                continue
            n = len(grp)
            wins = (grp["pnl_pts"] > 0).sum()
            wr = wins / n * 100
            avg_pnl = grp["pnl_pts"].mean()
            total_pnl = grp["pnl_pts"].sum()
            rows.append({
                "品种": sym, "方向": direction, "分组": label,
                "笔数": n, "占比%": round(n / n_total * 100, 1),
                "WR%": round(wr, 1), "均PnL(pt)": round(avg_pnl, 1),
                "总PnL(pt)": round(total_pnl, 1),
            })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Step 3: 方案对比
# ────────────────────────────────────────────────────────────────────────────

def step3_scheme_comparison(sym: str, db: DBManager) -> pd.DataFrame:
    """
    运行 baseline / scheme_A / scheme_B，对比总PnL / 笔数 / WR / 均PnL / BE滑点。
    事后模拟，与真实集成可能有偏差。
    """
    results = {}
    for scheme in ["baseline", "scheme_A", "scheme_B"]:
        all_trades = []
        for td in DATES:
            trades = run_day_with_br(sym, td, db, scheme=scheme)
            all_trades.extend(trades)
        full = [t for t in all_trades if not t.get("partial")]
        total_pnl = sum(t["pnl_pts"] for t in full)
        n = len(full)
        wins = sum(1 for t in full if t["pnl_pts"] > 0)
        wr = wins / n * 100 if n > 0 else 0
        avg_pnl = total_pnl / n if n > 0 else 0
        # Breakeven滑点: avg_pnl / 2（双边）
        be = avg_pnl / 2 if avg_pnl > 0 else 0
        results[scheme] = {
            "方案": scheme,
            "品种": sym,
            "总PnL(pt)": round(total_pnl, 0),
            "笔数": n,
            "WR%": round(wr, 1),
            "均PnL(pt)": round(avg_pnl, 1),
            "BE滑点(pt)": round(be, 1),
        }

    df = pd.DataFrame(list(results.values()))
    # 计算相对baseline的增量
    base_pnl = results["baseline"]["总PnL(pt)"]
    df["vs_baseline"] = df["总PnL(pt)"].apply(
        lambda x: f"{x - base_pnl:+.0f}pt ({(x - base_pnl) / abs(base_pnl) * 100:+.1f}%)"
        if base_pnl != 0 else "N/A"
    )
    return df


# ────────────────────────────────────────────────────────────────────────────
# Step 4: 影线分析
# ────────────────────────────────────────────────────────────────────────────

def step4_shadow_analysis(trades_df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """
    分析长影线（>0.5）是否预示趋势力竭（对应逆势交易）。
    上影线 + 做多 = 力竭；下影线 + 做空 = 力竭
    """
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for direction in ["LONG", "SHORT"]:
        sub = trades_df[trades_df["direction"] == direction].copy()
        if sub.empty:
            continue

        if direction == "LONG":
            # 上影线大 → 顶部试探，对做多不利
            shadow_col = "entry_us"
            shadow_label = "上影线(LONG方向)"
        else:
            # 下影线大 → 底部试探，对做空不利
            shadow_col = "entry_ls"
            shadow_label = "下影线(SHORT方向)"

        for thresh, label in [(0.5, ">0.5 长影线"), (0.3, "0.3~0.5 中影线"), (0.0, "<0.3 短影线")]:
            if thresh == 0.5:
                mask = sub[shadow_col] > 0.5
            elif thresh == 0.3:
                mask = (sub[shadow_col] > 0.3) & (sub[shadow_col] <= 0.5)
            else:
                mask = sub[shadow_col] <= 0.3

            grp = sub[mask]
            if len(grp) == 0:
                rows.append({
                    "品种": sym, "方向": direction, "影线类型": shadow_label,
                    "分组": label, "笔数": 0,
                    "WR%": "-", "均PnL(pt)": "-", "总PnL(pt)": "-",
                })
                continue
            n = len(grp)
            wins = (grp["pnl_pts"] > 0).sum()
            wr = wins / n * 100
            rows.append({
                "品种": sym, "方向": direction, "影线类型": shadow_label,
                "分组": label, "笔数": n,
                "WR%": round(wr, 1),
                "均PnL(pt)": round(grp["pnl_pts"].mean(), 1),
                "总PnL(pt)": round(grp["pnl_pts"].sum(), 1),
            })

        # 额外：对侧影线（不利影线）
        opp_col = "entry_ls" if direction == "LONG" else "entry_us"
        opp_label = "下影线(LONG支撑)" if direction == "LONG" else "上影线(SHORT压力)"
        for thresh, label in [(0.5, ">0.5"), (0.3, "0.3~0.5"), (0.0, "<0.3")]:
            if thresh == 0.5:
                mask = sub[opp_col] > 0.5
            elif thresh == 0.3:
                mask = (sub[opp_col] > 0.3) & (sub[opp_col] <= 0.5)
            else:
                mask = sub[opp_col] <= 0.3

            grp = sub[mask]
            if len(grp) == 0:
                continue
            n = len(grp)
            wins = (grp["pnl_pts"] > 0).sum()
            wr = wins / n * 100
            rows.append({
                "品种": sym, "方向": direction, "影线类型": opp_label,
                "分组": label, "笔数": n,
                "WR%": round(wr, 1),
                "均PnL(pt)": round(grp["pnl_pts"].mean(), 1),
                "总PnL(pt)": round(grp["pnl_pts"].sum(), 1),
            })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# 全量Bar的Body Ratio分布统计
# ────────────────────────────────────────────────────────────────────────────

def br_distribution_stats(sym: str, db: DBManager) -> pd.DataFrame:
    """统计回测日期范围内的body_ratio分布。"""
    spot_sym = _SPOT_SYM[sym]
    rows = []
    for td in DATES:
        date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"
        bars = db.query_df(
            f"SELECT open, high, low, close FROM index_min "
            f"WHERE symbol='{spot_sym}' AND period=300 "
            f"AND datetime LIKE '{date_dash}%'"
        )
        if bars is None or bars.empty:
            continue
        for c in ["open", "high", "low", "close"]:
            bars[c] = bars[c].astype(float)
        bars["br"] = bars.apply(
            lambda r: body_ratio(r["open"], r["high"], r["low"], r["close"]), axis=1
        )
        rows.append({
            "日期": td,
            "bars": len(bars),
            "br_mean": round(bars["br"].mean(), 4),
            "br_std": round(bars["br"].std(), 4),
            "br_pos%": round((bars["br"] > 0.2).mean() * 100, 1),
            "br_neg%": round((bars["br"] < -0.2).mean() * 100, 1),
            "br_doji%": round((bars["br"].abs() < 0.2).mean() * 100, 1),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def _print_table(df: pd.DataFrame, title: str):
    print(f"\n{'═' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")
    if df.empty:
        print("  (无数据)")
        return
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Body Ratio 调节器可行性分析")
    parser.add_argument("--symbol", default="IM,IC",
                        help="品种列表，逗号分隔，如 IM,IC")
    parser.add_argument("--skip-corr", action="store_true",
                        help="跳过相关性步骤（全量数据，较慢）")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    db = DBManager(ConfigLoader().get_db_path())

    print("\n" + "=" * 80)
    print(f"  K线实体占比 Body Ratio 调节器可行性分析")
    print(f"  品种: {', '.join(symbols)} | 回测日期: {len(DATES)}天")
    print(f"  IM threshold=60, IC threshold=65")
    print("=" * 80)

    all_step2_tables = []
    all_step4_tables = []
    all_scheme_tables = []

    # ── Step 1: 相关性
    if not args.skip_corr:
        print("\n【Step 1】Body Ratio 与 M/V/Q Proxy 的 Pearson 相关性")
        print("  目标: |r(body_ratio, M)| < 0.5 → body_ratio独立于M，可作为独立调节器")
        corr_frames = []
        for sym in symbols:
            print(f"  计算 {sym} 相关性（全量数据）...")
            cdf = step1_correlation(sym, db)
            corr_frames.append(cdf)
        if corr_frames:
            _print_table(pd.concat(corr_frames, ignore_index=True), "相关性分析结果")
    else:
        print("\n【Step 1】跳过（--skip-corr）")

    # ── 运行基线回测（同时收集所有trade数据）
    print("\n【Step 2/3/4 预处理】运行基线回测收集 body_ratio 数据...")
    all_baseline_trades = {}
    for sym in symbols:
        print(f"  {sym}: 回测 {len(DATES)} 天...")
        trades = []
        for td in DATES:
            day_trades = run_day_with_br(sym, td, db, scheme="baseline")
            trades.extend(day_trades)
        all_baseline_trades[sym] = trades
        full = [t for t in trades if not t.get("partial")]
        n = len(full)
        wins = sum(1 for t in full if t["pnl_pts"] > 0)
        wr = wins / n * 100 if n > 0 else 0
        total_pnl = sum(t["pnl_pts"] for t in full)
        print(f"  {sym}: {n}笔, WR={wr:.1f}%, PnL={total_pnl:+.0f}pt")

    # ── Step 2: 预测力
    print("\n【Step 2】Body Ratio 交易预测力")
    print("  注意：entry_br = 开仓时 bar_T-1 的 body_ratio（防lookahead）")
    for sym in symbols:
        trades = all_baseline_trades[sym]
        full = [t for t in trades if not t.get("partial")]
        if not full:
            continue
        df = pd.DataFrame(full)
        result = step2_predictive_power(df, sym)
        all_step2_tables.append(result)
        _print_table(result, f"Step2: {sym} 按Body Ratio分组的交易预测力")

    # ── Step 3: 方案对比
    print("\n【Step 3】调节器方案对比（事后模拟，与真实集成可能有偏差）")
    print("  方案A：方向矛盾×0.85，十字星×0.90，一致不变")
    print("  方案B：3根同向且最后一根强：×1.05；2/3反向×0.90")
    for sym in symbols:
        print(f"\n  {sym}: 运行 scheme_A...")
        scheme_trades_A = []
        for td in DATES:
            scheme_trades_A.extend(run_day_with_br(sym, td, db, scheme="scheme_A"))
        print(f"  {sym}: 运行 scheme_B...")
        scheme_trades_B = []
        for td in DATES:
            scheme_trades_B.extend(run_day_with_br(sym, td, db, scheme="scheme_B"))

        # 汇总
        def _summarize(trades, scheme_name):
            full = [t for t in trades if not t.get("partial")]
            total_pnl = sum(t["pnl_pts"] for t in full)
            n = len(full)
            wins = sum(1 for t in full if t["pnl_pts"] > 0)
            wr = wins / n * 100 if n > 0 else 0
            avg_pnl = total_pnl / n if n > 0 else 0
            be = avg_pnl / 2 if avg_pnl > 0 else 0
            return {
                "方案": scheme_name, "品种": sym,
                "总PnL(pt)": round(total_pnl, 0), "笔数": n,
                "WR%": round(wr, 1), "均PnL(pt)": round(avg_pnl, 1),
                "BE滑点(pt)": round(be, 1),
            }

        baseline_trades = all_baseline_trades[sym]
        s_base = _summarize(baseline_trades, "baseline")
        s_A    = _summarize(scheme_trades_A,  "scheme_A(方向一致性)")
        s_B    = _summarize(scheme_trades_B,  "scheme_B(3根一致性)")

        scheme_df = pd.DataFrame([s_base, s_A, s_B])
        base_pnl = s_base["总PnL(pt)"]
        scheme_df["vs_baseline"] = scheme_df["总PnL(pt)"].apply(
            lambda x: f"{x - base_pnl:+.0f}pt ({(x - base_pnl) / abs(base_pnl) * 100:+.1f}%)"
            if base_pnl != 0 else "N/A"
        )
        all_scheme_tables.append(scheme_df)
        _print_table(scheme_df, f"Step3: {sym} 方案对比")

    # ── Step 4: 影线分析
    print("\n【Step 4】影线分析（长影线是否预示趋势力竭）")
    for sym in symbols:
        trades = all_baseline_trades[sym]
        full = [t for t in trades if not t.get("partial")]
        if not full:
            continue
        df = pd.DataFrame(full)
        result = step4_shadow_analysis(df, sym)
        all_step4_tables.append(result)
        _print_table(result, f"Step4: {sym} 影线分析")

    # ── 综合结论
    print("\n" + "=" * 80)
    print("  综合结论")
    print("=" * 80)

    print("""
1. Body Ratio 独立性（Step1结论区）
   ─ 若 |r(BR, M)| < 0.5：BR 与 M 独立，可作为独立调节器维度
   ─ 若 |r(BR, M)| ≥ 0.5：BR 与 M 高度相关，引入后可能是重复调节

2. 预测力（Step2结论区）
   ─ 关注「阴线矛盾（做多）」和「阳线矛盾（做空）」组：
     若其 WR 和均PnL 明显低于「一致」组，则方向矛盾惩罚有效
   ─ 十字星组若接近均值，则不建议惩罚（只惩罚矛盾即可）
   ─ 「强阳/强阴确认」组若明显优于其他组，则可考虑奖励

3. 最佳接入方式（Step3结论区）
   ─ 方案A（方向一致性）：更直接，只惩罚矛盾；推荐若Step2显示矛盾组差
   ─ 方案B（3根一致性）：奖励趋势连贯性；推荐若连续方向比单根更有信息
   ─ 两方案均为「事后模拟」，真实集成需在 score_all 后、阈值判断前应用乘数
   ─ 建议：先用事后模拟确认方向，再做最小化代码改动验证

4. 影线辅助价值（Step4结论区）
   ─ 若长上影线（>0.5）+ LONG 的 WR 明显低：上影压力有实际预测价值
   ─ 若长下影线（>0.5）+ SHORT 的 WR 明显低：下影支撑有实际预测价值
   ─ 影线可作为 BR 调节器的附加条件（例：BR < -0.2 且上影 > 0.5 惩罚更重）

集成建议（若方案A有效）：
   在 score_all 返回后、effective_threshold 判断前：
       br = body_ratio(last_sig_bar)
       if direction=="LONG"  and br < -0.2: score = int(score * 0.85)
       elif direction=="SHORT" and br > 0.2: score = int(score * 0.85)
       elif abs(br) < 0.2:                  score = int(score * 0.90)
   这与 daily_mult / intraday_filter 的应用位置一致。
""")

    print("=" * 80)
    print("  分析完成。")
    print("=" * 80)


if __name__ == "__main__":
    main()
