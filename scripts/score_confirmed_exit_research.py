#!/usr/bin/env python3
"""
score_confirmed_exit_research.py
---------------------------------
验证"ME/TC/MID_BREAK触发时检查score，score仍高则不退出"的效果。

关键约束：无lookahead
- check_exit用prev_score（上一根bar的score），而非当前bar
- 这和实盘monitor一致：monitor每5分钟bar结束时先check_exit再score_all

用法：
    python scripts/score_confirmed_exit_research.py
    python scripts/score_confirmed_exit_research.py --symbol IM
    python scripts/score_confirmed_exit_research.py --symbol IC
    python scripts/score_confirmed_exit_research.py --date 20260402  # 单日详细
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

# 和baseline一致的常量
IM_MULT = 200
IC_MULT = 200

# ME/TC/MID_BREAK可被score阻止
SCORE_CONFIRMABLE_REASONS = {"MOMENTUM_EXHAUSTED", "TREND_COMPLETE", "MID_BREAK"}

# 安全阀：无条件退出（不管score多高）
UNCONDITIONAL_EXIT_REASONS = {"STOP_LOSS", "EOD_CLOSE", "LUNCH_CLOSE", "TIME_STOP", "EOD_FORCE"}


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
    resampled = df.resample("15min", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index(drop=True)


def _calc_minutes(t1: str, t2: str) -> int:
    try:
        h1, m1 = int(t1[:2]), int(t1[3:5])
        h2, m2 = int(t2[:2]), int(t2[3:5])
        return (h2 * 60 + m2) - (h1 * 60 + m1)
    except Exception:
        return 0


def run_day_both(sym: str, td: str, db: DBManager,
                 verbose: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Run both baseline and score-confirmed-exit for one day.

    Returns:
        (baseline_trades, confirmed_trades, blocked_events)
        blocked_events: list of dicts describing each time a ME/TC was blocked by score
    """
    from strategies.intraday.A_share_momentum_signal_v2 import (
        SignalGeneratorV2, SentimentData, check_exit,
        is_open_allowed, SIGNAL_ROUTING, SYMBOL_PROFILES, _DEFAULT_PROFILE,
        STOP_LOSS_PCT, TRAILING_STOP_HIVOL, TRAILING_STOP_NORMAL,
    )

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
        return [], [], []

    for c in ["open", "high", "low", "close", "volume"]:
        all_bars[c] = all_bars[c].astype(float)

    today_mask = all_bars["datetime"].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()
    if not today_indices:
        return [], [], []

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

    _ver = SIGNAL_ROUTING.get(sym, "v2")
    gen = SignalGeneratorV2({"min_signal_score": 60})

    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    effective_threshold = _sym_prof.get("signal_threshold", 60)

    daily_df = None
    if daily_all is not None:
        daily_df = daily_all[daily_all["trade_date"] < td].tail(30).reset_index(drop=True)
        if daily_df.empty:
            daily_df = None

    prev_c = 0.0
    if daily_df is not None and len(daily_df) >= 1:
        prev_rows = daily_df[daily_df["trade_date"] < td]
        if len(prev_rows) > 0:
            prev_c = float(prev_rows.iloc[-1]["close"])

    _today_open = 0.0
    _gap_pct = 0.0
    if prev_c > 0 and today_indices:
        _today_open = float(all_bars.loc[today_indices[0], "open"])
        _gap_pct = (_today_open - prev_c) / prev_c

    COOLDOWN_MINUTES = 15

    # ─── shared helper to build position dict ──────────────────────────────
    def _make_position(direction, entry_p, entry_time_utc, high, low, score, result, z_val):
        stop = (entry_p * (1 - STOP_LOSS_PCT) if direction == "LONG"
                else entry_p * (1 + STOP_LOSS_PCT))
        signal_price = float(bar_5m_signal.iloc[-1]["close"])
        recent_20 = bar_5m_signal.tail(20)
        min_20 = float(recent_20["low"].min())
        max_20 = float(recent_20["high"].max())
        if direction == "SHORT" and min_20 > 0:
            _rebound_pct = (signal_price - min_20) / min_20
        elif direction == "LONG" and max_20 > 0:
            _rebound_pct = (max_20 - signal_price) / max_20
        else:
            _rebound_pct = 0.0
        _gap_aligned = ((_gap_pct < 0 and direction == "SHORT") or
                        (_gap_pct > 0 and direction == "LONG"))
        return {
            "entry_price": entry_p,
            "direction": direction,
            "entry_time_utc": entry_time_utc,
            "highest_since": high,
            "lowest_since": low,
            "stop_loss": stop,
            "volume": 1,
            "half_closed": False,
            "bars_below_mid": 0,
            "entry_score": score,
            "entry_v_score": result.get("s_volatility", 0),
            "entry_daily_mult": result.get("daily_mult", 1.0),
            "entry_raw_total": result.get("raw_total", 0),
            "entry_gap_pct": _gap_pct,
            "entry_gap_aligned": _gap_aligned,
            "entry_rebound_pct": _rebound_pct,
            "entry_total_drop": (signal_price - prev_c) / prev_c if prev_c > 0 else 0.0,
            "entry_intraday_drop": (signal_price - _today_open) / _today_open if _today_open > 0 else 0.0,
            "entry_filter_mult": result.get("intraday_filter", 1.0),
            "entry_zscore": z_val,
        }

    def _make_trade(position, exit_time_bj, exit_p, reason, elapsed):
        entry_p = position["entry_price"]
        if position["direction"] == "LONG":
            pnl_pts = exit_p - entry_p
        else:
            pnl_pts = entry_p - exit_p
        return {
            "entry_time": _utc_to_bj(position["entry_time_utc"]),
            "entry_price": entry_p,
            "exit_time": exit_time_bj,
            "exit_price": exit_p,
            "direction": position["direction"],
            "pnl_pts": pnl_pts,
            "reason": reason,
            "minutes": elapsed,
            "entry_score": position.get("entry_score", 0),
            "entry_v_score": position.get("entry_v_score", 0),
            "entry_daily_mult": position.get("entry_daily_mult", 1.0),
            "entry_raw_total": position.get("entry_raw_total", 0),
            "entry_gap_pct": position.get("entry_gap_pct", 0.0),
            "entry_gap_aligned": position.get("entry_gap_aligned", False),
            "entry_rebound_pct": position.get("entry_rebound_pct", 0.0),
            "entry_total_drop": position.get("entry_total_drop", 0.0),
            "entry_intraday_drop": position.get("entry_intraday_drop", 0.0),
            "entry_filter_mult": position.get("entry_filter_mult", 1.0),
            "entry_zscore": position.get("entry_zscore"),
        }

    # ─── run both simulations in one pass ──────────────────────────────────
    # Baseline: standard exit logic
    pos_b: Optional[Dict] = None
    trades_b: List[Dict] = []
    last_exit_utc_b = ""
    last_exit_dir_b = ""

    # Score-confirmed: ME/TC/MID_BREAK gated by prev_score
    pos_c: Optional[Dict] = None
    trades_c: List[Dict] = []
    last_exit_utc_c = ""
    last_exit_dir_c = ""
    prev_score_c = 0        # score from previous bar (no lookahead)
    prev_direction_c = ""   # direction from previous bar

    # Blocked events log
    blocked_events: List[Dict] = []

    # Shared result buffer (computed once per bar)
    bar_5m_signal: pd.DataFrame = pd.DataFrame()

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
        bj_time = _utc_to_bj(utc_hm)

        z_val = (signal_price - ema20) / std20 if std20 > 0 else None
        bar_15m = _build_15m_from_5m(bar_5m_signal)

        result = gen.score_all(
            sym, bar_5m_signal, bar_15m, daily_df, None, sentiment,
            zscore=z_val, is_high_vol=is_high_vol, d_override=d_override,
        )

        score = result["total"] if result else 0
        direction = result["direction"] if result else ""

        # ── BASELINE ─────────────────────────────────────────────────────
        def _process_baseline():
            nonlocal pos_b, last_exit_utc_b, last_exit_dir_b

            if pos_b is not None:
                stop_price = pos_b.get("stop_loss", 0)
                bar_stopped = False
                if stop_price > 0:
                    if pos_b["direction"] == "LONG" and low <= stop_price:
                        bar_stopped = True
                    elif pos_b["direction"] == "SHORT" and high >= stop_price:
                        bar_stopped = True

                if bar_stopped:
                    ep = stop_price
                    trades_b.append(_make_trade(pos_b, bj_time, ep, "STOP_LOSS",
                                                _calc_minutes(pos_b["entry_time_utc"], utc_hm)))
                    last_exit_utc_b = utc_hm
                    last_exit_dir_b = pos_b["direction"]
                    pos_b = None
                else:
                    if pos_b["direction"] == "LONG":
                        pos_b["highest_since"] = max(pos_b["highest_since"], high)
                    else:
                        pos_b["lowest_since"] = min(pos_b["lowest_since"], low)

                    reverse_score = 0
                    if result and direction and direction != pos_b["direction"]:
                        reverse_score = score

                    exit_info = check_exit(
                        pos_b, price, bar_5m_signal,
                        bar_15m if not bar_15m.empty else None,
                        utc_hm, reverse_score, is_high_vol=is_high_vol, symbol=sym,
                    )

                    if exit_info["should_exit"]:
                        reason = exit_info["exit_reason"]
                        exit_vol = exit_info["exit_volume"]
                        elapsed = _calc_minutes(pos_b["entry_time_utc"], utc_hm)
                        if exit_vol >= pos_b["volume"]:
                            trades_b.append(_make_trade(pos_b, bj_time, price, reason, elapsed))
                            last_exit_utc_b = utc_hm
                            last_exit_dir_b = pos_b["direction"]
                            pos_b = None
                        else:
                            # partial
                            t = _make_trade(pos_b, bj_time, price, reason + "(半仓)", elapsed)
                            t["partial"] = True
                            trades_b.append(t)
                            pos_b["volume"] -= exit_vol
                            pos_b["half_closed"] = True

            if pos_b is None:
                in_cd = False
                if last_exit_utc_b and direction == last_exit_dir_b:
                    if 0 < _calc_minutes(last_exit_utc_b, utc_hm) < COOLDOWN_MINUTES:
                        in_cd = True
                if (result and not in_cd and score >= effective_threshold
                        and direction and is_open_allowed(utc_hm)):
                    ep = price
                    pos_b = _make_position(direction, ep, utc_hm, high, low, score, result, z_val)

        _process_baseline()

        # ── SCORE-CONFIRMED EXIT ─────────────────────────────────────────
        def _process_confirmed():
            nonlocal pos_c, last_exit_utc_c, last_exit_dir_c
            nonlocal prev_score_c, prev_direction_c

            if pos_c is not None:
                stop_price = pos_c.get("stop_loss", 0)
                bar_stopped = False
                if stop_price > 0:
                    if pos_c["direction"] == "LONG" and low <= stop_price:
                        bar_stopped = True
                    elif pos_c["direction"] == "SHORT" and high >= stop_price:
                        bar_stopped = True

                if bar_stopped:
                    ep = stop_price
                    trades_c.append(_make_trade(pos_c, bj_time, ep, "STOP_LOSS",
                                                _calc_minutes(pos_c["entry_time_utc"], utc_hm)))
                    last_exit_utc_c = utc_hm
                    last_exit_dir_c = pos_c["direction"]
                    pos_c = None
                else:
                    if pos_c["direction"] == "LONG":
                        pos_c["highest_since"] = max(pos_c["highest_since"], high)
                    else:
                        pos_c["lowest_since"] = min(pos_c["lowest_since"], low)

                    reverse_score = 0
                    if result and direction and direction != pos_c["direction"]:
                        reverse_score = score

                    exit_info = check_exit(
                        pos_c, price, bar_5m_signal,
                        bar_15m if not bar_15m.empty else None,
                        utc_hm, reverse_score, is_high_vol=is_high_vol, symbol=sym,
                    )

                    if exit_info["should_exit"]:
                        reason = exit_info["exit_reason"]
                        exit_vol = exit_info["exit_volume"]
                        elapsed = _calc_minutes(pos_c["entry_time_utc"], utc_hm)

                        # ── Score gate ──────────────────────────────────
                        if (reason in SCORE_CONFIRMABLE_REASONS
                                and prev_score_c >= effective_threshold
                                and prev_direction_c == pos_c["direction"]):
                            # Blocked: score still strong, don't exit
                            blocked_events.append({
                                "date": td,
                                "block_time": bj_time,
                                "reason_blocked": reason,
                                "prev_score": prev_score_c,
                                "prev_direction": prev_direction_c,
                                "position_dir": pos_c["direction"],
                                "entry_time": _utc_to_bj(pos_c["entry_time_utc"]),
                                "entry_price": pos_c["entry_price"],
                                "price_at_block": price,
                                "unrealized_pnl_at_block": (
                                    (price - pos_c["entry_price"]) if pos_c["direction"] == "LONG"
                                    else (pos_c["entry_price"] - price)
                                ),
                                "pos_ref": pos_c,  # keep ref so we can compute final PnL later
                            })
                            # Do NOT exit – continue holding
                        else:
                            # Normal exit (score low or unconditional reason)
                            if exit_vol >= pos_c["volume"]:
                                t = _make_trade(pos_c, bj_time, price, reason, elapsed)
                                # Tag if this trade had any blocks during its lifetime
                                trades_c.append(t)
                                last_exit_utc_c = utc_hm
                                last_exit_dir_c = pos_c["direction"]
                                pos_c = None
                            else:
                                t = _make_trade(pos_c, bj_time, price, reason + "(半仓)", elapsed)
                                t["partial"] = True
                                trades_c.append(t)
                                pos_c["volume"] -= exit_vol
                                pos_c["half_closed"] = True

            if pos_c is None:
                in_cd = False
                if last_exit_utc_c and direction == last_exit_dir_c:
                    if 0 < _calc_minutes(last_exit_utc_c, utc_hm) < COOLDOWN_MINUTES:
                        in_cd = True
                if (result and not in_cd and score >= effective_threshold
                        and direction and is_open_allowed(utc_hm)):
                    ep = price
                    pos_c = _make_position(direction, ep, utc_hm, high, low, score, result, z_val)

            # Update prev_score AFTER processing (no lookahead)
            prev_score_c = score
            prev_direction_c = direction

        _process_confirmed()

    # ── Force close any remaining positions ──────────────────────────────
    if today_indices:
        last_idx = today_indices[-1]
        last_price = float(all_bars.loc[last_idx, "close"])
        last_dt = str(all_bars.loc[last_idx, "datetime"])[11:16]
        last_bj = _utc_to_bj(last_dt)

        if pos_b is not None:
            elapsed = _calc_minutes(pos_b["entry_time_utc"], last_dt)
            trades_b.append(_make_trade(pos_b, last_bj, last_price, "EOD_FORCE", elapsed))

        if pos_c is not None:
            elapsed = _calc_minutes(pos_c["entry_time_utc"], last_dt)
            trades_c.append(_make_trade(pos_c, last_bj, last_price, "EOD_FORCE", elapsed))

    # Post-process blocked events: fill in final exit info
    for ev in blocked_events:
        # Find the trade in trades_c that has the same entry
        for t in trades_c:
            if (t["entry_time"] == _utc_to_bj(ev["pos_ref"]["entry_time_utc"])
                    and t["direction"] == ev["pos_ref"]["direction"]):
                ev["final_exit_time"] = t["exit_time"]
                ev["final_exit_reason"] = t["reason"]
                ev["final_pnl"] = t["pnl_pts"]
                ev["extra_minutes"] = _calc_minutes(
                    ev["block_time"].replace(":", "")[0:2] + ":" + ev["block_time"][3:5],
                    t["exit_time"].replace(":", "")[0:2] + ":" + t["exit_time"][3:5]
                )
                # Extra PnL = final PnL - what PnL would have been if we exited at block_time
                ev["extra_pnl"] = t["pnl_pts"] - ev["unrealized_pnl_at_block"]
                break

    return trades_b, trades_c, blocked_events


def _summarize(trades: List[Dict], label: str, sym: str) -> Dict:
    full = [t for t in trades if not t.get("partial")]
    n = len(full)
    wins = len([t for t in full if t["pnl_pts"] > 0])
    pnl = sum(t["pnl_pts"] for t in full)
    wr = wins / n * 100 if n > 0 else 0
    avg = pnl / n if n > 0 else 0
    return {"label": label, "sym": sym, "n": n, "wins": wins, "pnl": pnl, "wr": wr, "avg": avg}


def run_multi(sym: str, dates: List[str], db: DBManager,
              verbose_dates: Optional[List[str]] = None) -> Tuple[Dict, Dict, List[Dict]]:
    """Run all dates and aggregate."""
    all_trades_b: List[Dict] = []
    all_trades_c: List[Dict] = []
    all_blocked: List[Dict] = []

    verbose_dates = verbose_dates or []

    for td in dates:
        tb, tc, blocked = run_day_both(sym, td, db, verbose=(td in verbose_dates))
        all_trades_b.extend(tb)
        all_trades_c.extend(tc)
        all_blocked.extend(blocked)
        if len(dates) <= 3:
            print(f"  {td}: baseline={len([t for t in tb if not t.get('partial')])}笔"
                  f" PnL={sum(t['pnl_pts'] for t in tb if not t.get('partial')):+.0f}pt"
                  f"  | confirmed={len([t for t in tc if not t.get('partial')])}笔"
                  f" PnL={sum(t['pnl_pts'] for t in tc if not t.get('partial')):+.0f}pt"
                  f"  | blocked={len(blocked)}次")

    sum_b = _summarize(all_trades_b, "Baseline", sym)
    sum_c = _summarize(all_trades_c, "ME/TC score确认", sym)
    return sum_b, sum_c, all_blocked


def print_blocked_analysis(blocked: List[Dict], sym: str, effective_threshold: int):
    print(f"\n{'─'*70}")
    print(f" ME/TC/MID_BREAK被score阻止明细 | {sym}")
    print(f"{'─'*70}")
    if not blocked:
        print("  （无阻止事件）")
        return

    n_blocked = len(blocked)
    n_improved = len([e for e in blocked if e.get("extra_pnl", 0) > 0])
    n_hurt = len([e for e in blocked if e.get("extra_pnl", 0) < 0])
    total_extra = sum(e.get("extra_pnl", 0) for e in blocked)
    avg_extra = total_extra / n_blocked if n_blocked > 0 else 0

    # Extra minutes: calc from block_time and final_exit_time
    extra_mins = []
    for e in blocked:
        if "final_exit_time" in e:
            try:
                bh, bm = int(e["block_time"][:2]), int(e["block_time"][3:5])
                fh, fm = int(e["final_exit_time"][:2]), int(e["final_exit_time"][3:5])
                extra_mins.append((fh * 60 + fm) - (bh * 60 + bm))
            except Exception:
                pass
    avg_extra_min = sum(extra_mins) / len(extra_mins) if extra_mins else 0

    # Exit reason distribution after blocking
    final_reasons: Dict[str, int] = {}
    for e in blocked:
        r = e.get("final_exit_reason", "UNKNOWN")
        final_reasons[r] = final_reasons.get(r, 0) + 1

    print(f"\n  总阻止次数: {n_blocked}")
    print(f"  阻止后改善(extra_pnl>0): {n_improved}次 ({n_improved/n_blocked*100:.0f}%)")
    print(f"  阻止后恶化(extra_pnl<0): {n_hurt}次 ({n_hurt/n_blocked*100:.0f}%)")
    print(f"  平均额外持仓: {avg_extra_min:.1f}分钟")
    print(f"  额外PnL合计: {total_extra:+.0f}pt  平均: {avg_extra:+.1f}pt/次")

    print(f"\n  阻止后最终exit reason分布:")
    for r, cnt in sorted(final_reasons.items(), key=lambda x: -x[1]):
        print(f"    {r:<30} {cnt}次")

    print(f"\n  详细明细 (prev_score≥{effective_threshold}时阻止):")
    print(f"  {'日期':<10} {'阻止时间':<8} {'方向':<6} {'原因':<20} {'prevscore':<10}"
          f" {'阻止时浮盈':>10} {'最终PnL':>8} {'额外PnL':>8} {'最终原因'}")
    print(f"  {'─'*110}")
    for e in sorted(blocked, key=lambda x: (x["date"], x["block_time"])):
        fin_pnl = e.get("final_pnl", float("nan"))
        extra_pnl = e.get("extra_pnl", float("nan"))
        fin_reason = e.get("final_exit_reason", "?")[:20]
        print(f"  {e['date']:<10} {e['block_time']:<8} {e['position_dir']:<6} "
              f"{e['reason_blocked']:<20} {e['prev_score']:<10}"
              f" {e['unrealized_pnl_at_block']:>+10.0f}pt "
              f"{fin_pnl:>+8.0f}pt {extra_pnl:>+8.0f}pt  {fin_reason}")


def print_comparison_table(results: List[Tuple[Dict, Dict]], dates: List[str]):
    """Print comparison table for all symbols."""
    print(f"\n{'='*100}")
    print(f" ME/TC score确认 vs Baseline 对比 | {len(dates)}天 ({dates[0]}~{dates[-1]})")
    print(f"{'='*100}")
    print(f"  {'方案':<30} {'PnL(pt)':>8} {'笔数':>6} {'WR':>6} {'均PnL':>7} {'vs Baseline':>12}")
    print(f"  {'─'*80}")

    grand_b = 0
    grand_c = 0
    grand_nb = 0
    grand_nc = 0
    grand_wb = 0
    grand_wc = 0

    for sb, sc in results:
        sym = sb["sym"]
        delta_pnl = sc["pnl"] - sb["pnl"]
        delta_tag = f"{delta_pnl:+.0f}pt"
        print(f"\n  [{sym}]")
        print(f"  {'Baseline':<30} {sb['pnl']:>+8.0f} {sb['n']:>6} {sb['wr']:>5.1f}% "
              f"{sb['avg']:>+7.1f}    {'─':>12}")
        print(f"  {'ME/TC score确认':<30} {sc['pnl']:>+8.0f} {sc['n']:>6} {sc['wr']:>5.1f}% "
              f"{sc['avg']:>+7.1f}    {delta_tag:>12}")
        grand_b += sb["pnl"]
        grand_c += sc["pnl"]
        grand_nb += sb["n"]
        grand_nc += sc["n"]
        grand_wb += sb["wins"]
        grand_wc += sc["wins"]

    print(f"\n  {'─'*80}")
    grand_delta = grand_c - grand_b
    wr_b = grand_wb / grand_nb * 100 if grand_nb > 0 else 0
    wr_c = grand_wc / grand_nc * 100 if grand_nc > 0 else 0
    avg_b = grand_b / grand_nb if grand_nb > 0 else 0
    avg_c = grand_c / grand_nc if grand_nc > 0 else 0
    print(f"  {'合计 Baseline':<30} {grand_b:>+8.0f} {grand_nb:>6} {wr_b:>5.1f}%"
          f" {avg_b:>+7.1f}    {'─':>12}")
    print(f"  {'合计 ME/TC score确认':<30} {grand_c:>+8.0f} {grand_nc:>6} {wr_c:>5.1f}%"
          f" {avg_c:>+7.1f}    {grand_delta:>+12.0f}pt")
    print(f"\n  增量: {grand_delta:+.0f}pt ({'+' if grand_delta >= 0 else ''}{grand_delta / abs(grand_b) * 100 if grand_b != 0 else 0:.1f}%相对baseline)")


def print_single_day_detail(sym: str, td: str,
                             trades_b: List[Dict], trades_c: List[Dict],
                             blocked: List[Dict]):
    """Print detailed per-trade comparison for one day."""
    print(f"\n{'='*90}")
    print(f" {td} 单日对比 | {sym}")
    print(f"{'='*90}")

    full_b = [t for t in trades_b if not t.get("partial")]
    full_c = [t for t in trades_c if not t.get("partial")]

    print(f"\n  Baseline trades:")
    pnl_b = 0
    for i, t in enumerate(full_b):
        d = "L" if t["direction"] == "LONG" else "S"
        pnl_b += t["pnl_pts"]
        print(f"    #{i+1} {t['entry_time']} {d}@{t['entry_price']:.0f}"
              f" → {t['exit_time']} @{t['exit_price']:.0f}"
              f"  {t['pnl_pts']:>+6.0f}pt  {t['reason']}  {t['minutes']}min")
    print(f"    Total baseline: {pnl_b:+.0f}pt ({len(full_b)}笔)")

    print(f"\n  Score-confirmed trades:")
    pnl_c = 0
    for i, t in enumerate(full_c):
        d = "L" if t["direction"] == "LONG" else "S"
        pnl_c += t["pnl_pts"]
        print(f"    #{i+1} {t['entry_time']} {d}@{t['entry_price']:.0f}"
              f" → {t['exit_time']} @{t['exit_price']:.0f}"
              f"  {t['pnl_pts']:>+6.0f}pt  {t['reason']}  {t['minutes']}min")
    print(f"    Total confirmed: {pnl_c:+.0f}pt ({len(full_c)}笔)")

    day_blocked = [e for e in blocked if e["date"] == td]
    if day_blocked:
        print(f"\n  Blocked events this day:")
        for e in day_blocked:
            print(f"    {e['block_time']} {e['reason_blocked']} blocked (prev_score={e['prev_score']},"
                  f" {e['position_dir']})  unrealized={e['unrealized_pnl_at_block']:+.0f}pt"
                  f" → final={e.get('final_pnl', '?'):+.0f}pt"
                  f" extra={e.get('extra_pnl', '?'):+.0f}pt"
                  f" → {e.get('final_exit_reason', '?')}")

    print(f"\n  Delta: {pnl_c - pnl_b:+.0f}pt")


def main():
    parser = argparse.ArgumentParser(description="ME/TC score确认退出 vs Baseline 回测研究")
    parser.add_argument("--symbol", default="ALL", help="IM / IC / ALL (default: ALL)")
    parser.add_argument("--date", default="", help="YYYYMMDD 单日详细模式（可逗号分隔）")
    args = parser.parse_args()

    db = get_db()

    # Full date list
    ALL_DATES = (
        "20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,"
        "20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,"
        "20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,"
        "20260319,20260320,20260323,20260324,20260325,20260326,20260327,20260328,"
        "20260401,20260402"
    ).split(",")

    if args.date:
        # Single/specific date mode: show detail
        specific_dates = [d.strip() for d in args.date.split(",")]
        syms = ["IM", "IC"] if args.symbol == "ALL" else [args.symbol]

        for sym in syms:
            from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES, _DEFAULT_PROFILE
            _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
            eff_thr = _sym_prof.get("signal_threshold", 60)

            print(f"\nRunning {sym} for dates: {specific_dates}  threshold={eff_thr}")
            tb_all, tc_all, blocked_all = [], [], []
            for td in specific_dates:
                tb, tc, blocked = run_day_both(sym, td, db)
                tb_all.extend(tb)
                tc_all.extend(tc)
                blocked_all.extend(blocked)
                print_single_day_detail(sym, td, tb, tc, blocked)

            print_blocked_analysis(blocked_all, sym, eff_thr)
        return

    # Full backtest mode
    syms = ["IM", "IC"] if args.symbol == "ALL" else [args.symbol]
    results = []
    all_blocked_combined: List[Dict] = []

    for sym in syms:
        from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES, _DEFAULT_PROFILE
        _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
        eff_thr = _sym_prof.get("signal_threshold", 60)

        print(f"\n{'='*60}")
        print(f" Running {sym}  threshold={eff_thr}  dates={len(ALL_DATES)}")
        print(f"{'='*60}")

        sb, sc, blocked = run_multi(sym, ALL_DATES, db)
        results.append((sb, sc))
        all_blocked_combined.extend(blocked)

        print_blocked_analysis(blocked, sym, eff_thr)

    print_comparison_table(results, ALL_DATES)

    # 4/2 single day detail if in date list
    if "20260402" in ALL_DATES:
        syms_detail = ["IM", "IC"] if args.symbol == "ALL" else [args.symbol]
        print(f"\n{'='*60}")
        print(f" 4/2单日详细对比")
        print(f"{'='*60}")
        for sym in syms_detail:
            tb, tc, blocked = run_day_both(sym, "20260402", db)
            print_single_day_detail(sym, "20260402", tb, tc, blocked)

    print()


if __name__ == "__main__":
    main()
