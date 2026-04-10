#!/usr/bin/env python3
"""
backtest_signals_day.py
-----------------------
Replay intraday signals with full open/hold/close cycle for a specific day.

Usage:
    python scripts/backtest_signals_day.py --symbol IM --date 20260324
    python scripts/backtest_signals_day.py --symbol IM --date 20260320
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

IM_MULT = 200  # IM合约乘数
SLIPPAGE_PTS = 0  # 滑点（点），通过 --slippage 参数设置


def _utc_to_bj(utc_str: str) -> str:
    """Convert UTC HH:MM to Beijing HH:MM."""
    h = int(utc_str[:2]) + 8
    if h >= 24:
        h -= 24
    return f"{h:02d}:{utc_str[3:5]}"


def _build_15m_from_5m(bar_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample 5min bars to 15min bars.

    使用 label='left', closed='left' 和TQ的15分钟bar对齐：
    TQ定义: 09:30 bar = 09:30~09:45数据, 09:45 bar = 09:45~10:00数据
    之前用 label='right', closed='right' 导致聚合区间偏移一个周期。
    """
    if len(bar_5m) < 3:
        return pd.DataFrame()
    df = bar_5m.copy()
    df["dt"] = pd.to_datetime(df["datetime"] if "datetime" in df.columns
                               else df.index)
    df = df.set_index("dt")
    resampled = df.resample("15min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index(drop=True)


def run_day(sym: str, td: str, db: DBManager, verbose: bool = True,
            slippage: float = 0, version: str = "auto") -> List[Dict]:
    """Replay one day. Returns list of completed trades.

    Args:
        slippage: points of slippage per trade (applied adversely on entry and exit)
        version: "v2", "v3", or "auto" (use SIGNAL_ROUTING)
    """
    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"

    # Load spot index 5min bars (preferred) or fallback to futures
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
        # Fallback: futures_min
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

    # 历史同时段volume profile（Q分用分位数法，消除跨日偏差）
    from strategies.intraday.A_share_momentum_signal_v2 import compute_volume_profile
    vol_profile = compute_volume_profile(all_bars, before_date=td, lookback_days=20)

    # Load ALL daily data (truncate per replay date inside loop to avoid future leakage)
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

    # Z-Score: compute per replay date (using data up to that date)
    spot_all = db.query_df(
        f"SELECT trade_date, close FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if spot_all is not None:
        spot_all["close"] = spot_all["close"].astype(float)

    def _zscore_for_date(target_date: str):
        """Compute EMA20/STD20/Z using only data up to target_date."""
        if spot_all is None or spot_all.empty:
            return 0.0, 0.0
        sub = spot_all[spot_all["trade_date"] < target_date].tail(30)
        if len(sub) < 20:
            return 0.0, 0.0
        closes = sub["close"].values
        ema = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])
        std = float(pd.Series(closes).rolling(20).std().iloc[-1])
        return ema, std

    ema20, std20 = _zscore_for_date(td)  # initial for this day

    # GARCH regime (use data up to replay date, not latest)
    is_high_vol = True
    dmo = db.query_df(
        "SELECT garch_forecast_vol FROM daily_model_output "
        "WHERE underlying='IM' AND garch_forecast_vol > 0 "
        f"AND trade_date < '{td}' "
        "ORDER BY trade_date DESC LIMIT 1"
    )
    if dmo is not None and not dmo.empty:
        is_high_vol = (float(dmo.iloc[0].iloc[0]) * 100 / 24.9) > 1.2

    # Sentiment (use data up to replay date, not latest)
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

    # Select signal generator version
    _ver = version if version != "auto" else SIGNAL_ROUTING.get(sym, "v2")
    if _ver == "v3":
        gen = SignalGeneratorV3({"min_signal_score": 60})
    else:
        gen = SignalGeneratorV2({"min_signal_score": 60})

    # Per-symbol threshold（IC=65等，从SYMBOL_PROFILES读取）
    from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES, _DEFAULT_PROFILE
    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    effective_threshold = _sym_prof.get("signal_threshold", _SIGNAL_THRESHOLD)

    # Morning Briefing d_override（和monitor一致）
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
                if verbose:
                    print(f"  Briefing d_override: LONG={d_long} SHORT={d_short}")
    except Exception:
        pass

    # Position tracker
    position: Optional[Dict] = None
    completed_trades: List[Dict] = []
    suppressed_signals: List[Dict] = []   # signals killed by daily_mult
    rows: List[str] = []
    # Cooldown tracker: prevent same-direction re-entry within 15min
    last_exit_utc: str = ""
    last_exit_dir: str = ""
    # 开盘振幅过滤（215天验证：<0.4%日均亏-21pt，过滤+274pt）
    from strategies.intraday.A_share_momentum_signal_v2 import check_low_amplitude
    _low_amplitude: Optional[bool] = None  # None=未判断, True=低振幅, False=正常
    COOLDOWN_MINUTES = 15

    # Truncate daily data to replay date (no future leakage, strict < to exclude today)
    daily_df = None
    if daily_all is not None:
        daily_df = daily_all[daily_all["trade_date"] < td].tail(30).reset_index(drop=True)
        if daily_df.empty:
            daily_df = None

    # Compute gap_pct for signal quality analysis
    _gap_pct = 0.0
    prev_c = 0.0
    if daily_df is not None and len(daily_df) >= 2:
        prev_rows = daily_df[daily_df["trade_date"] < td]
        if len(prev_rows) > 0:
            prev_c = float(prev_rows.iloc[-1]["close"])
    # Use first 5min bar's open as today's open
    _today_open = 0.0
    if prev_c > 0 and today_indices:
        _today_open = float(all_bars.loc[today_indices[0], "open"])
        _gap_pct = (_today_open - prev_c) / prev_c

    if verbose:
        W = 115
        print(f"\n{'=' * W}")
        print(f" Signal+Exit Replay | {sym} | {td} | HiVol={is_high_vol}"
              f" | prevClose={prev_c:.0f}")
        print(f"{'=' * W}")
        print(f" BJ Time |  Price  |   Z   | RSI | Dir |  M  V  Q |  dm  |  f   |  t  |  s  | Raw>Flt | Action")
        print(f"---------+---------+-------+-----+-----+----------+------+------+-----+-----+---------+--------")

    for idx in today_indices:
        bar_5m = all_bars.loc[:idx].tail(200).copy()
        if len(bar_5m) < 15:
            continue

        # 信号/执行对齐（与实盘monitor一致，2026-04-10修正）：
        # Monitor: bar T完成后，信号数据包含bar T，entry在bar T+5min的市场价
        # Backtest等价: 信号从bar_5m（含当前bar），entry用当前bar close
        # 注：当前bar close ≈ bar完成时的市场价，这是最合理的近似
        # 旧版排除当前bar (bar_5m=bar_5m[:-1])导致信号滞后1根bar
        if len(bar_5m) < 16:  # 需要至少16根bar (15根历史 + 1根当前)
            continue

        price = float(bar_5m.iloc[-1]["close"])      # 执行价格 = 当前bar收盘
        high = float(bar_5m.iloc[-1]["high"])
        low = float(bar_5m.iloc[-1]["low"])
        signal_price = float(bar_5m.iloc[-1]["close"])  # 信号价 = 当前bar收盘（同源）
        dt_str = str(all_bars.loc[idx, "datetime"])
        utc_hm = dt_str[11:16]
        bj_time = _utc_to_bj(utc_hm)

        z_val = (signal_price - ema20) / std20 if std20 > 0 else None

        # Build 15m bars from 5m data, 排除最后一根forming 15m bar
        # 和TQ原生15m一致
        bar_15m_full = _build_15m_from_5m(bar_5m)
        bar_15m = bar_15m_full.iloc[:-1] if len(bar_15m_full) > 1 else bar_15m_full

        # Score using all completed bars (including current bar, matching monitor)
        result = gen.score_all(
            sym, bar_5m, bar_15m, daily_df, None, sentiment,
            zscore=z_val, is_high_vol=is_high_vol, d_override=d_override,
            vol_profile=vol_profile,
        )

        score = result["total"] if result else 0
        direction = result["direction"] if result else ""
        pre_z = result.get("pre_z_total", score) if result else 0
        z_filt = result.get("z_filter", "") if result else ""
        rsi = result.get("rsi", 50) if result else 50

        # Check exit first (if we have a position)
        action_str = ""
        if position is not None:
            # P0: Bar high/low 止损检查（优先级最高，在更新极值之前）
            # 实盘中止损单挂在stop_price，bar内触及即成交
            stop_price = position.get("stop_loss", 0)
            bar_stopped = False
            if stop_price > 0:
                if position["direction"] == "LONG" and low <= stop_price:
                    bar_stopped = True
                elif position["direction"] == "SHORT" and high >= stop_price:
                    bar_stopped = True

            if bar_stopped:
                # 止损触发：exit_price = stop_price（不是bar close）
                entry_p = position["entry_price"]
                exit_p = stop_price - slippage if position["direction"] == "LONG" else stop_price + slippage
                if position["direction"] == "LONG":
                    pnl_pts = exit_p - entry_p
                else:
                    pnl_pts = entry_p - exit_p
                elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)
                reason = "STOP_LOSS"
                action_str = (f"EXIT {reason} "
                              f"{'+'if pnl_pts>=0 else ''}{pnl_pts:.0f}pt "
                              f"{elapsed}min")
                completed_trades.append({
                    "entry_time": _utc_to_bj(position["entry_time_utc"]),
                    "entry_price": entry_p,
                    "exit_time": bj_time,
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
                })
                last_exit_utc = utc_hm
                last_exit_dir = position["direction"]
                position = None
            else:
                # 未触发止损 → 正常流程：更新极值 → check_exit
                # Update extremes
                if position["direction"] == "LONG":
                    position["highest_since"] = max(position["highest_since"], high)
                else:
                    position["lowest_since"] = min(position["lowest_since"], low)

                # Check reverse signal
                reverse_score = 0
                if result and direction and direction != position["direction"]:
                    reverse_score = score

                exit_info = check_exit(
                    position, price, bar_5m,
                    bar_15m if not bar_15m.empty else None,
                    utc_hm, reverse_score, is_high_vol=is_high_vol,
                    symbol=sym,
                )

                if exit_info["should_exit"]:
                    exit_vol = exit_info["exit_volume"]
                    reason = exit_info["exit_reason"]
                    entry_p = position["entry_price"]
                    # Apply slippage adversely on exit
                    exit_p = price - slippage if position["direction"] == "LONG" else price + slippage
                    if position["direction"] == "LONG":
                        pnl_pts = exit_p - entry_p
                    else:
                        pnl_pts = entry_p - exit_p

                    elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)

                    if exit_vol >= position["volume"]:
                        # Full close
                        action_str = (f"EXIT {reason} "
                                      f"{'+'if pnl_pts>=0 else ''}{pnl_pts:.0f}pt "
                                      f"{elapsed}min")
                        completed_trades.append({
                            "entry_time": _utc_to_bj(position["entry_time_utc"]),
                            "entry_price": entry_p,
                            "exit_time": bj_time,
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
                        })
                        last_exit_utc = utc_hm
                        last_exit_dir = position["direction"]
                        position = None
                    else:
                        # Partial close
                        action_str = (f"PARTIAL {reason} "
                                      f"{exit_vol}手 {'+'if pnl_pts>=0 else ''}{pnl_pts:.0f}pt")
                        position["volume"] -= exit_vol
                        position["half_closed"] = True
                        completed_trades.append({
                            "entry_time": _utc_to_bj(position["entry_time_utc"]),
                            "entry_price": entry_p,
                            "exit_time": bj_time,
                            "exit_price": exit_p,
                            "direction": position["direction"],
                            "pnl_pts": pnl_pts,
                            "reason": reason + "(半仓)",
                            "minutes": elapsed,
                            "partial": True,
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
                        })

        # Check entry (no position, score >= threshold, time allowed, cooldown passed)
        in_cooldown = False
        if last_exit_utc and direction == last_exit_dir:
            cd_elapsed = _calc_minutes(last_exit_utc, utc_hm)
            if 0 < cd_elapsed < COOLDOWN_MINUTES:
                in_cooldown = True

        # 开盘振幅过滤：10:00(UTC 02:00)后判断前6根bar
        if _low_amplitude is None and utc_hm >= "02:00":
            bar_idx_in_day = today_indices.index(idx) if idx in today_indices else 0
            if bar_idx_in_day >= 6:
                today_first6 = all_bars.loc[today_indices[:6]]
                _low_amplitude = check_low_amplitude(today_first6)
                if _low_amplitude and verbose:
                    print(f"  [AMP-FILTER] 开盘30min振幅<0.4%，后续不开新仓")
            else:
                _low_amplitude = False

        if (position is None and not action_str and result and not in_cooldown
                and score >= effective_threshold and direction and is_open_allowed(utc_hm)
                and not _low_amplitude):
            # Apply slippage adversely on entry
            entry_p = price + slippage if direction == "LONG" else price - slippage
            # Compute rebound/pullback from recent 20-bar extreme (use signal bars)
            recent_20 = bar_5m.tail(20)
            min_20 = float(recent_20["low"].min())
            max_20 = float(recent_20["high"].max())
            if direction == "SHORT" and min_20 > 0:
                _rebound_pct = (signal_price - min_20) / min_20
            elif direction == "LONG" and max_20 > 0:
                _rebound_pct = (max_20 - signal_price) / max_20
            else:
                _rebound_pct = 0.0
            # Gap alignment: 低开+做空=True, 高开+做多=True
            _gap_aligned = ((_gap_pct < 0 and direction == "SHORT") or
                            (_gap_pct > 0 and direction == "LONG"))
            _sl_pct = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE).get("stop_loss_pct", STOP_LOSS_PCT)
            stop = entry_p * (1 - _sl_pct) if direction == "LONG" else entry_p * (1 + _sl_pct)
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
                # 信号质量字段（供分析用）
                "entry_score": score,
                "entry_v_score": result.get("s_volatility", 0),
                "entry_daily_mult": result.get("daily_mult", 1.0),
                "entry_raw_total": result.get("raw_total", 0),
                "entry_s_breakout": result.get("s_breakout", 0),
                "entry_gap_pct": _gap_pct,
                "entry_gap_aligned": _gap_aligned,
                "entry_rebound_pct": _rebound_pct,
                "entry_total_drop": (signal_price - prev_c) / prev_c if prev_c > 0 else 0.0,
                "entry_intraday_drop": (signal_price - _today_open) / _today_open if _today_open > 0 else 0.0,
                "entry_filter_mult": result.get("intraday_filter", 1.0),
                "entry_zscore": z_val,
            }
            slip_tag = f" slip={slippage:.0f}" if slippage > 0 else ""
            action_str = f"OPEN {direction} @{entry_p:.0f} stop={stop:.0f}{slip_tag}"

        # Track signals suppressed by daily_mult or intraday_filter
        if (result and direction and position is None and not action_str
                and score < effective_threshold and is_open_allowed(utc_hm)):
            dm = result.get("daily_mult", 1.0)
            idf = result.get("intraday_filter", 1.0)
            raw = result.get("raw_total", 0)
            tw = result.get("time_weight", 1.0)
            sm = result.get("sentiment_mult", 1.0)
            change_pct = (signal_price - prev_c) / prev_c if prev_c > 0 else 0.0
            # What would score be with daily_mult=1.0?
            hyp_no_dm = int(round(max(0, min(100, raw * 1.0 * idf * tw * sm))))
            if dm < 0.8 and hyp_no_dm >= effective_threshold:
                suppressed_signals.append({
                    "time": bj_time, "price": signal_price, "direction": direction,
                    "raw_total": raw, "daily_mult": dm, "intraday_filter": idf,
                    "score_actual": score, "score_hypothetical": hyp_no_dm,
                    "entry_total_drop": change_pct, "suppressor": "daily_mult",
                })
            # What would score be with intraday_filter=1.0?
            hyp_no_idf = int(round(max(0, min(100, raw * dm * 1.0 * tw * sm))))
            if idf < 1.0 and hyp_no_idf >= effective_threshold and score < effective_threshold:
                suppressed_signals.append({
                    "time": bj_time, "price": signal_price, "direction": direction,
                    "raw_total": raw, "daily_mult": dm, "intraday_filter": idf,
                    "score_actual": score, "score_hypothetical": hyp_no_idf,
                    "entry_total_drop": change_pct, "suppressor": "intraday_filter",
                })

        # Build position status for display
        pos_info = ""
        if position is not None and not action_str.startswith("OPEN"):
            entry_p = position["entry_price"]
            trail_pct = TRAILING_STOP_HIVOL if is_high_vol else TRAILING_STOP_NORMAL
            if position["direction"] == "LONG":
                pnl = price - entry_p
                trail_ref = position["highest_since"]
                trail_price = trail_ref * (1 - trail_pct)
            else:
                pnl = entry_p - price
                trail_ref = position["lowest_since"]
                trail_price = trail_ref * (1 + trail_pct)
            if not action_str:
                action_str = (f"hold {'+'if pnl>=0 else ''}{pnl:.0f}pt "
                              f"trail={trail_price:.0f}")

        if verbose:
            d_ch = "L" if direction == "LONG" else ("S" if direction == "SHORT" else "-")
            z_s = f"{z_val:+5.2f}" if z_val is not None else "  --"
            s_m = result["s_momentum"] if result else 0
            s_v = result["s_volatility"] if result else 0
            s_q = result["s_volume"] if result else 0
            s_b = result.get("s_breakout", 0) if result else 0
            dm = result.get("daily_mult", 1.0) if result else 1.0
            d_f = result.get("intraday_filter", 1.0) if result else 1.0
            tw = result.get("time_weight", 1.0) if result else 1.0
            sm = result.get("sentiment_mult", 1.0) if result else 1.0

            marker = ""
            if action_str.startswith("OPEN"):
                marker = " ⚡"
            elif action_str.startswith("EXIT") or action_str.startswith("PARTIAL"):
                marker = " ✅"

            brk_s = f" B{s_b:d}" if s_b > 0 else ""
            print(f" {bj_time}  | {signal_price:7.0f} | {z_s:>5} | {rsi:3.0f}"
                  f" |   {d_ch} | {s_m:2d} {s_v:2d} {s_q:2d}"
                  f" | {dm:4.2f} | {d_f:4.2f} | {tw:.1f} | {sm:.2f}"
                  f" | {pre_z:3d}>{score:<3d} | {action_str}{brk_s}{marker}")

    # Force close any remaining position
    if position is not None:
        last_price = float(all_bars.loc[today_indices[-1], "close"])
        entry_p = position["entry_price"]
        # Apply slippage adversely on force close
        exit_p = last_price - slippage if position["direction"] == "LONG" else last_price + slippage
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
            "entry_v_score": position.get("entry_v_score", 0),
            "entry_daily_mult": position.get("entry_daily_mult", 1.0),
            "entry_raw_total": position.get("entry_raw_total", 0),
            "entry_gap_pct": position.get("entry_gap_pct", 0.0),
            "entry_gap_aligned": position.get("entry_gap_aligned", False),
            "entry_rebound_pct": position.get("entry_rebound_pct", 0.0),
        })

    # Summary
    if verbose:
        print(f"\n{'=' * W}")
        if completed_trades:
            print(f"\n === Trade Summary {td} ===")
            total_pnl = 0
            for i, t in enumerate(completed_trades):
                d = "L" if t["direction"] == "LONG" else "S"
                pnl = t["pnl_pts"]
                total_pnl += pnl
                print(f"  #{i+1}  {t['entry_time']} {d}@{t['entry_price']:.0f}"
                      f" → {t['exit_time']} @{t['exit_price']:.0f}"
                      f"  {'+'if pnl>=0 else ''}{pnl:.0f}pt"
                      f"  {t['reason']}  {t['minutes']}min")
            yuan = total_pnl * IM_MULT
            n = len([t for t in completed_trades if not t.get("partial")])
            wins = len([t for t in completed_trades if t["pnl_pts"] > 0 and not t.get("partial")])
            wr = wins / n * 100 if n > 0 else 0
            print(f"\n  Total: {'+'if total_pnl>=0 else ''}{total_pnl:.0f}pt"
                  f" = {'+'if yuan>=0 else ''}{yuan:,.0f}元(1手)"
                  f"  Trades: {n}  WinRate: {wr:.0f}%")
        else:
            print(f"\n  No trades for {td}")
        print()

    # Compute theoretical PnL for suppressed signals (hold to close)
    if today_indices:
        day_close = float(all_bars.loc[today_indices[-1], "close"])
        for sig in suppressed_signals:
            if sig["direction"] == "LONG":
                sig["pnl_pts"] = day_close - sig["price"]
            else:
                sig["pnl_pts"] = sig["price"] - day_close
            sig["suppressed"] = True
    # Attach suppressed signals as attribute via subclass
    class _TradeList(list):
        pass
    result_list = _TradeList(completed_trades)
    result_list._suppressed = suppressed_signals
    return result_list


# Import the constant we need
from strategies.intraday.A_share_momentum_signal_v2 import (
    STOP_LOSS_PCT, TRAILING_STOP_HIVOL, TRAILING_STOP_NORMAL,
    SYMBOL_PROFILES, _DEFAULT_PROFILE,
)


def _calc_minutes(t1: str, t2: str) -> int:
    """Calculate minutes between two HH:MM strings."""
    try:
        h1, m1 = int(t1[:2]), int(t1[3:5])
        h2, m2 = int(t2[:2]), int(t2[3:5])
        return (h2 * 60 + m2) - (h1 * 60 + m1)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="IM")
    parser.add_argument("--date", default="20260324", help="YYYYMMDD or comma-separated or YYYYMMDD-YYYYMMDD range")
    parser.add_argument("--slippage", type=float, default=0, help="Slippage in points per trade (e.g. 5)")
    parser.add_argument("--threshold", type=int, default=60, help="Signal threshold (default 60)")
    parser.add_argument("--version", choices=["v2", "v3", "auto"], default="auto",
                        help="Signal version: v2, v3, or auto (use SIGNAL_ROUTING)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity sweep: slippage 0/5/10 × threshold 50/55/60")
    args = parser.parse_args()

    db = get_db()

    # Expand date range
    if "-" in args.date and len(args.date) == 17:
        # YYYYMMDD-YYYYMMDD range
        start, end = args.date.split("-")
        all_dates_df = db.query_df(
            f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
            f"WHERE symbol='000852' AND period=300 ORDER BY d"
        )
        if all_dates_df is not None and not all_dates_df.empty:
            dates = [d.replace("-", "") for d in all_dates_df["d"].tolist()
                     if start <= d.replace("-", "") <= end]
        else:
            dates = [start]
    else:
        dates = [d.strip() for d in args.date.split(",")]

    if args.sensitivity:
        _run_sensitivity(args.symbol, dates, db, version=args.version)
        return

    # Override threshold in signal generator
    _patch_threshold(args.threshold)

    all_trades = []
    for td in dates:
        trades = run_day(args.symbol, td, db, verbose=(len(dates) <= 3),
                         slippage=args.slippage, version=args.version)
        all_trades.extend([(td, t) for t in trades])

    if len(dates) > 1:
        _print_multi_day_summary(args.symbol, dates, all_trades, args.slippage, args.threshold)


def _patch_threshold(threshold: int):
    """Patch the signal threshold in SignalGeneratorV2 at module level."""
    # The threshold is checked in run_day via `score >= 55`, we patch globally
    global _SIGNAL_THRESHOLD
    _SIGNAL_THRESHOLD = threshold


# Module-level threshold (patched by _patch_threshold)
_SIGNAL_THRESHOLD = 60  # Q分改为分位数法后恢复60（方案E曾降至55）


def _run_sensitivity(sym: str, dates: List[str], db: DBManager,
                     version: str = "auto"):
    """Run sensitivity sweep across slippage × threshold combinations."""
    slippages = [0, 1, 2, 3, 5]
    thresholds = [55, 60, 65]

    print(f"\n{'='*80}")
    print(f" SENSITIVITY ANALYSIS | {sym} | {len(dates)} days "
          f"({dates[0]}~{dates[-1]}) | version={version}")
    print(f"{'='*80}")

    results = {}
    for thr in thresholds:
        _patch_threshold(thr)
        for slip in slippages:
            all_trades = []
            for td in dates:
                trades = run_day(sym, td, db, verbose=False, slippage=slip,
                                 version=version)
                all_trades.extend(trades)
            full_trades = [t for t in all_trades if not t.get("partial")]
            total_pnl = sum(t["pnl_pts"] for t in full_trades)
            n_trades = len(full_trades)
            wins = len([t for t in full_trades if t["pnl_pts"] > 0])
            win_days = len(set())  # placeholder
            results[(thr, slip)] = {
                "pnl": total_pnl, "n": n_trades,
                "wr": wins / n_trades * 100 if n_trades > 0 else 0,
                "yuan": total_pnl * IM_MULT,
            }

    # Print matrix
    col_hdr = '|  '.join(f'slip={s}pt' for s in slippages)
    sep_w = 16 * len(slippages) + 10
    print(f"\n  PnL (points) | Slippage →")
    print(f"  Threshold ↓  |  {col_hdr}  |")
    print(f"  {'─'*14}+{'─'*sep_w}")
    for thr in thresholds:
        row = []
        for slip in slippages:
            r = results[(thr, slip)]
            pnl = r["pnl"]
            row.append(f"{'+'if pnl>=0 else ''}{pnl:5.0f}({r['n']:2d}T)")
        print(f"  thr={thr:3d}       | {'  | '.join(row)}  |")

    print(f"\n  Yuan (1手IM) | Slippage →")
    print(f"  Threshold ↓  |  {col_hdr}  |")
    print(f"  {'─'*14}+{'─'*sep_w}")
    for thr in thresholds:
        row = []
        for slip in slippages:
            r = results[(thr, slip)]
            yuan = r["yuan"]
            row.append(f"{'+'if yuan>=0 else ''}{yuan:>8,.0f}")
        print(f"  thr={thr:3d}       | {'  | '.join(row)}  |")

    print(f"\n  WinRate      | Slippage →")
    print(f"  Threshold ↓  |  {col_hdr}  |")
    print(f"  {'─'*14}+{'─'*sep_w}")
    for thr in thresholds:
        row = []
        for slip in slippages:
            r = results[(thr, slip)]
            row.append(f"  {r['wr']:5.1f}%  ")
        print(f"  thr={thr:3d}       | {'  | '.join(row)}  |")

    # Breakeven analysis
    print(f"\n  --- Breakeven analysis (thr=60) ---")
    for slip in slippages:
        r = results[(60, slip)]
        tag = "✅" if r["pnl"] > 0 else "❌"
        print(f"  slip={slip}pt: {'+'if r['pnl']>=0 else ''}{r['pnl']:.0f}pt "
              f"({r['wr']:.1f}% WR, {r['n']}T)  {tag}")
    base_r = results[(60, 0)]
    if base_r["n"] > 0:
        avg = base_r["pnl"] / base_r["n"]
        print(f"\n  Avg PnL/trade (slip=0): {avg:+.1f}pt")
        print(f"  → breakeven slippage ≈ {avg/2:.1f}pt per side"
              f" ({avg/2/7500*10000:.1f} bps on IM@7500)")
    print()


def _print_multi_day_summary(sym: str, dates: List[str], all_trades: List,
                              slippage: float, threshold: int):
    """Print multi-day summary with slippage/threshold info."""
    slip_tag = f" slip={slippage:.0f}pt" if slippage > 0 else ""
    thr_tag = f" thr={threshold}" if threshold != 60 else ""
    print(f"\n{'='*70}")
    print(f" Multi-Day Summary | {sym}{slip_tag}{thr_tag} | {len(dates)} days")
    print(f"{'='*70}")
    grand_pnl = 0
    win_days = 0
    for td in dates:
        day_trades = [t for d, t in all_trades if d == td and not t.get("partial")]
        pnl = sum(t["pnl_pts"] for t in day_trades)
        grand_pnl += pnl
        if pnl > 0:
            win_days += 1
        n = len(day_trades)
        if n > 0:
            print(f"  {td}: {n} trades  {'+'if pnl>=0 else ''}{pnl:.0f}pt")
    yuan = grand_pnl * IM_MULT
    total_days = len([td for td in dates
                      if any(d == td for d, t in all_trades)])
    wr_day = win_days / total_days * 100 if total_days > 0 else 0
    print(f"\n  Grand Total: {'+'if grand_pnl>=0 else ''}{grand_pnl:.0f}pt"
          f" = {'+'if yuan>=0 else ''}{yuan:,.0f}元(1手)"
          f"  | {total_days}天 盈利{win_days}天({wr_day:.0f}%)")


if __name__ == "__main__":
    main()
