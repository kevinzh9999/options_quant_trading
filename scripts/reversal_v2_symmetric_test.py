#!/usr/bin/env python3
"""测试对称逻辑：v2信号可以平掉reversal持仓并反手开仓。

对比三组：
  A: v2 baseline（无reversal）
  B: v2 + reversal（当前逻辑，reversal能平v2，v2不能平reversal）
  C: v2 + reversal + 对称（v2也能平reversal并反手）

IS: 20250101 ~ 20260416
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from collections import defaultdict
from datetime import time as _time
import numpy as np


def run_day_symmetric(sym, td, long_n, short_n, bear_thr, rev_thr, min_depth,
                      enable_reversal=True, v2_can_close_rev=False):
    """v2 + slope reversal, 可选对称逻辑。"""
    from data.storage.db_manager import get_db
    from strategies.intraday.A_share_momentum_signal_v2 import (
        SignalGeneratorV2, SYMBOL_PROFILES, _DEFAULT_PROFILE,
        check_exit, check_low_amplitude, compute_volume_profile, _get_utc_time,
    )
    from strategies.intraday.reversal_signal import ReversalDetectorSlope
    from scripts.backtest_signals_day import _build_15m_from_5m, _utc_to_bj, _calc_minutes

    db = get_db()
    _SPOT = {"IM": "000852", "IC": "000905", "IF": "000300", "IH": "000016"}
    spot_sym = _SPOT[sym]
    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"

    all_bars = db.query_df(
        f"SELECT datetime, open, high, low, close, volume FROM index_min "
        f"WHERE symbol='{spot_sym}' AND period=300 ORDER BY datetime")
    if all_bars is None or all_bars.empty:
        return []
    for c in ["open", "high", "low", "close", "volume"]:
        all_bars[c] = all_bars[c].astype(float)

    today_mask = all_bars["datetime"].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()
    if not today_indices:
        return []

    vol_profile = compute_volume_profile(all_bars, before_date=td, lookback_days=20)
    gen = SignalGeneratorV2({"min_signal_score": 50})
    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    effective_threshold = _sym_prof.get("signal_threshold", 60)
    _sl_pct = _sym_prof.get("stop_loss_pct", 0.005)

    NO_TRADE_BEFORE = _time(1, 25)
    NO_TRADE_AFTER = _time(7, 5)
    REV_NO_OPEN_BEFORE = _time(2, 25)
    REV_NO_OPEN_AFTER = _time(6, 30)

    position = None
    completed = []
    _low_amplitude = None

    # Slope reversal detector
    det = ReversalDetectorSlope(sym, config={
        "enabled": True, "method": "slope",
        "long_n": long_n, "short_n": short_n,
        "bear_thr": bear_thr, "rev_thr": rev_thr, "min_depth": min_depth,
    })

    for idx in today_indices:
        bar_open = float(all_bars.loc[idx, 'open'])
        bar_close = float(all_bars.loc[idx, 'close'])
        bar_high = float(all_bars.loc[idx, 'high'])
        bar_low = float(all_bars.loc[idx, 'low'])
        price = bar_close

        bar_5m = all_bars.loc[:idx].tail(199).copy()
        if len(bar_5m) < 16:
            continue

        utc_time = _get_utc_time(bar_5m)
        if not utc_time or utc_time < NO_TRADE_BEFORE or utc_time > NO_TRADE_AFTER:
            # still need to feed detector and gen.update
            if enable_reversal:
                det.update(bar_open, bar_high, bar_low, bar_close)
            gen.update(sym, bar_5m, _build_15m_from_5m(bar_5m), None, None,
                       sentiment=None, zscore=None, is_high_vol=True,
                       d_override=None, vol_profile=vol_profile)
            continue

        _h, _m = utc_time.hour, utc_time.minute
        _m += 5
        if _m >= 60:
            _h += 1; _m -= 60
        utc_hm = f"{_h:02d}:{_m:02d}"
        bj_time = _utc_to_bj(utc_hm)

        bar_15m_full = _build_15m_from_5m(bar_5m)
        bar_15m = bar_15m_full.iloc[:-1] if len(bar_15m_full) > 1 else bar_15m_full

        action_str = ""

        # ── reversal检测 ──
        reversal_direction = None
        reversal_depth = 0
        if enable_reversal:
            rev = det.update(bar_open, bar_high, bar_low, bar_close)
            if rev is not None:
                rev_in_window = REV_NO_OPEN_BEFORE <= utc_time <= REV_NO_OPEN_AFTER
                if rev_in_window:
                    reversal_direction = rev.direction
                    reversal_depth = rev.depth

        # ── reversal处理（优先于v2） ──
        if reversal_direction:
            # 持有反向仓 → 平仓 + 反手
            if position is not None and position["direction"] != reversal_direction:
                entry_p = position["entry_price"]
                pnl = (price - entry_p) if position["direction"] == "LONG" else (entry_p - price)
                elapsed = _calc_minutes(position.get("entry_time_utc", "00:00"), utc_hm)
                completed.append({
                    'entry_time': _utc_to_bj(position.get("entry_time_utc", "")),
                    'exit_time': bj_time, 'direction': position["direction"],
                    'pnl': pnl, 'reason': 'REVERSAL_EXIT',
                    'source': position.get('source', 'v2'), 'minutes': elapsed,
                })
                position = None
                action_str = "REV_EXIT"

            # 无持仓 → 开reversal仓
            if position is None:
                entry_p = price
                stop = entry_p * (1 - _sl_pct) if reversal_direction == "LONG" else entry_p * (1 + _sl_pct)
                position = {
                    "entry_price": entry_p, "direction": reversal_direction,
                    "entry_time_utc": utc_hm, "highest_since": bar_high,
                    "lowest_since": bar_low, "stop_loss": stop, "volume": 1,
                    "half_closed": False, "bars_below_mid": 0, "source": "reversal",
                }
                action_str = "REV_OPEN"

        # ── v2 exit ──
        if position is not None and not action_str:
            if position["direction"] == "LONG":
                position["highest_since"] = max(position["highest_since"], bar_high)
            else:
                position["lowest_since"] = min(position["lowest_since"], bar_low)

            result = gen.score_all(sym, bar_5m, bar_15m, None, None, None,
                                   zscore=None, is_high_vol=True, d_override=None,
                                   vol_profile=vol_profile)
            direction = result.get("direction", "") if result else ""
            score = result.get("total", 0) if result else 0
            reverse_score = score if direction and direction != position["direction"] else 0

            exit_info = check_exit(
                position, price, bar_5m,
                bar_15m if not bar_15m.empty else None,
                utc_hm, reverse_score, is_high_vol=True, symbol=sym)

            if exit_info["should_exit"]:
                entry_p = position["entry_price"]
                pnl = (price - entry_p) if position["direction"] == "LONG" else (entry_p - price)
                elapsed = _calc_minutes(position.get("entry_time_utc", "00:00"), utc_hm)
                completed.append({
                    'entry_time': _utc_to_bj(position.get("entry_time_utc", "")),
                    'exit_time': bj_time, 'direction': position["direction"],
                    'pnl': pnl, 'reason': exit_info["exit_reason"],
                    'source': position.get('source', 'v2'), 'minutes': elapsed,
                })
                position = None
                action_str = "V2_EXIT"

        # ── v2 entry（含对称逻辑） ──
        if not action_str:
            sig_v2 = gen.update(sym, bar_5m, bar_15m, None, None,
                             sentiment=None, zscore=None, is_high_vol=True,
                             d_override=None, vol_profile=vol_profile)

            if _low_amplitude is None and utc_hm >= "02:00":
                bar_idx = today_indices.index(idx) if idx in today_indices else 0
                if bar_idx >= 6:
                    today_first6 = all_bars.loc[today_indices[:6]]
                    _low_amplitude = check_low_amplitude(today_first6)
                else:
                    _low_amplitude = False

            if sig_v2 is not None and sig_v2.score >= effective_threshold and not _low_amplitude:
                if position is None:
                    # 无持仓 → 正常开仓
                    entry_p = price
                    stop = entry_p * (1 - _sl_pct) if sig_v2.direction == "LONG" else entry_p * (1 + _sl_pct)
                    position = {
                        "entry_price": entry_p, "direction": sig_v2.direction,
                        "entry_time_utc": utc_hm, "highest_since": bar_high,
                        "lowest_since": bar_low, "stop_loss": stop, "volume": 1,
                        "half_closed": False, "bars_below_mid": 0,
                        "entry_score": sig_v2.score, "source": "v2",
                    }
                elif (v2_can_close_rev
                      and position.get("source") == "reversal"
                      and position["direction"] != sig_v2.direction):
                    # 对称逻辑：持有reversal反向仓 → 平reversal + 开v2
                    entry_p = position["entry_price"]
                    pnl = (price - entry_p) if position["direction"] == "LONG" else (entry_p - price)
                    elapsed = _calc_minutes(position.get("entry_time_utc", "00:00"), utc_hm)
                    completed.append({
                        'entry_time': _utc_to_bj(position.get("entry_time_utc", "")),
                        'exit_time': bj_time, 'direction': position["direction"],
                        'pnl': pnl, 'reason': 'V2_OVERRIDE_REV',
                        'source': 'reversal', 'minutes': elapsed,
                    })
                    # 开v2仓
                    entry_p = price
                    stop = entry_p * (1 - _sl_pct) if sig_v2.direction == "LONG" else entry_p * (1 + _sl_pct)
                    position = {
                        "entry_price": entry_p, "direction": sig_v2.direction,
                        "entry_time_utc": utc_hm, "highest_since": bar_high,
                        "lowest_since": bar_low, "stop_loss": stop, "volume": 1,
                        "half_closed": False, "bars_below_mid": 0,
                        "entry_score": sig_v2.score, "source": "v2",
                    }
        else:
            gen.update(sym, bar_5m, bar_15m, None, None,
                       sentiment=None, zscore=None, is_high_vol=True,
                       d_override=None, vol_profile=vol_profile)
            if _low_amplitude is None and utc_hm >= "02:00":
                bar_idx = today_indices.index(idx) if idx in today_indices else 0
                if bar_idx >= 6:
                    today_first6 = all_bars.loc[today_indices[:6]]
                    _low_amplitude = check_low_amplitude(today_first6)
                else:
                    _low_amplitude = False

    # EOD
    if position is not None:
        entry_p = position["entry_price"]
        last_price = float(all_bars.loc[today_indices[-1], "close"])
        pnl = (last_price - entry_p) if position["direction"] == "LONG" else (entry_p - last_price)
        completed.append({
            'entry_time': _utc_to_bj(position.get("entry_time_utc", "")),
            'exit_time': '15:00', 'direction': position["direction"],
            'pnl': pnl, 'reason': 'EOD_CLOSE',
            'source': position.get('source', 'v2'), 'minutes': 0,
        })
    return completed


# ── multiprocessing ──

_G_MODE = None

def _init(mode):
    global _G_MODE
    _G_MODE = mode

def _worker(args):
    sym, td, mode = args
    # IM slope参数: sN6 rev3.0 d25
    long_n, short_n, bear_thr, rev_thr, min_depth = 8, 6, -1.5, 3.0, 25

    if mode == "A":
        trades = run_day_symmetric(sym, td, long_n, short_n, bear_thr, rev_thr, min_depth,
                                    enable_reversal=False, v2_can_close_rev=False)
    elif mode == "B":
        trades = run_day_symmetric(sym, td, long_n, short_n, bear_thr, rev_thr, min_depth,
                                    enable_reversal=True, v2_can_close_rev=False)
    elif mode == "C":
        trades = run_day_symmetric(sym, td, long_n, short_n, bear_thr, rev_thr, min_depth,
                                    enable_reversal=True, v2_can_close_rev=True)
    else:
        trades = []

    pnl = sum(t['pnl'] for t in trades)
    n = len(trades)
    rev_n = sum(1 for t in trades if t.get('source') == 'reversal')
    rev_pnl = sum(t['pnl'] for t in trades if t.get('source') == 'reversal')
    override_n = sum(1 for t in trades if t.get('reason') == 'V2_OVERRIDE_REV')
    override_pnl = sum(t['pnl'] for t in trades if t.get('reason') == 'V2_OVERRIDE_REV')
    return (sym, td, mode, n, pnl, rev_n, rev_pnl, override_n, override_pnl)


def main():
    from data.storage.db_manager import get_db

    db = get_db()
    sym = "IM"  # slope reversal只对IM有效

    dates = db.query_df(
        "SELECT DISTINCT trade_date FROM index_daily "
        "WHERE ts_code='000852.SH' AND trade_date >= '20250101' AND trade_date <= '20260416' "
        "ORDER BY trade_date")
    all_dates = dates['trade_date'].tolist()
    print(f"IM IS: {len(all_dates)}天 (20250101~20260416)")

    tasks = []
    for td in all_dates:
        for mode in ["A", "B", "C"]:
            tasks.append((sym, td, mode))

    print(f"共{len(tasks)}个任务 (3配置 × {len(all_dates)}天)")
    with Pool(7) as p:
        results = p.map(_worker, tasks)

    COST = 147; MULT = 200
    data = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'rev_n': 0, 'rev_pnl': 0.0,
                                 'override_n': 0, 'override_pnl': 0.0})
    daily = defaultdict(list)  # mode -> [(td, pnl)]

    for sym_r, td, mode, n, pnl, rev_n, rev_pnl, ov_n, ov_pnl in results:
        data[mode]['n'] += n
        data[mode]['pnl'] += pnl
        data[mode]['rev_n'] += rev_n
        data[mode]['rev_pnl'] += rev_pnl
        data[mode]['override_n'] += ov_n
        data[mode]['override_pnl'] += ov_pnl
        daily[mode].append((td, pnl))

    print(f"\n{'='*90}")
    print(f" IM Slope Reversal 对称逻辑测试 | IS {len(all_dates)}天")
    print(f"{'='*90}")
    print(f" {'Config':30s} | {'N':>5s} {'PnL_pt':>8s} {'Net_¥':>10s} | {'RevN':>5s} {'RevPnL':>8s} | {'OvN':>4s} {'OvPnL':>7s}")
    print(f" {'-'*30}+{'-'*27}+{'-'*17}+{'-'*14}")

    for mode, label in [("A", "v2 baseline (no rev)"),
                         ("B", "v2 + rev (当前, 不对称)"),
                         ("C", "v2 + rev + 对称 (v2可平rev)")]:
        d = data[mode]
        net = d['pnl'] * MULT - d['n'] * COST
        rpnl = d['rev_pnl'] * MULT
        opnl = d['override_pnl'] * MULT
        print(f" {label:30s} | {d['n']:5d} {d['pnl']:+8.0f} {net:+10.0f} | "
              f"{d['rev_n']:5d} {rpnl:+8.0f} | {d['override_n']:4d} {opnl:+7.0f}")

    # B vs A, C vs A, C vs B
    a_net = data['A']['pnl'] * MULT - data['A']['n'] * COST
    b_net = data['B']['pnl'] * MULT - data['B']['n'] * COST
    c_net = data['C']['pnl'] * MULT - data['C']['n'] * COST
    print(f"\n Δ(B-A) rev增量:       {b_net - a_net:+.0f}¥ ({(b_net-a_net)/MULT:+.0f}pt等价)")
    print(f" Δ(C-A) rev+对称增量:  {c_net - a_net:+.0f}¥ ({(c_net-a_net)/MULT:+.0f}pt等价)")
    print(f" Δ(C-B) 对称逻辑增量:  {c_net - b_net:+.0f}¥ ({(c_net-b_net)/MULT:+.0f}pt等价)")
    print(f"\n 对称逻辑触发: {data['C']['override_n']}次, "
          f"被平rev仓PnL: {data['C']['override_pnl']:+.0f}pt")

    # 逐日C-B差异（找出对称逻辑改善/恶化最大的日子）
    b_daily = {td: pnl for td, pnl in daily['B']}
    c_daily = {td: pnl for td, pnl in daily['C']}
    diffs = [(td, c_daily.get(td, 0) - b_daily.get(td, 0)) for td in all_dates]
    diffs.sort(key=lambda x: x[1])

    print(f"\n 对称逻辑逐日差异 Top/Bottom 5:")
    print(f"   恶化最大:")
    for td, d in diffs[:5]:
        if d != 0:
            print(f"     {td}: {d:+.1f}pt")
    print(f"   改善最大:")
    for td, d in diffs[-5:]:
        if d != 0:
            print(f"     {td}: {d:+.1f}pt")

    # 单日贡献检查
    total_delta = sum(d for _, d in diffs)
    if total_delta != 0:
        max_single = max(abs(d) for _, d in diffs)
        print(f"\n 单日最大贡献: {max_single:.1f}pt / 总{abs(total_delta):.1f}pt "
              f"= {max_single/abs(total_delta)*100:.0f}%")


if __name__ == '__main__':
    main()
