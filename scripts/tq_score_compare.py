#!/usr/bin/env python3
"""逐bar score对比：TqBacktest(monitor) vs 自研backtest(归档数据)。

bar数据已验证完全一致，所以score差异=信号逻辑差异。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONUNBUFFERED'] = '1'

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from tqsdk import TqApi, TqBacktest, BacktestFinished, TqAuth
from datetime import datetime
from typing import Dict
import pandas as pd


def run_compare(td: str, sym: str = 'IM'):
    y, m, d = int(td[:4]), int(td[4:6]), int(td[6:8])
    date_dash = f"{y}-{m:02d}-{d:02d}"
    SPOT_TQ = {"IM": "SSE.000852", "IC": "SSE.000905"}
    SPOT_SYM = {"IM": "000852", "IC": "000905"}
    spot_tq = SPOT_TQ[sym]
    spot_sym = SPOT_SYM[sym]

    # ── 自研backtest侧：用归档数据计算每根bar的score ──
    from data.storage.db_manager import get_db
    from strategies.intraday.A_share_momentum_signal_v2 import (
        SignalGeneratorV2, SentimentData, compute_volume_profile,
    )
    from scripts.backtest_signals_day import _build_15m_from_5m

    db = get_db()
    all_bars = db.query_df(
        f"SELECT datetime, open, high, low, close, volume FROM index_min "
        f"WHERE symbol='{spot_sym}' AND period=300 ORDER BY datetime"
    )
    for c in ['open','high','low','close','volume']:
        all_bars[c] = all_bars[c].astype(float)

    today_mask = all_bars['datetime'].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()

    daily_df = db.query_df(
        f"SELECT trade_date, close as open, close as high, close as low, close, 0 as volume "
        f"FROM index_daily WHERE ts_code='{spot_sym}.SH' AND trade_date < '{td}' ORDER BY trade_date"
    )
    daily_df = daily_df.tail(30).reset_index(drop=True)
    daily_df['close'] = daily_df['close'].astype(float)

    spot_all = db.query_df(
        f"SELECT close FROM index_daily WHERE ts_code='{spot_sym}.SH' "
        f"AND trade_date < '{td}' ORDER BY trade_date DESC LIMIT 30"
    )
    closes = spot_all['close'].astype(float).iloc[::-1].reset_index(drop=True)
    ema20 = float(closes.ewm(span=20).mean().iloc[-1])
    std20 = float(closes.rolling(20).std().iloc[-1])

    sentiment = None
    sdf = db.query_df(
        f"SELECT atm_iv, atm_iv_market, vrp, term_structure_shape, rr_25d "
        f"FROM daily_model_output WHERE underlying='IM' AND trade_date < '{td}' "
        f"ORDER BY trade_date DESC LIMIT 2"
    )
    if sdf is not None and len(sdf) >= 2:
        cur, prev = sdf.iloc[0], sdf.iloc[1]
        sentiment = SentimentData(
            atm_iv=float(cur.get('atm_iv_market') or cur.get('atm_iv') or 0),
            atm_iv_prev=float(prev.get('atm_iv_market') or prev.get('atm_iv') or 0),
            rr_25d=float(cur.get('rr_25d') or 0),
            rr_25d_prev=float(prev.get('rr_25d') or 0),
            vrp=float(cur.get('vrp') or 0),
            term_structure=str(cur.get('term_structure_shape') or ''),
        )

    vol_profile = compute_volume_profile(
        all_bars[['datetime','volume']], before_date=td, lookback_days=20)

    is_high_vol = True
    dmo = db.query_df(
        f"SELECT garch_forecast_vol FROM daily_model_output "
        f"WHERE underlying='IM' AND garch_forecast_vol > 0 AND trade_date < '{td}' "
        f"ORDER BY trade_date DESC LIMIT 1"
    )
    if dmo is not None and not dmo.empty:
        is_high_vol = (float(dmo.iloc[0].iloc[0]) * 100 / 24.9) > 1.2

    # 自研backtest每根bar的score
    gen_bt = SignalGeneratorV2({'min_signal_score': 60})
    bt_scores = {}
    for idx in today_indices:
        bar_5m = all_bars.loc[:idx].tail(199).copy()  # 199=TQ completed bars
        bar_5m.index = pd.to_datetime(bar_5m['datetime'])
        if len(bar_5m) < 16:
            continue
        bar_15m_full = _build_15m_from_5m(bar_5m)
        bar_15m = bar_15m_full.iloc[:-1] if len(bar_15m_full) > 1 else bar_15m_full
        price = float(bar_5m.iloc[-1]['close'])
        # 外部数据固定中性值（与monitor/backtest一致）
        result = gen_bt.score_all(
            sym, bar_5m, bar_15m, None, None, None,
            zscore=None, is_high_vol=True, d_override=None,
            vol_profile=vol_profile,
        )
        dt_str = str(all_bars.loc[idx, 'datetime'])
        utc_hm = dt_str[11:16]
        bj_h = int(utc_hm[:2]) + 8
        bj_time = f"{bj_h:02d}:{utc_hm[3:5]}"
        bt_scores[bj_time] = {
            'score': result['total'] if result else 0,
            'dir': result['direction'] if result else '-',
            'M': result.get('s_momentum', 0) if result else 0,
            'V': result.get('s_volatility', 0) if result else 0,
            'Q': result.get('s_volume', 0) if result else 0,
            'sent': result.get('sentiment_mult', 1.0) if result else 1.0,
            'n_bars': len(bar_5m),
        }

    # ── TqBacktest侧：通过monitor计算score ──
    from strategies.intraday.monitor import IntradayMonitor
    from strategies.intraday.strategy import IntradayConfig

    config = IntradayConfig()
    config.universe = {sym}
    config.tradeable = {sym}
    monitor = IntradayMonitor(config)
    monitor._load_daily_data()
    monitor._load_sentiment()

    # 覆盖所有可能有未来数据泄漏的数据（TqBacktest回测历史日期时datetime.now()!=回测日期）
    monitor._sentiment = sentiment  # sentiment
    monitor._vol_profiles[sym] = vol_profile  # vol_profile
    monitor._daily_data[sym] = daily_df  # daily（关键！影响daily_mult和intraday_filter）
    # zscore也需要覆盖
    monitor._zscore_params[sym] = {"ema20": ema20, "std20": std20, "index": f"{spot_sym}.SH"}

    auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))
    api = TqApi(backtest=TqBacktest(
        start_dt=datetime(y, m, d, 0, 0, 0),
        end_dt=datetime(y, m, d, 23, 59, 59),
    ), auth=auth)

    spot_k5 = {sym: api.get_kline_serial(spot_tq, 300, 200)}
    spot_k15 = {sym: api.get_kline_serial(spot_tq, 900, 100)}
    from utils.cffex_calendar import _near_month_by_expiry
    fut_sym = f"CFFEX.{sym}{_near_month_by_expiry()}"
    monitor._tq_symbols[sym] = fut_sym
    fut_quotes = {sym: api.get_quote(fut_sym)}

    monitor._warmup_done = False
    monitor._bars_since_start = 0

    tq_scores = {}
    tq_bar_dump = {}  # 记录每根bar的close用于对比

    # Monkey-patch score recording
    orig_score_all = monitor.signal_v2.score_all
    def _patched_score_all(symbol, bar_5m, bar_15m, *args, **kwargs):
        result = orig_score_all(symbol, bar_5m, bar_15m, *args, **kwargs)
        if result and bar_5m is not None and len(bar_5m) > 0:
            ts = bar_5m.index[-1]
            bj = (ts + pd.Timedelta(hours=8)).strftime('%H:%M')
            tq_scores[bj] = {
                'score': result['total'],
                'dir': result['direction'],
                'M': result.get('s_momentum', 0),
                'V': result.get('s_volatility', 0),
                'Q': result.get('s_volume', 0),
                'sent': result.get('sentiment_mult', 1.0),
                'n_bars': len(bar_5m),
            }
            # 记录bar数据用于对比
            tq_bar_dump[bj] = [float(c) for c in bar_5m['close'].tail(20)]
            # 记录15m数据
            if bar_15m is not None and len(bar_15m) > 0:
                tq_bar_dump[f"{bj}_15m"] = [float(c) for c in bar_15m['close'].tail(3)]
        return result
    monitor.signal_v2.score_all = _patched_score_all

    try:
        while True:
            api.wait_update()
            changed = []
            for s in [sym]:
                sk = spot_k5.get(s)
                if sk is not None and api.is_changing(sk) and len(sk) >= 2:
                    cdt = int(sk.iloc[-2]['datetime'])
                    prev = monitor._last_bar_time.get(s)
                    if prev != cdt:
                        monitor._last_bar_time[s] = cdt
                        changed.append(s)
            if changed:
                monitor._on_new_bar(changed, spot_k5, spot_k15, fut_quotes, None)
    except BacktestFinished:
        pass
    finally:
        api.close()

    # ── 对比输出 ──
    all_times = sorted(set(list(bt_scores.keys()) + list(tq_scores.keys())))
    print(f"\n{'='*80}")
    print(f"  {td} {sym} 逐bar score对比 (自研BT vs TqBT/monitor)")
    print(f"{'='*80}")
    print(f"  {'BJ':>6s} | {'BT score':>8s} {'BT M':>4s} {'BT V':>4s} {'BT Q':>4s} {'sent':>5s} {'bars':>4s}"
          f" | {'TQ score':>8s} {'TQ M':>4s} {'TQ V':>4s} {'TQ Q':>4s} {'sent':>5s} {'bars':>4s} | {'Δ':>3s}")
    print(f"  {'-'*76}")
    diff_count = 0
    for t in all_times:
        bt = bt_scores.get(t, {})
        tq = tq_scores.get(t, {})
        bs = bt.get('score', 0)
        ts_ = tq.get('score', 0)
        delta = bs - ts_ if (bt and tq) else 0
        flag = ' ⚠' if delta != 0 else ''
        if delta != 0:
            diff_count += 1
            # 有差异时打印最后20根bar close对比
            bt_bar = all_bars.loc[:today_indices[all_times.index(t)]].tail(199)
            bt_closes = [float(c) for c in bt_bar['close'].tail(20)]
            tq_closes = tq_bar_dump.get(t, [])
            if bt_closes and tq_closes and len(bt_closes) == len(tq_closes):
                bar_diffs = [(i, bc, tc) for i, (bc, tc) in enumerate(zip(bt_closes, tq_closes)) if abs(bc-tc) > 0.01]
                if bar_diffs:
                    print(f"    ↳ bar close差异（最近20根中）: {len(bar_diffs)}根不同")
                    for bi, bc, tc in bar_diffs[:3]:
                        print(f"      [{bi-20}] BT={bc:.1f} TQ={tc:.1f} Δ={tc-bc:+.1f}")
                else:
                    # 5m close一致，检查15m
                    bt_15m = _build_15m_from_5m(bt_bar)
                    bt_15m = bt_15m.iloc[:-1] if len(bt_15m) > 1 else bt_15m
                    tq_15m = tq_bar_dump.get(f"{t}_15m", [])
                    bt_15m_c = [f'{float(c):.1f}' for c in bt_15m['close'].tail(3)]
                    tq_15m_c = [f'{c:.1f}' for c in tq_15m] if tq_15m else ['?']
                    match_15m = bt_15m_c == tq_15m_c
                    flag_15m = "✓" if match_15m else "✗ 15m不同!"
                    print(f"    ↳ 5m一致。15m: BT={bt_15m_c} TQ={tq_15m_c} {flag_15m}")
        bt_str = f"{bs:>8d} {bt.get('M',0):>4d} {bt.get('V',0):>4d} {bt.get('Q',0):>4d} {bt.get('sent',1.0):>5.2f} {bt.get('n_bars',0):>4d}" if bt else f"{'(无)':>31s}"
        tq_str = f"{ts_:>8d} {tq.get('M',0):>4d} {tq.get('V',0):>4d} {tq.get('Q',0):>4d} {tq.get('sent',1.0):>5.2f} {tq.get('n_bars',0):>4d}" if tq else f"{'(无)':>31s}"
        print(f"  {t:>6s} | {bt_str} | {tq_str} | {delta:>+3d}{flag}")
    print(f"\n  差异bar数: {diff_count}/{len(all_times)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='20260408')
    parser.add_argument('--symbol', default='IM')
    args = parser.parse_args()
    run_compare(args.date, args.symbol)
