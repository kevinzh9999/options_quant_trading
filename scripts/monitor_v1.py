#!/usr/bin/env python3
"""v1信号系统独立Monitor。

完全独立于v2 monitor，有自己的TQ连接、bar构建、shadow position管理。
用v1_im/v1_ic评分函数打分，写入shadow_trades_new_mapping表。

用法：
    python scripts/monitor_v1.py              # 实盘模式
    python scripts/monitor_v1.py --backtest 20260408   # TqBacktest模式
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import time as _time
from datetime import datetime, date, time
from typing import Dict, Optional
import numpy as np
import pandas as pd

from data.storage.db_manager import get_db
from strategies.intraday.A_share_momentum_signal_v2 import (
    SignalGeneratorV2, SYMBOL_PROFILES, _DEFAULT_PROFILE,
    check_low_amplitude, check_exit, _get_utc_time,
    NO_TRADE_BEFORE, NO_TRADE_AFTER, compute_volume_profile,
)
from strategies.intraday.experimental.signal_new_mapping_v1_im import (
    score as score_im, THRESHOLD as THR_IM,
)
from strategies.intraday.experimental.signal_new_mapping_v1_ic import (
    score as score_ic, THRESHOLD as THR_IC,
)
from scripts.backtest_signals_day import _build_15m_from_5m

SYMBOLS = ["IM", "IC"]
SPOT_TQ = {"IM": "SSE.000852", "IC": "SSE.000905"}
SPOT_SYM = {"IM": "000852", "IC": "000905"}

_NO_OPEN_LUNCH_START = time(3, 20)
_NO_OPEN_LUNCH_END = time(5, 5)
_NO_OPEN_EOD = time(6, 30)


class MonitorV1:
    """v1独立Monitor进程。"""

    def __init__(self, backtest_date: str = None):
        self._db = get_db()
        self._gen = SignalGeneratorV2({"min_signal_score": 20})
        self._backtest_date = backtest_date  # None=实盘, "20260408"=回测
        self._shadow_positions: Dict[str, Dict] = {}
        self._low_amp: Dict[str, Optional[bool]] = {}
        self._daily_trade_count: Dict[str, int] = {}
        self._vol_profiles: Dict[str, dict] = {}
        self._last_bar_time: Dict[str, int] = {}
        self._panel_data: Dict[str, dict] = {}  # sym -> 最新bar的v1得分
        self._closed_pnl: Dict[str, float] = {}  # sym -> 已平仓PnL

        # 加载vol_profile
        today_str = datetime.now().strftime("%Y%m%d")
        for sym in SYMBOLS:
            spot = SPOT_SYM[sym]
            bar_all = self._db.query_df(
                f"SELECT datetime, volume FROM index_min "
                f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime"
            )
            if bar_all is not None and len(bar_all) > 0:
                bar_all['volume'] = bar_all['volume'].astype(float)
                self._vol_profiles[sym] = compute_volume_profile(
                    bar_all, before_date=today_str, lookback_days=20)
        print(f"  [V1] Vol profile加载完成: {list(self._vol_profiles.keys())}")

    def on_bar(self, sym: str, b5: pd.DataFrame, b15: pd.DataFrame):
        """处理一根新的已完成bar。"""
        if len(b5) < 20:
            return

        utc_time = _get_utc_time(b5)
        if not utc_time:
            return
        if utc_time < NO_TRADE_BEFORE or utc_time > NO_TRADE_AFTER:
            return

        # bar_start + 5min = 执行时间（与v2 monitor/backtest一致）
        _h, _m = utc_time.hour, utc_time.minute
        _m += 5
        if _m >= 60:
            _h += 1; _m -= 60
        utc_hm = f"{_h:02d}:{_m:02d}"
        bj_h = (_h + 8) % 24
        bj_hm = f"{bj_h:02d}:{_m:02d}"

        # 开盘振幅过滤
        if sym not in self._low_amp:
            self._low_amp[sym] = None
        if self._low_amp[sym] is None and utc_hm >= "02:00":
            today_bars = self._extract_today_bars(b5)
            if len(today_bars) >= 6:
                self._low_amp[sym] = check_low_amplitude(today_bars.iloc[:6])

        # 每根bar都计算v1得分（面板输出用）
        vp = self._vol_profiles.get(sym)
        result = self._gen.score_all(
            sym, b5, b15 if b15 is not None and len(b15) > 0 else None,
            None, None, None,
            zscore=None, is_high_vol=True, d_override=None,
            vol_profile=vp,
        )
        v1r = None
        v1_score = 0
        direction = ""
        if result and result.get('direction'):
            direction = result['direction']
            raw_mom = result.get('raw_mom_5m', 0.0)
            raw_atr = result.get('raw_atr_ratio', 0.0)
            raw_vpct = result.get('raw_vol_pct', -1.0)
            raw_vratio = result.get('raw_vol_ratio', -1.0)

            # Gap计算
            today_bars = self._extract_today_bars(b5)
            gap_aligned = False
            if len(today_bars) > 0 and len(b5) > len(today_bars):
                today_open = float(today_bars.iloc[0]['open'])
                prev_close = float(b5.iloc[-(len(today_bars) + 1)]['close'])
                if prev_close > 0:
                    gap_pct = (today_open - prev_close) / prev_close
                    gap_aligned = (gap_pct > 0 and direction == 'LONG') or \
                                  (gap_pct < 0 and direction == 'SHORT')

            if sym == 'IM':
                v1r = score_im(raw_mom, raw_atr, raw_vpct, raw_vratio, bj_h, gap_aligned)
                threshold = THR_IM
                version = "v1_im"
            else:
                v1r = score_ic(raw_mom, raw_atr, raw_vpct, raw_vratio, bj_h, gap_aligned)
                threshold = THR_IC
                version = "v1_ic"
            v1_score = v1r['total_score']

        # 面板输出：每根bar打印v1得分
        cur_price = float(b5.iloc[-1]['close'])
        pos_tag = ""
        if sym in self._shadow_positions:
            sp = self._shadow_positions[sym]
            d = "L" if sp['direction'] == "LONG" else "S"
            pnl = (cur_price - sp['entry_price']) if sp['direction'] == 'LONG' else (sp['entry_price'] - cur_price)
            pos_tag = f" POS:{d}@{sp['entry_price']:.0f} {pnl:+.0f}pt"
        dir_tag = "^L" if direction == "LONG" else ("vS" if direction == "SHORT" else "--")
        m_s = v1r['m_score'] if v1r else 0
        v_s = v1r['v_score'] if v1r else 0
        q_s = v1r['q_score'] if v1r else 0
        sess = v1r.get('session_bonus', 0) if v1r else 0
        gap_b = v1r.get('gap_bonus', 0) if v1r else 0
        self._panel_data[sym] = {
            'bj_hm': bj_hm, 'price': cur_price, 'dir': dir_tag,
            'm': m_s, 'v': v_s, 'q': q_s, 'sess': sess, 'gap': gap_b,
            'total': v1_score, 'pos': pos_tag,
        }

        # 持仓中：检查exit
        if sym in self._shadow_positions:
            self._check_exit(sym, b5, b15, utc_hm)
            return

        # 无持仓：检查入场条件
        if not v1r or v1_score < threshold:
            return
        if self._low_amp.get(sym):
            return
        if _NO_OPEN_LUNCH_START <= utc_time <= _NO_OPEN_LUNCH_END:
            return
        if utc_time >= _NO_OPEN_EOD:
            return
        if self._daily_trade_count.get(sym, 0) >= 5:
            return

        # 入场
        entry_price = float(b5.iloc[-1]['close'])
        prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
        sl_pct = prof.get('stop_loss_pct', 0.005)
        stop = entry_price * (1 - sl_pct) if direction == 'LONG' else entry_price * (1 + sl_pct)

        self._shadow_positions[sym] = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': utc_hm,
            'entry_score': v1_score,
            'stop_loss': stop,
            'highest_since': float(b5.iloc[-1]['high']),
            'lowest_since': float(b5.iloc[-1]['low']),
            'bars_held': 0,
            'signal_version': version,
            'raw_mom_5m': raw_mom,
            'raw_atr_ratio': raw_atr,
            'raw_vol_pct': raw_vpct,
            'v1_m': v1r['m_score'],
            'v1_v': v1r['v_score'],
            'v1_q': v1r['q_score'],
            'session_bonus': v1r['session_bonus'],
            'gap_bonus': v1r['gap_bonus'],
            'gap_aligned': gap_aligned,
            'entry_session': f"{bj_h:02d}:00-{bj_h+1:02d}:00",
        }

        print(f"  [V1] {sym} OPEN {direction} @{entry_price:.0f} score={v1_score} [{version}] {bj_hm}")

    def _check_exit(self, sym: str, b5: pd.DataFrame, b15, utc_hm: str):
        """复用v2的check_exit。"""
        pos = self._shadow_positions[sym]
        cur_price = float(b5.iloc[-1]['close'])
        high = float(b5.iloc[-1]['high'])
        low = float(b5.iloc[-1]['low'])

        if pos['direction'] == 'LONG':
            pos['highest_since'] = max(pos['highest_since'], high)
        else:
            pos['lowest_since'] = min(pos['lowest_since'], low)
        pos['bars_held'] += 1

        exit_info = check_exit(
            pos, cur_price, b5,
            b15 if b15 is not None and len(b15) > 0 else None,
            utc_hm, reverse_signal_score=0, is_high_vol=True,
            symbol=sym, spot_price=0,
        )

        if exit_info['should_exit']:
            exit_price = cur_price
            entry_price = pos['entry_price']
            pnl = (exit_price - entry_price) if pos['direction'] == 'LONG' else (entry_price - exit_price)

            try:
                eh, em = int(pos['entry_time'][:2]), int(pos['entry_time'][3:5])
                xh, xm = int(utc_hm[:2]), int(utc_hm[3:5])
                hold_min = (xh - eh) * 60 + (xm - em)
            except:
                hold_min = 0

            trade_date = self._backtest_date or datetime.now().strftime("%Y%m%d")
            entry_bj = f"{(eh+8)%24:02d}:{em:02d}"
            exit_bj = f"{(xh+8)%24:02d}:{xm:02d}"

            self._write_shadow_trade({
                'trade_date': trade_date,
                'symbol': sym,
                'direction': pos['direction'],
                'entry_time': entry_bj,
                'entry_price': entry_price,
                'entry_score': pos['entry_score'],
                'entry_dm': 1.0, 'entry_f': 1.0, 'entry_t': 1.0, 'entry_s': 0,
                'entry_m': pos['v1_m'],
                'entry_v': pos['v1_v'],
                'entry_q': pos['v1_q'],
                'exit_time': exit_bj,
                'exit_price': exit_price,
                'exit_reason': exit_info['exit_reason'],
                'pnl_pts': pnl,
                'hold_minutes': hold_min,
                'operator_action': 'SHADOW',
                'is_executed': 0,
                'raw_mom_5m': pos['raw_mom_5m'],
                'raw_atr_ratio': pos['raw_atr_ratio'],
                'raw_vol_pct': pos['raw_vol_pct'],
                'entry_session': pos['entry_session'],
                'session_bonus': pos['session_bonus'],
                'gap_aligned': 1 if pos['gap_aligned'] else 0,
                'gap_bonus': pos['gap_bonus'],
                'signal_version': pos['signal_version'],
            })

            print(f"  [V1] {sym} CLOSE {exit_info['exit_reason']} pnl={pnl:+.1f}pt {entry_bj}→{exit_bj}")

            del self._shadow_positions[sym]
            self._daily_trade_count[sym] = self._daily_trade_count.get(sym, 0) + 1
            self._closed_pnl[sym] = self._closed_pnl.get(sym, 0) + pnl

    def _write_shadow_trade(self, trade: Dict):
        cols = list(trade.keys())
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        vals = [trade[c] for c in cols]
        try:
            self._db._conn.execute(
                f"INSERT INTO shadow_trades_new_mapping ({col_names}) VALUES ({placeholders})", vals)
            self._db._conn.commit()
        except Exception as e:
            print(f"  [V1] shadow write failed: {e}")

    def _extract_today_bars(self, bar_5m: pd.DataFrame) -> pd.DataFrame:
        if len(bar_5m) < 2:
            return bar_5m
        idx = bar_5m.index
        diffs = idx.to_series().diff()
        gaps = diffs[diffs > pd.Timedelta(minutes=30)]
        if len(gaps) > 0:
            pos = bar_5m.index.get_loc(gaps.index[-1])
            return bar_5m.iloc[pos:]
        return bar_5m

    def print_panel(self):
        """打印v1面板（每根bar后调用，类似v2的status面板）。"""
        if not self._panel_data:
            return
        # 只在两个品种都有数据时打印（避免半屏输出）
        if len(self._panel_data) < 2:
            return
        bj_hm = list(self._panel_data.values())[0].get('bj_hm', '??:??')
        closed_pnl = sum(self._closed_pnl.values())
        float_pnl = 0.0
        pos_parts = []
        for sym in SYMBOLS:
            if sym in self._shadow_positions:
                sp = self._shadow_positions[sym]
                pd_info = self._panel_data.get(sym, {})
                cp = pd_info.get('price', sp['entry_price'])
                pnl = (cp - sp['entry_price']) if sp['direction'] == 'LONG' else (sp['entry_price'] - cp)
                float_pnl += pnl
                d = "L" if sp['direction'] == 'LONG' else 'S'
                pos_parts.append(f"{sym}{d}1")
        total_pnl = closed_pnl + float_pnl
        trades_n = sum(self._daily_trade_count.values()) + len(self._shadow_positions)

        print(f"\n  v1 Monitor | {bj_hm}")
        print(f"  {'SYM':4s} | {'PRICE':>8s} | {'DIR':3s} | {'M':>3s} {'V':>3s} {'Q':>3s} {'S':>3s} {'G':>3s} | {'TOT':>3s} | POS")
        print(f"  -----+----------+-----+---------------------+-----+----")
        for sym in SYMBOLS:
            p = self._panel_data.get(sym)
            if not p:
                continue
            print(f"  {sym:4s} | {p['price']:8.1f} | {p['dir']:3s} | "
                  f"{p['m']:3d} {p['v']:3d} {p['q']:3d} {p['sess']:+3d} {p['gap']:3d} | "
                  f"{p['total']:3d} |{p['pos']}")
        pos_str = ' '.join(pos_parts) if pos_parts else 'none'
        print(f"  POS: {pos_str}  P&L: {total_pnl:+.0f}pt"
              f" (已平{closed_pnl:+.0f} 浮盈{float_pnl:+.0f})  Trades: {trades_n}")

    def on_day_end(self):
        for sym in list(self._shadow_positions.keys()):
            print(f"  [V1] {sym} 日末强平")
            del self._shadow_positions[sym]
        self._low_amp.clear()
        self._daily_trade_count.clear()


def run_live():
    """实盘模式：独立TQ连接。"""
    from dotenv import load_dotenv
    load_dotenv()
    from tqsdk import TqApi, TqAuth

    print("=" * 50)
    print("  v1 Monitor 独立进程 (实盘)")
    print("=" * 50)

    monitor = MonitorV1()

    auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))
    api = TqApi(auth=auth)

    spot_klines = {}
    for sym in SYMBOLS:
        spot_klines[sym] = api.get_kline_serial(SPOT_TQ[sym], 300, 200)

    last_bar_time: Dict[str, int] = {}

    print("  v1 Monitor running...\n")

    while True:
        api.wait_update()
        any_changed = False
        for sym in SYMBOLS:
            sk = spot_klines.get(sym)
            if sk is None or not api.is_changing(sk) or len(sk) < 2:
                continue
            completed_dt = int(sk.iloc[-2]["datetime"])
            if last_bar_time.get(sym) == completed_dt:
                continue
            last_bar_time[sym] = completed_dt
            any_changed = True

            # 构建bar数据
            completed = sk.iloc[:-1]
            df = completed[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(completed["datetime"], unit="ns")
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = df[c].astype(float)

            # 15m重采样
            b15 = None
            if len(df) >= 3:
                b15_full = _build_15m_from_5m(df)
                if len(b15_full) > 1:
                    b15 = b15_full.iloc[:-1]
                elif len(b15_full) > 0:
                    b15 = b15_full

            monitor.on_bar(sym, df, b15)

        if any_changed:
            monitor.print_panel()


def run_backtest(td: str):
    """TqBacktest模式：用TQ回放驱动v1 monitor。"""
    from dotenv import load_dotenv
    load_dotenv()
    from tqsdk import TqApi, TqBacktest, BacktestFinished, TqAuth

    y, m, d = int(td[:4]), int(td[4:6]), int(td[6:8])

    print(f"  v1 TqBacktest: {td}")

    monitor = MonitorV1(backtest_date=td)

    # 覆盖vol_profile到回测日期之前
    for sym in SYMBOLS:
        spot = SPOT_SYM[sym]
        bar_all = monitor._db.query_df(
            f"SELECT datetime, volume FROM index_min "
            f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime"
        )
        if bar_all is not None and len(bar_all) > 0:
            bar_all['volume'] = bar_all['volume'].astype(float)
            monitor._vol_profiles[sym] = compute_volume_profile(
                bar_all, before_date=td, lookback_days=20)

    auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))
    api = TqApi(backtest=TqBacktest(
        start_dt=datetime(y, m, d, 0, 0, 0),
        end_dt=datetime(y, m, d, 23, 59, 59),
    ), auth=auth)

    spot_klines = {}
    for sym in SYMBOLS:
        spot_klines[sym] = api.get_kline_serial(SPOT_TQ[sym], 300, 200)

    last_bar_time: Dict[str, int] = {}

    try:
        while True:
            api.wait_update()
            for sym in SYMBOLS:
                sk = spot_klines.get(sym)
                if sk is None or not api.is_changing(sk) or len(sk) < 2:
                    continue
                completed_dt = int(sk.iloc[-2]["datetime"])
                if last_bar_time.get(sym) == completed_dt:
                    continue
                last_bar_time[sym] = completed_dt

                completed = sk.iloc[:-1]
                df = completed[["open", "high", "low", "close", "volume"]].copy()
                df.index = pd.to_datetime(completed["datetime"], unit="ns")
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = df[c].astype(float)

                b15 = None
                if len(df) >= 3:
                    b15_full = _build_15m_from_5m(df)
                    if len(b15_full) > 1:
                        b15 = b15_full.iloc[:-1]
                    elif len(b15_full) > 0:
                        b15 = b15_full

                monitor.on_bar(sym, df, b15)

            # TqBacktest模式不打印面板（太多输出）

    except BacktestFinished:
        pass
    finally:
        api.close()

    # 返回今天的v1 shadow trades
    trades = monitor._db.query_df(
        f"SELECT * FROM shadow_trades_new_mapping WHERE trade_date='{td}' ORDER BY entry_time"
    )
    return trades


def main():
    parser = argparse.ArgumentParser(description="v1信号系统独立Monitor")
    parser.add_argument("--backtest", type=str, default=None, help="TqBacktest日期 YYYYMMDD")
    args = parser.parse_args()

    if args.backtest:
        trades = run_backtest(args.backtest)
        if trades is not None and len(trades) > 0:
            print(f"\n  v1 shadow trades ({len(trades)}笔):")
            for _, t in trades.iterrows():
                print(f"    {t['symbol']} {t['direction']} {t['entry_time']}→{t['exit_time']} "
                      f"{t['exit_reason']} {float(t['pnl_pts']):+.1f}pt [{t['signal_version']}]")
        else:
            print("\n  v1 shadow trades: 0笔")
    else:
        run_live()


if __name__ == "__main__":
    main()
