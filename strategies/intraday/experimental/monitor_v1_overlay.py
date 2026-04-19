"""v1评分系统的Monitor Overlay。

不是独立进程，而是寄生在现有monitor上的旁路评分系统。
在monitor的_on_new_bar里被调用，用v1评分函数重新打分，
管理独立的shadow positions，写入shadow_trades_new_mapping表。

用法：
    在monitor.py的_on_new_bar末尾加一行：
        self._v1_overlay.on_bar(bar_data, bar_15m_data, current_time_utc)
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, Optional

from strategies.intraday.experimental.signal_new_mapping_v1_im import (
    score as score_im, THRESHOLD as THR_IM,
)
from strategies.intraday.experimental.signal_new_mapping_v1_ic import (
    score as score_ic, THRESHOLD as THR_IC,
)
from strategies.intraday.A_share_momentum_signal_v2 import (
    SignalGeneratorV2, SYMBOL_PROFILES, _DEFAULT_PROFILE,
    NO_TRADE_BEFORE, NO_TRADE_AFTER, check_low_amplitude,
    _get_utc_time,
)
from strategies.intraday.atomic_factors import momentum, momentum_direction, atr_ratio


# 交易时间窗口（跟v2一致）
_NO_OPEN_EOD = time(6, 30)   # 14:30 BJ
_NO_OPEN_LUNCH_START = time(3, 20)  # 11:20 BJ
_NO_OPEN_LUNCH_END = time(5, 5)     # 13:05 BJ


class MonitorV1Overlay:
    """v1评分的shadow trade管理器。"""

    def __init__(self, db, tmp_dir: str = "tmp"):
        self._db = db
        self._tmp_dir = tmp_dir
        self._shadow_positions: Dict[str, Dict] = {}  # sym -> position
        self._gen = SignalGeneratorV2({"min_signal_score": 20})  # 用v2获取raw值
        self._vol_profiles: Dict[str, dict] = {}
        self._low_amp: Dict[str, Optional[bool]] = {}
        self._daily_trade_count: Dict[str, int] = {}
        self._max_daily_trades = 5

    def set_vol_profiles(self, vp: Dict):
        """从主monitor接收vol_profiles。"""
        self._vol_profiles = vp

    def on_bar(self, bar_data: Dict[str, pd.DataFrame],
               bar_15m_data: Dict[str, pd.DataFrame],
               current_time_utc: str):
        """在主monitor的_on_new_bar末尾调用。"""
        for sym, b5 in bar_data.items():
            if sym not in ('IM', 'IC'):
                continue
            if b5 is None or len(b5) < 20:
                continue

            b15 = bar_15m_data.get(sym)
            utc_time = _get_utc_time(b5)
            if not utc_time:
                continue

            # 交易时间检查
            if utc_time < NO_TRADE_BEFORE or utc_time > NO_TRADE_AFTER:
                continue

            utc_hm = utc_time.strftime("%H:%M")

            # 开盘振幅过滤（跟v2一致）
            if sym not in self._low_amp:
                self._low_amp[sym] = None
            if self._low_amp[sym] is None and utc_hm >= "02:00":
                today_bars = self._extract_today_bars(b5)
                if len(today_bars) >= 6:
                    self._low_amp[sym] = check_low_amplitude(today_bars.iloc[:6])

            # 持仓管理：检查exit
            if sym in self._shadow_positions:
                self._check_exit(sym, b5, b15, utc_hm)
                continue  # 有持仓时不检查入场

            # 无持仓：检查入场
            if self._low_amp.get(sym):
                continue  # 低振幅日不开仓

            # 午餐/尾盘限制
            if _NO_OPEN_LUNCH_START <= utc_time <= _NO_OPEN_LUNCH_END:
                continue
            if utc_time >= _NO_OPEN_EOD:
                continue

            # 日内交易次数限制
            if self._daily_trade_count.get(sym, 0) >= self._max_daily_trades:
                continue

            # 用v2的score_all获取raw值和方向
            vp = self._vol_profiles.get(sym)
            result = self._gen.score_all(
                sym, b5, b15 if b15 is not None and len(b15) > 0 else None,
                None, None, None,
                zscore=None, is_high_vol=True, d_override=None,
                vol_profile=vp,
            )
            if not result or not result.get('direction'):
                continue

            direction = result['direction']
            raw_mom = result.get('raw_mom_5m', 0.0)
            raw_atr = result.get('raw_atr_ratio', 0.0)
            raw_vpct = result.get('raw_vol_pct', -1.0)
            raw_vratio = result.get('raw_vol_ratio', -1.0)

            # 解析BJ小时
            bj_h = (utc_time.hour + 8) % 24

            # Gap计算
            today_bars = self._extract_today_bars(b5)
            gap_pct = 0.0
            gap_aligned = False
            if len(today_bars) > 0:
                today_open = float(today_bars.iloc[0]['open'])
                # 简化：用第一根bar的open vs 前一根bar的close
                if len(b5) > len(today_bars):
                    prev_close = float(b5.iloc[-(len(today_bars)+1)]['close'])
                    if prev_close > 0:
                        gap_pct = (today_open - prev_close) / prev_close
                        gap_aligned = (gap_pct > 0 and direction == 'LONG') or \
                                      (gap_pct < 0 and direction == 'SHORT')

            # v1评分
            if sym == 'IM':
                v1_result = score_im(raw_mom, raw_atr, raw_vpct, raw_vratio, bj_h, gap_aligned)
                threshold = THR_IM
                version = "v1_im"
            else:
                v1_result = score_ic(raw_mom, raw_atr, raw_vpct, raw_vratio, bj_h, gap_aligned)
                threshold = THR_IC
                version = "v1_ic"

            v1_score = v1_result['total_score']

            if v1_score < threshold:
                continue

            # 入场！创建shadow position
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
                'v1_m': v1_result['m_score'],
                'v1_v': v1_result['v_score'],
                'v1_q': v1_result['q_score'],
                'session_bonus': v1_result['session_bonus'],
                'gap_bonus': v1_result['gap_bonus'],
                'gap_aligned': gap_aligned,
                'entry_session': f"{bj_h:02d}:00-{bj_h+1:02d}:00",
            }

    def _check_exit(self, sym: str, b5: pd.DataFrame, b15, utc_hm: str):
        """复用v2的check_exit逻辑。"""
        from strategies.intraday.A_share_momentum_signal_v2 import check_exit

        pos = self._shadow_positions[sym]
        cur_price = float(b5.iloc[-1]['close'])
        high = float(b5.iloc[-1]['high'])
        low = float(b5.iloc[-1]['low'])

        # 更新极值
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
            if pos['direction'] == 'LONG':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            # 计算持仓时间
            try:
                eh, em = int(pos['entry_time'][:2]), int(pos['entry_time'][3:5])
                xh, xm = int(utc_hm[:2]), int(utc_hm[3:5])
                hold_min = (xh - eh) * 60 + (xm - em)
            except:
                hold_min = 0

            # 写入数据库
            trade_date = datetime.now().strftime("%Y%m%d")
            entry_bj = f"{(eh+8)%24:02d}:{em:02d}" if 'eh' in dir() else pos['entry_time']
            exit_bj = f"{(xh+8)%24:02d}:{xm:02d}" if 'xh' in dir() else utc_hm

            self._write_shadow_trade({
                'trade_date': trade_date,
                'symbol': sym,
                'direction': pos['direction'],
                'entry_time': entry_bj,
                'entry_price': entry_price,
                'entry_score': pos['entry_score'],
                'entry_dm': 1.0,
                'entry_f': 1.0,
                'entry_t': 1.0,
                'entry_s': 0,
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

            del self._shadow_positions[sym]
            self._daily_trade_count[sym] = self._daily_trade_count.get(sym, 0) + 1

    def _write_shadow_trade(self, trade: Dict):
        """写入shadow_trades_new_mapping表。"""
        cols = list(trade.keys())
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        vals = [trade[c] for c in cols]
        try:
            self._db._conn.execute(
                f"INSERT INTO shadow_trades_new_mapping ({col_names}) VALUES ({placeholders})",
                vals)
            self._db._conn.commit()
        except Exception as e:
            print(f"  [V1-SHADOW] write failed: {e}")

    def _extract_today_bars(self, bar_5m: pd.DataFrame) -> pd.DataFrame:
        """提取今天的bar（简化版，跟v2的逻辑一致）。"""
        if len(bar_5m) < 2:
            return bar_5m
        idx = bar_5m.index if isinstance(bar_5m.index, pd.DatetimeIndex) else pd.to_datetime(bar_5m.get('datetime', bar_5m.index))
        diffs = idx.to_series().diff()
        gaps = diffs[diffs > pd.Timedelta(minutes=30)]
        if len(gaps) > 0:
            last_gap_idx = gaps.index[-1]
            pos = bar_5m.index.get_loc(last_gap_idx)
            return bar_5m.iloc[pos:]
        return bar_5m

    def on_day_end(self):
        """日末清理。"""
        # 强制平仓所有shadow持仓
        for sym in list(self._shadow_positions.keys()):
            pos = self._shadow_positions[sym]
            print(f"  [V1-SHADOW] {sym} 日末强平 (entry={pos['entry_price']:.0f})")
            del self._shadow_positions[sym]
        self._low_amp.clear()
        self._daily_trade_count.clear()
