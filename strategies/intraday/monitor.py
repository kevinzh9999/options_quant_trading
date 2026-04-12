"""
monitor.py
----------
盘中实时信号监控 + 数据记录。

用法：
    python -m strategies.intraday.monitor

连接天勤实时行情，每根5分钟K线结束时：
  1. 运行信号评估，输出结果
  2. 记录盘口快照 → orderbook_snapshots
  3. 记录信号日志 → signal_log（无信号也记录 score=0）
  4. 信号 >= 55 时交互式确认 → trade_decisions
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 加载 .env（需在 os.getenv 调用之前）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

from strategies.intraday.strategy import IntradayStrategy, IntradayConfig
from strategies.intraday.signal import IntradaySignal
from strategies.intraday.A_share_momentum_signal_v2 import (
    SignalGeneratorV2, SignalGeneratorV3, SIGNAL_ROUTING,
    SentimentData, check_exit,
)


# ---------------------------------------------------------------------------
# 数据库记录
# ---------------------------------------------------------------------------

class IntradayRecorder:
    """盘中数据记录器。写入与 daily_record 同一个数据库。"""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            from config.config_loader import ConfigLoader
            db_path = ConfigLoader().get_db_path()
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """确保盘中记录表存在。"""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                symbol       TEXT NOT NULL,
                datetime     TEXT NOT NULL,
                bid_price1   REAL,
                ask_price1   REAL,
                bid_volume1  REAL,
                ask_volume1  REAL,
                last_price   REAL,
                volume       REAL,
                PRIMARY KEY (symbol, datetime)
            );
            CREATE TABLE IF NOT EXISTS signal_log (
                datetime          TEXT NOT NULL,
                symbol            TEXT NOT NULL,
                direction         TEXT,
                score             INT,
                score_breakout    INT,
                score_vwap        INT,
                score_multiframe  INT,
                score_volume      INT,
                score_daily       INT,
                score_orderbook   INT,
                action_taken      TEXT,
                reason            TEXT,
                PRIMARY KEY (datetime, symbol)
            );
            CREATE TABLE IF NOT EXISTS trade_decisions (
                datetime          TEXT NOT NULL PRIMARY KEY,
                symbol            TEXT,
                signal_score      INT,
                signal_direction  TEXT,
                decision          TEXT,
                manual_note       TEXT,
                created_at        TEXT
            );
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date       TEXT,
                symbol           TEXT,
                direction        TEXT,
                entry_time       TEXT,
                entry_price      REAL,
                entry_score      INT,
                entry_dm         REAL,
                entry_f          REAL,
                entry_t          REAL,
                entry_s          REAL,
                entry_m          INT,
                entry_v          INT,
                entry_q          INT,
                exit_time        TEXT,
                exit_price       REAL,
                exit_reason      TEXT,
                pnl_pts          REAL,
                hold_minutes     INT,
                operator_action  TEXT,
                is_executed      INT DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_shadow_date ON shadow_trades (trade_date);
        """)
        # 新增列（已存在时忽略）
        for col, ctype in [
            ("score_v2", "INT"),
            ("direction_v2", "TEXT"),
            ("score_v3", "INT"),
            ("direction_v3", "TEXT"),
            ("style_v3", "TEXT"),
            ("signal_version", "TEXT"),
            # 研究指标（不参与评分，供未来迭代分析）
            ("adx_14", "REAL"),
            ("body_ratio", "REAL"),
            ("vwap_offset", "REAL"),
            ("style_spread", "REAL"),
            ("cross_rank", "INT"),
        ]:
            try:
                conn.execute(
                    f"ALTER TABLE signal_log ADD COLUMN {col} {ctype}")
            except Exception:
                pass
        conn.commit()
        conn.close()

    def record_shadow_trade(self, trade: Dict) -> None:
        """Write a completed shadow trade to the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO shadow_trades "
                "(trade_date, symbol, direction, entry_time, entry_price, "
                "entry_score, entry_dm, entry_f, entry_t, entry_s, "
                "entry_m, entry_v, entry_q, "
                "exit_time, exit_price, exit_reason, pnl_pts, hold_minutes, "
                "operator_action, is_executed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (trade.get("trade_date"), trade.get("symbol"),
                 trade.get("direction"), trade.get("entry_time"),
                 trade.get("entry_price"), trade.get("entry_score"),
                 trade.get("entry_dm"), trade.get("entry_f"),
                 trade.get("entry_t"), trade.get("entry_s"),
                 trade.get("entry_m"), trade.get("entry_v"),
                 trade.get("entry_q"), trade.get("exit_time"),
                 trade.get("exit_price"), trade.get("exit_reason"),
                 trade.get("pnl_pts"), trade.get("hold_minutes"),
                 trade.get("operator_action"), trade.get("is_executed", 0)),
            )
            conn.commit()
        except Exception as e:
            print(f"  [WARN] shadow_trades write failed: {e}")
        finally:
            conn.close()

    def record_orderbook(
        self, dt_str: str, symbol: str, quote: Dict,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO orderbook_snapshots "
            "(symbol, datetime, bid_price1, ask_price1, bid_volume1, "
            "ask_volume1, last_price, volume) VALUES (?,?,?,?,?,?,?,?)",
            (symbol, dt_str,
             quote.get("bid_price1"), quote.get("ask_price1"),
             quote.get("bid_volume1"), quote.get("ask_volume1"),
             quote.get("last_price"), quote.get("volume", 0)),
        )
        conn.commit()
        conn.close()

    def record_signal(
        self,
        dt_str: str,
        symbol: str,
        signal: Optional[IntradaySignal],
        action_taken: str,
        v2_score: int = 0,
        v2_direction: str = "",
        v3_score: int = 0,
        v3_direction: str = "",
        v3_style: str = "",
        signal_version: str = "",
        score_detail: dict | None = None,
    ) -> None:
        """Write signal to signal_log with optional v2/v3 dimension detail."""
        sd = score_detail or {}
        if signal is not None:
            comp = signal.components
            row = (
                dt_str, symbol, signal.direction, signal.score,
                comp.get("opening_breakout", {}).get("score", 0),
                comp.get("vwap", {}).get("score", 0),
                comp.get("multi_tf", {}).get("score", 0),
                comp.get("volume", {}).get("score", 0),
                comp.get("daily_levels", {}).get("score", 0),
                comp.get("orderbook", {}).get("score", 0),
                action_taken,
                signal.reason,
                v2_score or 0, v2_direction or "",
                v3_score or 0, v3_direction or "", v3_style or "",
                signal_version or "",
                # dimension detail
                sd.get("s_momentum", 0),
                sd.get("s_volatility", 0),
                sd.get("s_volume", 0),
                sd.get("intraday_filter"),
                sd.get("time_weight"),
                sd.get("sentiment_mult"),
                sd.get("zscore"),
                sd.get("rsi"),
                sd.get("pre_z_total"),
                sd.get("total"),
                sd.get("z_filter", ""),
            )
        else:
            row = (dt_str, symbol, None, 0, 0, 0, 0, 0, 0, 0, "NONE", "",
                   v2_score or 0, v2_direction or "",
                   v3_score or 0, v3_direction or "", v3_style or "",
                   signal_version or "",
                   sd.get("s_momentum", 0), sd.get("s_volatility", 0),
                   sd.get("s_volume", 0), sd.get("intraday_filter"),
                   sd.get("time_weight"), sd.get("sentiment_mult"),
                   sd.get("zscore"), sd.get("rsi"),
                   sd.get("pre_z_total"), sd.get("total"),
                   sd.get("z_filter", ""))

        # 追加研究指标列（不影响核心信号记录）
        extra = (
            sd.get("adx_14"),
            sd.get("body_ratio"),
            sd.get("vwap_offset"),
            sd.get("style_spread"),
            sd.get("cross_rank"),
        )
        full_row = row + extra

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO signal_log "
            "(datetime, symbol, direction, score, score_breakout, score_vwap, "
            "score_multiframe, score_volume, score_daily, score_orderbook, "
            "action_taken, reason, score_v2, direction_v2, "
            "score_v3, direction_v3, style_v3, signal_version, "
            "s_momentum, s_volatility, s_quality, intraday_filter, "
            "time_mult, sentiment_mult, z_score, rsi, "
            "raw_score, filtered_score, filter_reason, "
            "adx_14, body_ratio, vwap_offset, style_spread, cross_rank) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            full_row,
        )
        conn.commit()
        conn.close()

    def record_decision(
        self,
        dt_str: str,
        symbol: str,
        score: int,
        direction: str,
        decision: str,
        note: str = "",
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO trade_decisions "
            "(datetime, symbol, signal_score, signal_direction, decision, "
            "manual_note, created_at) VALUES (?,?,?,?,?,?,?)",
            (dt_str, symbol, score, direction, decision, note,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# 研究指标计算（不参与评分，只记录到signal_log供未来分析）
# 所有函数fail-safe：任何异常返回None，绝不影响信号流程
# ---------------------------------------------------------------------------

def _calc_research_indicators(
    bar_data: Dict[str, pd.DataFrame],
    sym: str,
    spot_map: Dict[str, str] = None,
) -> Dict[str, Any]:
    """计算研究指标，返回dict供signal_log记录。全部try/except包裹。"""
    result: Dict[str, Any] = {}
    b5 = bar_data.get(sym)
    if b5 is None or len(b5) < 20:
        return result

    close = b5["close"].astype(float)
    high = b5["high"].astype(float)
    low = b5["low"].astype(float)
    opn = b5["open"].astype(float)
    vol = b5["volume"].astype(float) if "volume" in b5.columns else None

    # --- ADX(14) ---
    try:
        period = 14
        if len(close) >= period * 2:
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            # 只保留较大的那个
            plus_dm[plus_dm < minus_dm] = 0
            minus_dm[minus_dm < plus_dm] = 0
            tr = pd.concat([high - low, (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
            minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
            denom = plus_di + minus_di
            dx = 100 * (plus_di - minus_di).abs() / denom.replace(0, float("nan"))
            adx = dx.ewm(span=period, adjust=False).mean()
            result["adx_14"] = round(float(adx.iloc[-1]), 2) if not pd.isna(adx.iloc[-1]) else None
    except Exception:
        pass

    # --- Body Ratio（最后一根completed bar）---
    try:
        hl = float(high.iloc[-1]) - float(low.iloc[-1])
        if hl > 1e-5:
            result["body_ratio"] = round(
                (float(close.iloc[-1]) - float(opn.iloc[-1])) / hl, 3)
    except Exception:
        pass

    # --- VWAP Offset（日内累计VWAP偏离度）---
    try:
        if vol is not None and len(vol) > 5:
            tp = (high + low + close) / 3
            cum_tpv = (tp * vol).cumsum()
            cum_vol = vol.cumsum().replace(0, float("nan"))
            vwap = cum_tpv / cum_vol
            cur_vwap = float(vwap.iloc[-1])
            cur_close = float(close.iloc[-1])
            if cur_vwap > 0:
                result["vwap_offset"] = round(
                    (cur_close - cur_vwap) / cur_vwap * 100, 4)
    except Exception:
        pass

    # --- Style Spread（IM-IH return差，衡量风格分化）---
    try:
        _sym_to_spot = spot_map or {
            "IM": "000852.SH", "IC": "000905.SH",
            "IF": "000300.SH", "IH": "000016.SH"}
        im_bars = bar_data.get("IM")
        ih_bars = bar_data.get("IH")
        if im_bars is not None and ih_bars is not None and len(im_bars) >= 2 and len(ih_bars) >= 2:
            im_ret = (float(im_bars["close"].iloc[-1]) - float(im_bars["close"].iloc[-2])) / float(im_bars["close"].iloc[-2])
            ih_ret = (float(ih_bars["close"].iloc[-1]) - float(ih_bars["close"].iloc[-2])) / float(ih_bars["close"].iloc[-2])
            result["style_spread"] = round((im_ret - ih_ret) * 100, 4)
    except Exception:
        pass

    # --- Cross Rank（4品种return排名，1=最强）---
    try:
        returns = {}
        for s in ["IM", "IC", "IF", "IH"]:
            b = bar_data.get(s)
            if b is not None and len(b) >= 2:
                c = b["close"].astype(float)
                returns[s] = (float(c.iloc[-1]) - float(c.iloc[-2])) / float(c.iloc[-2])
        if len(returns) >= 2 and sym in returns:
            sorted_syms = sorted(returns, key=returns.get, reverse=True)
            result["cross_rank"] = sorted_syms.index(sym) + 1
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# 监控器
# ---------------------------------------------------------------------------

class IntradayMonitor:
    """盘中实时监控 + 数据记录。"""

    def __init__(self, config: IntradayConfig | None = None):
        self.config = config or IntradayConfig()
        self.strategy = IntradayStrategy(self.config)
        self.symbols = list(self.config.universe)
        self.recorder = IntradayRecorder()
        _debug = "--debug" in sys.argv
        # v2 / v3 信号生成器
        self.signal_v2 = SignalGeneratorV2({
            "min_signal_score": self.config.min_signal_score,
            "debug": _debug,
        })
        self.signal_v3 = SignalGeneratorV3({
            "min_signal_score": self.config.min_signal_score,
            "debug": _debug,
        })
        # 追踪最后处理的K线时间，避免重复处理
        self._last_bar_time: Dict[str, str] = {}
        # 已经prompt过的 (symbol, bar_dt) 集合，防止同一根bar重复触发
        self._prompted_bars: set = set()
        # 开盘振幅过滤：per-symbol，每天重置
        self._low_amplitude: Dict[str, bool] = {}
        # 日线数据（启动时从数据库加载）
        self._daily_data: Dict[str, pd.DataFrame] = {}
        # 最新各维度评分（供面板显示）— 路由后的最终得分
        self._latest_scores_v2: Dict[str, Dict] = {}
        self._latest_scores_v3: Dict[str, Dict] = {}
        # 面板辅助显示数据（VWAP, 趋势, 区间等）
        self._display_data: Dict[str, Dict] = {}
        # Z-Score 参数（EMA20 和 STD20，从现货指数日线计算，无换月跳变）
        self._zscore_params: Dict[str, Dict] = {}
        # 品种→TQ具体合约代码映射（如 IF→CFFEX.IF2604）
        self._tq_symbols: Dict[str, str] = {}
        # 品种→现货指数TQ代码映射
        _SPOT_TQ = {"IM": "SSE.000852", "IF": "SSE.000300",
                     "IH": "SSE.000016", "IC": "SSE.000905"}
        self._spot_tq: Dict[str, str] = {s: _SPOT_TQ[s] for s in self.symbols if s in _SPOT_TQ}
        # 账户权益（启动时缓存，用于计算建议手数）
        self._account_equity: float = 0.0
        # 波动率区间判断（GARCH ratio > 1.2 = 高波动）
        self._is_high_vol: bool = True
        # 期权情绪数据（启动时加载，盘中定期刷新）
        self._sentiment: Optional[SentimentData] = None
        # Morning Briefing d_override（启动时加载）
        self._d_override: Dict[str, float] | None = None
        # Q分历史同时段volume profile（启动时从index_min加载）
        self._vol_profiles: Dict[str, Dict[str, list]] = {}
        # 影子交易簿：记录所有信号的完整生命周期，key=symbol
        self._shadow_positions: Dict[str, Dict] = {}
        # 影子交易累计已平仓PnL（点数）和笔数，用于面板显示
        self._shadow_closed_pnl: float = 0.0
        self._shadow_closed_count: int = 0
        # 项目 tmp 目录（信号文件、持仓文件等）
        from config.config_loader import ConfigLoader
        self._tmp_dir: str = ConfigLoader().get_tmp_dir()
        self._signal_file: str = os.path.join(self._tmp_dir, "signal_pending.json")
        # v1独立进程运行，不在v2 monitor里嵌入

    def _load_daily_data(self) -> None:
        """从数据库加载日线数据 + 解析近月合约 + 用现货指数算Z-Score。"""
        # 品种→现货指数映射（无换月跳变，Z-Score更准确）
        _SPOT_INDEX = {"IM": "000852.SH", "IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH"}

        try:
            from data.storage.db_manager import DBManager, get_db
            from config.config_loader import ConfigLoader
            db = get_db()

            # 1. 解析合约（离线 fallback，TQ连接后会按持仓量重新选择）
            from utils.cffex_calendar import get_main_contract
            for sym in self.symbols:
                self._tq_symbols[sym] = get_main_contract(sym)
            print(f"  合约(离线): {list(self._tq_symbols.values())}")

            # 2. 加载现货指数日线数据（用于信号计算：daily_mult, intraday_return）
            for sym in self.symbols:
                idx_code = _SPOT_INDEX.get(sym)
                if not idx_code:
                    continue
                df = db.query_df(
                    "SELECT trade_date, close as open, close as high, "
                    "close as low, close, 0 as volume "
                    "FROM index_daily WHERE ts_code = ? "
                    "ORDER BY trade_date DESC LIMIT 30",
                    (idx_code,),
                )
                if df is not None and len(df) > 0:
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    df["close"] = df["close"].astype(float)
                    self._daily_data[sym] = df
                    print(f"  日线(现货): {sym} via {idx_code} {len(df)} 天")

            # 3. Z-Score用现货指数（无换月跳变）
            for sym in self.symbols:
                idx_code = _SPOT_INDEX.get(sym)
                if not idx_code:
                    continue
                idx_df = db.query_df(
                    "SELECT close FROM index_daily WHERE ts_code = ? "
                    "ORDER BY trade_date DESC LIMIT 30",
                    (idx_code,),
                )
                if idx_df is not None and len(idx_df) >= 20:
                    closes = idx_df["close"].astype(float).iloc[::-1].reset_index(drop=True)
                    ema20 = float(closes.ewm(span=20).mean().iloc[-1])
                    std20 = float(closes.rolling(20).std().iloc[-1])
                    if std20 > 0:
                        self._zscore_params[sym] = {
                            "ema20": ema20, "std20": std20, "index": idx_code,
                        }
                        z_now = (float(closes.iloc[-1]) - ema20) / std20
                        print(f"  Z-Score: {sym} via {idx_code}"
                              f" EMA20={ema20:.0f} STD={std20:.0f} Z={z_now:+.2f}")

            # 4. 波动率区间判断
            # 优先用 daily_model_output 的GARCH（每日重新拟合，更灵敏）
            # fallback 到 volatility_history（全历史单次拟合，较平滑）
            try:
                current_garch = None
                # 优先: daily_model_output (每日拟合，值如0.46=46%)
                dmo = db.query_df(
                    "SELECT garch_forecast_vol FROM daily_model_output "
                    "WHERE underlying='IM' AND garch_forecast_vol > 0 "
                    "ORDER BY trade_date DESC LIMIT 1"
                )
                if dmo is not None and not dmo.empty:
                    current_garch = float(dmo.iloc[0]["garch_forecast_vol"]) * 100

                # fallback: volatility_history (值如25.17=25.17%)
                if current_garch is None or current_garch <= 0:
                    gh = db.query_df(
                        "SELECT garch_sigma FROM volatility_history "
                        "WHERE garch_sigma > 0 ORDER BY trade_date DESC LIMIT 1"
                    )
                    if gh is not None and not gh.empty:
                        current_garch = float(gh.iloc[0]["garch_sigma"])

                # 长期均值从 volatility_history（稳定的参照系）
                garch_mean_df = db.query_df(
                    "SELECT AVG(garch_sigma) as avg_g FROM volatility_history "
                    "WHERE garch_sigma > 0"
                )
                if current_garch and garch_mean_df is not None and not garch_mean_df.empty:
                    long_run_garch = float(garch_mean_df.iloc[0]["avg_g"])
                    if long_run_garch > 0:
                        ratio = current_garch / long_run_garch
                        self._is_high_vol = ratio > 1.2
                        regime = "HIGH" if self._is_high_vol else "NORMAL"
                        print(f"  Vol regime: GARCH={current_garch:.1f}%"
                              f" / mean={long_run_garch:.1f}%"
                              f" = {ratio:.2f}x → {regime}")
            except Exception:
                pass

            # 5. Morning Briefing d_override — 已禁用
            # 215天回测验证：d_override 导致 -794pt (-19.2% 劣化)
            # 纯算法 dm (1.1/0.9) 已是最优，briefing 的激进覆盖(0.5/1.2)过度干预
            # self._d_override 保持 None，score_all 会使用算法 dm
            pass

            # 6. Q分历史同时段volume profile（分位数法）
            _SPOT_SYM = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
            from strategies.intraday.A_share_momentum_signal_v2 import compute_volume_profile
            today_str = datetime.now().strftime("%Y%m%d")
            for sym in self.symbols:
                spot_sym = _SPOT_SYM.get(sym)
                if not spot_sym:
                    continue
                bar_all = db.query_df(
                    f"SELECT datetime, volume FROM index_min "
                    f"WHERE symbol='{spot_sym}' AND period=300 ORDER BY datetime"
                )
                if bar_all is not None and len(bar_all) > 0:
                    bar_all["volume"] = bar_all["volume"].astype(float)
                    self._vol_profiles[sym] = compute_volume_profile(
                        bar_all, before_date=today_str, lookback_days=20)
                    n_slots = len(self._vol_profiles[sym])
                    print(f"  Vol profile: {sym} {n_slots} 时段 (过去20天)")

        except Exception as e:
            print(f"  [WARNING] daily data load failed: {e}")

    def _append_signal(self, signal_dict: dict) -> None:
        """追加信号到JSON列表（解决同根K线CLOSE被OPEN覆盖的竞态）。"""
        import json as _json
        signals = []
        if os.path.exists(self._signal_file):
            try:
                with open(self._signal_file, "r") as f:
                    data = _json.load(f)
                signals = data if isinstance(data, list) else [data]
            except (ValueError, IOError):
                signals = []
        signals.append(signal_dict)
        try:
            with open(self._signal_file, "w") as f:
                _json.dump(signals, f, indent=2)
        except Exception as e:
            print(f"  [WARN] 写入信号文件失败: {e}")

    def _write_signal_file(self, action: dict, dt_str: str) -> None:
        """构建开仓信号并追加到JSON文件供 order_executor 读取。"""
        direction = action.get("direction", "")
        bid1 = action.get("bid1", 0)
        ask1 = action.get("ask1", 0)
        last = action.get("last", 0)
        sym = action.get("symbol", "")
        signal = {
            "timestamp": dt_str,
            "symbol": sym,
            "contract": self._tq_symbols.get(sym, ""),
            "direction": direction,
            "action": action.get("action", "OPEN"),
            "score": action.get("score", 0),
            "bid1": bid1,
            "ask1": ask1,
            "last": last,
            "suggested_lots": self._calc_suggested_lots(last, sym),
            "limit_price": bid1 if direction == "SHORT" else ask1,
            "reason": action.get("reason", ""),
        }
        self._append_signal(signal)

    _CONTRACT_MULT = {"IF": 300, "IH": 300, "IM": 200, "IC": 200}

    def _calc_suggested_lots(self, entry_price: float, symbol: str = "IM") -> int:
        """Fixed Risk 0.5%: 建议手数 = 最大亏损 / 每手止损金额。"""
        if self._account_equity <= 0 or entry_price <= 0:
            return 1
        mult = self._CONTRACT_MULT.get(symbol, 200)
        risk_per_trade = self._account_equity * 0.005
        stop_loss_amount = entry_price * mult * 0.005
        if stop_loss_amount <= 0:
            return 1
        return max(1, int(risk_per_trade / stop_loss_amount))

    def _load_account_equity(self) -> None:
        """从 account_snapshots 读取最新权益（用于计算建议手数）。"""
        try:
            from data.storage.db_manager import DBManager, get_db
            from config.config_loader import ConfigLoader
            db = get_db()
            df = db.query_df(
                "SELECT balance FROM account_snapshots "
                "ORDER BY trade_date DESC LIMIT 1"
            )
            if df is not None and not df.empty:
                self._account_equity = float(df.iloc[0]["balance"])
                print(f"  账户权益: {self._account_equity:,.0f}")
        except Exception as e:
            print(f"  [警告] 账户权益加载失败: {e}")

    # ------------------------------------------------------------------
    # 盘中重启：状态持久化 & 恢复
    # ------------------------------------------------------------------

    def _save_shadow_state(self) -> None:
        """将活跃shadow持仓+信号去重状态持久化到JSON文件。

        包含: shadow_positions, prompted_bars（信号去重防重启重复）
        使用原子写入（先写tmp再rename）防止crash导致JSON损坏。
        """
        import json as _json
        state = {
            "trade_date": datetime.now().strftime("%Y%m%d"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "positions": self._shadow_positions,
            "prompted_bars": [list(k) for k in self._prompted_bars],
        }
        path = os.path.join(self._tmp_dir, "shadow_state.json")
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                _json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)  # 原子替换
        except Exception as e:
            print(f"  [WARN] shadow_state write failed: {e}")

    def _restore_daily_state(self) -> None:
        """从持久化文件+数据库恢复当天状态，防止重启后突破持仓限制。

        恢复三个层面:
          1. _shadow_positions（活跃shadow持仓）→ 从 shadow_state.json
          2. position_mgr（占位，使 can_open/total_net_lots 正确拦截）
          3. risk_mgr + daily_trades（当天已完成交易的计数/PnL）→ 从 shadow_trades 表
        """
        import json as _json
        trade_date = datetime.now().strftime("%Y%m%d")

        # --- 1. 从 shadow_state.json 恢复活跃的 shadow 持仓 ---
        state_path = os.path.join(self._tmp_dir, "shadow_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    state = _json.load(f)
                if state.get("trade_date") == trade_date:
                    restored = state.get("positions", {})
                    self._shadow_positions = restored
                    # 注入 position_mgr 占位
                    for sym, sp in restored.items():
                        self.strategy.position_mgr.inject_position(
                            sym, sp["direction"], sp["entry_price"],
                            sp.get("entry_time_bj", ""),
                            sp.get("entry_score", 0),
                        )
                    if restored:
                        parts = [f"{s} {sp['direction']}@{sp['entry_price']:.0f}"
                                 for s, sp in restored.items()]
                        print(f"  恢复shadow持仓: {', '.join(parts)}")
                    # 不恢复prompted_bars：重启后应该允许对当前bar重新评估
                    # 之前恢复全部prompted_bars会导致高分信号被静默丢弃
                    # 最多恢复当前活跃shadow持仓的品种，防止对已有持仓重复发开仓信号
                    pb_syms = set(restored.keys())  # 已有shadow持仓的品种
                    pb = state.get("prompted_bars", [])
                    pb_kept = 0
                    for item in pb:
                        if isinstance(item, list) and len(item) == 2 and item[0] in pb_syms:
                            self._prompted_bars.add(tuple(item))
                            pb_kept += 1
                    if pb_kept:
                        print(f"  恢复信号去重: {pb_kept}/{len(pb)}个(仅活跃持仓品种)")
                else:
                    # 非当天的状态文件，忽略
                    pass
            except Exception as e:
                print(f"  [WARN] shadow_state恢复失败: {e}")

        # --- 2. 从 shadow_trades 恢复当天已平仓交易计数 ---
        try:
            conn = sqlite3.connect(self.recorder.db_path)
            rows = conn.execute(
                "SELECT symbol, pnl_pts FROM shadow_trades WHERE trade_date = ?",
                (trade_date,),
            ).fetchall()
            if rows:
                for symbol, pnl_pts in rows:
                    self.strategy.position_mgr.daily_trades += 1
                    pts = pnl_pts or 0
                    pnl_money = pts * self._CONTRACT_MULT.get(symbol, 200)
                    self.strategy.risk_mgr.on_trade_complete(pnl_money, symbol)
                    self._shadow_closed_pnl += pts
                    self._shadow_closed_count += 1
                print(f"  恢复当日交易: {len(rows)}笔已平仓, "
                      f"累计{self._shadow_closed_pnl:+.0f}pt")

            # --- 3. 安全网：从 order_log 交叉验证是否有孤立持仓 ---
            # executor 已 FILLED 的 OPEN 单如果没有对应的 LOCK/CLOSE，
            # 说明有真实持仓但 shadow 丢失了——必须恢复
            open_fills = conn.execute(
                "SELECT symbol, direction, filled_price, datetime "
                "FROM order_log "
                "WHERE datetime LIKE ? AND action='OPEN' AND status='FILLED'",
                (f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}%",),
            ).fetchall()
            closed_syms = set()
            lock_rows = conn.execute(
                "SELECT symbol FROM order_log "
                "WHERE datetime LIKE ? AND action IN ('LOCK','CLOSE') "
                "AND status='FILLED'",
                (f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}%",),
            ).fetchall()
            for r in lock_rows:
                closed_syms.add(r[0])

            for sym, direction, price, dt in open_fills:
                if sym in closed_syms:
                    continue  # 已有平仓/锁仓记录
                if sym in self._shadow_positions:
                    continue  # shadow 已恢复
                # 孤立持仓：executor 开了仓但 shadow 丢失
                entry_time_bj = dt[11:16] if len(dt) > 15 else "09:30"
                entry_time_utc = entry_time_bj  # 近似
                print(f"  ⚠ 发现孤立持仓: {sym} {direction}@{price}"
                      f" (order_log有FILLED但shadow缺失)")
                self._shadow_positions[sym] = {
                    "direction": direction,
                    "entry_time_utc": entry_time_utc,
                    "entry_time_bj": entry_time_bj,
                    "entry_price": float(price),
                    "highest_since": float(price),
                    "lowest_since": float(price),
                    "volume": 1,
                    "bars_below_mid": 0,
                    "entry_score": 0,
                    "entry_dm": 0, "entry_f": 0, "entry_t": 0, "entry_s": 0,
                    "entry_m": 0, "entry_v": 0, "entry_q": 0,
                    "operator_action": "RECOVERED",
                    "is_executed": 1,
                    "fut_symbol": self._tq_symbols.get(sym, ""),
                }
                self.strategy.position_mgr.inject_position(
                    sym, direction, float(price), entry_time_bj, 0)
                self._save_shadow_state()
                print(f"    → 已恢复shadow持仓，退出信号将正常生成")

            conn.close()
        except Exception as e:
            print(f"  [WARN] shadow_trades/order_log恢复失败: {e}")

    def _load_sentiment(self) -> None:
        """从 daily_model_output 加载情绪数据（与backtest统一数据源）。

        使用daily_model_output而非vol_monitor_snapshots，原因：
        1. 与backtest使用完全相同的数据源和字段（atm_iv_market）
        2. 按trade_date过滤，避免取到错误日期的数据
        3. vol_monitor_snapshots无trade_date字段，盘前启动可能取到旧数据
        """
        try:
            from data.storage.db_manager import get_db
            db = get_db()

            today_str = datetime.now().strftime("%Y%m%d")
            dmo = db.query_df(
                "SELECT atm_iv, atm_iv_market, vrp, rr_25d, term_structure_shape "
                "FROM daily_model_output WHERE underlying='IM' "
                f"AND trade_date < '{today_str}' "
                "ORDER BY trade_date DESC LIMIT 2"
            )
            if dmo is not None and len(dmo) >= 1:
                cur = dmo.iloc[0]
                prev = dmo.iloc[1] if len(dmo) >= 2 else cur
                iv_cur = float(cur.get("atm_iv_market") or cur.get("atm_iv") or 0)
                iv_prev = float(prev.get("atm_iv_market") or prev.get("atm_iv") or 0)
                self._sentiment = SentimentData(
                    atm_iv=iv_cur,
                    atm_iv_prev=iv_prev,
                    rr_25d=float(cur.get("rr_25d") or 0),
                    rr_25d_prev=float(prev.get("rr_25d") or 0),
                    vrp=float(cur.get("vrp") or 0),
                    term_structure=str(cur.get("term_structure_shape") or ""),
                )
                print(f"  情绪数据: IV={iv_cur*100:.1f}%"
                      f"  RR={self._sentiment.rr_25d*100:+.1f}pp"
                      f"  VRP={self._sentiment.vrp*100:+.1f}%")
        except Exception as e:
            print(f"  [警告] 情绪数据加载失败: {e}")

    def _update_futures_positions(self, api) -> None:
        """从 TQ 读取期货实盘持仓（所有活跃合约），写入共享文件供 executor 对账。"""
        try:
            from utils.cffex_calendar import active_im_months
            today_str = datetime.now().strftime("%Y%m%d")
            active_months = active_im_months(today_str)

            positions = {}
            # 先订阅所有候选合约的 quote（TQ 要求先订阅才能读到 position）
            for sym in ["IM", "IF", "IH", "IC"]:
                for month in active_months:
                    try:
                        api.get_quote(f"CFFEX.{sym}{month}")
                    except Exception:
                        pass

            for sym in ["IM", "IF", "IH", "IC"]:
                total_long = 0
                total_short = 0
                total_long_today = 0
                total_short_today = 0
                last_price = 0.0
                float_pnl_long = 0.0
                float_pnl_short = 0.0
                held_contract = ""
                for month in active_months:
                    contract = f"CFFEX.{sym}{month}"
                    try:
                        pos = api.get_position(contract)
                        if pos.pos_long > 0 or pos.pos_short > 0:
                            total_long += pos.pos_long
                            total_short += pos.pos_short
                            total_long_today += pos.pos_long_today
                            total_short_today += pos.pos_short_today
                            float_pnl_long += float(pos.float_profit_long or 0)
                            float_pnl_short += float(pos.float_profit_short or 0)
                            last_price = float(pos.last_price or 0)
                            held_contract = contract
                    except Exception:
                        pass
                if total_long > 0 or total_short > 0:
                    positions[sym] = {
                        "contract": held_contract,
                        "long": total_long,
                        "short": total_short,
                        "long_today": total_long_today,
                        "short_today": total_short_today,
                        "last_price": last_price,
                        "float_profit_long": float_pnl_long,
                        "float_profit_short": float_pnl_short,
                    }
            import json as _json
            path = os.path.join(self._tmp_dir, "futures_positions.json")
            with open(path, "w") as f:
                _json.dump({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "positions": positions,
                }, f, indent=2)
        except Exception as e:
            print(f"  [WARN] 期货持仓写出失败: {e}")

    def run(self) -> None:
        """主循环：连接天勤获取实时K线。"""
        from data.sources.tq_client import TqClient

        creds = {
            "auth_account":    os.getenv("TQ_ACCOUNT", ""),
            "auth_password":   os.getenv("TQ_PASSWORD", ""),
            "broker_id":       os.getenv("TQ_BROKER", ""),
            "account_id":      os.getenv("TQ_ACCOUNT_ID", ""),
            "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
        }
        if not creds["auth_account"] or not creds["auth_password"]:
            print("请在 .env 中设置 TQ_ACCOUNT 和 TQ_PASSWORD")
            return

        # 加载日线数据 + 情绪数据 + 账户权益 + 恢复盘中状态
        self._load_daily_data()
        self._load_sentiment()
        self._load_account_equity()
        self._restore_daily_state()

        client = TqClient(**creds)
        client.connect()
        api = client._api

        try:
            # 用 TQ 按持仓量批量选择主力合约
            from utils.cffex_calendar import active_im_months, _near_month_by_expiry
            import time as _time
            today_str = datetime.now().strftime("%Y%m%d")
            active_months = active_im_months(today_str)
            print(f"  活跃合约月份: {active_months}")
            all_quotes = {}  # (sym, contract) -> quote
            for sym in self.symbols:
                for month in active_months:
                    contract = f"CFFEX.{sym}{month}"
                    try:
                        all_quotes[(sym, contract)] = api.get_quote(contract)
                    except Exception:
                        pass
            # 等待数据到达（最多5秒，拿到什么算什么）
            if all_quotes:
                api.wait_update(deadline=_time.time() + 5)
            for sym in self.symbols:
                best, max_oi = None, 0
                for (s, contract), quote in all_quotes.items():
                    if s != sym:
                        continue
                    try:
                        oi = int(quote.open_interest or 0)
                        if oi > max_oi:
                            max_oi = oi
                            best = contract
                    except Exception:
                        pass
                if best:
                    self._tq_symbols[sym] = best
                    print(f"  {sym} → {best} (OI={max_oi:,})")
                else:
                    fallback = f"CFFEX.{sym}{_near_month_by_expiry()}"
                    self._tq_symbols[sym] = fallback
                    print(f"  {sym} → {fallback} (OI不可用,用近月)")

            # 订阅现货指数K线（信号计算用）
            spot_klines_5m: Dict[str, pd.DataFrame] = {}
            spot_klines_15m: Dict[str, pd.DataFrame] = {}
            spot_quotes: Dict = {}  # DEBUG: 暂未使用
            # 订阅期货K线和行情（下单参考+归档+贴水计算）
            fut_klines_5m: Dict[str, pd.DataFrame] = {}
            fut_quotes: Dict = {}

            for sym in self.symbols:
                # 现货指数
                spot_sym = self._spot_tq.get(sym)
                if spot_sym:
                    spot_klines_5m[sym] = api.get_kline_serial(spot_sym, 300, 200)
                    spot_klines_15m[sym] = api.get_kline_serial(spot_sym, 900, 100)

                # 期货
                fut_sym = self._tq_symbols[sym]
                fut_klines_5m[sym] = api.get_kline_serial(fut_sym, 300, 200)
                fut_quotes[sym] = api.get_quote(fut_sym)

            contracts_str = " | ".join(f"{s}={self._tq_symbols.get(s,'?')}" for s in self.symbols)
            print(f"\n{'=' * 70}")
            print(f"  Intraday Monitor | {contracts_str}")
            print(f"  DB: {self.recorder.db_path}")

            # Wait for first complete bar: skip bars until we see a bar
            # whose datetime is on a standard 5-min boundary
            self._warmup_done = False
            self._bars_since_start = 0
            print(f"  Waiting for first aligned bar...")
            print(f"{'=' * 70}\n")

            while True:
                try:
                    api.wait_update()
                except Exception as e:
                    now_h = datetime.now().hour
                    now_m = datetime.now().minute
                    if now_h >= 15 and now_m >= 5:
                        print(f"\n  Market closed, exiting normally")
                        break
                    print(f"\n  [TQ] connection error: {e}, retrying in 5s...")
                    import time as _time
                    _time.sleep(5)
                    continue

                # 检查是否有新K线（用现货指数触发信号）
                # 重要：先收集所有变化的品种，再统一处理一次
                # 避免每个品种变化都触发全品种处理，导致同一根bar的逻辑重复执行
                changed_syms = []
                for sym in self.symbols:
                    sk = spot_klines_5m.get(sym)
                    if sk is not None and api.is_changing(sk):
                        # per-symbol去重：用倒数第二根bar的datetime
                        if len(sk) >= 2:
                            completed_dt = int(sk.iloc[-2]["datetime"])
                            prev_dt = self._last_bar_time.get(sym)
                            if prev_dt != completed_dt:
                                self._last_bar_time[sym] = completed_dt
                                changed_syms.append(sym)
                bar_updated = len(changed_syms) > 0
                if bar_updated:
                    self._on_new_bar(
                        changed_syms, spot_klines_5m, spot_klines_15m,
                        fut_quotes, fut_klines_5m,
                    )

                # 每次有新bar时刷新期货持仓（跟随5分钟频率）
                if bar_updated:
                    self._update_futures_positions(api)

        except KeyboardInterrupt:
            print("\n  Monitor stopped by user")
        finally:
            client.disconnect()

    def _on_new_bar(
        self,
        changed_syms: list,
        spot_klines_5m: Dict,
        spot_klines_15m: Dict,
        fut_quotes: Dict,
        fut_klines_5m: Dict = None,
    ) -> None:
        """新K线到达时的处理——用现货K线驱动信号计算。

        changed_syms: 本次有新bar的品种列表（去重已在调用方完成）。
        全品种信号计算和shadow检查只执行一次。
        """
        if not changed_syms:
            return

        # Warmup: skip first bar（用第一个变化品种的时间戳判断）
        if not getattr(self, "_warmup_done", True):
            self._bars_since_start = getattr(self, "_bars_since_start", 0) + 1
            if self._bars_since_start <= 1:
                k5 = spot_klines_5m.get(changed_syms[0])
                if k5 is not None and len(k5) >= 2:
                    completed_dt = int(k5.iloc[-2]["datetime"])
                    try:
                        ts = pd.Timestamp(completed_dt, unit="ns")
                        if ts.minute % 5 != 0:
                            print(f"  [WARMUP] Skipping misaligned bar {ts}")
                            return
                    except Exception:
                        pass
                print(f"  [WARMUP] First aligned bar received, starting signals")
                self._warmup_done = True

        # 从bar数据推导当前时间（实盘和TqBacktest通用）
        # 不依赖datetime.now()，避免TqBacktest模式下系统时间与模拟时间不一致
        _ref_k5 = spot_klines_5m.get(changed_syms[0])
        if _ref_k5 is not None and len(_ref_k5) >= 2:
            _bar_utc = pd.Timestamp(int(_ref_k5.iloc[-2]["datetime"]), unit="ns")
            # bar时间是K线开始时间，+5min = K线结束 ≈ 当前时刻
            _now_utc = _bar_utc + pd.Timedelta(minutes=5)
            _now_bj = _now_utc + pd.Timedelta(hours=8)
        else:
            _now_utc = pd.Timestamp(datetime.utcnow())
            _now_bj = pd.Timestamp(datetime.now())
        current_time_utc = _now_utc.strftime("%Y-%m-%d %H:%M:%S")

        # 构建现货bar数据（用于信号计算）
        bar_data: Dict[str, pd.DataFrame] = {}
        bar_15m_data: Dict[str, pd.DataFrame] = {}

        for sym in self.symbols:
            k5 = spot_klines_5m.get(sym)
            if k5 is not None and len(k5) > 1:
                # TQ的最后一根bar是当前正在形成的（未完成），排除它
                # 只用已完成的bar，和backtest行为一致
                completed = k5.iloc[:-1]
                df = completed[["open", "high", "low", "close", "volume"]].copy()
                df.index = pd.to_datetime(completed["datetime"], unit="ns")
                if len(df) > 0:
                    bar_data[sym] = df

            # 15m从5m重采样（不用TQ原生15m）
            # TQ原生15m的partial更新行为不确定，从5m重采样是确定性的
            # 与backtest使用完全相同的_build_15m_from_5m逻辑
            if sym in bar_data and len(bar_data[sym]) >= 3:
                from scripts.backtest_signals_day import _build_15m_from_5m
                b15_full = _build_15m_from_5m(bar_data[sym])
                if len(b15_full) > 1:
                    bar_15m_data[sym] = b15_full.iloc[:-1]
                elif len(b15_full) > 0:
                    bar_15m_data[sym] = b15_full

        # 构建盘口数据（期货行情，用于下单参考和盘口记录）
        quote_dict: Dict[str, Dict] = {}
        for sym in self.symbols:
            q = fut_quotes.get(sym)
            if q is not None:
                qd = {
                    "bid_price1": float(q.bid_price1),
                    "ask_price1": float(q.ask_price1),
                    "bid_volume1": int(q.bid_volume1),
                    "ask_volume1": int(q.ask_volume1),
                    "last_price": float(q.last_price),
                    "volume": int(getattr(q, "volume", 0)),
                }
                quote_dict[sym] = qd
                # 记录盘口快照（fail-safe，不阻塞信号流程）
                try:
                    self.recorder.record_orderbook(current_time_utc, sym, qd)
                except Exception:
                    pass

        # 计算各品种的路由版本评分（含Z-Score过滤 + 波动率区间）
        # 必须在策略on_bar之前计算，以便过滤actions
        for sym in self.symbols:
            if sym not in bar_data or len(bar_data[sym]) < 2:
                continue
            b5 = bar_data[sym]
            b15 = bar_15m_data.get(sym)
            daily = self._daily_data.get(sym)
            qd = quote_dict.get(sym)

            # 计算当前Z-Score
            zp = self._zscore_params.get(sym)
            cur_price = float(b5.iloc[-1]["close"])
            z_val = None
            if zp and cur_price > 0 and zp["std20"] > 0:
                z_val = (cur_price - zp["ema20"]) / zp["std20"]

            # 路由版本评分
            # 外部数据固定为中性值（消除backtest/monitor差异源，只保留vol_profile）
            # TODO: 验证对齐后恢复动态值
            ver = SIGNAL_ROUTING.get(sym, "v2")
            _vp = self._vol_profiles.get(sym)
            if ver == "v3":
                sc3 = self.signal_v3.score_all(
                    sym, b5, b15, None, qd, None,
                    zscore=None, is_high_vol=True, d_override=None,
                    vol_profile=_vp)
                if sc3:
                    self._latest_scores_v3[sym] = sc3
            else:
                sc2 = self.signal_v2.score_all(
                    sym, b5, b15, None, qd, None,
                    zscore=None, is_high_vol=True, d_override=None,
                    vol_profile=_vp)
                if sc2:
                    self._latest_scores_v2[sym] = sc2

            # 面板辅助显示数据（VWAP, 趋势, 开盘区间, BOLL）
            self._display_data[sym] = self._calc_display_data(b5, b15)

        # 运行策略（外部数据固定为中性值，与score_all调用一致）
        # TODO: 验证对齐后恢复动态值
        actions = self.strategy.on_bar(
            bar_data, bar_15m_data, None, current_time_utc,
            quote_data=quote_dict,
            zscore_params={},
            is_high_vol=True,
            sentiment=None,
            d_override=None,
            vol_profiles=self._vol_profiles,
        )

        # Z-Score过滤actions：策略层不知道Z-Score，这里做最终拦截
        filtered_actions = []
        for act in actions:
            if act.get("action") != "OPEN":
                filtered_actions.append(act)
                continue
            sym = act.get("symbol", "")
            sc = self._get_routed_score(sym)
            if sc and sc.get("z_filter"):
                # Z-Score过滤生效，阻断此action
                print(f"  [Z-FILTER] {sym} {act.get('direction','')} "
                      f"score={act.get('score',0)} BLOCKED by {sc['z_filter']}")
                continue
            filtered_actions.append(act)
        actions = filtered_actions

        # 开盘振幅过滤：10:00后如果前30min振幅<0.4%，阻断OPEN
        # 215天验证：过滤13天低振幅日，+274pt改善
        from strategies.intraday.A_share_momentum_signal_v2 import check_low_amplitude
        if current_time_utc >= "02:00":  # 10:00 BJ = 02:00 UTC
            amp_filtered = []
            for act in actions:
                if act.get("action") != "OPEN":
                    amp_filtered.append(act)
                    continue
                sym = act.get("symbol", "")
                b5 = bar_data.get(sym)
                if b5 is not None and not self._low_amplitude.get(sym, False):
                    # 首次检查（只做一次/天）
                    if sym not in self._low_amplitude:
                        # 必须提取当天bar再判断振幅，b5包含多天历史
                        # check_low_amplitude取iloc[:6]，如果传全量历史会取到前几天的bar
                        today_b5 = self._extract_today_bars(b5)
                        is_low = check_low_amplitude(today_b5)
                        self._low_amplitude[sym] = is_low
                        if is_low:
                            print(f"  [AMP-FILTER] {sym} 开盘30min振幅<0.4%，今日不开新仓")
                if self._low_amplitude.get(sym, False):
                    print(f"  [AMP-FILTER] {sym} {act.get('direction','')} "
                          f"score={act.get('score',0)} BLOCKED (低振幅日)")
                    # 撤销strategy.on_bar已创建的position_mgr占位
                    self.strategy.position_mgr.remove_by_symbol(sym)
                    continue
                amp_filtered.append(act)
            actions = amp_filtered

        # 暂存信号日志所需数据（延迟到信号发出之后再写DB，避免阻塞信号流程）
        _signal_log_data = []
        action_syms = {a["symbol"] for a in actions if a.get("action") == "OPEN"}
        for sym in self.symbols:
            sig = self._get_latest_signal(sym, bar_data, bar_15m_data, quote_dict)
            action_taken = "OPEN" if sym in action_syms else ("SKIP" if sig else "NONE")
            sc2 = self._latest_scores_v2.get(sym, {})
            sc3 = self._latest_scores_v3.get(sym, {})
            ver = SIGNAL_ROUTING.get(sym, "v2")
            routed_sc = sc3 if ver == "v3" else sc2
            zp = self._zscore_params.get(sym)
            cur_z = None
            if zp and sym in bar_data and len(bar_data[sym]) > 0:
                cp = float(bar_data[sym].iloc[-1]["close"])
                if cp > 0 and zp["std20"] > 0:
                    cur_z = (cp - zp["ema20"]) / zp["std20"]
            detail = {**routed_sc, "zscore": cur_z} if routed_sc else {"zscore": cur_z}
            _signal_log_data.append((sym, sig, action_taken, sc2, sc3, ver, detail))

        # 附加盘口数据到 action（供面板和决策确认显示）
        for act in actions:
            if act.get("action") == "OPEN":
                sym = act.get("symbol", "")
                qd = quote_dict.get(sym, {})
                act["bid1"] = qd.get("bid_price1", 0)
                act["ask1"] = qd.get("ask_price1", 0)
                act["bid_vol1"] = qd.get("bid_volume1", 0)
                act["ask_vol1"] = qd.get("ask_volume1", 0)
                act["last"] = qd.get("last_price", 0)

        # 更新影子持仓（每根K线用与回测相同的exit逻辑检查）
        # 注意：shadow entry_price是期货价格，所以check_exit也必须用期货价格
        # 否则贴水导致现货-期货价差 > 0.5%，SHORT持仓一开仓就假触发STOP_LOSS
        utc_hm = _now_utc.strftime("%H:%M")
        trade_date = _now_bj.strftime("%Y%m%d")
        for sym in list(self._shadow_positions.keys()):
            sp = self._shadow_positions[sym]
            b5 = bar_data.get(sym)
            if b5 is None or len(b5) < 2:
                continue
            # 价格分两层：期货价用于止损/PnL，现货价用于Bollinger zone
            spot_price = float(b5.iloc[-1]["close"])
            fq = fut_quotes.get(sym)
            fut_price = 0.0
            if fq is not None:
                try:
                    fp = float(fq.last_price)
                    if fp > 0:
                        fut_price = fp
                except Exception:
                    pass
            # exit信号判断全部用现货价格（与backtest一致，确保exit时点对齐）
            # 期货价格只用于PnL计算
            cur_price = spot_price

            b15 = bar_15m_data.get(sym)
            b15_arg = b15 if (b15 is not None and len(b15) > 0) else None
            # Update extremes — 用现货价格（与信号数据同源）
            if sp["direction"] == "LONG":
                sp["highest_since"] = max(sp.get("highest_since", sp["entry_price"]), cur_price)
            else:
                sp["lowest_since"] = min(sp.get("lowest_since", sp["entry_price"]), cur_price)
            exit_info = check_exit(
                sp, cur_price, b5, b15_arg,
                utc_hm, reverse_signal_score=0, is_high_vol=True,
                symbol=sym, spot_price=spot_price,
            )
            if exit_info["should_exit"]:
                reason = exit_info["exit_reason"]
                entry_p = sp["entry_price"]
                # 用期货价格计算PnL（entry是期货价，exit也用期货）
                fut_exit = cur_price  # fallback
                fq = fut_quotes.get(sym)
                if fq is not None:
                    try:
                        fp = float(fq.last_price)
                        if fp > 0:
                            fut_exit = fp
                    except Exception:
                        pass
                pnl_pts = (fut_exit - entry_p) if sp["direction"] == "LONG" else (entry_p - fut_exit)
                try:
                    e_h, e_m = int(sp["entry_time_utc"][:2]), int(sp["entry_time_utc"][3:5])
                    c_h, c_m = int(utc_hm[:2]), int(utc_hm[3:5])
                    hold_min = (c_h * 60 + c_m) - (e_h * 60 + e_m)
                except Exception:
                    hold_min = 0
                self.recorder.record_shadow_trade({
                    "trade_date": trade_date,
                    "symbol": sym,
                    "direction": sp["direction"],
                    "entry_time": sp["entry_time_bj"],
                    "entry_price": entry_p,
                    "entry_score": sp.get("entry_score", 0),
                    "entry_dm": sp.get("entry_dm", 0),
                    "entry_f": sp.get("entry_f", 0),
                    "entry_t": sp.get("entry_t", 0),
                    "entry_s": sp.get("entry_s", 0),
                    "entry_m": sp.get("entry_m", 0),
                    "entry_v": sp.get("entry_v", 0),
                    "entry_q": sp.get("entry_q", 0),
                    "exit_time": _now_bj.strftime("%H:%M"),
                    "exit_price": fut_exit,
                    "exit_reason": reason,
                    "pnl_pts": round(pnl_pts, 1),
                    "hold_minutes": hold_min,
                    "operator_action": sp.get("operator_action", ""),
                    "is_executed": sp.get("is_executed", 0),
                })
                d_s = sp["direction"]
                d_cn = "LONG" if d_s == "LONG" else "SHORT"
                print(f"\n *** EXIT: {sym} {d_cn} → {reason}  "
                      f"PnL={pnl_pts:+.0f}pt ***")

                # 写平仓信号供executor
                # 获取期货盘口价格
                fq = fut_quotes.get(sym)
                exit_bid = exit_ask = exit_last = cur_price
                if fq is not None:
                    try:
                        exit_bid = float(fq.bid_price1) or cur_price
                        exit_ask = float(fq.ask_price1) or cur_price
                        exit_last = float(fq.last_price) or cur_price
                    except Exception:
                        pass
                close_signal = {
                    "timestamp": _now_bj.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sym,
                    "contract": self._tq_symbols.get(sym, ""),
                    "direction": d_s,
                    "action": "CLOSE",
                    "reason": reason,
                    "bid1": exit_bid,
                    "ask1": exit_ask,
                    "last": exit_last,
                    "suggested_lots": self._calc_suggested_lots(entry_p, sym),
                    "limit_price": exit_bid if d_cn == "LONG" else exit_ask,
                    "pnl_pts": round(pnl_pts, 1),
                }
                self._append_signal(close_signal)
                print(f"     → 已写入signal_pending.json(CLOSE)，等待executor确认")

                self._shadow_closed_pnl += pnl_pts
                self._shadow_closed_count += 1
                del self._shadow_positions[sym]
                self.strategy.position_mgr.remove_by_symbol(sym)
                self._save_shadow_state()

        # 持久化shadow状态（每根bar都保存，防crash丢失）
        # 注意：不能只在有持仓时保存——空持仓时也需要保存prompted_bars
        # 否则crash后恢复到旧的有持仓状态，导致幽灵持仓
        self._save_shadow_state()

        # 打印状态面板（传入现货bar和期货行情）
        self._print_status(bar_data, fut_quotes, actions)

        # 写入信号文件（供 order_executor 读取）+ 注册shadow持仓
        # Monitor不prompt，executor负责确认
        # bar_dt用bar_data的实际时间戳（不用_last_bar_time，因为不同品种触发时该值不同步）
        for act in actions:
            sym = act.get("symbol", "")
            b = bar_data.get(sym)
            if b is not None and len(b) > 0 and isinstance(b.index, pd.DatetimeIndex):
                bar_dt = int(b.index[-1].timestamp() * 1e9)
            else:
                bar_dt = self._last_bar_time.get(sym)
            key = (sym, bar_dt)

            if act.get("action") == "OPEN":
                if key in self._prompted_bars:
                    # 撤销strategy.on_bar已创建的position_mgr占位
                    # 否则孤儿占位永远不会清除，阻止后续所有同品种信号
                    self.strategy.position_mgr.remove_by_symbol(sym)
                    continue
                self._prompted_bars.add(key)

                direction = act.get("direction", "")
                bid1 = act.get("bid1", 0)
                ask1 = act.get("ask1", 0)
                last = act.get("last", 0)
                sugg_lots = self._calc_suggested_lots(last, sym)

                # 写信号JSON供executor（只有实盘品种）
                if sym in self.strategy.tradeable:
                    self._write_signal_file(act, current_time_utc)

                # 面板打印信号（不等确认）
                d_cn = "LONG" if direction == "LONG" else "SHORT"
                if d_cn == "LONG":
                    limit_s = f"排队{ask1:.1f}/吃{ask1 + 0.2:.1f}" if ask1 > 0 else ""
                else:
                    limit_s = f"排队{bid1:.1f}/吃{bid1 - 0.2:.1f}" if bid1 > 0 else ""
                print(f"\n *** SIGNAL: {sym} {d_cn} score={act.get('score', 0)} ***")
                print(f"     盘口: {limit_s}  建议{sugg_lots}手")
                print(f"     → 已写入signal_pending.json，等待executor确认")

                # 记录信号决策（fail-safe）
                try:
                    self.recorder.record_decision(
                        current_time_utc, sym, act.get("score", 0),
                        direction, "SIGNAL")
                except Exception:
                    pass

                # 注册shadow持仓（所有信号自动进入）
                # entry_price用现货close（与exit判断同源，确保backtest/monitor信号一致）
                # 期货bid/ask只用于executor下单和PnL计算
                b = bar_data.get(sym)
                spot_entry = float(b.iloc[-1]["close"]) if (b is not None and len(b) > 0) else 0
                entry_price = spot_entry if spot_entry > 0 else (ask1 or last or 0)
                sc = self._get_routed_score(sym) or {}
                self._shadow_positions[sym] = {
                    "direction": direction,
                    "entry_time_utc": _now_utc.strftime("%H:%M"),
                    "entry_time_bj": _now_bj.strftime("%H:%M"),
                    "entry_price": float(entry_price),
                    "highest_since": float(entry_price),
                    "lowest_since": float(entry_price),
                    "volume": 1,
                    "bars_below_mid": 0,
                    "entry_score": act.get("score", 0),
                    "entry_dm": sc.get("daily_mult", 0),
                    "entry_f": sc.get("intraday_filter", 0),
                    "entry_t": sc.get("time_weight", 0),
                    "entry_s": sc.get("sentiment_mult", 0),
                    "entry_m": sc.get("s_momentum", 0),
                    "entry_v": sc.get("s_volatility", 0),
                    "entry_q": sc.get("s_volume", 0),
                    "operator_action": "SIGNAL",
                    "is_executed": 0,
                    "fut_symbol": self._tq_symbols.get(sym, ""),
                }
                self._save_shadow_state()

        # 延迟写入信号日志（信号发出之后，不阻塞信号流程）
        # 研究指标在这里计算，即使耗时也不影响信号时效
        for sym, sig, action_taken, sc2, sc3, ver, detail in _signal_log_data:
            try:
                ri = _calc_research_indicators(bar_data, sym)
                detail.update(ri)
            except Exception:
                pass
            self.recorder.record_signal(
                current_time_utc, sym, sig, action_taken,
                v2_score=sc2.get("total", 0),
                v2_direction=sc2.get("direction", ""),
                v3_score=sc3.get("total", 0),
                v3_direction=sc3.get("direction", ""),
                v3_style=sc3.get("style", ""),
                signal_version=ver,
                score_detail=detail,
            )

    def _get_latest_signal(
        self,
        symbol: str,
        bar_data: Dict[str, pd.DataFrame],
        bar_15m_data: Dict[str, pd.DataFrame],
        quote_dict: Dict[str, Dict],
    ) -> Optional[IntradaySignal]:
        """独立调用信号生成器获取评分（用于日志记录）。"""
        if symbol not in bar_data or len(bar_data[symbol]) < 2:
            return None
        b15 = bar_15m_data.get(symbol)
        daily = self._daily_data.get(symbol)
        qd = quote_dict.get(symbol)
        _vp = self._vol_profiles.get(symbol)
        return self.strategy.signal_gen.update(
            symbol, bar_data[symbol], b15, daily, qd, vol_profile=_vp)

    @staticmethod
    def _extract_today_bars(bar_5m: pd.DataFrame) -> pd.DataFrame:
        """提取当天的bar（通过检测>30分钟的gap识别日间分界）。"""
        if len(bar_5m) < 2:
            return bar_5m
        idx = bar_5m.index
        diffs = idx.to_series().diff()
        gaps = diffs[diffs > pd.Timedelta(minutes=30)]
        if len(gaps) > 0:
            pos = bar_5m.index.get_loc(gaps.index[-1])
            return bar_5m.iloc[pos:]
        return bar_5m

    @staticmethod
    def _calc_display_data(
        bar_5m: pd.DataFrame, bar_15m: pd.DataFrame | None,
    ) -> Dict:
        """计算面板辅助显示数据（VWAP, 趋势, 开盘区间, BOLL）。"""
        result: Dict = {}

        # 提取当日 bars
        if isinstance(bar_5m.index, pd.DatetimeIndex):
            last_date = bar_5m.index[-1].date()
            today = bar_5m[bar_5m.index.date == last_date]
        else:
            today = bar_5m

        # VWAP
        if len(today) >= 3:
            closes = today["close"].values.astype(float)
            volumes = today["volume"].values.astype(float)
            cum_vol = np.cumsum(volumes)
            cum_pv = np.cumsum(closes * volumes)
            if cum_vol[-1] > 0:
                result["vwap"] = cum_pv[-1] / cum_vol[-1]
                result["vwap_dev"] = (closes[-1] - result["vwap"]) / result["vwap"]

        # 开盘区间（前6根5分钟K线）
        if len(today) > 6:
            opening = today.iloc[:6]
            result["or_high"] = float(opening["high"].max())
            result["or_low"] = float(opening["low"].min())

        # 趋势方向（SMA10 vs SMA30）
        for label, df, fast, slow in [
            ("trend_5m", bar_5m, 10, 30),
            ("trend_15m", bar_15m, 10, 30),
        ]:
            if df is None or len(df) < slow:
                result[label] = ""
                continue
            c = df["close"].astype(float)
            f_val = float(c.rolling(fast).mean().iloc[-1])
            s_val = float(c.rolling(slow).mean().iloc[-1])
            cur = float(c.iloc[-1])
            if pd.isna(f_val) or pd.isna(s_val):
                result[label] = ""
            elif f_val > s_val and cur > f_val:
                result[label] = "LONG"
            elif f_val < s_val and cur < f_val:
                result[label] = "SHORT"
            else:
                result[label] = ""

        # BOLL 位置（5m）
        if len(bar_5m) >= 20:
            c = bar_5m["close"].astype(float)
            m = float(c.rolling(20).mean().iloc[-1])
            s = float(c.rolling(20).std().iloc[-1])
            cur = float(c.iloc[-1])
            if not pd.isna(m) and not pd.isna(s) and s > 0:
                if cur > m + 2 * s:
                    result["boll_5m"] = "上轨上"
                elif cur > m:
                    result["boll_5m"] = "中轨上"
                elif cur > m - 2 * s:
                    result["boll_5m"] = "中轨下"
                else:
                    result["boll_5m"] = "下轨下"

        return result

    def _get_routed_score(self, sym: str) -> Optional[Dict]:
        """返回品种路由后的信号评分。"""
        ver = SIGNAL_ROUTING.get(sym, "v2")
        if ver == "v3":
            return self._latest_scores_v3.get(sym)
        return self._latest_scores_v2.get(sym)

    def _print_status(
        self,
        bar_data: Dict[str, pd.DataFrame],
        quotes: Dict,
        actions: list,
    ) -> None:
        """打印状态面板（全ASCII表头 + 固定列宽 + 竖线分隔）。"""
        now = datetime.now()
        bj_time = now.strftime("%Y-%m-%d %H:%M:%S")

        # 纯 ASCII 表头 (SPOT=现货信号基准, FUT=期货下单参考)
        HDR = (" SYM  |    SPOT  |     FUT  | DEV%   | Z-SCORE | RSI "
               "|  5m  | 15m  |  BOLL  | RAW>FLT | REASON")
        SEP = ("------+----------+----------+--------+---------+-----"
               "+------+------+--------+---------+--------------")
        W = len(HDR)

        vol_tag = "HiVol" if self._is_high_vol else "Normal"
        print(f"\n{'=' * W}")
        print(f" Intraday Monitor | {bj_time} | {vol_tag}")
        print(f"{'=' * W}")
        print(HDR)
        print(SEP)

        def _trend(d: str) -> str:
            if d == "LONG":
                return " ^L "
            elif d == "SHORT":
                return " vS "
            return " -- "

        def _dir_ch(d: str) -> str:
            if d == "LONG":
                return "^"
            elif d == "SHORT":
                return "v"
            return " "

        for sym in self.symbols:
            # Spot price (from spot index K-lines, used for signals)
            spot_price = 0.0
            if sym in bar_data and len(bar_data[sym]) > 0:
                spot_price = float(bar_data[sym].iloc[-1]["close"])
            # Futures price (from TQ quote, for order reference)
            fut_price = 0.0
            q = quotes.get(sym)
            if q is not None:
                try:
                    fut_price = float(q.last_price) if hasattr(q, "last_price") else 0
                except Exception:
                    pass
            price = spot_price  # signals use spot

            sc = self._get_routed_score(sym)
            ver = SIGNAL_ROUTING.get(sym, "v2")
            dd = self._display_data.get(sym, {})

            # Z-Score
            zp = self._zscore_params.get(sym)
            if zp and price > 0 and zp["std20"] > 0:
                z_val = (price - zp["ema20"]) / zp["std20"]
                z_str = f"{z_val:+5.2f}"
                if z_val <= -2.0 or z_val >= 2.0:
                    z_str += "!"
                else:
                    z_str += " "
            else:
                z_str = "  --  "

            if sc:
                dev_pct = dd.get("vwap_dev", 0) * 100
                dev_s = f"{dev_pct:+5.2f}%"
                t5 = _trend(dd.get("trend_5m", ""))
                t15 = _trend(dd.get("trend_15m", ""))
                boll_map = {"上轨上": "UP+", "中轨上": "MID+", "中轨下": "MID-",
                            "下轨下": "DN-"}
                boll_s = boll_map.get(dd.get("boll_5m", "--"), dd.get("boll_5m", "--")[:4])

                total = sc["total"]
                direction = sc["direction"]
                pre_z = sc.get("pre_z_total", total)
                z_filt = sc.get("z_filter", "")
                rsi_val = sc.get("rsi", 50)
                rsi_note = sc.get("rsi_note", "")

                # RSI display
                rsi_s = f"{rsi_val:4.0f}"

                # Score: RAW>FLT column
                d_ch = _dir_ch(direction)
                if z_filt and pre_z != total:
                    score_s = f"{d_ch}{pre_z:2d}>{total:<2d}"
                else:
                    score_s = f"  {d_ch}{total:3d} "

                # Reason column
                reason = z_filt or rsi_note or ""

                print(f" {sym:<4s} | {spot_price:8.1f} | {fut_price:8.1f}"
                      f" | {dev_s:>6s}"
                      f" | {z_str:>7s} | {rsi_s:>3s}"
                      f" | {t5:>4s} | {t15:>4s}"
                      f" | {boll_s:>6s} | {score_s:>7s} | {reason}")

                # 维度明细行
                parts = [f"M{sc['s_momentum']}", f"V{sc['s_volatility']}",
                         f"Q{sc['s_volume']}"]
                s_rev = sc.get("s_reversal", 0)
                if s_rev > 0:
                    parts.append(f"R{s_rev}")
                s_brk = sc.get("s_breakout", 0)
                if s_brk > 0:
                    parts.append(f"B{s_brk}")
                parts.append(f"dm{sc['daily_mult']:.1f}")
                idf = sc.get("intraday_filter", 1.0)
                if idf != 1.0:
                    parts.append(f"f{idf:.2f}")
                parts.append(f"t{sc['time_weight']:.1f}")
                sent_m = sc.get("sentiment_mult", 1.0)
                if sent_m != 1.0:
                    parts.append(f"s{sent_m:.2f}")
                print(f"       {ver}[{'/'.join(parts)}]")
            else:
                print(f" {sym:<4s} | {spot_price:8.1f} | {fut_price:8.1f}"
                      f" | {'--':>6s}"
                      f" | {z_str:>7s} | {'--':>3s}"
                      f" | {'--':>4s} | {'--':>4s}"
                      f" | {'--':>6s} | {'--':>7s} | --")

        print(SEP)

        # 持仓
        nets = self.strategy.position_mgr.get_net_positions()
        if nets:
            pos_parts = []
            for s, v in nets.items():
                d = "L" if v > 0 else "S"
                pos_parts.append(f"{s}{d}{abs(v)}")
            print(f" POS: {' | '.join(pos_parts)}")
        else:
            print(f" POS: none")

        # Shadow P&L: 已平仓累计 + 活跃持仓浮盈（全部用期货价，点数）
        floating_pts = 0.0
        shadow_parts = []
        for s_sym, sp in self._shadow_positions.items():
            b5 = bar_data.get(s_sym)
            if b5 is not None and len(b5) > 0:
                cur = 0.0
                fq = quotes.get(s_sym)
                if fq is not None:
                    try:
                        cur = float(fq.last_price)
                    except Exception:
                        pass
                if cur <= 0:
                    cur = float(b5.iloc[-1]["close"])
                pnl = (cur - sp["entry_price"]) if sp["direction"] == "LONG" \
                    else (sp["entry_price"] - cur)
                floating_pts += pnl
                d = "L" if sp["direction"] == "LONG" else "S"
                exec_flag = "*" if sp.get("is_executed") else ""
                shadow_parts.append(
                    f"{s_sym}{exec_flag} {d}@{sp['entry_price']:.0f}"
                    f" {'+' if pnl >= 0 else ''}{pnl:.0f}pt")
        total_pts = self._shadow_closed_pnl + floating_pts
        total_trades = self._shadow_closed_count + len(self._shadow_positions)
        print(f" P&L: {total_pts:+.0f}pt"
              f" (已平{self._shadow_closed_pnl:+.0f} 浮盈{floating_pts:+.0f})"
              f"  Trades: {total_trades}")
        if shadow_parts:
            print(f" SHADOW: {' | '.join(shadow_parts)}")

        print(f"{'=' * W}")


def main():
    monitor = IntradayMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
