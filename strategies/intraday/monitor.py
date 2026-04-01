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

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO signal_log "
            "(datetime, symbol, direction, score, score_breakout, score_vwap, "
            "score_multiframe, score_volume, score_daily, score_orderbook, "
            "action_taken, reason, score_v2, direction_v2, "
            "score_v3, direction_v3, style_v3, signal_version, "
            "s_momentum, s_volatility, s_quality, intraday_filter, "
            "time_mult, sentiment_mult, z_score, rsi, "
            "raw_score, filtered_score, filter_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
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
        # 影子交易簿：记录所有信号的完整生命周期，key=symbol
        self._shadow_positions: Dict[str, Dict] = {}
        # 项目 tmp 目录（信号文件、持仓文件等）
        from config.config_loader import ConfigLoader
        self._tmp_dir: str = ConfigLoader().get_tmp_dir()
        self._signal_file: str = os.path.join(self._tmp_dir, "signal_pending.json")

    def _load_daily_data(self) -> None:
        """从数据库加载日线数据 + 解析近月合约 + 用现货指数算Z-Score。"""
        # 品种→现货指数映射（无换月跳变，Z-Score更准确）
        _SPOT_INDEX = {"IM": "000852.SH", "IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH"}

        try:
            from data.storage.db_manager import DBManager
            from config.config_loader import ConfigLoader
            db = DBManager(ConfigLoader().get_db_path())

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
        signal = {
            "timestamp": dt_str,
            "symbol": action.get("symbol", ""),
            "direction": direction,
            "action": action.get("action", "OPEN"),
            "score": action.get("score", 0),
            "bid1": bid1,
            "ask1": ask1,
            "last": last,
            "suggested_lots": self._calc_suggested_lots(last, action.get("symbol", "IM")),
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
            from data.storage.db_manager import DBManager
            from config.config_loader import ConfigLoader
            db = DBManager(ConfigLoader().get_db_path())
            df = db.query_df(
                "SELECT balance FROM account_snapshots "
                "ORDER BY trade_date DESC LIMIT 1"
            )
            if df is not None and not df.empty:
                self._account_equity = float(df.iloc[0]["balance"])
                print(f"  账户权益: {self._account_equity:,.0f}")
        except Exception as e:
            print(f"  [警告] 账户权益加载失败: {e}")

    def _load_sentiment(self) -> None:
        """从 vol_monitor_snapshots 或 daily_model_output 加载情绪数据。"""
        try:
            from data.storage.db_manager import DBManager
            from config.config_loader import ConfigLoader
            db = DBManager(ConfigLoader().get_db_path())

            # 优先用 vol_monitor_snapshots（盘中最新）
            snap = db.query_df(
                "SELECT atm_iv, vrp, rr_25d, term_structure_shape "
                "FROM vol_monitor_snapshots ORDER BY datetime DESC LIMIT 2"
            )
            if snap is not None and len(snap) >= 1:
                cur = snap.iloc[0]
                prev = snap.iloc[1] if len(snap) >= 2 else cur
                self._sentiment = SentimentData(
                    atm_iv=float(cur.get("atm_iv") or 0),
                    atm_iv_prev=float(prev.get("atm_iv") or 0),
                    rr_25d=float(cur.get("rr_25d") or 0),
                    rr_25d_prev=float(prev.get("rr_25d") or 0),
                    vrp=float(cur.get("vrp") or 0),
                    term_structure=str(cur.get("term_structure_shape") or ""),
                )
                print(f"  情绪数据: IV={self._sentiment.atm_iv*100:.1f}%"
                      f"  RR={self._sentiment.rr_25d*100:+.1f}pp"
                      f"  VRP={self._sentiment.vrp*100:+.1f}%")
                return

            # Fallback: daily_model_output（昨日+前日）
            dmo = db.query_df(
                "SELECT atm_iv, atm_iv_market, vrp, rr_25d, term_structure_shape "
                "FROM daily_model_output WHERE underlying='IM' "
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
                print(f"  情绪数据(EOD): IV={iv_cur*100:.1f}%"
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

        # 加载日线数据 + 情绪数据 + 账户权益
        self._load_daily_data()
        self._load_sentiment()
        self._load_account_equity()

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
                bar_updated = False
                for sym in self.symbols:
                    sk = spot_klines_5m.get(sym)
                    if sk is not None and api.is_changing(sk):
                        bar_updated = True
                        self._on_new_bar(
                            sym, spot_klines_5m, spot_klines_15m,
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
        updated_sym: str,
        spot_klines_5m: Dict,
        spot_klines_15m: Dict,
        fut_quotes: Dict,
        fut_klines_5m: Dict = None,
    ) -> None:
        """新K线到达时的处理——用现货K线驱动信号计算。"""
        k5 = spot_klines_5m.get(updated_sym)
        if k5 is None or len(k5) < 2:
            return
        # 去重：用倒数第二根bar的datetime
        completed_dt = int(k5.iloc[-2]["datetime"])
        prev_dt = self._last_bar_time.get(updated_sym)
        if prev_dt == completed_dt:
            return
        self._last_bar_time[updated_sym] = completed_dt

        # Warmup: skip first bar
        if not getattr(self, "_warmup_done", True):
            self._bars_since_start = getattr(self, "_bars_since_start", 0) + 1
            if self._bars_since_start <= 1:
                try:
                    ts = pd.Timestamp(completed_dt, unit="ns")
                    if ts.minute % 5 != 0:
                        print(f"  [WARMUP] Skipping misaligned bar {ts}")
                        return
                except Exception:
                    pass
                print(f"  [WARMUP] First aligned bar received, starting signals")
                self._warmup_done = True

        now = datetime.now()
        current_time_utc = now.strftime("%Y-%m-%d %H:%M:%S")

        # 构建现货bar数据（用于信号计算）
        bar_data: Dict[str, pd.DataFrame] = {}
        bar_15m_data: Dict[str, pd.DataFrame] = {}

        for sym in self.symbols:
            k5 = spot_klines_5m.get(sym)
            if k5 is not None and len(k5) > 0:
                df = k5[["open", "high", "low", "close", "volume"]].copy()
                df.index = pd.to_datetime(k5["datetime"], unit="ns")
                if len(df) > 0:
                    bar_data[sym] = df

            k15 = spot_klines_15m.get(sym)
            if k15 is not None and len(k15) > 0:
                df = k15[["open", "high", "low", "close", "volume"]].copy()
                df.index = pd.to_datetime(k15["datetime"], unit="ns")
                if len(df) > 0:
                    bar_15m_data[sym] = df

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
                # 记录盘口快照
                self.recorder.record_orderbook(current_time_utc, sym, qd)

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

            # 路由版本评分（含情绪乘数 + Z-Score过滤 + 波动率区间）
            ver = SIGNAL_ROUTING.get(sym, "v2")
            hv = self._is_high_vol
            if ver == "v3":
                sc3 = self.signal_v3.score_all(
                    sym, b5, b15, daily, qd, self._sentiment,
                    zscore=z_val, is_high_vol=hv)
                if sc3:
                    self._latest_scores_v3[sym] = sc3
            else:
                sc2 = self.signal_v2.score_all(
                    sym, b5, b15, daily, qd, self._sentiment,
                    zscore=z_val, is_high_vol=hv)
                if sc2:
                    self._latest_scores_v2[sym] = sc2

            # 面板辅助显示数据（VWAP, 趋势, 开盘区间, BOLL）
            self._display_data[sym] = self._calc_display_data(b5, b15)

        # 运行策略（传入zscore/is_high_vol/sentiment参数，和面板评分一致）
        actions = self.strategy.on_bar(
            bar_data, bar_15m_data, self._daily_data or None, current_time_utc,
            quote_data=quote_dict,
            zscore_params=self._zscore_params,
            is_high_vol=self._is_high_vol,
            sentiment=self._sentiment,
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

        # 记录信号日志（每个品种都记，无信号也记 score=0，含维度明细）
        action_syms = {a["symbol"] for a in actions if a.get("action") == "OPEN"}
        for sym in self.symbols:
            sig = self._get_latest_signal(sym, bar_data, bar_15m_data, quote_dict)
            action_taken = "OPEN" if sym in action_syms else ("SKIP" if sig else "NONE")
            sc2 = self._latest_scores_v2.get(sym, {})
            sc3 = self._latest_scores_v3.get(sym, {})
            ver = SIGNAL_ROUTING.get(sym, "v2")
            # 构建维度明细（从路由版本的score_all结果）
            routed_sc = sc3 if ver == "v3" else sc2
            # 补充zscore（score_all的返回值中没有存zscore的值，从zscore_params算）
            zp = self._zscore_params.get(sym)
            cur_z = None
            if zp and sym in bar_data and len(bar_data[sym]) > 0:
                cp = float(bar_data[sym].iloc[-1]["close"])
                if cp > 0 and zp["std20"] > 0:
                    cur_z = (cp - zp["ema20"]) / zp["std20"]
            detail = {**routed_sc, "zscore": cur_z} if routed_sc else {"zscore": cur_z}
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
        utc_hm = datetime.utcnow().strftime("%H:%M")
        trade_date = datetime.now().strftime("%Y%m%d")
        for sym in list(self._shadow_positions.keys()):
            sp = self._shadow_positions[sym]
            b5 = bar_data.get(sym)
            if b5 is None or len(b5) < 2:
                continue
            cur_price = float(b5.iloc[-1]["close"])
            high = float(b5.iloc[-1]["high"])
            low = float(b5.iloc[-1]["low"])
            b15 = bar_15m_data.get(sym)
            b15_arg = b15 if (b15 is not None and len(b15) > 0) else None
            # Update extremes
            if sp["direction"] == "LONG":
                sp["highest_since"] = max(sp.get("highest_since", sp["entry_price"]), high)
            else:
                sp["lowest_since"] = min(sp.get("lowest_since", sp["entry_price"]), low)
            exit_info = check_exit(
                sp, cur_price, b5, b15_arg,
                utc_hm, reverse_signal_score=0, is_high_vol=self._is_high_vol,
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
                    "exit_time": datetime.now().strftime("%H:%M"),
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
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sym,
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

                del self._shadow_positions[sym]

        # 打印状态面板（传入现货bar和期货行情）
        self._print_status(bar_data, fut_quotes, actions)

        # 写入信号文件（供 order_executor 读取）+ 注册shadow持仓
        # Monitor不prompt，executor负责确认
        for act in actions:
            sym = act.get("symbol", "")
            bar_dt = self._last_bar_time.get(sym)
            key = (sym, bar_dt)

            if act.get("action") == "OPEN":
                if key in self._prompted_bars:
                    continue
                self._prompted_bars.add(key)

                direction = act.get("direction", "")
                bid1 = act.get("bid1", 0)
                ask1 = act.get("ask1", 0)
                last = act.get("last", 0)
                sugg_lots = self._calc_suggested_lots(last, sym)

                # 写信号JSON供 executor
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

                # 记录信号决策
                self.recorder.record_decision(
                    current_time_utc, sym, act.get("score", 0),
                    direction, "SIGNAL")

                # 注册shadow持仓（所有信号自动进入）
                entry_price = (ask1 or last or 0) if direction == "LONG" \
                    else (bid1 or last or 0)
                sc = self._get_routed_score(sym) or {}
                self._shadow_positions[sym] = {
                    "direction": direction,
                    "entry_time_utc": datetime.utcnow().strftime("%H:%M"),
                    "entry_time_bj": datetime.now().strftime("%H:%M"),
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
        return self.strategy.signal_gen.update(symbol, bar_data[symbol], b15, daily, qd)

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

        summary = self.strategy.position_mgr.get_exposure_summary()
        print(f" P&L: {summary['daily_pnl']:+,.0f}  Trades: {summary['daily_trades']}")

        # 影子持仓状态
        if self._shadow_positions:
            shadow_parts = []
            for s_sym, sp in self._shadow_positions.items():
                b5 = bar_data.get(s_sym)
                if b5 is not None and len(b5) > 0:
                    # 用期货last_price（与entry_price基准一致），fallback现货close
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
                    d = "L" if sp["direction"] == "LONG" else "S"
                    exec_flag = "*" if sp.get("is_executed") else ""
                    shadow_parts.append(
                        f"{s_sym}{exec_flag} {d}@{sp['entry_price']:.0f}"
                        f" {'+' if pnl >= 0 else ''}{pnl:.0f}pt")
            if shadow_parts:
                print(f" SHADOW: {' | '.join(shadow_parts)}")

        print(f"{'=' * W}")


def main():
    monitor = IntradayMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
