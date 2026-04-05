"""
daily_record.py
---------------
每日数据记录主脚本，支持四个子命令：

  snapshot  盘中运行 — 抓取指定合约行情快照存入 tq_snapshots
  eod       收盘后运行 — 完整记录：账户/持仓/成交/Tushare 增量/模型输出
              内含数据时效性检查：若 Tushare 数据未更新则等待重试（最多3次×5分钟），
              超时后 TQ 快照正常保存，模型输出跳过。
  model     补跑模型 — 跳过 TQ/Tushare，仅运行模型计算和写入 daily_model_output
              用于 eod 因数据延迟跳过模型输出后的手动补跑
  note      写交易笔记 — 记录市场观察、交易理由、复盘总结到 daily_notes

用法示例：
    python scripts/daily_record.py snapshot
    python scripts/daily_record.py snapshot --symbols CFFEX.MO2606-C-7800 CFFEX.IM2606
    python scripts/daily_record.py eod
    python scripts/daily_record.py eod --date 20260316 --no-tq
    python scripts/daily_record.py model --date 20260317
    python scripts/daily_record.py note --date 20260317
    python scripts/daily_record.py note --date 20260317 \\
        --market "CSI1000 跌1.2%，波动率大幅上升" \\
        --rationale "IV 溢价明显，维持做空波动率结构" \\
        --deviations "无偏离" \\
        --lessons "注意 MO2604 临近到期 Gamma 加速"

TQ 凭证通过环境变量配置：
    TQ_ACCOUNT          天勤平台账户（邮箱）
    TQ_PASSWORD         天勤平台密码
    TQ_BROKER_ID        期货公司代码，如 H宏源期货
    TQ_ACCOUNT_ID       资金账号
    TQ_BROKER_PASSWORD  账户密码
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置（允许从任意目录运行）
# ---------------------------------------------------------------------------

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 加载项目根目录的 .env 文件（需在任何 os.getenv 调用之前）
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass  # python-dotenv 未安装时静默跳过

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DB_PATH: str     = os.path.join(ROOT, "data", "storage", "trading.db")
RISK_FREE: float = 0.02
CONTRACT_MULT: int = 100

# 默认监控合约（snapshot 子命令无 --symbols 时使用）
# TQ 格式：EXCHANGE.SYMBOL
# 注意：CFFEX 股指期货只有当月/下月/当季/隔季四个合约，需定期更新
DEFAULT_SNAPSHOT_SYMBOLS: list[str] = [
    "CFFEX.IM2604",
    "CFFEX.IM2606",
    "CFFEX.MO2604-P-7200",
    "CFFEX.MO2605-P-7200",
    "CFFEX.MO2606-C-7800",
    "CFFEX.MO2606-C-8400",
]

# VRP 信号阈值
VRP_SELL_THRESHOLD  =  0.02   # VRP > 2% → 做空波动率
VRP_BUY_THRESHOLD   = -0.02   # VRP < -2% → 做多波动率


# ---------------------------------------------------------------------------
# TQ 凭证加载
# ---------------------------------------------------------------------------

def _tq_credentials() -> dict[str, str]:
    """从环境变量读取 TQ 凭证。返回 dict，缺字段时对应值为空字符串。"""
    return {
        "auth_account":    os.getenv("TQ_ACCOUNT", ""),
        "auth_password":   os.getenv("TQ_PASSWORD", ""),
        "broker_id":       os.getenv("TQ_BROKER", ""),   # .env 用 TQ_BROKER，非 TQ_BROKER_ID
        "account_id":      os.getenv("TQ_ACCOUNT_ID", ""),
        "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
    }


def _has_tq_auth() -> bool:
    """是否配置了最低限度的 TQ 行情权限（账户+密码）。"""
    creds = _tq_credentials()
    return bool(creds["auth_account"] and creds["auth_password"])


def _has_tq_broker() -> bool:
    """是否配置了实盘账户（账户/持仓/成交需要）。"""
    creds = _tq_credentials()
    return bool(
        creds["auth_account"] and creds["auth_password"]
        and creds["broker_id"] and creds["account_id"] and creds["broker_password"]
    )


# ---------------------------------------------------------------------------
# DB 连接
# ---------------------------------------------------------------------------

def _open_db():
    """打开 trading.db，同时确保新表已创建。"""
    from data.storage.db_manager import DBManager
    db = DBManager(DB_PATH)
    db.initialize_tables()
    return db


# ---------------------------------------------------------------------------
# 数据时效性检查
# ---------------------------------------------------------------------------

@dataclass
class DataFreshness:
    futures_date: str   # MAX(trade_date) in futures_daily for IM.CFX
    options_date: str   # MAX(trade_date) in options_daily for MO*
    expected_date: str  # target trade_date

    @property
    def futures_ok(self) -> bool:
        return self.futures_date == self.expected_date

    @property
    def options_ok(self) -> bool:
        return self.options_date == self.expected_date

    @property
    def is_fresh(self) -> bool:
        return self.futures_ok and self.options_ok


def _check_data_freshness(db, expected_date: str) -> DataFreshness:
    """
    查询 futures_daily（IM.CFX）和 options_daily（MO*）的最新 trade_date，
    与 expected_date 对比，返回 DataFreshness。
    """
    futures_date = ""
    options_date = ""

    try:
        row = db.query_df(
            "SELECT MAX(trade_date) AS dt FROM futures_daily WHERE ts_code='IM.CFX'"
        )
        if row is not None and not row.empty and row["dt"].iloc[0]:
            futures_date = str(row["dt"].iloc[0]).replace("-", "")
    except Exception as e:
        logger.warning("查询 futures_daily 最新日期失败: %s", e)

    try:
        row = db.query_df(
            "SELECT MAX(trade_date) AS dt FROM options_daily WHERE ts_code LIKE 'MO%'"
        )
        if row is not None and not row.empty and row["dt"].iloc[0]:
            options_date = str(row["dt"].iloc[0]).replace("-", "")
    except Exception as e:
        logger.warning("查询 options_daily 最新日期失败: %s", e)

    return DataFreshness(
        futures_date=futures_date,
        options_date=options_date,
        expected_date=expected_date,
    )


def _format_freshness_status(
    freshness: DataFreshness,
    model_skipped: bool = False,
    model_ok: bool = False,
) -> str:
    """
    生成数据状态摘要行，例如：
      数据状态: 期货日线 ✓ (20260317) | 期权日线 ✗ (20260316) | 模型输出 ⏭ 已跳过
    """
    def _fmt(ok: bool, dt: str) -> str:
        mark = "✓" if ok else "✗"
        return f"{mark} ({dt})" if dt else f"{mark} (无数据)"

    fut = _fmt(freshness.futures_ok, freshness.futures_date)
    opt = _fmt(freshness.options_ok, freshness.options_date)

    if model_skipped:
        mdl = "⏭ 已跳过"
    elif model_ok:
        mdl = "✓"
    else:
        mdl = "✗ 失败"

    return f"数据状态: 期货日线 {fut} | 期权日线 {opt} | 模型输出 {mdl}"


# ---------------------------------------------------------------------------
# ── SNAPSHOT 子命令 ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def save_tq_snapshots(
    tq_api,
    symbols: list[str],
    snapshot_time: str,
    db,
) -> int:
    """
    从 TQ API 拉取 symbols 的行情快照并批量写入 tq_snapshots 表。

    关键：先对所有合约调用 get_quote()（注册订阅），然后一次 wait_update()
    等待所有数据到达，最后逐个读取。避免逐个 wait_update 导致后续合约超时。

    Returns
    -------
    int
        成功写入的条数
    """
    FIELDS = (
        "last_price", "bid_price1", "ask_price1",
        "bid_volume1", "ask_volume1", "volume",
        "open_interest", "highest", "lowest", "open",
    )

    # Step 1: 批量订阅所有合约（get_quote 只注册订阅，不阻塞）
    quotes: dict = {}
    for symbol in symbols:
        try:
            quotes[symbol] = tq_api.get_quote(symbol)
        except Exception as e:
            logger.warning("订阅失败 %s: %s", symbol, e)

    if not quotes:
        return 0

    # Step 2: 一次 wait_update 等待所有合约数据到达
    try:
        tq_api.wait_update(deadline=time.time() + 30)
    except Exception as e:
        logger.warning("wait_update 超时: %s", e)

    # Step 3: 读取所有 quote 数据，跳过无效合约（last_price 为 NaN 或 0）
    records = []
    for symbol, quote in quotes.items():
        try:
            last_px = getattr(quote, "last_price", None)
            if last_px is None or (isinstance(last_px, float) and (np.isnan(last_px) or last_px <= 0)):
                logger.warning("合约 %s 无有效行情（可能不存在），跳过", symbol)
                continue
            row = {
                "snapshot_time": snapshot_time,
                "symbol":        symbol,
            }
            for f in FIELDS:
                v = getattr(quote, f, None)
                row[f] = float(v) if v is not None else None
            records.append(row)
        except Exception as e:
            logger.warning("读取快照失败 %s: %s", symbol, e)

    if not records:
        return 0

    df = pd.DataFrame(records)
    try:
        db.upsert_dataframe("tq_snapshots", df)
        return len(records)
    except Exception as e:
        logger.error("写入 tq_snapshots 失败: %s", e)
        return 0


def cmd_snapshot(args: argparse.Namespace) -> None:
    """盘中快照子命令：抓取行情快照存入 tq_snapshots。"""
    symbols = args.symbols or DEFAULT_SNAPSHOT_SYMBOLS
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[snapshot] {now_str}  抓取 {len(symbols)} 个合约行情快照...")

    if not _has_tq_auth():
        print("  [警告] 未配置 TQ_ACCOUNT/TQ_PASSWORD 环境变量，无法连接天勤，跳过")
        return

    try:
        from data.sources.tq_client import TqClient
        creds = _tq_credentials()
        client = TqClient(**creds)
        client.connect()
        db = _open_db()
        n = save_tq_snapshots(client._api, symbols, now_str, db)
        client.disconnect()
        print(f"  已写入 {n} 条快照  →  tq_snapshots")
    except Exception as e:
        logger.error("snapshot 子命令失败: %s", e)
        raise


# ---------------------------------------------------------------------------
# ── EOD 子命令 ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _eod_account_snapshot(account_ref, trade_date: str, db) -> dict | None:
    """
    从已填充的 TQ account 引用中读取账户快照并写入 account_snapshots。
    account_ref 为 api.get_account() 返回的对象，调用前需已完成 wait_update。

    首次写入有效；若当日已有记录则返回数据库中的版本（避免结算后 TQ 数据清零覆盖）。
    """
    try:
        # 先查数据库，若已有当日记录则直接返回
        existing = db.query_df(
            "SELECT * FROM account_snapshots WHERE trade_date = ?", (trade_date,)
        )
        if existing is not None and len(existing) > 0:
            row = existing.iloc[0].to_dict()
            logger.info("账户快照已存在（%s），使用数据库版本", row.get("snapshot_time", ""))
            return row

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        balance = float(account_ref.balance or 0)
        margin  = float(account_ref.margin  or 0)

        row = {
            "trade_date":   trade_date,
            "snapshot_time": now_str,
            "balance":      balance,
            "available":    float(account_ref.available    or 0),
            "margin":       margin,
            "margin_ratio": round(margin / balance, 6) if balance > 0 else 0.0,
            "float_profit": float(account_ref.float_profit  or 0),
            "close_profit": float(account_ref.close_profit  or 0),
            "commission":   float(account_ref.commission    or 0),
            "risk_ratio":   float(getattr(account_ref, "risk_ratio", 0) or 0),
        }
        db.upsert_dataframe("account_snapshots", pd.DataFrame([row]))
        logger.info("账户快照已写入: balance=%.2f  margin=%.2f", balance, margin)
        return row
    except Exception as e:
        logger.warning("账户快照失败（可能为非交易时段）: %s", e)
        return None


def _eod_position_snapshot(position_ref, trade_date: str, db) -> list[dict]:
    """
    从已填充的 TQ position 引用中读取持仓并写入 position_snapshots。
    position_ref 为 api.get_position() 返回的对象，调用前需已完成 wait_update。

    首次写入有效；若当日已有记录则返回数据库中的版本（避免结算后 TQ 数据清零覆盖）。
    """
    try:
        # 先查数据库，若已有当日记录则直接返回
        existing = db.query_df(
            "SELECT * FROM position_snapshots WHERE trade_date = ?", (trade_date,)
        )
        if existing is not None and len(existing) > 0:
            rows = existing.to_dict("records")
            logger.info("持仓快照已存在（%d 条），使用数据库版本", len(rows))
            return rows

        rows = []
        for symbol, pos in position_ref.items():
            for direction, vol_attr, today_attr, price_attr, profit_attr, margin_attr in [
                ("LONG",  "volume_long",  "volume_long_today",
                 "open_price_long",  "float_profit_long",  "margin_long"),
                ("SHORT", "volume_short", "volume_short_today",
                 "open_price_short", "float_profit_short", "margin_short"),
            ]:
                vol = int(getattr(pos, vol_attr, 0) or 0)
                if vol <= 0:
                    continue
                rows.append({
                    "trade_date":     trade_date,
                    "symbol":         symbol,
                    "direction":      direction,
                    "volume":         vol,
                    "volume_today":   int(getattr(pos, today_attr,  0) or 0),
                    "open_price_avg": float(getattr(pos, price_attr,  0) or 0),
                    "last_price":     float(pos.last_price or 0),
                    "float_profit":   float(getattr(pos, profit_attr, 0) or 0),
                    "margin":         float(getattr(pos, margin_attr, 0) or 0),
                })
        if rows:
            db.upsert_dataframe("position_snapshots", pd.DataFrame(rows))
            logger.info("持仓快照已写入: %d 条", len(rows))
        else:
            logger.info("当前无持仓记录")
        return rows
    except Exception as e:
        logger.warning("持仓快照失败: %s", e)
        return []


def _eod_trade_records(trade_ref, trade_date: str, db) -> int:
    """
    从已填充的 TQ trade 引用中读取当日成交流水并写入 trade_records。
    trade_ref 为 api.get_trade() 返回的对象，调用前需已完成 wait_update。

    首次写入有效；若当日已有记录则保留数据库版本（避免结算后 TQ 数据清零覆盖）。
    """
    try:
        # 先查数据库，若已有当日记录则直接返回
        existing = db.query_df(
            "SELECT * FROM trade_records WHERE trade_date = ?", (trade_date,)
        )
        if existing is not None and len(existing) > 0:
            logger.info("成交记录已存在（%d 条），使用数据库版本", len(existing))
            return len(existing)

        rows = []
        for order_id, trade in trade_ref.items():
            # TQ trade 字段参考天勤文档 get_trade()
            trade_time = ""
            if hasattr(trade, "trade_date_time"):
                try:
                    ts = pd.Timestamp(trade.trade_date_time, unit="ns", tz="UTC")
                    ts_sh = ts.tz_convert("Asia/Shanghai")
                    trade_time = ts_sh.strftime("%H:%M:%S")
                except Exception:
                    pass

            rows.append({
                "trade_date":    trade_date,
                "trade_time":    trade_time,
                "symbol":        str(getattr(trade, "instrument_id", "") or ""),
                "direction":     str(getattr(trade, "direction",     "") or "").upper(),
                "offset":        str(getattr(trade, "offset",        "") or "").upper(),
                "volume":        int(getattr(trade,  "volume",       0)  or 0),
                "price":         float(getattr(trade, "price",       0)  or 0),
                "commission":    float(getattr(trade, "commission",  0)  or 0),
                "order_id":      str(order_id),
                "strategy_name": "manual",
                "notes":         "",
            })

        if rows:
            db.upsert_dataframe("trade_records", pd.DataFrame(rows))
            logger.info("成交记录已写入: %d 条", len(rows))
        else:
            logger.info("当日无成交记录")
        return len(rows)
    except Exception as e:
        logger.warning("成交记录获取失败: %s", e)
        return 0


def _eod_archive_minute_bars(api, trade_date: str, db) -> int:
    """
    拉取当天 IF/IH/IM/IC 5分钟K线，追加写入 futures_min 表。
    每天 ~48 根 × 4 品种 = ~192 条，INSERT OR IGNORE 避免重复。
    返回新写入条数。
    """
    symbols = ["IF", "IH", "IM", "IC"]
    total = 0
    try:
        from utils.cffex_calendar import get_main_contract
        for sym in symbols:
            full_sym = get_main_contract(sym, api=api)
            klines = api.get_kline_serial(full_sym, 300, 200)
            api.wait_update(deadline=time.time() + 10)

            if klines is None or len(klines) == 0:
                continue

            df = klines[["open", "high", "low", "close", "volume"]].copy()
            df["datetime"] = pd.to_datetime(klines["datetime"], unit="ns")
            # 只保留当日数据
            td_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
            df = df[df["datetime"].dt.strftime("%Y-%m-%d") == td_str]

            if df.empty:
                continue

            df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df["symbol"] = sym
            df["period"] = 300

            conn = db._conn if hasattr(db, "_conn") else db.get_connection()
            for _, row in df.iterrows():
                conn.execute(
                    "INSERT OR IGNORE INTO futures_min "
                    "(symbol, datetime, period, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["symbol"], row["datetime"], row["period"],
                     row["open"], row["high"], row["low"],
                     row["close"], row["volume"]),
                )
            conn.commit()
            n = len(df)
            total += n
            logger.info("归档 %s 5m K线: %d 根", sym, n)
    except Exception as e:
        logger.warning("期货分钟线归档失败: %s", e)

    # 归档现货指数5分钟K线到 index_min 表
    spot_syms = {
        "SSE.000852": "000852", "SSE.000300": "000300",
        "SSE.000016": "000016", "SSE.000905": "000905",
    }
    try:
        for tq_sym, db_sym in spot_syms.items():
            klines = api.get_kline_serial(tq_sym, 300, 200)
            api.wait_update(deadline=time.time() + 10)
            if klines is None or len(klines) == 0:
                continue
            df = klines[["open", "high", "low", "close", "volume"]].copy()
            df["datetime"] = pd.to_datetime(klines["datetime"], unit="ns")
            td_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
            df = df[df["datetime"].dt.strftime("%Y-%m-%d") == td_str]
            if df.empty:
                continue
            df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df["symbol"] = db_sym
            df["period"] = 300
            conn = db._conn if hasattr(db, "_conn") else db.get_connection()
            for _, row in df.iterrows():
                conn.execute(
                    "INSERT OR IGNORE INTO index_min "
                    "(symbol, datetime, period, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["symbol"], row["datetime"], row["period"],
                     row["open"], row["high"], row["low"],
                     row["close"], row["volume"]),
                )
            conn.commit()
            n = len(df)
            total += n
            logger.info("归档 %s 现货5m K线: %d 根", db_sym, n)
    except Exception as e:
        logger.warning("现货指数分钟线归档失败: %s", e)

    return total


def _eod_update_market_data() -> None:
    """
    触发 Tushare 增量数据下载（期货日线 + 期权日线）。
    调用已有的 download_scripts，子进程输出实时流向终端（不捕获）。
    失败时打印警告，不中断主流程。
    """
    # 期货脚本通常很快（几十个合约只拉今天）；
    # 期权脚本首次下载合约信息可能较慢，给 600 秒。
    scripts = [
        (os.path.join(ROOT, "data", "download_scripts", "download_futures_daily.py"), 300),
        (os.path.join(ROOT, "data", "download_scripts", "download_options_daily.py"), 600),
        (os.path.join(ROOT, "data", "download_scripts", "download_index_daily.py"),   120),
    ]
    for script_path, timeout_s in scripts:
        if not os.path.exists(script_path):
            logger.warning("下载脚本不存在，跳过: %s", script_path)
            continue
        print(f"  → {Path(script_path).name} --update", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, script_path, "--update"],
                # 不捕获 stdout，让进度信息实时显示在终端
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                cwd=ROOT,
            )
            if result.returncode == 0:
                logger.info("数据更新完成: %s", Path(script_path).name)
            else:
                logger.warning(
                    "数据更新返回非零 (%d): %s\n%s",
                    result.returncode, Path(script_path).name,
                    result.stderr[-500:] if result.stderr else "",
                )
        except subprocess.TimeoutExpired:
            logger.warning(
                "数据更新超时（%ds），跳过: %s", timeout_s, Path(script_path).name,
            )
        except Exception as e:
            logger.warning("数据更新失败，跳过: %s  错误: %s", Path(script_path).name, e)


def _eod_model_output(trade_date: str, db) -> dict | None:
    """
    计算并写入当日模型输出到 daily_model_output：
      - GARCH 当日条件波动率 & 5日预测
      - 20日 / 60日历史 RV
      - ATM IV（MO 期权链）
      - VRP = ATM_IV - GARCH_forecast
      - VRP 历史百分位
      - 信号方向
      - 组合 Greeks（基于当日持仓快照）
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        from models.volatility.garch_model import GJRGARCHModel
        from scripts.portfolio_analysis import (
            _parse_mo, _build_expire_map, _get_mo_chain,
            get_futures_prices_by_expiry,
            get_atm_iv,
            get_iv_for_positions,
            calc_implied_forwards_by_expiry,
            calc_portfolio,
        )
    except ImportError as e:
        logger.warning("模型模块导入失败，跳过模型输出: %s", e)
        return None

    # ── 1. IM.CFX 日线，计算 RV & GARCH ────────────────────────────────
    try:
        df = db.query_df(
            "SELECT trade_date, close FROM futures_daily "
            f"WHERE ts_code='IM.CFX' AND trade_date <= '{trade_date}' "
            "ORDER BY trade_date ASC"
        )
        if df is None or len(df) < 62:
            logger.warning("IM.CFX 日线数据不足 62 条，跳过模型输出")
            return None

        # 最终数据时效性守卫：拒绝写入过时数据
        latest_date = str(df["trade_date"].iloc[-1]).replace("-", "")
        if latest_date != trade_date:
            logger.warning(
                "数据时效性守卫：IM.CFX 最新日期 %s ≠ 目标 %s，拒绝写入模型输出",
                latest_date, trade_date,
            )
            return None

        close = pd.Series(
            df["close"].astype(float).values,
            index=pd.to_datetime(df["trade_date"]),
        ).dropna()
        returns = np.log(close / close.shift(1)).dropna()

        rv_5d  = float(returns.iloc[-5:].std(ddof=1)) * np.sqrt(252) if len(returns) >= 5 else 0.0
        rv_20d = float(returns.rolling(20).std().dropna().iloc[-1]) * np.sqrt(252)
        rv_60d = float(returns.rolling(60).std().dropna().iloc[-1]) * np.sqrt(252)

        model = GJRGARCHModel()
        model.fit(returns)
        garch_current_vol  = float(model.forecast_period_avg(horizon=1))
        garch_forecast_vol = float(model.forecast_period_avg(horizon=5))
    except Exception as e:
        logger.warning("GARCH 计算失败: %s", e)
        return None

    # ── 2. MO 期权链 → ATM IV（与 portfolio_analysis.py 完全一致的定价逻辑）──
    # 使用 get_atm_iv()：14天过滤 + PCP 隐含 Forward Price
    atm_iv = None
    spot = None
    chain_df = pd.DataFrame()
    expire_map: dict = {}
    futures_prices: dict = {}
    latest_opt_date: str | None = None
    try:
        spot_row = db.query_df(
            "SELECT close FROM futures_daily "
            "WHERE ts_code='IM.CFX' ORDER BY trade_date DESC LIMIT 1"
        )
        spot = float(spot_row["close"].iloc[0]) if spot_row is not None and not spot_row.empty else None

        latest_row = db.query_df(
            "SELECT MAX(trade_date) as dt FROM options_daily WHERE ts_code LIKE 'MO%'"
        )
        latest_opt_date = (
            str(latest_row["dt"].iloc[0])
            if latest_row is not None and not latest_row.empty
            else None
        )

        if spot and latest_opt_date:
            expire_map = _build_expire_map(db)
            chain_df   = _get_mo_chain(db, latest_opt_date, expire_map)

            if not chain_df.empty:
                expire_months = sorted(chain_df["expire_month"].unique())
                # 与 portfolio_analysis.py 完全相同：按到期月获取对应期货价格
                futures_prices = get_futures_prices_by_expiry(
                    db, expire_months, spot, trade_date=latest_opt_date
                )
                # 与 portfolio_analysis.py 完全相同：ATM IV 使用 PCP 隐含 Forward
                atm_iv, atm_iv_src = get_atm_iv(spot, chain_df, trade_date, futures_prices)
                logger.info("ATM IV: %.2f%%  (%s)", atm_iv * 100, atm_iv_src)
    except Exception as e:
        logger.warning("ATM IV 计算失败: %s", e)

    # ── 2b. Market ATM IV（期货价格based，用于 VRP 和情绪指标）──────────
    # 避免 Forward-based IV 的循环论证：Forward 由期权价格反推 → IV 天然"合理" → VRP 被低估
    atm_iv_market = None
    try:
        if spot and not chain_df.empty and expire_map:
            from scripts.vol_monitor import calc_market_atm_iv, _find_near_month
            expire_months = sorted(chain_df["expire_month"].unique())
            near_month = _find_near_month(expire_months, expire_map, trade_date)
            if near_month:
                near_ed = expire_map.get(near_month, "")
                atm_iv_market = calc_market_atm_iv(
                    chain_df, near_month, spot, near_ed, trade_date
                )
                if atm_iv_market and atm_iv_market > 0:
                    logger.info(
                        "Market ATM IV: %.2f%%  (期货价格based, vs 结构IV %.2f%%)",
                        atm_iv_market * 100,
                        atm_iv * 100 if atm_iv else 0,
                    )
                else:
                    atm_iv_market = None
    except Exception as e:
        logger.warning("Market ATM IV 计算失败: %s", e)

    # ── 3. VRP & 百分位（使用 market IV，避免循环论证）─────────────────
    vrp = None
    vrp_percentile = None
    signal = "NEUTRAL"
    # Blended RV + GARCH sanity check
    rv_max = max(rv_5d, rv_20d) if rv_5d > 0 and rv_20d > 0 else rv_20d
    _garch_reliable = not (garch_forecast_vol > 0 and rv_max > 0
                           and garch_forecast_vol > rv_max * 1.4)
    if _garch_reliable and garch_forecast_vol > 0 and rv_5d > 0 and rv_20d > 0:
        blended_rv = 0.4 * rv_5d + 0.4 * rv_20d + 0.2 * garch_forecast_vol
    elif rv_5d > 0 and rv_20d > 0:
        blended_rv = 0.5 * rv_5d + 0.5 * rv_20d
    else:
        blended_rv = rv_20d if rv_20d > 0 else garch_forecast_vol

    # VRP = IV - Blended RV
    iv_for_vrp = atm_iv_market if atm_iv_market else atm_iv
    if iv_for_vrp is not None:
        vrp = iv_for_vrp - blended_rv
        # 历史 VRP 百分位（从 daily_model_output 历史记录计算）
        try:
            hist = db.query_df(
                "SELECT vrp FROM daily_model_output "
                "WHERE underlying='IM' AND vrp IS NOT NULL "
                "ORDER BY trade_date ASC"
            )
            if hist is not None and len(hist) >= 20:
                vrp_pct = float(np.mean(hist["vrp"].values <= vrp) * 100)
                vrp_percentile = round(vrp_pct, 1)
        except Exception:
            pass

        # 主信号：基于IV历史分位（不依赖GARCH）
        iv_pctile_hist = None
        try:
            iv_hist = db.query_df(
                "SELECT atm_iv FROM volatility_history "
                "WHERE atm_iv IS NOT NULL AND atm_iv > 0"
            )
            if iv_hist is not None and len(iv_hist) > 50:
                iv_val_pct = (iv_for_vrp * 100) if iv_for_vrp < 1 else iv_for_vrp
                hist_vals = iv_hist["atm_iv"].values
                hist_pct = hist_vals if hist_vals.mean() > 1 else hist_vals * 100
                iv_pctile_hist = round(float(np.mean(hist_pct <= iv_val_pct) * 100), 1)
                if iv_pctile_hist >= 90:
                    signal = "SELL_VOL"
                elif iv_pctile_hist <= 10:
                    signal = "BUY_VOL"
                else:
                    signal = "NEUTRAL"
            else:
                # fallback to VRP-based signal
                if vrp > VRP_SELL_THRESHOLD:
                    signal = "SELL_VOL"
                elif vrp < VRP_BUY_THRESHOLD:
                    signal = "BUY_VOL"
                else:
                    signal = "NEUTRAL"
        except Exception:
            if vrp > VRP_SELL_THRESHOLD:
                signal = "SELL_VOL"
            elif vrp < VRP_BUY_THRESHOLD:
                signal = "BUY_VOL"
            else:
                signal = "NEUTRAL"

    # ── 4. 组合 Greeks（与 portfolio_analysis.py 完全相同的定价逻辑）──────
    # 使用隐含 Forward Price + 各到期月期货价格 + 市场 IV，与 portfolio_analysis 对齐
    net_delta = net_gamma = net_theta = net_vega = None
    try:
        pos_df = db.query_df(
            f"SELECT * FROM position_snapshots WHERE trade_date='{trade_date}'"
        )
        if pos_df is not None and not pos_df.empty and spot and not chain_df.empty:
            from data.sources.tq_client import TqClient

            # 只处理期权持仓（symbol 含 "-C-" 或 "-P-"）
            opt_mask = (
                pos_df["symbol"].str.contains("-C-", na=False) |
                pos_df["symbol"].str.contains("-P-", na=False)
            )
            opt_df = pos_df[opt_mask].copy()

            if not opt_df.empty and expire_map:
                # 将 TQ 代码转换为 ts_code（同 portfolio_analysis.py 的 build_positions 逻辑）
                positions = []
                for _, row in opt_df.iterrows():
                    ts_code = TqClient.convert_symbol_tq_to_ts(row["symbol"])
                    parsed = _parse_mo(ts_code)
                    if parsed is None:
                        continue
                    expire_month, cp, strike = parsed
                    exp_date = expire_map.get(expire_month, "")
                    if not exp_date:
                        continue
                    vol = int(row["volume"]) * (1 if row["direction"] == "LONG" else -1)
                    positions.append({
                        "ts_code":      ts_code,
                        "strike_price": strike,
                        "call_put":     cp,
                        "expire_date":  exp_date,
                        "volume":       vol,
                    })

                if positions:
                    # Step 1: PCP 隐含 Forward（与 portfolio_analysis.py Step 3c 相同）
                    implied_forwards = calc_implied_forwards_by_expiry(
                        chain_df, futures_prices, spot
                    )
                    # Step 2: 市场 IV（与 portfolio_analysis.py Step 4 相同）
                    iv_map = get_iv_for_positions(
                        positions, spot, chain_df, futures_prices, implied_forwards
                    )
                    # Step 3: Greeks（与 portfolio_analysis.py Step 7 相同）
                    if iv_map:
                        greeks_result, _ = calc_portfolio(
                            spot, positions, iv_map, futures_prices, implied_forwards
                        )
                        net_delta = greeks_result["net_delta"]
                        net_gamma = greeks_result["net_gamma"]
                        net_theta = greeks_result["net_theta"]
                        net_vega  = greeks_result["net_vega"]
    except Exception as e:
        logger.warning("组合 Greeks 计算失败: %s", e)

    # ── 5. 贴水信号计算 ────────────────────────────────────────────────
    discount_rate_iml1 = None
    discount_rate_iml2 = None
    discount_rate_iml3 = None
    discount_signal    = "NONE"
    recommended_contract = None
    try:
        from strategies.discount_capture.signal import DiscountSignal
        disc_gen = DiscountSignal(db)
        disc_df = disc_gen.calculate_discount(trade_date)
        if disc_df is not None and not disc_df.empty:
            from utils.cffex_calendar import _IML_CODES
            # IML_CODES = ["IM.CFX", "IML1.CFX", "IML2.CFX", "IML3.CFX"]
            # disc_df rows are sorted by days_to_expiry; iml_code identifies position
            iml_map = {
                "IML1.CFX": "discount_rate_iml1",
                "IML2.CFX": "discount_rate_iml2",
                "IML3.CFX": "discount_rate_iml3",
            }
            for _, drow in disc_df.iterrows():
                iml_code = str(drow["iml_code"])
                field = iml_map.get(iml_code)
                if field and drow["spot_price"] > 0:
                    raw_rate = float(drow["absolute_discount"]) / float(drow["spot_price"])
                    if field == "discount_rate_iml1":
                        discount_rate_iml1 = round(raw_rate, 6)
                    elif field == "discount_rate_iml2":
                        discount_rate_iml2 = round(raw_rate, 6)
                    elif field == "discount_rate_iml3":
                        discount_rate_iml3 = round(raw_rate, 6)

        sig_result = disc_gen.generate_signal(trade_date)
        discount_signal      = sig_result.get("signal", "NONE")
        recommended_contract = sig_result.get("recommended_contract")
        logger.info(
            "贴水信号: IML1=%s  IML2=%s  IML3=%s  signal=%s  推荐=%s",
            f"{discount_rate_iml1*100:.2f}%" if discount_rate_iml1 else "N/A",
            f"{discount_rate_iml2*100:.2f}%" if discount_rate_iml2 else "N/A",
            f"{discount_rate_iml3*100:.2f}%" if discount_rate_iml3 else "N/A",
            discount_signal, recommended_contract,
        )
    except Exception as e:
        logger.warning("贴水信号计算失败: %s", e)

    # ── 6. 写入 daily_model_output ────────────────────────────────────
    row = {
        "trade_date":             trade_date,
        "underlying":             "IM",
        "garch_current_vol":      round(garch_current_vol,  6),
        "garch_forecast_vol":     round(garch_forecast_vol, 6),
        "realized_vol_20d":       round(rv_20d,              6),
        "realized_vol_60d":       round(rv_60d,              6),
        "atm_iv":                 round(atm_iv, 6) if atm_iv else None,
        "atm_iv_market":          round(atm_iv_market, 6) if atm_iv_market else None,
        "vrp":                    round(vrp, 6)    if vrp    else None,
        "vrp_percentile":         vrp_percentile,
        "signal":                 signal,
        "net_delta":              round(net_delta, 4) if net_delta is not None else None,
        "net_gamma":              round(net_gamma, 6) if net_gamma is not None else None,
        "net_theta":              round(net_theta, 4) if net_theta is not None else None,
        "net_vega":               round(net_vega,  4) if net_vega  is not None else None,
        "discount_rate_iml1":     discount_rate_iml1,
        "discount_rate_iml2":     discount_rate_iml2,
        "discount_rate_iml3":     discount_rate_iml3,
        "discount_signal":        discount_signal,
        "recommended_contract":   recommended_contract,
        "garch_5d_forecast_date": trade_date,   # 标记本条记录的预测日期（用于回溯查询）
        "rv_5d_actual":           None,          # 5日后回填
        "forecast_error":         None,          # 5日后回填
        "iv_percentile_hist":     iv_pctile_hist,
        "signal_primary":         signal,        # 基于IV分位的主信号
        "garch_reliable":         1 if _garch_reliable else 0,
    }
    # 追加研究指标（fail-safe，不影响核心模型写入）
    try:
        from scripts.vol_monitor import calc_skew_table, calc_rr_bf
        if near_month and not chain_df.empty and expire_map:
            _skew_fwd = spot or 0  # fallback到现货
            _skew_ed = expire_map.get(near_month, "")
            _skew_df = calc_skew_table(chain_df, near_month, _skew_fwd, _skew_ed, trade_date)
            _rr, _ = calc_rr_bf(_skew_df)
            if _rr != 0:
                row["rr_25d"] = round(_rr, 6)
    except Exception as e:
        logger.debug("RR计算跳过: %s", e)
    try:
        from scripts.morning_briefing import _calc_hurst
        _hdf = db.query_df(
            "SELECT close FROM index_daily WHERE ts_code='000852.SH' "
            "AND trade_date<=? ORDER BY trade_date DESC LIMIT 60",
            (trade_date,))
        if _hdf is not None and len(_hdf) >= 60:
            row["hurst_60d"] = round(_calc_hurst(_hdf["close"].astype(float).values[::-1]), 4)
    except Exception as e:
        logger.debug("Hurst计算跳过: %s", e)
    try:
        db.upsert_dataframe("daily_model_output", pd.DataFrame([row]))
        logger.info(
            "模型输出已写入: RV20=%.2f%%  GARCH=%.2f%%  结构IV=%s  市场IV=%s  VRP=%s  信号=%s",
            rv_20d * 100, garch_forecast_vol * 100,
            f"{atm_iv*100:.2f}%" if atm_iv else "N/A",
            f"{atm_iv_market*100:.2f}%" if atm_iv_market else "N/A",
            f"{vrp*100:+.2f}%"   if vrp    else "N/A",
            signal,
        )
    except Exception as e:
        logger.error("写入 daily_model_output 失败: %s", e)

    return row


def _eod_pnl_attribution(trade_date: str, db) -> dict | None:
    """
    计算当日 P&L 归因并写入 daily_model_output 的 pnl_* 字段。

    使用前一交易日的 Greeks（BOD）和当日价格/IV 变动，
    通过一阶泰勒展开将总盈亏分解为 Delta/Gamma/Theta/Vega 贡献。
    """
    try:
        from analysis.pnl_attribution import PnLAttributor

        attributor = PnLAttributor(db)
        result = attributor.attribute_daily_pnl(trade_date)

        if result is None:
            logger.info("P&L 归因: 数据不足，跳过")
            return None

        # 用 UPDATE 写入（不覆盖已有的其他字段）
        db._conn.execute(
            "UPDATE daily_model_output SET "
            "pnl_total=?, pnl_realized=?, pnl_unrealized=?, "
            "pnl_delta=?, pnl_gamma=?, pnl_theta=?, pnl_vega=?, pnl_residual=? "
            "WHERE trade_date=? AND underlying='IM'",
            (
                result.total_pnl, result.realized_pnl, result.unrealized_pnl,
                result.delta_pnl, result.gamma_pnl, result.theta_pnl,
                result.vega_pnl, result.residual_pnl,
                trade_date,
            ),
        )
        db._conn.commit()

        logger.info(
            "P&L 归因: total=%s  Δ=%s  Γ=%s  Θ=%s  V=%s  残差=%s  解释率=%.0f%%",
            f"{result.total_pnl:+,.0f}", f"{result.delta_pnl:+,.0f}",
            f"{result.gamma_pnl:+,.0f}", f"{result.theta_pnl:+,.0f}",
            f"{result.vega_pnl:+,.0f}", f"{result.residual_pnl:+,.0f}",
            result.explained_ratio * 100,
        )
        return {
            "total": result.total_pnl,
            "realized": result.realized_pnl,
            "unrealized": result.unrealized_pnl,
            "delta": result.delta_pnl,
            "gamma": result.gamma_pnl,
            "theta": result.theta_pnl,
            "vega": result.vega_pnl,
            "residual": result.residual_pnl,
            "explained_ratio": result.explained_ratio,
        }
    except Exception as e:
        logger.warning("P&L 归因计算失败: %s", e)
        return None


def _backfill_forecast_accuracy(db, trade_date: str) -> dict | None:
    """
    回填5个交易日前那条 daily_model_output 记录的 rv_5d_actual 和 forecast_error。

    逻辑：
      1. 找到今天往前数第5个交易日（从 IM.CFX 日线数据取日期序列）
      2. 读取那天的 garch_forecast_vol
      3. 用那天到今天这5个交易日的对数收益率计算年化RV
      4. 更新该记录：rv_5d_actual = 实际RV, forecast_error = forecast - actual

    Returns
    -------
    dict or None
        { 'forecast_date', 'forecast_vol', 'actual_rv', 'error', 'error_pp' }
        回填成功时返回，否则返回 None
    """
    try:
        # 取最近 15 个交易日的 IM.CFX 收盘价（含今天），用于确定"5日前"日期和计算RV
        df_closes = db.query_df(
            "SELECT trade_date, close FROM futures_daily "
            "WHERE ts_code='IM.CFX' ORDER BY trade_date DESC LIMIT 15"
        )
        if df_closes is None or len(df_closes) < 7:
            return None

        df_closes = df_closes.sort_values("trade_date").reset_index(drop=True)
        dates  = df_closes["trade_date"].tolist()
        closes = df_closes["close"].astype(float).values

        # 确保 trade_date 在序列末尾（或取最后一条）
        if dates[-1] != trade_date:
            # 今天可能还没写入期货日线，用最新一条
            logger.debug("backfill: 今日 %s 不在期货日线末尾，跳过", trade_date)
            return None

        if len(dates) < 6:
            return None

        # 5个交易日前的日期 = dates[-6]（今天是 dates[-1]，向前5步）
        forecast_date = dates[-6]

        # 读取那天的模型预测
        df_hist = db.query_df(
            "SELECT garch_forecast_vol, rv_5d_actual FROM daily_model_output "
            "WHERE trade_date=? AND underlying='IM'",
            (forecast_date,),
        )
        if df_hist is None or df_hist.empty:
            return None

        forecast_vol = df_hist["garch_forecast_vol"].iloc[0]
        already_filled = df_hist["rv_5d_actual"].iloc[0]

        if forecast_vol is None or pd.isna(forecast_vol):
            return None
        forecast_vol = float(forecast_vol)

        # 如果已经填过了，直接返回存量数据
        if already_filled is not None and not pd.isna(already_filled):
            actual_rv = float(already_filled)
            error = forecast_vol - actual_rv
            return {
                "forecast_date": forecast_date,
                "forecast_vol":  forecast_vol,
                "actual_rv":     actual_rv,
                "error":         error,
                "error_pp":      error * 100,
                "already_filled": True,
            }

        # 用最近5根日线（dates[-5:]）的对数收益率计算年化5日RV
        # closes[-6] 是 forecast_date 收盘，closes[-5:]~closes[-1] 是随后5天收盘
        log_rets = np.diff(np.log(closes[-6:]))   # 5个收益率
        actual_rv = float(np.std(log_rets, ddof=1) * np.sqrt(252))

        error = forecast_vol - actual_rv

        # 写回数据库：只更新这两列（不用 upsert，避免覆盖其他字段）
        db._conn.execute(
            "UPDATE daily_model_output SET rv_5d_actual=?, forecast_error=? "
            "WHERE trade_date=? AND underlying='IM'",
            (round(actual_rv, 6), round(error, 6), forecast_date),
        )
        db._conn.commit()
        logger.info(
            "回填预测精度: forecast_date=%s  forecast=%.2f%%  actual=%.2f%%  error=%+.2fpp",
            forecast_date, forecast_vol * 100, actual_rv * 100, error * 100,
        )

        return {
            "forecast_date":  forecast_date,
            "forecast_vol":   forecast_vol,
            "actual_rv":      actual_rv,
            "error":          error,
            "error_pp":       error * 100,
            "already_filled": False,
        }

    except Exception as e:
        logger.warning("回填预测精度失败: %s", e)
        return None


def _calc_market_regime(db, model: dict | None) -> dict | None:
    """
    从 IM.CFX 日线数据计算市场状态评估指标。

    Returns dict with keys:
        garch_cond_vol, garch_long_vol, garch_ratio, garch_state
        rv5, rv20, rv60, rv_trend
        price_20d_low, price_20d_high, price_current, price_range_pct, price_pos_pct
        consec_days, consec_dir
        vrp_percentile
        summary
    """
    try:
        df = db.query_df(
            "SELECT trade_date, close FROM futures_daily "
            "WHERE ts_code='IM.CFX' ORDER BY trade_date DESC LIMIT 280"
        )
        if df is None or len(df) < 25:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)
        closes = df["close"].astype(float).values

        # Log returns for RV calculation
        log_rets = np.diff(np.log(closes))

        def ann_rv(n: int) -> float:
            if len(log_rets) < n:
                return float("nan")
            return float(np.std(log_rets[-n:], ddof=1) * np.sqrt(252))

        rv5  = ann_rv(5)
        rv20 = ann_rv(20)
        rv60 = ann_rv(60)
        rv252 = ann_rv(252)

        # RV trend direction
        if not (np.isnan(rv5) or np.isnan(rv20) or np.isnan(rv60)):
            if rv5 < rv20 < rv60:
                rv_trend = "下降"
            elif rv5 > rv20 > rv60:
                rv_trend = "上升"
            elif rv5 > rv20:
                rv_trend = "短期回升"
            else:
                rv_trend = "震荡"
        else:
            rv_trend = "数据不足"

        # GARCH state
        garch_cond  = model.get("garch_current_vol") if model else None
        garch_long  = rv252 if not np.isnan(rv252) else (model.get("realized_vol_60d") if model else None)
        garch_ratio = None
        garch_state = "未知"
        if garch_cond and garch_long and garch_long > 0:
            garch_ratio = garch_cond / garch_long
            if garch_ratio > 1.3:
                garch_state = "偏高"
            elif garch_ratio < 0.8:
                garch_state = "偏低"
            else:
                garch_state = "正常"

        # 20-day price range & current position
        if len(closes) >= 20:
            recent20 = closes[-20:]
            p_low  = float(np.min(recent20))
            p_high = float(np.max(recent20))
            p_cur  = float(closes[-1])
            p_range_pct = (p_high - p_low) / p_low * 100 if p_low > 0 else 0.0
            p_pos_pct   = (p_cur - p_low) / (p_high - p_low) * 100 if p_high > p_low else 50.0
        else:
            p_low = p_high = p_cur = float(closes[-1])
            p_range_pct = 0.0
            p_pos_pct = 50.0

        # Consecutive up/down days
        if len(closes) >= 2:
            diffs = np.diff(closes[-11:])  # last 10 changes
            consec = 1
            direction = "涨" if diffs[-1] > 0 else "跌" if diffs[-1] < 0 else "平"
            for d in reversed(diffs[:-1]):
                if direction == "涨" and d > 0:
                    consec += 1
                elif direction == "跌" and d < 0:
                    consec += 1
                else:
                    break
        else:
            consec = 0
            direction = "平"

        # VRP percentile
        vrp_pct = model.get("vrp_percentile") if model else None

        # Summary text
        parts = []
        if garch_state != "未知":
            parts.append(f"波动率{garch_state}")
        if rv_trend in ("上升", "下降"):
            parts.append(f"RV{rv_trend}中")
        elif rv_trend == "短期回升":
            parts.append("短期波动率回升")
        if p_range_pct < 5:
            parts.append("窄幅震荡")
        elif p_range_pct > 10:
            parts.append("宽幅波动")
        if vrp_pct is not None:
            if vrp_pct >= 70:
                parts.append("VRP高位，卖方有优势")
            elif vrp_pct <= 30:
                parts.append("VRP低位，买方有优势")
            else:
                parts.append("VRP中性")

        summary = "，".join(parts) if parts else "数据不足，无法综合判断"

        return {
            "garch_cond_vol":  garch_cond,
            "garch_long_vol":  garch_long,
            "garch_ratio":     garch_ratio,
            "garch_state":     garch_state,
            "rv5":             rv5,
            "rv20":            rv20,
            "rv60":            rv60,
            "rv_trend":        rv_trend,
            "price_20d_low":   p_low,
            "price_20d_high":  p_high,
            "price_current":   p_cur,
            "price_range_pct": p_range_pct,
            "price_pos_pct":   p_pos_pct,
            "consec_days":     consec,
            "consec_dir":      direction,
            "vrp_percentile":  vrp_pct,
            "summary":         summary,
        }

    except Exception as e:
        logger.warning("计算市场状态失败: %s", e)
        return None


def _print_market_regime(regime: dict) -> None:
    """打印市场状态评估段落。"""
    print()
    print("【市场状态评估】")

    # GARCH σ state
    g_cond = regime["garch_cond_vol"]
    g_long = regime["garch_long_vol"]
    g_ratio = regime["garch_ratio"]
    if g_cond is not None and g_long is not None and g_ratio is not None:
        dir_hint = "→ 倾向回落" if g_ratio > 1.3 else "→ 倾向上升" if g_ratio < 0.8 else "→ 正常区间"
        print(
            f"  GARCH σ状态    : {regime['garch_state']}"
            f"（条件{g_cond*100:.2f}% / 长期{g_long*100:.1f}% = {g_ratio:.2f}x）{dir_hint}"
        )
    else:
        print("  GARCH σ状态    : 数据不足")

    # RV trend
    rv5  = regime["rv5"]
    rv20 = regime["rv20"]
    rv60 = regime["rv60"]
    if not any(np.isnan(x) for x in [rv5, rv20, rv60]):
        # Arrow direction
        if rv5 < rv20 and rv20 < rv60:
            arrow = "→ 波动率在下降"
        elif rv5 > rv20 and rv20 > rv60:
            arrow = "→ 波动率在上升"
        elif rv5 > rv20:
            arrow = "→ 短期波动率回升"
        else:
            arrow = "→ 震荡"
        print(
            f"  RV趋势        : 5d={rv5*100:.1f}% / 20d={rv20*100:.1f}% / 60d={rv60*100:.1f}%  {arrow}"
        )
    else:
        print("  RV趋势        : 数据不足")

    # 20-day price range
    p_low  = regime["price_20d_low"]
    p_high = regime["price_20d_high"]
    rng    = regime["price_range_pct"]
    pos    = regime["price_pos_pct"]
    if rng < 5:
        width_desc = "窄幅震荡"
    elif rng > 10:
        width_desc = "宽幅波动"
    else:
        width_desc = "正常波幅"
    print(f"  20日价格范围   : {p_low:.0f} ~ {p_high:.0f}（振幅 {rng:.1f}%，{width_desc}）")
    print(f"  当前位置       : 范围 {pos:.0f}%（{'偏上方' if pos > 60 else '偏下方' if pos < 40 else '中间'}）")

    # Consecutive days
    c_days = regime["consec_days"]
    c_dir  = regime["consec_dir"]
    print(f"  连续涨跌       : 连{c_dir} {c_days} 天")

    # VRP percentile
    vrp_pct = regime["vrp_percentile"]
    if vrp_pct is not None:
        if vrp_pct >= 70:
            vrp_desc = "高位，卖方有优势"
        elif vrp_pct <= 30:
            vrp_desc = "低位，买方有优势"
        else:
            vrp_desc = "中性"
        print(f"  VRP百分位      : {vrp_pct:.0f}%（{vrp_desc}）")
    else:
        print("  VRP百分位      : 暂无数据")

    # Summary
    print()
    print(f"  综合判断: {regime['summary']}")


def _print_eod_summary(
    trade_date: str,
    account: dict | None,
    positions: list[dict],
    n_trades: int,
    model: dict | None,
    freshness: "DataFreshness | None" = None,
    model_skipped: bool = False,
    model_ok: bool = False,
    db=None,
    backfill: dict | None = None,
) -> None:
    """打印收盘后数据摘要报告。"""
    sep = "═" * 70

    print()
    print(sep)
    print(f"  每日记录摘要  |  {trade_date}")
    print(sep)

    # 账户快照
    print()
    print("【账户快照】")
    if account:
        print(f"  账户权益   : {account['balance']:>14,.2f} 元")
        print(f"  可用资金   : {account['available']:>14,.2f} 元")
        print(f"  占用保证金 : {account['margin']:>14,.2f} 元  ({account['margin_ratio']*100:.1f}%)")
        print(f"  浮动盈亏   : {account['float_profit']:>+14,.2f} 元")
        print(f"  当日平盈亏 : {account['close_profit']:>+14,.2f} 元")
        print(f"  当日手续费 : {account['commission']:>14,.2f} 元")
    else:
        print("  （TQ 未连接或非交易时段，无账户数据）")

    # 持仓快照
    print()
    print(f"【持仓快照】  共 {len(positions)} 条")
    if positions:
        print(f"  {'合约':<40} {'方向':<6} {'手数':>5} {'开仓价':>10} {'浮盈':>12}")
        print(f"  {'-'*70}")
        for p in positions:
            print(
                f"  {p['symbol']:<40} {p['direction']:<6} {p['volume']:>5}"
                f"  {p['open_price_avg']:>10.2f}  {p['float_profit']:>+12.2f}"
            )

    # 成交记录
    print()
    print(f"【成交记录】  当日 {n_trades} 笔成交")

    # 模型输出
    print()
    print("【模型输出（IM 标的）】")
    if model:
        print(f"  20日历史RV  : {model['realized_vol_20d']*100:>7.2f}%")
        print(f"  60日历史RV  : {model['realized_vol_60d']*100:>7.2f}%")
        print(f"  GARCH条件σ  : {model['garch_current_vol']*100:>7.2f}%")
        print(f"  GARCH 5日预 : {model['garch_forecast_vol']*100:>7.2f}%")
        if model["atm_iv"]:
            print(f"  ATM IV      : {model['atm_iv']*100:>7.2f}%")
        if model["vrp"] is not None:
            vrp_pct = f"  历史百分位 {model['vrp_percentile']}%" if model["vrp_percentile"] else ""
            print(f"  VRP         : {model['vrp']*100:>+7.2f}%{vrp_pct}")
        print(f"  信号        : {model['signal']}")
        if model.get("discount_signal"):
            d1 = model.get("discount_rate_iml1")
            d2 = model.get("discount_rate_iml2")
            d3 = model.get("discount_rate_iml3")
            print(f"  贴水信号    : {model['discount_signal']}"
                  + (f"  推荐={model['recommended_contract']}" if model.get("recommended_contract") else ""))
            parts = []
            if d1 is not None:
                parts.append(f"IML1={d1*100:+.2f}%")
            if d2 is not None:
                parts.append(f"IML2={d2*100:+.2f}%")
            if d3 is not None:
                parts.append(f"IML3={d3*100:+.2f}%")
            if parts:
                print(f"  贴水率      : {' | '.join(parts)}")
        if model["net_delta"] is not None:
            print(f"  净Delta     : {model['net_delta']:>+10.2f} 元/点")
            print(f"  净Theta     : {model['net_theta']:>+10.2f} 元/天")
            print(f"  净Vega      : {model['net_vega']:>+10.2f} 元/1%σ")
    else:
        print("  （模型输出未能计算）")

    # 预测回溯验证
    if backfill is not None:
        fd   = backfill["forecast_date"]
        fv   = backfill["forecast_vol"] * 100
        av   = backfill["actual_rv"]    * 100
        ep   = backfill["error_pp"]
        bias = "高估" if ep > 0 else "低估"
        print()
        print("【预测回溯验证】")
        print(f"  5日前 GARCH 预测: {fv:.1f}%  →  实际 RV: {av:.1f}%  →  误差: {ep:+.1f}pp（{bias}）"
              f"  [{fd}]")

    # 市场状态评估
    if db is not None:
        regime = _calc_market_regime(db, model)
        if regime is not None:
            _print_market_regime(regime)

    # 数据状态
    if freshness is not None:
        print()
        print(_format_freshness_status(freshness, model_skipped=model_skipped, model_ok=model_ok))

    print()
    print(sep)
    print()

    # 生成 Markdown 报告并持久化
    try:
        md = _generate_eod_markdown(
            trade_date, account, positions, n_trades, model,
            regime=_calc_market_regime(db, model) if db else None,
            backfill=backfill,
            db=db,
        )
        from utils.report_writer import save_report
        fp = save_report(trade_date, "eod", md)
        print(f"报告已保存: {fp}")
    except Exception as e:
        logger.warning("EOD 报告保存失败: %s", e)


def _generate_eod_markdown(
    trade_date: str,
    account: dict | None,
    positions: list[dict],
    n_trades: int,
    model: dict | None,
    regime: dict | None = None,
    backfill: dict | None = None,
    db=None,
) -> str:
    """生成 EOD Markdown 报告。"""
    lines: list[str] = []

    td_fmt = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
    lines.append(f"# EOD 每日记录 | {td_fmt}")

    # 账户快照（优先用传入数据，fallback 从数据库读取）
    acct = account
    if not acct and db is not None:
        try:
            acct_df = db.query_df(
                "SELECT * FROM account_snapshots WHERE trade_date = ?", (trade_date,)
            )
            if acct_df is not None and len(acct_df) > 0:
                acct = acct_df.iloc[0].to_dict()
        except Exception:
            pass

    lines.append("\n## 账户快照")
    if acct:
        lines.append(f"- 账户权益: {acct['balance']:,.2f} 元")
        lines.append(f"- 可用资金: {acct['available']:,.2f} 元")
        lines.append(f"- 占用保证金: {acct['margin']:,.2f} 元 ({acct['margin_ratio']*100:.1f}%)")
        lines.append(f"- 浮动盈亏: {acct['float_profit']:+,.2f} 元")
        lines.append(f"- 当日平盈亏: {acct['close_profit']:+,.2f} 元")
        lines.append(f"- 当日手续费: {acct['commission']:,.2f} 元")
    else:
        lines.append("（TQ 未连接或非交易时段，无账户数据）")

    # 持仓快照（优先用传入数据，fallback 从数据库读取）
    pos_list = positions
    if not pos_list and db is not None:
        try:
            pos_df = db.query_df(
                "SELECT * FROM position_snapshots WHERE trade_date = ?", (trade_date,)
            )
            if pos_df is not None and len(pos_df) > 0:
                pos_list = pos_df.to_dict("records")
        except Exception:
            pass

    lines.append(f"\n## 持仓快照（共 {len(pos_list)} 条）")
    if pos_list:
        lines.append("| 合约 | 方向 | 手数 | 开仓价 | 浮盈 |")
        lines.append("|------|------|------|--------|------|")
        for p in pos_list:
            lines.append(
                f"| {p['symbol']} | {p['direction']} | {p['volume']} "
                f"| {p['open_price_avg']:.2f} | {p['float_profit']:+,.2f} |"
            )

    # 成交记录（从数据库读取，避免结算后 TQ 数据清零）
    trade_df = None
    if db is not None:
        try:
            trade_df = db.query_df(
                "SELECT trade_time, symbol, direction, offset, volume, price "
                "FROM trade_records WHERE trade_date = ? ORDER BY trade_time",
                (trade_date,),
            )
        except Exception:
            pass

    if trade_df is not None and len(trade_df) > 0:
        lines.append(f"\n## 成交记录（{len(trade_df)} 笔）")
        lines.append("| 时间 | 合约 | 方向 | 开平 | 手数 | 价格 |")
        lines.append("|------|------|------|------|------|------|")
        for _, t in trade_df.iterrows():
            d_cn = "买" if str(t["direction"]).upper() == "BUY" else "卖"
            o_cn = {"OPEN": "开", "CLOSE": "平", "CLOSETODAY": "平今"}.get(
                str(t["offset"]).upper(), str(t["offset"])
            )
            lines.append(
                f"| {t['trade_time']} | {t['symbol']} | {d_cn} "
                f"| {o_cn} | {t['volume']} | {t['price']:.2f} |"
            )
    else:
        lines.append(f"\n## 成交记录\n- 当日无成交")

    # 模型输出
    lines.append("\n## 模型输出")
    if model:
        lines.append(f"- 20日历史RV: {model['realized_vol_20d']*100:.2f}%")
        lines.append(f"- 60日历史RV: {model['realized_vol_60d']*100:.2f}%")
        lines.append(f"- GARCH 条件σ: {model['garch_current_vol']*100:.2f}%")
        lines.append(f"- GARCH 5日预测: {model['garch_forecast_vol']*100:.2f}%")
        if model.get("atm_iv"):
            lines.append(f"- ATM IV: {model['atm_iv']*100:.2f}%")
        if model.get("vrp") is not None:
            vrp_str = f"- VRP: {model['vrp']*100:+.2f}%"
            if model.get("vrp_percentile"):
                vrp_str += f"（历史百分位 {model['vrp_percentile']}%）"
            lines.append(vrp_str)
        lines.append(f"- 信号: **{model['signal']}**")

        # Greeks
        if model.get("net_delta") is not None:
            lines.append("\n## 组合 Greeks")
            lines.append(f"- 净Delta: {model['net_delta']:+,.2f} 元/点")
            lines.append(f"- 净Theta: {model['net_theta']:+,.2f} 元/天")
            lines.append(f"- 净Vega: {model['net_vega']:+,.2f} 元/1%σ")

        # 贴水信号
        if model.get("discount_signal"):
            lines.append("\n## 贴水信号")
            parts = []
            for key, label in [("discount_rate_iml1", "IML1"),
                                ("discount_rate_iml2", "IML2"),
                                ("discount_rate_iml3", "IML3")]:
                v = model.get(key)
                if v is not None:
                    parts.append(f"{label}: {v*100:+.2f}%")
            if parts:
                lines.append(f"- {' | '.join(parts)}")
            sig = model["discount_signal"]
            rec = model.get("recommended_contract", "")
            lines.append(f"- 信号: **{sig}**" + (f"，推荐 {rec}" if rec else ""))
    else:
        lines.append("（模型输出未能计算）")

    # 预测回溯
    if backfill is not None:
        lines.append("\n## 预测回溯验证")
        fv = backfill["forecast_vol"] * 100
        av = backfill["actual_rv"] * 100
        ep = backfill["error_pp"]
        bias = "高估" if ep > 0 else "低估"
        lines.append(
            f"- GARCH 预测: {fv:.1f}% → 实际 RV: {av:.1f}% → "
            f"误差: {ep:+.1f}pp（{bias}）[{backfill['forecast_date']}]"
        )

    # 市场状态评估
    if regime is not None:
        lines.append("\n## 市场状态评估")
        g_cond = regime.get("garch_cond_vol")
        g_long = regime.get("garch_long_vol")
        g_ratio = regime.get("garch_ratio")
        if g_cond is not None and g_long is not None and g_ratio is not None:
            state = regime["garch_state"]
            lines.append(
                f"- GARCH σ状态: {state}"
                f"（条件{g_cond*100:.2f}% / 长期{g_long*100:.1f}% = {g_ratio:.2f}x）"
            )

        rv5 = regime.get("rv5", float("nan"))
        rv20 = regime.get("rv20", float("nan"))
        rv60 = regime.get("rv60", float("nan"))
        if not any(np.isnan(x) for x in [rv5, rv20, rv60]):
            if rv5 < rv20 and rv20 < rv60:
                arrow = "波动率在下降"
            elif rv5 > rv20 and rv20 > rv60:
                arrow = "波动率在上升"
            elif rv5 > rv20:
                arrow = "短期波动率回升"
            else:
                arrow = "震荡"
            lines.append(
                f"- RV趋势: 5d={rv5*100:.1f}% / 20d={rv20*100:.1f}% "
                f"/ 60d={rv60*100:.1f}% → {arrow}"
            )

        p_low = regime.get("price_20d_low", 0)
        p_high = regime.get("price_20d_high", 0)
        rng = regime.get("price_range_pct", 0)
        pos = regime.get("price_pos_pct", 0)
        lines.append(f"- 20日价格范围: {p_low:.0f} ~ {p_high:.0f}（振幅 {rng:.1f}%）")
        lines.append(
            f"- 当前位置: 范围 {pos:.0f}%"
            f"（{'偏上方' if pos > 60 else '偏下方' if pos < 40 else '中间'}）"
        )

        lines.append(f"\n综合判断: {regime.get('summary', '')}")

    return "\n".join(lines) + "\n"


def run_eod(
    trade_date: str | None = None,
    use_tq: bool = True,
    update_market_data: bool = True,
) -> None:
    """
    收盘后完整记录流程（可被 run_daily.py 直接调用）。

    Parameters
    ----------
    trade_date : str, optional
        交易日 YYYYMMDD，默认为今天
    use_tq : bool
        是否连接天勤抓取账户/持仓/成交（需要配置 TQ 凭证）
    update_market_data : bool
        是否触发 Tushare 增量数据下载
    """
    if not trade_date:
        trade_date = date.today().strftime("%Y%m%d")

    print(f"\n[EOD] 收盘后记录流程开始 — {trade_date}")

    db = _open_db()

    account_data: dict | None    = None
    positions:    list[dict]     = []
    n_trades:     int            = 0

    # ── a. TQ 账户/持仓/成交 ──────────────────────────────────────────
    if use_tq and _has_tq_broker():
        print("[EOD] a. 连接天勤，抓取账户/持仓/成交记录...")
        try:
            from data.sources.tq_client import TqClient
            creds = _tq_credentials()
            client = TqClient(**creds)
            client.connect()
            api = client._api

            # 先订阅全部账户数据，再用一次 wait_update 接收初始快照。
            # 收盘后服务器不再推送行情，加 deadline 防止永久阻塞。
            account_ref  = api.get_account()
            position_ref = api.get_position()
            trade_ref    = api.get_trade()
            got = api.wait_update(deadline=time.time() + 30)
            if not got:
                logger.info("TQ wait_update 超时（收盘后正常），使用当前缓存数据")

            account_data = _eod_account_snapshot(account_ref,  trade_date, db)
            positions    = _eod_position_snapshot(position_ref, trade_date, db)
            n_trades     = _eod_trade_records(trade_ref,        trade_date, db)

            # 归档当日5分钟K线
            n_bars = _eod_archive_minute_bars(api, trade_date, db)

            client.disconnect()
            print(f"  账户快照: {'✓' if account_data else '✗'}"
                  f"  持仓: {len(positions)} 条  成交: {n_trades} 笔"
                  f"  5m归档: {n_bars} 根")
        except Exception as e:
            logger.warning("TQ 连接/数据抓取失败: %s", e)
            print(f"  [警告] TQ 失败（{e}）")
    else:
        reason = "未配置实盘账户凭证" if use_tq else "--no-tq 已指定"
        print(f"[EOD] a. 跳过 TQ（{reason}）")

    # ── b. Tushare 增量数据更新 + 时效性检查（最多重试 3 次）────────────
    model_skipped = False

    # 先检查一次，若已是最新则完全跳过下载
    freshness = _check_data_freshness(db, trade_date)

    if not update_market_data:
        print("[EOD] b. 跳过 Tushare 数据更新（--no-update）")
    elif freshness.is_fresh:
        print(
            f"[EOD] b. 数据已是最新（期货 {freshness.futures_date}"
            f"  期权 {freshness.options_date}），跳过 Tushare 更新"
        )
    else:
        MAX_RETRIES = 3
        RETRY_WAIT  = 300  # 5 分钟

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"[EOD] b. Tushare 增量数据更新（第 {attempt}/{MAX_RETRIES} 次）...")
            _eod_update_market_data()

            freshness = _check_data_freshness(db, trade_date)
            if freshness.is_fresh:
                print(f"  数据时效性: 期货 {freshness.futures_date}  期权 {freshness.options_date}  ✓")
                break

            logger.warning(
                "数据未更新至 %s（期货: %s, 期权: %s）",
                trade_date, freshness.futures_date, freshness.options_date,
            )
            if attempt < MAX_RETRIES:
                print(f"  [警告] 数据尚未更新，等待 {RETRY_WAIT//60} 分钟后重试...")
                time.sleep(RETRY_WAIT)
            else:
                print("  [警告] 3 次重试后数据仍未更新，模型输出将跳过")
                model_skipped = True

    # ── c. 模型输出计算 & 写入 ─────────────────────────────────────────
    model_output = None
    if not model_skipped:
        print("[EOD] c. 运行模型计算（GARCH / IV / VRP / Greeks）...")
        model_output = _eod_model_output(trade_date, db)
    else:
        print("[EOD] c. 跳过模型计算（数据未更新）")

    model_ok = model_output is not None

    # ── c2. P&L 归因 ────────────────────────────────────────────────
    pnl_attr: dict | None = None
    if model_ok:
        print("[EOD] c2. 计算 P&L 归因（Delta/Gamma/Theta/Vega）...")
        pnl_attr = _eod_pnl_attribution(trade_date, db)
        if pnl_attr:
            print(f"  总盈亏: {pnl_attr['total']:+,.0f}元  "
                  f"Δ={pnl_attr['delta']:+,.0f}  Γ={pnl_attr['gamma']:+,.0f}  "
                  f"Θ={pnl_attr['theta']:+,.0f}  V={pnl_attr['vega']:+,.0f}  "
                  f"残差={pnl_attr['residual']:+,.0f}  "
                  f"解释率={pnl_attr['explained_ratio']:.0%}")
        else:
            print("  数据不足，跳过")

    # ── d. 回填5日前预测精度 ──────────────────────────────────────────
    backfill_result: dict | None = None
    if not model_skipped:
        print("[EOD] d. 回填5日前 GARCH 预测精度...")
        backfill_result = _backfill_forecast_accuracy(db, trade_date)
        if backfill_result:
            fd = backfill_result["forecast_date"]
            ep = backfill_result["error_pp"]
            tag = "（已填）" if backfill_result.get("already_filled") else "✓"
            print(f"  {tag} forecast_date={fd}  误差={ep:+.2f}pp")
        else:
            print("  数据不足，跳过")

    # ── e. 摘要报告 ───────────────────────────────────────────────────
    _print_eod_summary(
        trade_date, account_data, positions, n_trades, model_output,
        freshness=freshness, model_skipped=model_skipped, model_ok=model_ok,
        db=db, backfill=backfill_result,
    )


def run_model_only(trade_date: str | None = None) -> None:
    """
    仅重跑模型计算并写入 daily_model_output，跳过 TQ 和 Tushare。
    用于 eod 因数据延迟跳过模型输出后的手动补跑。
    """
    if not trade_date:
        trade_date = date.today().strftime("%Y%m%d")

    print(f"\n[model] 补跑模型输出 — {trade_date}")

    db = _open_db()

    freshness = _check_data_freshness(db, trade_date)
    if not freshness.is_fresh:
        logger.warning(
            "数据未更新至 %s（期货: %s, 期权: %s），继续运行但结果可能基于前一日数据",
            trade_date, freshness.futures_date, freshness.options_date,
        )
        print(f"  [警告] {_format_freshness_status(freshness)}")

    print("[model] 运行模型计算（GARCH / IV / VRP / Greeks）...")
    model_output = _eod_model_output(trade_date, db)
    model_ok = model_output is not None

    print()
    print(_format_freshness_status(freshness, model_skipped=False, model_ok=model_ok))

    if model_ok:
        print(
            f"  RV20={model_output['realized_vol_20d']*100:.2f}%"
            f"  GARCH={model_output['garch_forecast_vol']*100:.2f}%"
            f"  ATM_IV={model_output['atm_iv']*100:.2f}%" if model_output["atm_iv"] else ""
        )
        print(f"  信号: {model_output['signal']}")

        # P&L 归因
        print("[model] 计算 P&L 归因...")
        pnl_attr = _eod_pnl_attribution(trade_date, db)
        if pnl_attr:
            print(f"  总盈亏: {pnl_attr['total']:+,.0f}元  "
                  f"Δ={pnl_attr['delta']:+,.0f}  Γ={pnl_attr['gamma']:+,.0f}  "
                  f"Θ={pnl_attr['theta']:+,.0f}  V={pnl_attr['vega']:+,.0f}  "
                  f"残差={pnl_attr['residual']:+,.0f}")
    else:
        print("  模型输出未能写入（数据可能仍未更新）")


def cmd_model(args: argparse.Namespace) -> None:
    """补跑模型子命令。"""
    run_model_only(trade_date=args.date)


def cmd_eod(args: argparse.Namespace) -> None:
    """收盘后记录子命令。"""
    run_eod(
        trade_date=args.date,
        use_tq=not args.no_tq,
        update_market_data=not args.no_update,
    )


# ---------------------------------------------------------------------------
# ── NOTE 子命令 ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def save_daily_note(
    trade_date: str,
    db,
    market_observation: str = "",
    trade_rationale:    str = "",
    deviations:         str = "",
    lessons:            str = "",
) -> None:
    """写入每日交易笔记到 daily_notes 表（upsert）。"""
    row = {
        "trade_date":         trade_date,
        "market_observation": market_observation,
        "trade_rationale":    trade_rationale,
        "deviations":         deviations,
        "lessons":            lessons,
        "created_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    db.upsert_dataframe("daily_notes", pd.DataFrame([row]))
    print(f"  笔记已写入  →  daily_notes  ({trade_date})")


def _prompt(label: str, existing: str = "") -> str:
    """交互式单行输入（有预设值时显示）。"""
    if existing:
        print(f"  {label} [{existing}]: ", end="", flush=True)
    else:
        print(f"  {label}: ", end="", flush=True)
    try:
        val = input().strip()
        return val if val else existing
    except (EOFError, KeyboardInterrupt):
        return existing


def cmd_note(args: argparse.Namespace) -> None:
    """交易笔记子命令：CLI 参数或交互式输入，写入 daily_notes。"""
    trade_date = args.date or date.today().strftime("%Y%m%d")
    db = _open_db()

    # 读取已有笔记（方便修改）
    existing = {}
    try:
        row = db.query_df(
            f"SELECT * FROM daily_notes WHERE trade_date='{trade_date}'"
        )
        if row is not None and not row.empty:
            existing = row.iloc[0].to_dict()
            print(f"[note] 已找到 {trade_date} 的笔记，可在下方修改（直接回车保留原值）")
    except Exception:
        pass

    # 优先使用命令行参数，否则交互式输入
    if any([args.market, args.rationale, args.deviations, args.lessons]):
        market      = args.market      or existing.get("market_observation", "")
        rationale   = args.rationale   or existing.get("trade_rationale",    "")
        deviations  = args.deviations  or existing.get("deviations",         "")
        lessons     = args.lessons     or existing.get("lessons",             "")
    else:
        print(f"\n[note] 交易日 {trade_date} — 输入各项笔记（空行跳过保留原值）")
        market     = _prompt("市场观察",  existing.get("market_observation", ""))
        rationale  = _prompt("交易理由",  existing.get("trade_rationale",    ""))
        deviations = _prompt("偏离记录",  existing.get("deviations",         ""))
        lessons    = _prompt("复盘总结",  existing.get("lessons",             ""))

    save_daily_note(
        trade_date=trade_date,
        db=db,
        market_observation=market,
        trade_rationale=rationale,
        deviations=deviations,
        lessons=lessons,
    )


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="daily_record",
        description="每日数据记录：行情快照 / 收盘记录 / 交易笔记",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    # ── snapshot ──────────────────────────────────────────────────────
    sp = sub.add_parser("snapshot", help="盘中行情快照（写入 tq_snapshots）")
    sp.add_argument(
        "--symbols", nargs="+", metavar="SYMBOL",
        help="天勤合约代码列表，如 CFFEX.MO2604-P-7200。默认使用内置监控列表",
    )
    sp.set_defaults(func=cmd_snapshot)

    # ── eod ───────────────────────────────────────────────────────────
    ep = sub.add_parser("eod", help="收盘后完整记录（账户/持仓/成交/模型）")
    ep.add_argument(
        "--date",
        default=None,
        metavar="YYYYMMDD",
        help="指定交易日，默认今天",
    )
    ep.add_argument(
        "--no-tq",
        action="store_true",
        help="跳过 TQ 连接（仅运行模型计算）",
    )
    ep.add_argument(
        "--no-update",
        action="store_true",
        help="跳过 Tushare 增量数据更新",
    )
    ep.set_defaults(func=cmd_eod)

    # ── model ─────────────────────────────────────────────────────────
    mp = sub.add_parser("model", help="补跑模型输出（跳过 TQ/Tushare，仅写入 daily_model_output）")
    mp.add_argument(
        "--date",
        default=None,
        metavar="YYYYMMDD",
        help="指定交易日，默认今天",
    )
    mp.set_defaults(func=cmd_model)

    # ── note ──────────────────────────────────────────────────────────
    np_ = sub.add_parser("note", help="写交易日志（写入 daily_notes）")
    np_.add_argument("--date",       default=None, metavar="YYYYMMDD", help="交易日，默认今天")
    np_.add_argument("--market",     default=None, metavar="TEXT", help="市场观察")
    np_.add_argument("--rationale",  default=None, metavar="TEXT", help="交易理由")
    np_.add_argument("--deviations", default=None, metavar="TEXT", help="偏离记录")
    np_.add_argument("--lessons",    default=None, metavar="TEXT", help="复盘总结")
    np_.set_defaults(func=cmd_note)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
