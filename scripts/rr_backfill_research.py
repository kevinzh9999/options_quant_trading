"""
rr_backfill_research.py
-----------------------
用 options_daily 回算历史 25D Risk Reversal（RR），
并分析 RR 与日内回测 PnL 的关系。

研究范围：2026-02-04 ~ 2026-04-02（回测的 34 个交易日）

用法：
    python scripts/rr_backfill_research.py
    python scripts/rr_backfill_research.py --full_history   # 计算全量历史RR（2022-07起）
    python scripts/rr_backfill_research.py --no_backtest    # 只算RR，不跑回测
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager
from models.pricing.forward_price import calc_implied_forward
from models.pricing.implied_vol import calc_implied_vol
from models.pricing.greeks import calc_all_greeks
from scripts.vol_monitor import (
    _parse_mo, _T, _dte, RISK_FREE,
    calc_skew_table, calc_rr_bf, _interp_iv_at_delta,
)

# ---------------------------------------------------------------------------
# 回测日期（与CLAUDE.md中34天干净回测保持一致）
# ---------------------------------------------------------------------------
BACKTEST_DATES = [
    "20260204", "20260205", "20260206", "20260209", "20260210",
    "20260211", "20260212", "20260213", "20260225", "20260226",
    "20260227", "20260302", "20260303", "20260304", "20260305",
    "20260306", "20260309", "20260310", "20260311", "20260312",
    "20260313", "20260316", "20260317", "20260318", "20260319",
    "20260320", "20260323", "20260324", "20260325", "20260326",
    "20260327", "20260330", "20260331", "20260401", "20260402",
]

# 缺 20260224（MO数据有），但正式回测只到上面这些日期
# 注：实际backtest_dates是34天，如果shadow_trades有数据则用；否则调用run_day

IM_MULT = 200  # 点数→元


# ---------------------------------------------------------------------------
# Step 1: 构建 expire_map（所有 MO 合约的到期日映射）
# ---------------------------------------------------------------------------

def build_expire_map(db: DBManager) -> Dict[str, str]:
    """返回 {'2604': '20260417', ...}"""
    df = db.query_df(
        "SELECT DISTINCT ts_code, expire_date "
        "FROM options_contracts WHERE ts_code LIKE 'MO%'"
    )
    if df is None or df.empty:
        return {}
    result = {}
    for _, row in df.iterrows():
        p = _parse_mo(str(row["ts_code"]))
        if p and row["expire_date"]:
            result[p[0]] = str(row["expire_date"])
    return result


# ---------------------------------------------------------------------------
# Step 2: 构建当天期权链
# ---------------------------------------------------------------------------

def build_chain(db: DBManager, trade_date: str, expire_map: Dict[str, str]) -> pd.DataFrame:
    """加载当天所有 MO 合约，解析到期月/行权价/CP，过滤无效的到期月。"""
    df = db.query_df(
        f"SELECT ts_code, close, settle, volume, oi "
        f"FROM options_daily "
        f"WHERE ts_code LIKE 'MO%' AND trade_date='{trade_date}' AND close > 0"
    )
    if df is None or df.empty:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        p = _parse_mo(str(row["ts_code"]))
        if p is None:
            continue
        em, cp, strike = p
        ed = expire_map.get(em, "")
        if not ed:
            continue
        records.append({
            "ts_code": row["ts_code"],
            "expire_month": em,
            "expire_date": ed,
            "call_put": cp,
            "exercise_price": strike,
            "close": float(row["close"]),
            "volume": float(row["volume"]) if row["volume"] else 0.0,
            "oi": float(row["oi"]) if row["oi"] else 0.0,
        })
    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 3: 选近月（DTE >= min_dte）
# ---------------------------------------------------------------------------

def select_near_month(
    chain: pd.DataFrame,
    expire_map: Dict[str, str],
    trade_date: str,
    min_dte: int = 14,
) -> Optional[str]:
    """选 DTE >= min_dte 的最近到期月份，不足则换次月。"""
    if chain.empty:
        return None
    months = sorted(chain["expire_month"].unique())
    for m in months:
        ed = expire_map.get(m, "")
        if ed and _dte(ed, trade_date) >= min_dte:
            return m
    # fallback: 最近月
    return months[0] if months else None


# ---------------------------------------------------------------------------
# Step 4: 计算 PCP 隐含 Forward Price（或 fallback 到现货）
# ---------------------------------------------------------------------------

def get_forward_price(
    chain: pd.DataFrame,
    expire_month: str,
    expire_map: Dict[str, str],
    trade_date: str,
    spot_price: float,
    fut_prices: Dict[str, float],
) -> Tuple[float, str]:
    """
    计算 PCP 隐含远期价格。
    返回 (forward, method) 其中 method 是 'pcp' / 'futures' / 'spot'。
    """
    sub = chain[chain["expire_month"] == expire_month].copy()
    ed = expire_map.get(expire_month, "")
    T_val = _T(ed, trade_date) if ed else 0.033

    # 首选：从期货价（如有）作为参考，PCP 精细调整
    fut_price = fut_prices.get(expire_month, spot_price)

    try:
        fwd, n_est = calc_implied_forward(sub, T_val, RISK_FREE, fut_price)
        if n_est > 0:
            return fwd, "pcp"
    except Exception:
        pass

    # Fallback 1: 期货价
    if expire_month in fut_prices:
        return fut_prices[expire_month], "futures"

    # Fallback 2: 现货
    return spot_price, "spot"


# ---------------------------------------------------------------------------
# Step 5: 计算当天 25D RR
# ---------------------------------------------------------------------------

def calc_daily_rr(
    db: DBManager,
    trade_date: str,
    expire_map: Dict[str, str],
    spot_prices: Dict[str, float],
    all_fut_prices: Dict[str, Dict[str, float]],
) -> Optional[Dict]:
    """
    计算指定日期的 25D Risk Reversal。

    Returns:
        dict with keys: trade_date, near_month, dte, forward, method,
                        put_25d_iv, call_25d_iv, atm_iv, rr, bf,
                        spot_price, n_strikes
        None if data not available.
    """
    spot_price = spot_prices.get(trade_date)
    if spot_price is None:
        return None

    chain = build_chain(db, trade_date, expire_map)
    if chain.empty:
        return None

    near_month = select_near_month(chain, expire_map, trade_date)
    if near_month is None:
        return None

    fut_prices = all_fut_prices.get(trade_date, {})
    fwd, method = get_forward_price(chain, near_month, expire_map, trade_date, spot_price, fut_prices)

    expire_date = expire_map[near_month]
    dte_val = _dte(expire_date, trade_date)
    T_val = _T(expire_date, trade_date)

    # 计算 skew table（ATM ± 5档）
    skew = calc_skew_table(chain, near_month, fwd, expire_date, trade_date)
    if skew.empty:
        return None

    # 计算 RR/BF
    rr, bf = calc_rr_bf(skew)

    puts = skew[skew["cp"] == "P"].copy()
    calls = skew[skew["cp"] == "C"].copy()
    put_25d_iv = _interp_iv_at_delta(puts, -0.25)
    call_25d_iv = _interp_iv_at_delta(calls, 0.25)
    atm_iv = _interp_iv_at_delta(calls, 0.50)

    return {
        "trade_date": trade_date,
        "near_month": near_month,
        "dte": dte_val,
        "forward": round(fwd, 1),
        "method": method,
        "spot_price": round(spot_price, 1),
        "put_25d_iv": round(put_25d_iv * 100, 2),    # 转百分比
        "call_25d_iv": round(call_25d_iv * 100, 2),
        "atm_iv": round(atm_iv * 100, 2),
        "rr": round(rr * 100, 2),                    # 正值=Put skew（看跌偏向）
        "bf": round(bf * 100, 2),
        "n_strikes": len(skew) // 2,                 # 大约的行权价数
    }


# ---------------------------------------------------------------------------
# Step 6: 获取回测 PnL（调用 backtest_signals_day.run_day）
# ---------------------------------------------------------------------------

def get_backtest_pnl(
    db: DBManager,
    dates: List[str],
    symbols: List[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    逐日运行回测，返回每天每个品种的 PnL。
    columns: trade_date, symbol, pnl_pts, n_trades, win_rate
    """
    if symbols is None:
        symbols = ["IM", "IC"]

    try:
        from scripts.backtest_signals_day import run_day
    except ImportError as e:
        print(f"  [警告] 无法导入 run_day: {e}")
        return pd.DataFrame()

    rows = []
    for td in dates:
        for sym in symbols:
            try:
                trades = run_day(sym, td, db, verbose=verbose, slippage=0, version="auto")
                if not trades:
                    rows.append({"trade_date": td, "symbol": sym, "pnl_pts": 0.0,
                                 "n_trades": 0, "win_rate": None})
                    continue
                pnl = sum(t["pnl_pts"] for t in trades)
                n = len([t for t in trades if not t.get("partial")])
                wins = len([t for t in trades if t["pnl_pts"] > 0 and not t.get("partial")])
                wr = wins / n * 100 if n > 0 else None
                rows.append({"trade_date": td, "symbol": sym, "pnl_pts": round(pnl, 1),
                             "n_trades": n, "win_rate": round(wr, 1) if wr is not None else None})
            except Exception as e:
                if verbose:
                    print(f"  [警告] {sym} {td}: {e}")
                rows.append({"trade_date": td, "symbol": sym, "pnl_pts": None,
                             "n_trades": 0, "win_rate": None})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7: 辅助：加载现货价格 & 期货价格
# ---------------------------------------------------------------------------

def load_spot_prices(db: DBManager, dates: List[str]) -> Dict[str, float]:
    min_d, max_d = min(dates), max(dates)
    df = db.query_df(
        f"SELECT trade_date, close FROM index_daily "
        f"WHERE ts_code='000852.SH' AND trade_date>='{min_d}' AND trade_date<='{max_d}'"
    )
    if df is None or df.empty:
        return {}
    return dict(zip(df["trade_date"].astype(str), df["close"].astype(float)))


def load_futures_prices(db: DBManager, dates: List[str]) -> Dict[str, Dict[str, float]]:
    """返回 {trade_date: {expire_month: price}}"""
    import re as _re
    min_d, max_d = min(dates), max(dates)
    df = db.query_df(
        f"SELECT trade_date, ts_code, close FROM futures_daily "
        f"WHERE ts_code LIKE 'IM2%' AND ts_code LIKE '%.CFX' "
        f"AND trade_date>='{min_d}' AND trade_date<='{max_d}' AND close > 0"
    )
    result: Dict[str, Dict[str, float]] = {}
    if df is None or df.empty:
        return result
    pat = _re.compile(r"IM(\d{4})\.CFX")
    for _, row in df.iterrows():
        m = pat.match(str(row["ts_code"]))
        if not m:
            continue
        td = str(row["trade_date"])
        em = m.group(1)
        result.setdefault(td, {})[em] = float(row["close"])
    return result


# ---------------------------------------------------------------------------
# Step 8: 辅助：加载日线技术指标（Momentum/Volatility/Volume）
# ---------------------------------------------------------------------------

def load_daily_mvq(db: DBManager, dates: List[str]) -> pd.DataFrame:
    """
    加载 000852.SH 日线数据，计算用于相关性分析的日频指标：
    - momentum_5d: 5日涨幅
    - realized_vol_5d: 5日实现波动率
    - volume_ratio: 成交量 / 20日均量
    """
    min_d = min(dates)
    # 多拉 30 天用于滚动计算
    start_ext = "20260101"
    df = db.query_df(
        f"SELECT trade_date, close, volume FROM index_daily "
        f"WHERE ts_code='000852.SH' AND trade_date>='{start_ext}' "
        f"ORDER BY trade_date"
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    df["ret"] = df["close"].pct_change()
    df["momentum_5d"] = df["close"].pct_change(5) * 100
    df["realized_vol_5d"] = df["ret"].rolling(5).std() * np.sqrt(252) * 100
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["vol_ma20"].where(df["vol_ma20"] > 0)

    return df[df["trade_date"].isin(dates)][
        ["trade_date", "momentum_5d", "realized_vol_5d", "volume_ratio"]
    ].copy()


# ---------------------------------------------------------------------------
# Step 9: 分析函数
# ---------------------------------------------------------------------------

def analyze_rr_pnl_by_quantile(rr_df: pd.DataFrame, pnl_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 RR 分位（Q1~Q4）分组，统计 IM/IC 的平均 PnL。
    """
    merged = pnl_df.merge(rr_df[["trade_date", "rr"]], on="trade_date", how="left")
    merged = merged.dropna(subset=["rr", "pnl_pts"])

    if merged.empty or len(merged["rr"].dropna()) < 4:
        print("  [警告] RR数据不足，无法分位分析")
        return pd.DataFrame()

    merged["rr_q"] = pd.qcut(merged["rr"], q=4, labels=["Q1(最低)", "Q2", "Q3", "Q4(最高)"])

    rows = []
    for sym in merged["symbol"].unique():
        sub = merged[merged["symbol"] == sym]
        for q, g in sub.groupby("rr_q", observed=True):
            rows.append({
                "symbol": sym,
                "rr_quantile": q,
                "n_days": len(g),
                "avg_pnl": round(g["pnl_pts"].mean(), 1),
                "total_pnl": round(g["pnl_pts"].sum(), 1),
                "win_days": int((g["pnl_pts"] > 0).sum()),
                "avg_rr": round(g["rr"].mean(), 2),
            })
    return pd.DataFrame(rows)


def analyze_delta_rr_pnl(rr_df: pd.DataFrame, pnl_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 RR 日变化（delta_RR）分组，统计 PnL。
    """
    rr_sorted = rr_df.sort_values("trade_date").copy()
    rr_sorted["delta_rr"] = rr_sorted["rr"].diff()

    merged = pnl_df.merge(
        rr_sorted[["trade_date", "rr", "delta_rr"]], on="trade_date", how="left"
    )
    merged = merged.dropna(subset=["delta_rr", "pnl_pts"])

    if merged.empty or len(merged["delta_rr"].dropna()) < 4:
        return pd.DataFrame()

    merged["delta_rr_q"] = pd.qcut(
        merged["delta_rr"], q=4,
        labels=["Q1(急降)", "Q2", "Q3", "Q4(急升)"]
    )

    rows = []
    for sym in merged["symbol"].unique():
        sub = merged[merged["symbol"] == sym]
        for q, g in sub.groupby("delta_rr_q", observed=True):
            rows.append({
                "symbol": sym,
                "delta_rr_quantile": q,
                "n_days": len(g),
                "avg_pnl": round(g["pnl_pts"].mean(), 1),
                "total_pnl": round(g["pnl_pts"].sum(), 1),
                "avg_delta_rr": round(g["delta_rr"].mean(), 2),
            })
    return pd.DataFrame(rows)


def analyze_rr_correlations(
    rr_df: pd.DataFrame, pnl_df: pd.DataFrame, mvq_df: pd.DataFrame
) -> pd.DataFrame:
    """
    计算 RR / delta_RR 与 M/V/Q 及 PnL 的 Pearson 相关矩阵。
    """
    rr_sorted = rr_df.sort_values("trade_date").copy()
    rr_sorted["delta_rr"] = rr_sorted["rr"].diff()

    # PnL 合并（IM + IC 分别）
    base = rr_sorted.merge(mvq_df, on="trade_date", how="left")

    for sym in ["IM", "IC"]:
        sub_pnl = pnl_df[pnl_df["symbol"] == sym][["trade_date", "pnl_pts"]].copy()
        sub_pnl = sub_pnl.rename(columns={"pnl_pts": f"pnl_{sym.lower()}"})
        base = base.merge(sub_pnl, on="trade_date", how="left")

    cols = ["rr", "delta_rr", "atm_iv", "momentum_5d", "realized_vol_5d",
            "volume_ratio", "pnl_im", "pnl_ic"]
    existing = [c for c in cols if c in base.columns]
    corr_matrix = base[existing].corr(method="pearson").round(3)
    return corr_matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="25D Risk Reversal 历史回算与 PnL 分析")
    parser.add_argument("--full_history", action="store_true",
                        help="计算全量历史RR（从有MO数据的最早日期开始）")
    parser.add_argument("--no_backtest", action="store_true",
                        help="只计算RR，不运行回测（节省时间）")
    parser.add_argument("--symbols", default="IM,IC",
                        help="回测品种，逗号分隔（默认 IM,IC）")
    args = parser.parse_args()

    db = DBManager(ConfigLoader().get_db_path())
    symbols = [s.strip() for s in args.symbols.split(",")]

    print("=" * 70)
    print("  25D Risk Reversal 历史回算研究")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 确定要计算的日期列表
    # ------------------------------------------------------------------
    if args.full_history:
        # 全量：从有 MO 数据的所有交易日
        dates_df = db.query_df(
            "SELECT DISTINCT trade_date FROM options_daily "
            "WHERE ts_code LIKE 'MO%' ORDER BY trade_date"
        )
        compute_dates = dates_df["trade_date"].astype(str).tolist() if dates_df is not None else BACKTEST_DATES
        print(f"\n  全量模式：共 {len(compute_dates)} 个交易日")
    else:
        compute_dates = BACKTEST_DATES
        print(f"\n  回测期间模式：{len(compute_dates)} 个交易日（2026-02 ~ 2026-04）")

    # ------------------------------------------------------------------
    # Step 0: 加载基础数据
    # ------------------------------------------------------------------
    print("\n[Step 0] 加载基础数据...")

    expire_map = build_expire_map(db)
    print(f"  expire_map: {len(expire_map)} 个合约月份")

    spot_prices = load_spot_prices(db, compute_dates)
    print(f"  现货价格: {len(spot_prices)} 天")

    all_fut_prices = load_futures_prices(db, compute_dates)
    print(f"  期货价格: {len(all_fut_prices)} 天")

    # ------------------------------------------------------------------
    # Step 1: 逐日计算 RR
    # ------------------------------------------------------------------
    print(f"\n[Step 1] 逐日计算 25D RR...")

    rr_rows = []
    failed_dates = []
    for i, td in enumerate(compute_dates):
        if (i + 1) % 10 == 0 or (i + 1) == len(compute_dates):
            print(f"  进度: {i+1}/{len(compute_dates)}  ({td})")
        result = calc_daily_rr(db, td, expire_map, spot_prices, all_fut_prices)
        if result is not None:
            rr_rows.append(result)
        else:
            failed_dates.append(td)

    rr_df = pd.DataFrame(rr_rows)
    print(f"\n  成功计算 {len(rr_df)} 天 RR，失败 {len(failed_dates)} 天")
    if failed_dates:
        print(f"  失败日期: {failed_dates[:10]}{'...' if len(failed_dates) > 10 else ''}")

    if rr_df.empty:
        print("  [错误] 没有 RR 数据，退出")
        return

    # 计算 delta_RR
    rr_df = rr_df.sort_values("trade_date").reset_index(drop=True)
    rr_df["delta_rr"] = rr_df["rr"].diff()

    # ------------------------------------------------------------------
    # 输出 RR 时间序列
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [输出 1] RR 时间序列")
    print("=" * 70)
    display_cols = ["trade_date", "near_month", "dte", "spot_price", "forward",
                    "method", "put_25d_iv", "call_25d_iv", "atm_iv", "rr", "delta_rr"]
    display_df = rr_df[[c for c in display_cols if c in rr_df.columns]]
    print(display_df.to_string(index=False))

    # 统计汇总
    print(f"\n  RR 统计（百分比）:")
    print(f"  均值={rr_df['rr'].mean():.2f}%  中位数={rr_df['rr'].median():.2f}%  "
          f"标准差={rr_df['rr'].std():.2f}%")
    print(f"  最小={rr_df['rr'].min():.2f}%  最大={rr_df['rr'].max():.2f}%")
    print(f"  Q25={rr_df['rr'].quantile(0.25):.2f}%  Q75={rr_df['rr'].quantile(0.75):.2f}%")
    print(f"\n  ATM IV 统计:")
    print(f"  均值={rr_df['atm_iv'].mean():.2f}%  中位数={rr_df['atm_iv'].median():.2f}%  "
          f"标准差={rr_df['atm_iv'].std():.2f}%")

    # ------------------------------------------------------------------
    # Step 2: 回测 PnL（如需要）
    # ------------------------------------------------------------------
    if not args.no_backtest:
        backtest_dates = [d for d in BACKTEST_DATES if d in set(rr_df["trade_date"].tolist())]
        print(f"\n[Step 2] 运行回测 {symbols}，共 {len(backtest_dates)} 天...")
        print("  (这可能需要几分钟，请耐心等待...)")

        pnl_df = get_backtest_pnl(db, backtest_dates, symbols=symbols, verbose=False)

        if pnl_df.empty:
            print("  [警告] 回测结果为空，跳过 PnL 分析")
        else:
            # 回测汇总
            print("\n" + "=" * 70)
            print("  [输出 2] 回测 PnL 汇总")
            print("=" * 70)
            for sym in symbols:
                sub = pnl_df[pnl_df["symbol"] == sym].dropna(subset=["pnl_pts"])
                if sub.empty:
                    continue
                total = sub["pnl_pts"].sum()
                avg = sub["pnl_pts"].mean()
                wins = (sub["pnl_pts"] > 0).sum()
                n = len(sub)
                print(f"  {sym}: {n}天  总PnL={total:+.0f}pt  "
                      f"均PnL={avg:+.1f}pt  盈利天={wins}/{n}")

            # ------------------------------------------------------------------
            # 分析 1: RR 分位 × PnL
            # ------------------------------------------------------------------
            rr_bt = rr_df[rr_df["trade_date"].isin(backtest_dates)].copy()
            q_analysis = analyze_rr_pnl_by_quantile(rr_bt, pnl_df)

            if not q_analysis.empty:
                print("\n" + "=" * 70)
                print("  [输出 3] RR 分位 × PnL（RR 越高 = Put skew 越强 = 市场越悲观）")
                print("=" * 70)
                print(q_analysis.to_string(index=False))

                # 计算 RR 与 PnL 的单调性
                print("\n  解读：")
                for sym in symbols:
                    sub_q = q_analysis[q_analysis["symbol"] == sym]
                    if sub_q.empty:
                        continue
                    pnls = sub_q.sort_values("rr_quantile")["avg_pnl"].tolist()
                    monotone = "单调递增" if all(p2 > p1 for p1, p2 in zip(pnls, pnls[1:])) else \
                               "单调递减" if all(p2 < p1 for p1, p2 in zip(pnls, pnls[1:])) else \
                               "非单调"
                    print(f"  {sym}: RR-PnL 关系 = {monotone}  "
                          f"分位PnL = {[f'{p:+.0f}' for p in pnls]}")

            # ------------------------------------------------------------------
            # 分析 2: delta_RR × PnL
            # ------------------------------------------------------------------
            delta_analysis = analyze_delta_rr_pnl(rr_bt, pnl_df)

            if not delta_analysis.empty:
                print("\n" + "=" * 70)
                print("  [输出 4] delta_RR 分位 × PnL（delta_RR 急升 = 悲观情绪加剧）")
                print("=" * 70)
                print(delta_analysis.to_string(index=False))

            # ------------------------------------------------------------------
            # 分析 3: 相关性矩阵
            # ------------------------------------------------------------------
            mvq_df = load_daily_mvq(db, backtest_dates)

            print("\n" + "=" * 70)
            print("  [输出 5] Pearson 相关矩阵（RR / delta_RR / M / V / Q / PnL）")
            print("=" * 70)
            corr = analyze_rr_correlations(rr_bt, pnl_df, mvq_df)
            print(corr.to_string())

            # 高亮强相关
            print("\n  强相关对（|r| >= 0.3）：")
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    r = corr.iloc[i, j]
                    if abs(r) >= 0.3:
                        print(f"  {corr.columns[i]} vs {corr.columns[j]}: r={r:.3f}")

            # ------------------------------------------------------------------
            # 分析 4: RR 极值日的表现
            # ------------------------------------------------------------------
            print("\n" + "=" * 70)
            print("  [输出 6] RR 极值日（Top/Bottom 5）")
            print("=" * 70)
            rr_pnl_merged = pnl_df.merge(
                rr_bt[["trade_date", "rr", "delta_rr", "atm_iv", "spot_price"]],
                on="trade_date", how="left"
            )

            # 只看 IM
            im_data = rr_pnl_merged[rr_pnl_merged["symbol"] == "IM"].dropna(subset=["rr"])

            print("\n  Put skew 最强的5天（RR 最高，市场最悲观）：")
            top5 = im_data.nlargest(5, "rr")[
                ["trade_date", "rr", "delta_rr", "atm_iv", "spot_price", "pnl_pts"]
            ]
            print(top5.to_string(index=False))

            print("\n  Put skew 最弱的5天（RR 最低，市场最中性/乐观）：")
            bot5 = im_data.nsmallest(5, "rr")[
                ["trade_date", "rr", "delta_rr", "atm_iv", "spot_price", "pnl_pts"]
            ]
            print(bot5.to_string(index=False))

    # ------------------------------------------------------------------
    # 综合结论
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [综合结论]")
    print("=" * 70)
    print(f"\n  1. 数据覆盖：{len(rr_df)} 天 RR 成功计算")
    print(f"     正向方法（PCP）使用率：{(rr_df['method']=='pcp').sum()/len(rr_df)*100:.0f}%")
    print(f"     期货 fallback：{(rr_df['method']=='futures').sum()} 天")
    print(f"     现货 fallback：{(rr_df['method']=='spot').sum()} 天")

    print(f"\n  2. RR 水平（高RR=Put skew强=看跌偏向）")
    print(f"     回测期间均值：{rr_df['rr'].mean():.2f}%，范围 [{rr_df['rr'].min():.2f}%, {rr_df['rr'].max():.2f}%]")
    print(f"     RR > 5%（强Put skew）天数：{(rr_df['rr'] > 5).sum()}")
    print(f"     RR < 2%（弱Put skew）天数：{(rr_df['rr'] < 2).sum()}")

    print(f"\n  3. ATM IV 与 RR 相关：{rr_df[['atm_iv','rr']].corr().iloc[0,1]:.3f}")

    if not args.no_backtest and not pnl_df.empty:
        # Simple correlation: RR with IM PnL
        merged_simple = pnl_df[pnl_df["symbol"] == "IM"].merge(
            rr_df[["trade_date", "rr"]], on="trade_date", how="inner"
        ).dropna()
        if len(merged_simple) >= 5:
            r_im = merged_simple[["rr", "pnl_pts"]].corr().iloc[0, 1]
            print(f"\n  4. IM PnL 与 RR Pearson 相关系数：r = {r_im:.3f}")
            if abs(r_im) >= 0.3:
                direction = "负相关（Put skew强时IM趋势策略表现弱）" if r_im < 0 else "正相关"
                print(f"     结论：{direction}")
            else:
                print(f"     结论：相关性较弱（|r| < 0.3），RR 对 IM 日内 PnL 影响有限")

    print()

    # ------------------------------------------------------------------
    # 保存到 CSV（供进一步分析）
    # ------------------------------------------------------------------
    output_path = os.path.join(ROOT, "tmp", "rr_history.csv")
    os.makedirs(os.path.join(ROOT, "tmp"), exist_ok=True)
    rr_df.to_csv(output_path, index=False)
    print(f"  RR 历史数据已保存到：{output_path}")
    print()


if __name__ == "__main__":
    main()
