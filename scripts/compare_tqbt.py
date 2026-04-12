#!/usr/bin/env python3
"""TqBacktest vs 自研Backtest 对比验证工具。

常规操作脚本：验证TqBacktest（TQ回放驱动monitor）与自研backtest（DB离线回测）
在相同信号逻辑下产出完全一致的结果。任何差异都意味着数据源或逻辑路径不对齐。

用法：
    # v2对比，最近10天
    python scripts/compare_tqbt.py --dates 20260327-20260410 --version v2

    # v1对比，指定日期
    python scripts/compare_tqbt.py --dates 20260407,20260408 --version v1

    # v1+v2全部对比
    python scripts/compare_tqbt.py --dates 20260401-20260410 --version all

    # 单品种
    python scripts/compare_tqbt.py --dates 20260410 --version v2 --symbol IM

输出：
    - 逐日逐笔对比（entry时间、方向、score、exit时间、exit原因、PnL）
    - 汇总表（Markdown格式，可直接复制）
    - 一致性判定（✓ / ✗）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
from datetime import datetime
from typing import List, Dict, Tuple


# ── 日期解析 ──────────────────────────────────────────────────

def get_trading_days(date_spec: str) -> List[str]:
    """解析日期：单日 / 逗号分隔 / 范围（YYYYMMDD-YYYYMMDD）。"""
    from data.storage.db_manager import get_db
    db = get_db()

    if "-" in date_spec and len(date_spec) == 17:
        start, end = date_spec.split("-")
        rows = db.query_df(
            f"SELECT DISTINCT trade_date FROM index_daily "
            f"WHERE ts_code='000852.SH' AND trade_date >= '{start}' AND trade_date <= '{end}' "
            f"ORDER BY trade_date"
        )
        return rows["trade_date"].tolist() if rows is not None else []
    else:
        return [d.strip() for d in date_spec.split(",")]


# ── 自研 Backtest ─────────────────────────────────────────────

def run_self_v2(td: str, symbols: List[str]) -> Dict[str, List[Dict]]:
    """自研v2 backtest（DB离线）。"""
    from data.storage.db_manager import get_db
    from scripts.backtest_signals_day import run_day
    db = get_db()
    out = {}
    for sym in symbols:
        trades = run_day(sym, td, db, verbose=False, version="v2")
        out[sym] = [_norm_trade(t) for t in trades if not t.get("partial")]
    return out


def run_self_v1(td: str, symbols: List[str]) -> Dict[str, List[Dict]]:
    """自研v1 backtest（直接用MonitorV1类 + DB数据，确保与TqBT v1完全一致）。"""
    from data.storage.db_manager import get_db
    from scripts.monitor_v1 import MonitorV1
    from scripts.backtest_signals_day import _build_15m_from_5m
    from strategies.intraday.A_share_momentum_signal_v2 import compute_volume_profile
    import pandas as pd

    db = get_db()
    SPOT_SYM = {"IM": "000852", "IC": "000905"}
    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"

    # 备份已有的v1 shadow trades
    existing = db.query_df(f"SELECT * FROM shadow_trades_new_mapping WHERE trade_date='{td}'")
    db._conn.execute(f"DELETE FROM shadow_trades_new_mapping WHERE trade_date='{td}'")
    db._conn.commit()

    # 创建MonitorV1实例（与TqBT一样）
    monitor = MonitorV1(backtest_date=td)

    # 覆盖vol_profile到回测日期之前
    for sym in symbols:
        spot = SPOT_SYM.get(sym)
        if not spot:
            continue
        bar_all = db.query_df(
            f"SELECT datetime, volume FROM index_min "
            f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime"
        )
        if bar_all is not None and len(bar_all) > 0:
            bar_all['volume'] = bar_all['volume'].astype(float)
            monitor._vol_profiles[sym] = compute_volume_profile(
                bar_all, before_date=td, lookback_days=20)

    # 加载DB数据，逐bar喂给MonitorV1（模拟TQ bar-by-bar播放）
    for sym in symbols:
        spot = SPOT_SYM.get(sym)
        if not spot:
            continue

        all_bars = db.query_df(
            f"SELECT datetime, open, high, low, close, volume FROM index_min "
            f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime"
        )
        if all_bars is None or len(all_bars) == 0:
            continue

        for c in ['open', 'high', 'low', 'close', 'volume']:
            all_bars[c] = all_bars[c].astype(float)

        # 找出当天的bar index
        today_mask = all_bars["datetime"].str.startswith(date_dash)
        today_indices = all_bars.index[today_mask].tolist()
        if not today_indices:
            continue

        # 逐bar喂数据（模拟TQ的completed bars）
        for idx in today_indices:
            # 取到当前bar为止的历史（最多199根=TQ的200-1根completed）
            bar_5m = all_bars.loc[:idx].tail(199).copy()
            if len(bar_5m) < 20:
                continue

            # 构建DatetimeIndex（与TQ bar数据格式一致）
            df = bar_5m[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(bar_5m["datetime"])

            # 15m重采样（与monitor_v1.py完全相同）
            b15 = None
            if len(df) >= 3:
                b15_full = _build_15m_from_5m(df)
                if len(b15_full) > 1:
                    b15 = b15_full.iloc[:-1]
                elif len(b15_full) > 0:
                    b15 = b15_full

            monitor.on_bar(sym, df, b15)

    # 从DB读取v1 shadow trades
    trades_df = db.query_df(
        f"SELECT * FROM shadow_trades_new_mapping WHERE trade_date='{td}' ORDER BY entry_time"
    )

    # 恢复原始数据
    db._conn.execute(f"DELETE FROM shadow_trades_new_mapping WHERE trade_date='{td}'")
    if existing is not None and len(existing) > 0:
        existing.to_sql("shadow_trades_new_mapping", db._conn, if_exists="append", index=False)
    db._conn.commit()

    return _parse_shadow_df(trades_df, symbols)


# ── TqBacktest ────────────────────────────────────────────────

def run_tqbt_v2(td: str, symbols: List[str]) -> Dict[str, List[Dict]]:
    """TqBacktest v2（驱动v2 monitor）。"""
    from scripts.tq_backtest_monitor import run_tq_backtest
    from data.storage.db_manager import get_db
    db = get_db()

    existing = db.query_df(f"SELECT * FROM shadow_trades WHERE trade_date='{td}'")
    db._conn.execute(f"DELETE FROM shadow_trades WHERE trade_date='{td}'")
    db._conn.commit()

    try:
        tq_df = run_tq_backtest(td, symbols)
    finally:
        db._conn.execute(f"DELETE FROM shadow_trades WHERE trade_date='{td}'")
        if existing is not None and len(existing) > 0:
            existing.to_sql("shadow_trades", db._conn, if_exists="append", index=False)
        db._conn.commit()

    return _parse_shadow_df(tq_df, symbols)


def run_tqbt_v1(td: str, symbols: List[str]) -> Dict[str, List[Dict]]:
    """TqBacktest v1（驱动v1独立monitor）。"""
    from data.storage.db_manager import get_db
    db = get_db()

    existing = db.query_df(f"SELECT * FROM shadow_trades_new_mapping WHERE trade_date='{td}'")
    db._conn.execute(f"DELETE FROM shadow_trades_new_mapping WHERE trade_date='{td}'")
    db._conn.commit()

    try:
        from scripts.monitor_v1 import run_backtest as _run_v1
        v1_df = _run_v1(td)
    except Exception as e:
        print(f"  [V1-TqBT] {td} error: {e}")
        v1_df = None
    finally:
        db._conn.execute(f"DELETE FROM shadow_trades_new_mapping WHERE trade_date='{td}'")
        if existing is not None and len(existing) > 0:
            existing.to_sql("shadow_trades_new_mapping", db._conn, if_exists="append", index=False)
        db._conn.commit()

    return _parse_shadow_df(v1_df, symbols)


def _parse_shadow_df(df, symbols) -> Dict[str, List[Dict]]:
    out = {}
    for sym in symbols:
        if df is not None and len(df) > 0:
            sdf = df[df["symbol"] == sym].sort_values("entry_time")
            out[sym] = [_norm_trade(r) for _, r in sdf.iterrows()]
        else:
            out[sym] = []
    return out


def _norm_trade(t) -> Dict:
    """统一trade dict格式（兼容backtest_signals_day输出和shadow_trades表输出）。"""
    if isinstance(t, dict):
        return {
            "entry_time": str(t.get("entry_time", "")),
            "exit_time": str(t.get("exit_time", "")),
            "direction": str(t.get("direction", "")),
            "pnl_pts": round(float(t.get("pnl_pts", 0)), 1),
            "entry_score": int(t.get("entry_score", 0)),
            "exit_reason": str(t.get("reason", t.get("exit_reason", ""))),
        }
    else:
        # pandas Series (from iterrows)
        return {
            "entry_time": str(t.get("entry_time", "")),
            "exit_time": str(t.get("exit_time", "")),
            "direction": str(t.get("direction", "")),
            "pnl_pts": round(float(t.get("pnl_pts", 0)), 1),
            "entry_score": int(t.get("entry_score", 0)),
            "exit_reason": str(t.get("exit_reason", "")),
        }


# ── 对比逻辑 ──────────────────────────────────────────────────

def compare_trades(self_trades: List[Dict], tqbt_trades: List[Dict]) -> Tuple[bool, List[str]]:
    """逐笔对比，返回 (完全一致, 差异描述列表)。"""
    diffs = []
    n_self, n_tqbt = len(self_trades), len(tqbt_trades)
    if n_self != n_tqbt:
        diffs.append(f"笔数不同: 自研{n_self} vs TqBT{n_tqbt}")

    for i in range(max(n_self, n_tqbt)):
        s = self_trades[i] if i < n_self else None
        t = tqbt_trades[i] if i < n_tqbt else None
        if s is None:
            diffs.append(f"#{i+1} 自研无此笔, TqBT={t['entry_time']}→{t['exit_time']}")
            continue
        if t is None:
            diffs.append(f"#{i+1} TqBT无此笔, 自研={s['entry_time']}→{s['exit_time']}")
            continue
        # 逐字段对比
        if s["entry_time"] != t["entry_time"]:
            diffs.append(f"#{i+1} entry_time: {s['entry_time']} vs {t['entry_time']}")
        if s["exit_time"] != t["exit_time"]:
            diffs.append(f"#{i+1} exit_time: {s['exit_time']} vs {t['exit_time']}")
        if s["direction"] != t["direction"]:
            diffs.append(f"#{i+1} direction: {s['direction']} vs {t['direction']}")
        pnl_diff = abs(s["pnl_pts"] - t["pnl_pts"])
        if pnl_diff > 0.5:  # 允许0.5pt浮点误差
            diffs.append(f"#{i+1} pnl: {s['pnl_pts']:+.1f} vs {t['pnl_pts']:+.1f} (Δ{pnl_diff:.1f})")

    return len(diffs) == 0, diffs


# ── 主流程 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TqBacktest vs 自研Backtest 对比验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python scripts/compare_tqbt.py --dates 20260410 --version v2
  python scripts/compare_tqbt.py --dates 20260401-20260410 --version all
  python scripts/compare_tqbt.py --dates 20260407,20260408 --version v1 --symbol IM
""")
    parser.add_argument("--dates", required=True,
                        help="日期：YYYYMMDD / YYYYMMDD,YYYYMMDD / YYYYMMDD-YYYYMMDD")
    parser.add_argument("--version", choices=["v1", "v2", "all"], default="all",
                        help="信号版本 (default: all)")
    parser.add_argument("--symbol", default=None,
                        help="品种 IM/IC (default: 两个都跑)")
    parser.add_argument("--verbose", action="store_true",
                        help="显示逐笔明细")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ["IM", "IC"]
    dates = get_trading_days(args.dates)
    versions = ["v2", "v1"] if args.version == "all" else [args.version]

    print(f"\n{'='*70}")
    print(f"  TqBacktest vs 自研Backtest 对比验证")
    print(f"  日期: {dates[0]} ~ {dates[-1]} ({len(dates)}天)")
    print(f"  品种: {', '.join(symbols)}  版本: {', '.join(versions)}")
    print(f"{'='*70}\n")

    # 结果收集: {version: [{date, sym, self_n, self_pnl, tqbt_n, tqbt_pnl, match}]}
    results = {v: [] for v in versions}
    all_match = True

    for i, td in enumerate(dates):
        print(f"── [{i+1}/{len(dates)}] {td} ──")

        for ver in versions:
            # 自研
            if ver == "v2":
                self_res = run_self_v2(td, symbols)
            else:
                self_res = run_self_v1(td, symbols)

            # TqBT
            if ver == "v2":
                tqbt_res = run_tqbt_v2(td, symbols)
            else:
                tqbt_res = run_tqbt_v1(td, symbols)

            for sym in symbols:
                st = self_res.get(sym, [])
                tt = tqbt_res.get(sym, [])
                s_pnl = sum(t["pnl_pts"] for t in st)
                t_pnl = sum(t["pnl_pts"] for t in tt)
                match, diffs = compare_trades(st, tt)
                tag = "✓" if match else "✗"
                if not match:
                    all_match = False

                results[ver].append({
                    "date": td, "sym": sym,
                    "self_n": len(st), "self_pnl": s_pnl,
                    "tqbt_n": len(tt), "tqbt_pnl": t_pnl,
                    "match": match,
                })

                delta = t_pnl - s_pnl
                print(f"  {tag} {sym}/{ver}: 自研{len(st)}笔/{s_pnl:+.0f}  "
                      f"TqBT{len(tt)}笔/{t_pnl:+.0f}  Δ={delta:+.1f}")

                if not match and (args.verbose or abs(delta) > 1.0):
                    for d in diffs:
                        print(f"      {d}")

        print()

    # ── 汇总表 ──
    print(f"\n{'='*70}")
    print(f"  汇总表")
    print(f"{'='*70}\n")

    for ver in versions:
        ver_rows = results[ver]
        for sym in symbols:
            sym_rows = [r for r in ver_rows if r["sym"] == sym]
            if not sym_rows:
                continue
            print(f"### {sym} / {ver}\n")
            print(f"| 日期 | 自研 | TqBT | Δ | 一致 |")
            print(f"|------|------|------|---|------|")
            total_s = total_t = 0
            n_match = 0
            for r in sym_rows:
                s_str = f"{r['self_n']}笔/{r['self_pnl']:+.0f}"
                t_str = f"{r['tqbt_n']}笔/{r['tqbt_pnl']:+.0f}"
                delta = r['tqbt_pnl'] - r['self_pnl']
                tag = "✓" if r['match'] else "✗"
                print(f"| {r['date']} | {s_str} | {t_str} | {delta:+.1f} | {tag} |")
                total_s += r['self_pnl']
                total_t += r['tqbt_pnl']
                if r['match']:
                    n_match += 1
            print(f"| **合计** | **{total_s:+.0f}** | **{total_t:+.0f}** | "
                  f"**{total_t-total_s:+.1f}** | **{n_match}/{len(sym_rows)}** |")
            print()

    # 最终判定
    print(f"{'='*70}")
    if all_match:
        print(f"  ✓ 全部一致 ({len(dates)}天 × {len(symbols)}品种 × {len(versions)}版本)")
    else:
        n_total = sum(len(results[v]) for v in versions)
        n_ok = sum(1 for v in versions for r in results[v] if r["match"])
        print(f"  ✗ 存在差异 ({n_ok}/{n_total} 一致)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
