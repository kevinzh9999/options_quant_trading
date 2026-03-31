"""
波动率套利策略回测脚本。

用法:
    python -m strategies.vol_arb.backtest --start 20230101 --end 20260317
    python -m strategies.vol_arb.backtest --start 20230101 --end 20260317 --attribution
    python -m strategies.vol_arb.backtest --start 20230101 --end 20260317 --sensitivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_vol_arb_backtest(
    start_date: str = "20230101",
    end_date: str = "20260317",
    initial_capital: float = 5_000_000,
    config_overrides: dict | None = None,
    quiet: bool = False,
):
    """Execute vol_arb strategy backtest. Returns (report, strategy)."""
    from config.config_loader import ConfigLoader
    from data.storage.db_manager import DBManager
    from backtest.data_feed import DataFeed
    from backtest.broker import SimBroker
    from backtest.engine import BacktestEngine
    from strategies.vol_arb.strategy import VolArbStrategy, VolArbConfig

    config = ConfigLoader()
    db = DBManager(config.get_db_path())

    # Include "MO.CFX" so that DataFeed triggers options data loading (MO prefix → load
    # options_daily + options_contracts tables), and the engine injects options_chain into
    # market_data for the strategy.
    # Include "IC.CFX" for GARCH training — IC has data from 2015, much longer than IM (2022-07).
    symbols = ["IM.CFX", "IC.CFX", "MO.CFX"]
    feed = DataFeed(db, start_date, end_date, symbols)
    # MO/IO option tick = 0.2 points; slippage ~1 tick is realistic
    broker = SimBroker(initial_capital=initial_capital, slippage_points=0.2)
    engine = BacktestEngine(feed, broker)

    vol_config = VolArbConfig(
        strategy_id="vol_arb_IM",
        universe=["IM.CFX", "IC.CFX", "MO.CFX"],
        enabled=True,
        dry_run=False,
    )
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(vol_config, k):
                setattr(vol_config, k, v)

    strategy = VolArbStrategy(vol_config, db_manager=db)
    engine.add_strategy(strategy)

    if not quiet:
        print(f"[vol_arb backtest] {start_date} ~ {end_date}  capital={initial_capital:,.0f}")
    report = engine.run(show_progress=not quiet)
    if not quiet:
        report.print_report()
    return report, strategy


# ---------------------------------------------------------------------------
# Attribution analysis
# ---------------------------------------------------------------------------

def print_attribution(strategy, report=None) -> None:
    """Print detailed attribution analysis from strategy trade log."""
    trade_log = strategy.trade_log
    vrp_log = strategy._daily_vrp_log

    if not trade_log:
        print("\n[归因分析] 无交易记录。")
        return

    tdf = pd.DataFrame(trade_log)
    sep = "─" * 62

    print(f"\n{'═' * 62}")
    print(" 归因分析报告")
    print(f"{'═' * 62}")

    # --- 1. 按年度拆分收益 ---
    print(f"\n{sep}")
    print("【1. 年度收益拆分】")
    print(f"{sep}")
    tdf["year"] = tdf["entry_date"].str[:4]
    for year, grp in tdf.groupby("year"):
        total = grp["total_pnl"].sum()
        n = len(grp)
        wins = (grp["total_pnl"] > 0).sum()
        print(f"  {year}:  PnL={total:>+10,.0f}  交易={n}笔  胜率={wins/n*100:.0f}%")

    # --- 2. 按 VRP 水平拆分 ---
    print(f"\n{sep}")
    print("【2. VRP 信号来源拆分】")
    print(f"{sep}")
    bins = [(0.01, 0.02, "VRP 1-2%"), (0.02, 0.03, "VRP 2-3%"), (0.03, 0.05, "VRP 3-5%"), (0.05, 1.0, "VRP >5%")]
    for lo, hi, label in bins:
        sub = tdf[(tdf["entry_vrp"] >= lo) & (tdf["entry_vrp"] < hi)]
        if len(sub) == 0:
            print(f"  {label:<12}:  无交易")
            continue
        avg_pnl = sub["total_pnl"].mean()
        total = sub["total_pnl"].sum()
        n = len(sub)
        wr = (sub["total_pnl"] > 0).sum() / n * 100
        print(f"  {label:<12}:  {n}笔  平均PnL={avg_pnl:>+8,.0f}  合计={total:>+10,.0f}  胜率={wr:.0f}%")

    # GARCH level breakdown
    print()
    for level in ("GJR-GARCH", "GARCH", "EWMA"):
        sub = tdf[tdf["entry_garch_level"] == level]
        if len(sub) == 0:
            continue
        avg_pnl = sub["total_pnl"].mean()
        n = len(sub)
        print(f"  GARCH={level:<10}:  {n}笔  平均PnL={avg_pnl:>+8,.0f}")

    # --- 3. 止盈 vs 止损 vs 到期 ---
    print(f"\n{sep}")
    print("【3. 平仓原因分布】")
    print(f"{sep}")
    reason_map = {
        "take_profit": "止盈(70%)",
        "trailing_stop": "阶梯止盈",
        "condor_stop_loss": "Condor止损",
        "daily_loss_limit": "日亏损限额",
        "max_holding": "最长持仓",
        "dte_roll": "DTE滚动",
        "vrp_exit": "VRP反转",
        "expiry_settle": "到期结算",
        "exit": "普通退出",
    }
    for reason_key, reason_label in reason_map.items():
        sub = tdf[tdf["reason"] == reason_key]
        if len(sub) == 0:
            continue
        n = len(sub)
        avg_pnl = sub["total_pnl"].mean()
        total = sub["total_pnl"].sum()
        pct = n / len(tdf) * 100
        print(f"  {reason_label:<12}:  {n}笔({pct:.0f}%)  平均PnL={avg_pnl:>+8,.0f}  合计={total:>+10,.0f}")

    # --- 4. 持仓时间分析 ---
    print(f"\n{sep}")
    print("【4. 持仓时间分析】")
    print(f"{sep}")
    winners = tdf[tdf["total_pnl"] > 0]
    losers = tdf[tdf["total_pnl"] <= 0]
    if len(winners) > 0:
        print(f"  盈利交易:  平均持仓 {winners['holding_days'].mean():.1f} 天  ({len(winners)}笔)")
    if len(losers) > 0:
        print(f"  亏损交易:  平均持仓 {losers['holding_days'].mean():.1f} 天  ({len(losers)}笔)")
    print(f"  全部交易:  平均持仓 {tdf['holding_days'].mean():.1f} 天")

    # --- 5. DTE 分布分析 ---
    print(f"\n{sep}")
    print("【5. 入场 DTE 分布】")
    print(f"{sep}")
    dte_bins = [(20, 30, "20-30天"), (30, 45, "30-45天"), (45, 60, "45-60天")]
    for lo, hi, label in dte_bins:
        sub = tdf[(tdf["entry_dte"] >= lo) & (tdf["entry_dte"] < hi)]
        if len(sub) == 0:
            print(f"  {label:<10}:  无交易")
            continue
        avg_pnl = sub["total_pnl"].mean()
        n = len(sub)
        wr = (sub["total_pnl"] > 0).sum() / n * 100
        print(f"  {label:<10}:  {n}笔  平均PnL={avg_pnl:>+8,.0f}  胜率={wr:.0f}%")

    # --- 6. VRP 信号准确率 ---
    print(f"\n{sep}")
    print("【6. VRP 信号准确率】")
    print(f"{sep}")
    if vrp_log:
        vdf = pd.DataFrame(vrp_log)
        if "atm_iv" in vdf.columns and len(vdf) > 5:
            vdf["iv_5d_later"] = vdf["atm_iv"].shift(-5)
            vdf["iv_change_5d"] = vdf["iv_5d_later"] - vdf["atm_iv"]
            # When VRP > 2% (SELL_VOL signal), did IV actually decrease in next 5 days?
            sell_signals = vdf[vdf["vrp"] > 0.02].dropna(subset=["iv_change_5d"])
            if len(sell_signals) > 0:
                iv_down = (sell_signals["iv_change_5d"] < 0).sum()
                accuracy = iv_down / len(sell_signals) * 100
                avg_iv_chg = sell_signals["iv_change_5d"].mean()
                print(f"  VRP>2%时发出SELL_VOL信号 {len(sell_signals)} 次")
                print(f"  5日后IV下降比例: {accuracy:.1f}%")
                print(f"  5日后IV平均变化: {avg_iv_chg:+.4f}")
                avg_vrp = sell_signals["vrp"].mean()
                print(f"  信号时平均VRP: {avg_vrp:.4f}")
            else:
                print("  VRP>2%的交易日不足，无法统计")
        else:
            print("  VRP日志数据不足")

    # --- Summary stats ---
    print(f"\n{sep}")
    print("【汇总统计】")
    print(f"{sep}")
    total_pnl = tdf["total_pnl"].sum()
    n_trades = len(tdf)
    avg_pnl = tdf["total_pnl"].mean()
    avg_premium = tdf["net_premium_per_lot"].mean()
    avg_max_loss = tdf["max_loss_per_lot"].mean()
    print(f"  总交易笔数:  {n_trades}")
    print(f"  总PnL:       {total_pnl:>+10,.0f}")
    print(f"  平均PnL/笔:  {avg_pnl:>+10,.0f}")
    print(f"  平均权利金:  {avg_premium:>10.1f} 点/手")
    print(f"  平均最大亏损: {avg_max_loss:>10,.0f} 元/手")
    print(f"{'═' * 62}\n")


# ---------------------------------------------------------------------------
# Trade detail table
# ---------------------------------------------------------------------------

def print_trade_details(strategy, max_rows: int = 0) -> None:
    """Print condor trades with entry/exit dates, PnL, and close reason.

    Parameters
    ----------
    max_rows : int
        Max number of trades to print. 0 = all.
    """
    trade_log = strategy.trade_log
    if not trade_log:
        print("\n[交易明细] 无交易记录。")
        return

    tdf = pd.DataFrame(trade_log)
    sep = "─" * 100

    n_show = len(tdf) if max_rows == 0 else min(max_rows, len(tdf))
    title = f" Iron Condor 交易明细 (前{n_show}笔)" if max_rows > 0 else " Iron Condor 交易明细"

    print(f"\n{'═' * 100}")
    print(title)
    print(f"{'═' * 100}")
    print(f"  {'#':>3}  {'入场日期':<10}  {'退出日期':<10}  {'到期日':<10}  "
          f"{'手数':>4}  {'权利金':>8}  {'PnL':>10}  {'天数':>4}  {'交易日':>4}  {'平仓原因':<12}")
    print(f"  {sep}")

    reason_map = {
        "take_profit": "止盈(70%)",
        "trailing_stop": "阶梯止盈",
        "condor_stop_loss": "Condor止损",
        "daily_loss_limit": "日亏损限额",
        "max_holding": "最长持仓",
        "dte_roll": "DTE滚动",
        "vrp_exit": "VRP反转",
        "expiry_settle": "到期结算",
        "exit": "普通退出",
    }

    for i, row in tdf.head(n_show).iterrows():
        reason_label = reason_map.get(row["reason"], row["reason"])
        trade_days = row.get("holding_trade_days", "")
        trade_days_str = f"{trade_days:>4}" if trade_days != "" else "   -"
        print(
            f"  {i+1:>3}  {row['entry_date']:<10}  {row['exit_date']:<10}  "
            f"{row['expire_date']:<10}  {row['lots']:>4}  "
            f"{row['net_premium_per_lot']:>8.1f}  {row['total_pnl']:>+10,.0f}  "
            f"{row['holding_days']:>4}  {trade_days_str}  {reason_label:<12}"
        )

    total_pnl = tdf["total_pnl"].sum()
    if max_rows > 0 and len(tdf) > max_rows:
        print(f"  ... (共{len(tdf)}笔交易，仅显示前{max_rows}笔)")
    print(f"  {sep}")
    print(f"  {'合计':<66} {total_pnl:>+10,.0f}")
    print(f"{'═' * 100}\n")


# ---------------------------------------------------------------------------
# Parameter sensitivity test
# ---------------------------------------------------------------------------

def run_sensitivity_test(
    start_date: str = "20230101",
    end_date: str = "20260317",
    initial_capital: float = 5_000_000,
) -> None:
    """Run backtest with different parameter combinations and compare."""
    # --- VRP threshold × max holding days × stop-loss multiplier grid ---
    configs = [
        ("vrp1_h10_sl1.5", {"vrp_entry_threshold": 0.01, "max_holding_days": 10, "condor_stop_loss_mult": 1.5}),
        ("vrp1_h15_sl1.5", {"vrp_entry_threshold": 0.01, "max_holding_days": 15, "condor_stop_loss_mult": 1.5}),
        ("vrp1_h10_sl2.0", {"vrp_entry_threshold": 0.01, "max_holding_days": 10, "condor_stop_loss_mult": 2.0}),
        ("vrp1.5_h10_sl1.5", {"vrp_entry_threshold": 0.015, "max_holding_days": 10, "condor_stop_loss_mult": 1.5}),
        ("vrp1.5_h15_sl1.5", {"vrp_entry_threshold": 0.015, "max_holding_days": 15, "condor_stop_loss_mult": 1.5}),
        ("vrp2_h10_sl1.5", {"vrp_entry_threshold": 0.02, "max_holding_days": 10, "condor_stop_loss_mult": 1.5}),
        ("vrp2_h15_sl2.0", {"vrp_entry_threshold": 0.02, "max_holding_days": 15, "condor_stop_loss_mult": 2.0}),
        ("vrp1_h10_tp50", {"vrp_entry_threshold": 0.01, "max_holding_days": 10, "take_profit_pct": 0.50}),
    ]
    results = []
    for label, overrides in configs:
        report, strategy = run_vol_arb_backtest(
            start_date, end_date, initial_capital,
            config_overrides=overrides, quiet=True,
        )
        metrics = report.calculate_metrics()
        n = len(strategy.trade_log)
        tp = sum(1 for t in strategy.trade_log if t["reason"] == "take_profit")
        total_pnl = sum(t["total_pnl"] for t in strategy.trade_log)
        results.append({
            "Config": label,
            "Return": f"{metrics.get('total_return', 0)*100:+.1f}%",
            "MaxDD": f"{metrics.get('max_drawdown', 0)*100:.1f}%",
            "Sharpe": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "P/L": f"{metrics.get('profit_loss_ratio', 0):.2f}",
            "Trades": n,
            "TP%": f"{tp/max(n,1)*100:.0f}%",
            "PnL": f"{total_pnl:+,.0f}",
        })

    print(f"\n{'═' * 80}")
    print(" VRP阈值 × 持仓天数 × 止损倍数 参数敏感性测试")
    print(f"{'═' * 80}")
    rdf = pd.DataFrame(results)
    print(rdf.to_string(index=False))
    print(f"{'═' * 80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20230101")
    parser.add_argument("--end", default="20260317")
    parser.add_argument("--capital", type=float, default=5_000_000)
    parser.add_argument("--attribution", action="store_true", help="Print attribution analysis")
    parser.add_argument("--details", action="store_true", help="Print trade details table")
    parser.add_argument("--sensitivity", action="store_true", help="Run VRP threshold sensitivity test")
    args = parser.parse_args()

    report, strategy = run_vol_arb_backtest(args.start, args.end, args.capital)

    # Always print first 20 trade details
    print_trade_details(strategy, max_rows=20)

    if args.details:
        print_trade_details(strategy)

    if args.attribution:
        print_attribution(strategy, report)

    if args.sensitivity:
        run_sensitivity_test(args.start, args.end, args.capital)
