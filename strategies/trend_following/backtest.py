"""
趋势跟踪策略回测。

用法:
    python -m strategies.trend_following.backtest --start 20150105 --end 20260317
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_trend_backtest(
    symbols: Optional[List[str]] = None,
    start_date: str = "20150105",
    end_date: str = "20260317",
    initial_capital: float = 5_000_000,
    config_overrides: Optional[dict] = None,
):
    """Execute trend following backtest."""
    from config.config_loader import ConfigLoader
    from data.storage.db_manager import DBManager
    from backtest.data_feed import DataFeed
    from backtest.broker import SimBroker
    from backtest.engine import BacktestEngine
    from strategies.trend_following.strategy import TrendFollowingStrategy, TrendConfig

    if symbols is None:
        symbols = ["IF.CFX", "IC.CFX", "IH.CFX", "IM.CFX"]

    config = ConfigLoader()
    db = DBManager(config.get_db_path())

    feed = DataFeed(db, start_date, end_date, symbols)
    broker = SimBroker(initial_capital=initial_capital)
    engine = BacktestEngine(feed, broker)

    trend_config = TrendConfig(
        strategy_id="trend_following_multi",
        universe=symbols,
        enabled=True,
        dry_run=False,
    )
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(trend_config, k):
                setattr(trend_config, k, v)

    strategy = TrendFollowingStrategy(trend_config)
    engine.add_strategy(strategy)

    # Set per-symbol contract multipliers
    multipliers = {"IF.CFX": 300, "IH.CFX": 300, "IC.CFX": 200, "IM.CFX": 200}
    for sym in symbols:
        mult = multipliers.get(sym, 200)
        engine.set_symbol_params(sym, contract_multiplier=mult, margin_rate=0.15)

    print(f"[trend backtest] {start_date} ~ {end_date}  symbols={symbols}  capital={initial_capital:,.0f}")
    report = engine.run()
    report.print_report()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["IF.CFX", "IC.CFX", "IH.CFX", "IM.CFX"])
    parser.add_argument("--start", default="20150105")
    parser.add_argument("--end", default="20260317")
    parser.add_argument("--capital", type=float, default=5_000_000)
    args = parser.parse_args()
    run_trend_backtest(args.symbols, args.start, args.end, args.capital)
