"""
run_backtest.py — 回测主入口

用法:
    python run_backtest.py --strategy trend_following --start 20150105 --end 20260317
    python run_backtest.py --strategy vol_arb --start 20230101
    python run_backtest.py --list
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STRATEGY_RUNNERS = {
    "trend_following": "strategies.trend_following.backtest.run_trend_backtest",
    "vol_arb": "strategies.vol_arb.backtest.run_vol_arb_backtest",
    "discount_capture": "strategies.discount_capture.backtest.run_discount_backtest",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="回测主入口")
    parser.add_argument("--strategy", help="策略名称")
    parser.add_argument("--start", default="20220101")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=5_000_000)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--output", default=None)
    parser.add_argument("--list", action="store_true", dest="list_strategies")
    parser.add_argument(
        "--param",
        nargs="*",
        default=[],
        help="KEY=VALUE pairs to override strategy params",
    )
    args = parser.parse_args()

    if args.list_strategies:
        print("可用策略:")
        for name in STRATEGY_RUNNERS:
            print(f"  {name}")
        return

    if not args.strategy:
        parser.print_help()
        sys.exit(1)

    if args.strategy not in STRATEGY_RUNNERS:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {list(STRATEGY_RUNNERS.keys())}")
        sys.exit(1)

    # Parse --param KEY=VALUE overrides
    overrides: dict = {}
    for p in args.param or []:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                v = int(v)  # type: ignore[assignment]
            except ValueError:
                try:
                    v = float(v)  # type: ignore[assignment]
                except ValueError:
                    pass
            overrides[k] = v

    from datetime import date
    end = args.end or date.today().strftime("%Y%m%d")

    # Dynamically import and run the strategy's runner function
    module_path, func_name = STRATEGY_RUNNERS[args.strategy].rsplit(".", 1)
    module = importlib.import_module(module_path)
    runner = getattr(module, func_name)

    kwargs: dict = {
        "start_date": args.start,
        "end_date": end,
        "initial_capital": args.capital,
    }
    if args.symbols:
        kwargs["symbols"] = args.symbols
    if overrides:
        kwargs["config_overrides"] = overrides

    report = runner(**kwargs)

    if args.output and report is not None:
        report.save_to_csv(args.output)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
