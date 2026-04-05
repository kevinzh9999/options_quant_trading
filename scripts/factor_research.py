#!/usr/bin/env python3
"""
factor_research.py
------------------
因子研究脚本：加载215天5分钟K线，评估M分候选因子。

用法：
    python scripts/factor_research.py                # 评估全部M分候选
    python scripts/factor_research.py --category all  # 评估全部类别
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader
from models.factors.evaluator import FactorEvaluator
from models.factors.catalog_price import (
    MomSimple, MomEMA, MomLinReg, MomDecayLinear,
    MomRank, MomMultiScale, MomRiskAdjusted,
)
from models.factors.catalog_vol import (
    VolATRRatio, VolATRTrend, VolParkinson, VolReturnStd, VolBBWidth,
)
from models.factors.catalog_volume import (
    QtyRatio, QtyTrend, QtyPriceCorr, QtySignedFlow,
)
from models.factors.catalog_structure import (
    BollBreakout, BodyRatioFactor, PricePosition, RSIFactor,
)
from models.factors.catalog_alpha101 import (
    Alpha001, Alpha002, Alpha006, Alpha012, Alpha018, Alpha041, Alpha101,
)


def load_data(db: DBManager):
    """加载5分钟K线 + 日内振幅（用于regime分组）。"""
    print("Loading 5min bars...")
    bar_5m = db.query_df(
        "SELECT datetime, open, high, low, close, volume "
        "FROM index_min WHERE symbol='000852' AND period=300 "
        "ORDER BY datetime"
    )
    for c in ['open', 'high', 'low', 'close', 'volume']:
        bar_5m[c] = bar_5m[c].astype(float)
    bar_5m['datetime'] = pd.to_datetime(bar_5m['datetime'])
    bar_5m = bar_5m.set_index('datetime')
    print(f"  {len(bar_5m)} bars, {bar_5m.index[0]} ~ {bar_5m.index[-1]}")

    # 计算每根bar所属日期的日内振幅
    print("Computing daily range for regime analysis...")
    bar_5m['date'] = bar_5m.index.date
    daily_stats = bar_5m.groupby('date').agg(
        day_high=('high', 'max'),
        day_low=('low', 'min'),
        day_close=('close', 'last'),
    )
    daily_stats['range_pct'] = (daily_stats['day_high'] - daily_stats['day_low']) / daily_stats['day_close'] * 100

    # Map back to each bar
    daily_range = bar_5m['date'].map(daily_stats['range_pct'])
    daily_range.index = bar_5m.index

    bar_5m = bar_5m.drop(columns=['date'])

    return bar_5m, daily_range


def get_momentum_factors():
    """M分候选因子列表。"""
    return [
        # Baseline（当前系统使用）
        MomSimple(12),

        # 替代候选
        MomEMA(5, 20),
        MomLinReg(12),
        MomDecayLinear(12),
        MomRank(48),
        MomMultiScale(6, 18),
        MomRiskAdjusted(12, 20),

        # 参数变体
        MomSimple(6),     # 更短lookback
        MomSimple(18),    # 更长lookback
        MomLinReg(6),
        MomLinReg(18),
    ]


def main():
    parser = argparse.ArgumentParser(description="因子研究")
    parser.add_argument("--category", default="momentum",
                        help="momentum / all")
    parser.add_argument("--forward", default="1,3,5,10",
                        help="Forward periods (bar count)")
    args = parser.parse_args()

    db = DBManager(ConfigLoader().get_db_path())
    bar_5m, daily_range = load_data(db)

    forward_periods = [int(x) for x in args.forward.split(",")]

    evaluator = FactorEvaluator(
        bar_5m,
        forward_periods=forward_periods,
        daily_range=daily_range,
    )

    if args.category in ("momentum", "all"):
        print("\n" + "=" * 50)
        print("  M分候选因子评估")
        print("=" * 50)
        factors = get_momentum_factors()
        evaluator.print_report(factors)

    if args.category in ("volatility", "all"):
        print("\n" + "=" * 50)
        print("  V分候选因子评估")
        print("=" * 50)
        factors = [
            VolATRRatio(5, 40),   # baseline
            VolATRTrend(3, 8),
            VolParkinson(20),
            VolReturnStd(10, 40),
            VolBBWidth(20),
        ]
        evaluator.print_report(factors)

    if args.category in ("volume", "all"):
        print("\n" + "=" * 50)
        print("  Q分候选因子评估")
        print("=" * 50)
        factors = [
            QtyRatio(20),         # baseline
            QtyTrend(3, 10),
            QtyPriceCorr(10),
            QtySignedFlow(10),
        ]
        evaluator.print_report(factors)

    if args.category in ("structure", "all"):
        print("\n" + "=" * 50)
        print("  B分+结构因子评估")
        print("=" * 50)
        factors = [
            BollBreakout(20),     # baseline
            BodyRatioFactor(),
            PricePosition(240),
            RSIFactor(14),
        ]
        evaluator.print_report(factors)

    if args.category in ("alpha101", "all"):
        print("\n" + "=" * 50)
        print("  101 Alphas 经典因子评估")
        print("=" * 50)
        factors = [
            Alpha001(),
            Alpha002(),
            Alpha006(),
            Alpha012(),
            Alpha018(),
            Alpha041(),
            Alpha101(),
        ]
        evaluator.print_report(factors)


if __name__ == "__main__":
    main()
