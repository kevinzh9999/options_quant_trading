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
from models.factors.catalog_cross import (
    CrossMomentumSpread, CrossVolRatio, CrossCorrelation, CrossRank,
)
from models.factors.catalog_daily import (
    DailyBBWidth, DailyRangeMA, DailyConsecDays, DailyGapSize,
)

# 现货指数映射
_SPOT_MAP = {
    "IM": "000852", "IC": "000905", "IF": "000300", "IH": "000016",
}


def load_data(db: DBManager, with_cross=False):
    """加载5分钟K线 + 日内振幅 + 可选跨品种数据。"""
    print("Loading 5min bars (000852/IM)...")
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

    # 跨品种数据：merge其他3个品种的close
    if with_cross:
        print("Loading cross-asset data...")
        for sym, spot in [("IF", "000300"), ("IH", "000016"), ("IC", "000905")]:
            other = db.query_df(
                f"SELECT datetime, close FROM index_min "
                f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime"
            )
            if other is not None and not other.empty:
                other['close'] = other['close'].astype(float)
                other['datetime'] = pd.to_datetime(other['datetime'])
                other = other.set_index('datetime')
                bar_5m[f'close_{sym}'] = other['close']
                print(f"  {sym}({spot}): {len(other)} bars merged")

    # 日线特征映射到每根bar
    print("Loading daily features...")
    daily = db.query_df(
        "SELECT trade_date, open, high, low, close, volume "
        "FROM index_daily WHERE ts_code='000852.SH' ORDER BY trade_date"
    )
    if daily is not None and not daily.empty:
        for c in ['open', 'high', 'low', 'close', 'volume']:
            daily[c] = daily[c].astype(float)
        daily['range_pct'] = (daily['high'] - daily['low']) / daily['close'] * 100
        daily['gap_pct'] = (daily['open'] - daily['close'].shift(1)) / daily['close'].shift(1)

        # BB width
        daily['bb_std'] = daily['close'].rolling(20).std()
        daily['bb_ma'] = daily['close'].rolling(20).mean()
        daily['bb_width'] = (2 * daily['bb_std'] / daily['bb_ma']).fillna(0)

        # 5日平均振幅
        daily['range_ma5'] = daily['range_pct'].rolling(5).mean()

        # 连续同方向天数
        consec = np.zeros(len(daily))
        for i in range(1, len(daily)):
            if daily['close'].iloc[i] > daily['close'].iloc[i-1]:
                if consec[i-1] > 0:
                    consec[i] = consec[i-1] + 1
                else:
                    consec[i] = 1
            elif daily['close'].iloc[i] < daily['close'].iloc[i-1]:
                if consec[i-1] < 0:
                    consec[i] = consec[i-1] - 1
                else:
                    consec[i] = -1
        daily['consec_days'] = consec

        # 映射到bar_5m的每根bar（用前一日的日线特征，避免前瞻偏差）
        daily['next_date'] = daily['trade_date'].shift(-1)
        date_to_features = {}
        for _, row in daily.iterrows():
            nd = row['next_date']
            if pd.notna(nd):
                date_to_features[str(nd)] = {
                    'bb_width': row['bb_width'],
                    'daily_range': row['range_ma5'],
                    'consec_days': abs(row['consec_days']),
                    'gap_pct': 0,  # gap要用当天的open，特殊处理
                }
        # gap用当天的
        for _, row in daily.iterrows():
            td = str(row['trade_date'])
            if td in date_to_features:
                date_to_features[td]['gap_pct'] = abs(row['gap_pct']) if pd.notna(row['gap_pct']) else 0

        bar_5m['date_str'] = bar_5m.index.strftime('%Y%m%d')
        for feat in ['bb_width', 'daily_range', 'consec_days', 'gap_pct']:
            bar_5m[feat] = bar_5m['date_str'].map(
                lambda d, f=feat: date_to_features.get(d, {}).get(f, np.nan)
            )
        bar_5m = bar_5m.drop(columns=['date_str'])

    # 日内振幅（用于regime分组）
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
    need_cross = args.category in ("cross", "all")
    bar_5m, daily_range = load_data(db, with_cross=need_cross)

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

    if args.category in ("cross", "all"):
        print("\n" + "=" * 50)
        print("  跨品种因子评估")
        print("=" * 50)
        factors = [
            CrossMomentumSpread(12, 'close_IH'),
            CrossMomentumSpread(12, 'close_IF'),
            CrossVolRatio(20, 'close_IF'),
            CrossCorrelation(48, 'close_IH'),
            CrossRank(12),
        ]
        evaluator.print_report(factors)

    if args.category in ("daily", "all"):
        print("\n" + "=" * 50)
        print("  日线级别因子评估（盘前可知）")
        print("=" * 50)
        factors = [
            DailyBBWidth(),
            DailyRangeMA(),
            DailyConsecDays(),
            DailyGapSize(),
        ]
        evaluator.print_report(factors)


if __name__ == "__main__":
    main()
