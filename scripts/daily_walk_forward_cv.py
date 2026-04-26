#!/usr/bin/env python3
"""Daily prediction — Walk-forward Cross Validation

用 strategies/daily/factors.py 的 DailyFactorPipeline。
TimeSeriesSplit 5-fold，每 fold 都重新训练，看 IC 在不同时段是否稳定。

输出:
  - 每 fold 的 train/test IC
  - 跨 fold IC 均值 / 标准差 → 是否 robust
  - 跨 fold top-bot decile spread 均值
  - 全样本 feature importance
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from strategies.daily.factors import (
    build_default_pipeline, load_default_context, add_forward_returns,
)

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"


def main():
    from sklearn.model_selection import TimeSeriesSplit

    ctx = load_default_context(DB_PATH)
    pipeline_proto = build_default_pipeline()
    print(f"Factors: {len(pipeline_proto.factors)}")
    for f in pipeline_proto.factors:
        print(f"  [{f.category}] {f.name}")

    # Add forward returns to px_df for target
    horizons = [1, 3, 5, 10]
    px_with_fwd = add_forward_returns(ctx.px_df, horizons)

    # Build full feature matrix only over dates with IV data (post 2022-07)
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    eligible = px_with_fwd[
        px_with_fwd["trade_date"].isin(iv_dates)
    ].copy().reset_index(drop=True)
    print(f"\nEligible rows (with IV data): {len(eligible)}")

    dates = eligible["trade_date"].tolist()
    print("Computing feature matrix...")
    X_full = pipeline_proto.features_matrix(dates, ctx)
    print(f"  Feature matrix: {X_full.shape}")

    # Drop rows where any feature is NaN (XGB handles but we want clean for IC calc)
    valid_mask = ~np.isnan(X_full).any(axis=1)
    print(f"  Rows with all features non-NaN: {valid_mask.sum()}")
    # Also drop rows where target NaN; do per horizon

    for horizon in horizons:
        target_col = f"fwd_{horizon}d_ret"
        target_nan = eligible[target_col].isna().values
        full_mask = valid_mask & ~target_nan
        n_use = full_mask.sum()

        print(f"\n{'=' * 70}")
        print(f" Horizon: next-{horizon}d return  (使用 {n_use} 行)")
        print(f"{'=' * 70}")

        if n_use < 200:
            print(f"  样本不足，跳过")
            continue

        X = X_full[full_mask]
        y = eligible.loc[full_mask, target_col].values
        date_arr = np.array(dates)[full_mask]

        # Time-series 5-fold split
        tscv = TimeSeriesSplit(n_splits=5)
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            pipe = build_default_pipeline()
            pipe.train(X_tr, y_tr)
            pred_tr = pipe.predict(X_tr)
            pred_te = pipe.predict(X_te)

            ic_tr = np.corrcoef(pred_tr, y_tr)[0, 1] if y_tr.std() > 0 else 0
            ic_te = np.corrcoef(pred_te, y_te)[0, 1] if y_te.std() > 0 else 0

            # Top-bot decile spread (test)
            df_t = pd.DataFrame({"p": pred_te, "y": y_te})
            df_t["dec"] = pd.qcut(df_t["p"], 10, labels=False, duplicates="drop")
            top_y = df_t[df_t["dec"] == 9]["y"].mean()
            bot_y = df_t[df_t["dec"] == 0]["y"].mean()

            tr_dr_start = date_arr[train_idx[0]]
            tr_dr_end = date_arr[train_idx[-1]]
            te_dr_start = date_arr[test_idx[0]]
            te_dr_end = date_arr[test_idx[-1]]

            fold_results.append({
                "fold": fold_idx,
                "n_tr": len(train_idx),
                "n_te": len(test_idx),
                "tr_period": f"{tr_dr_start}~{tr_dr_end}",
                "te_period": f"{te_dr_start}~{te_dr_end}",
                "ic_tr": ic_tr,
                "ic_te": ic_te,
                "top_y": top_y,
                "bot_y": bot_y,
                "spread": top_y - bot_y,
            })

        print(f"\n  Fold-by-fold:")
        print(f"  {'fold':<6}{'n_tr':<6}{'n_te':<6}{'test period':<22}"
              f"{'tr IC':>7}{'te IC':>7}{'top y':>9}{'bot y':>9}{'spread':>8}")
        for r in fold_results:
            print(f"  {r['fold']:<6}{r['n_tr']:<6}{r['n_te']:<6}{r['te_period']:<22}"
                  f"{r['ic_tr']:>+7.3f}{r['ic_te']:>+7.3f}"
                  f"{r['top_y']:>+9.4f}{r['bot_y']:>+9.4f}{r['spread']:>+8.4f}")

        # Aggregate
        ic_te_arr = np.array([r["ic_te"] for r in fold_results])
        spread_arr = np.array([r["spread"] for r in fold_results])
        n_pos_ic = (ic_te_arr > 0).sum()
        print(f"\n  Aggregate (5 folds):")
        print(f"    test IC mean: {ic_te_arr.mean():+.4f}  std: {ic_te_arr.std():.4f}  "
              f"({n_pos_ic}/5 folds 正)")
        print(f"    spread mean:  {spread_arr.mean()*100:+.3f}pp  std: {spread_arr.std()*100:.3f}pp")

        # Final pipe trained on all data for feature importance
        pipe_full = build_default_pipeline()
        pipe_full.train(X, y)
        imp = pipe_full.feature_importance()
        print(f"\n  Feature importance (top 12, all-data):")
        for _, r in imp.head(12).iterrows():
            bar = "█" * int(r["importance"] * 80)
            print(f"    [{r['category']:<5}] {r['feature']:<25} {r['importance']:.4f}  {bar}")


if __name__ == "__main__":
    main()
