"""XGBoost training & prediction wrapper for Daily XGB strategy.

Caches trained model + features matrix on disk to avoid re-computing every signal day.
Re-trains every cfg.retrain_every trading days using all available history.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import DailyXGBConfig
from .factors import (
    DailyContext,
    build_full_pipeline,
    load_default_context,
    add_forward_returns,
)


@dataclass
class TrainedModel:
    """Container for trained pipeline + thresholds + metadata."""
    pipeline: object  # DailyFactorPipeline
    train_end_date: str
    n_train_samples: int
    train_ic: float
    top_threshold: float   # P80 of in-sample preds → LONG threshold
    bot_threshold: float   # P20 of in-sample preds → SHORT threshold


def _model_cache_path(cfg: DailyXGBConfig, train_end_date: str) -> Path:
    cache_dir = Path(cfg.log_dir) / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"model_{train_end_date}.pkl"


def train_model(ctx: DailyContext,
                 train_end_date: str,
                 cfg: DailyXGBConfig) -> TrainedModel:
    """Train fresh pipeline using data ≤ train_end_date (excluding HOLD_DAYS for target realization)."""
    pipeline = build_full_pipeline()

    # Build feature matrix on dates from start to train_end (with IV available)
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = ctx.px_df[ctx.px_df["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values

    end_idx = next((i for i, d in enumerate(dates) if d > train_end_date), len(dates))
    if end_idx <= cfg.initial_train_days:
        raise ValueError(
            f"Insufficient training data: only {end_idx} days ≤ {train_end_date}, "
            f"need at least {cfg.initial_train_days}")

    # Forward returns target (5-day, matching backtest)
    fwd5 = np.array([
        closes[i + cfg.hold_days_default] / closes[i] - 1
        if i + cfg.hold_days_default < len(closes)
        else np.nan
        for i in range(len(closes))
    ])

    cutoff = end_idx - cfg.hold_days_default  # need realized targets
    X_full = pipeline.features_matrix(dates[:cutoff], ctx)
    y = fwd5[:cutoff]
    valid = ~(np.isnan(X_full).any(axis=1) | np.isnan(y))
    X_v = X_full[valid]
    y_v = y[valid]

    if len(X_v) < cfg.initial_train_days:
        raise ValueError(f"Not enough valid training samples: {len(X_v)}")

    pipeline.train(
        X_v, y_v,
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        min_child_weight=cfg.xgb_min_child_weight,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        reg_alpha=cfg.xgb_reg_alpha,
        reg_lambda=cfg.xgb_reg_lambda,
        random_state=cfg.xgb_random_state,
    )
    pred_in = pipeline.predict(X_v)
    train_ic = float(np.corrcoef(pred_in, y_v)[0, 1])

    return TrainedModel(
        pipeline=pipeline,
        train_end_date=str(dates[end_idx - 1]),
        n_train_samples=int(len(X_v)),
        train_ic=train_ic,
        top_threshold=float(np.quantile(pred_in, 1 - cfg.top_pct)),
        bot_threshold=float(np.quantile(pred_in, cfg.bot_pct)),
    )


def save_model(model: TrainedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> TrainedModel:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_or_train_model(ctx: DailyContext,
                         train_end_date: str,
                         cfg: DailyXGBConfig,
                         force_retrain: bool = False) -> TrainedModel:
    """Cached training. Returns cached model if exists and not forced."""
    cache_path = _model_cache_path(cfg, train_end_date)
    if cache_path.exists() and not force_retrain:
        try:
            return load_model(cache_path)
        except Exception:
            pass  # fall through to retrain
    model = train_model(ctx, train_end_date, cfg)
    save_model(model, cache_path)
    return model


def predict_for_date(model: TrainedModel,
                      ctx: DailyContext,
                      date: str,
                      cfg: DailyXGBConfig) -> Tuple[Optional[float], Optional[str], np.ndarray]:
    """Compute features for a date, run model, return (pred, direction_or_None, feature_vec).

    direction:
      "LONG"  if pred ≥ top_threshold
      "SHORT" if pred ≤ bot_threshold
      None    otherwise (no signal)
    """
    feat = model.pipeline.features_for_date(date, ctx)
    if np.isnan(feat).any():
        return None, None, feat

    pred = float(model.pipeline.predict(feat.reshape(1, -1))[0])
    if pred >= model.top_threshold:
        direction = "LONG"
    elif pred <= model.bot_threshold:
        direction = "SHORT"
    else:
        direction = None
    return pred, direction, feat


__all__ = [
    "TrainedModel", "train_model", "save_model", "load_model",
    "get_or_train_model", "predict_for_date",
]
