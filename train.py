from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from pathlib import Path 

PRIMARY_KEYS = ["trend", "ts"]

BASELINE_8_FEATURES = [
    "post_count",
    "count_lag1",
    "delta_count",
    "share_of_attention",
    "rank_in_snapshot",
    "tokens_per_post",
    "hour_sin",
    "hour_cos",
]

LABEL = "label_count_next"

@dataclass(frozen=True)
class TrainConfig:
    test_fraction: float = 0.2          # last 20% timestamps as test
    ridge_alpha: float = 5.0            # regularization strength
    target_transform: str = "log1p"     # "log1p" or "none"
    clip_negative_preds: bool = True    # counts cannot be negative
    round_preds: bool = True            # for UI; can keep float if you prefer


def _time_based_split(df: pd.DataFrame, ts_col: str = "ts", test_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ts_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {ts_col}")

    unique_ts = np.array(sorted(df[ts_col].dropna().unique()))
    if len(unique_ts) < 5:
        raise ValueError(f"Not enough unique timestamps ({len(unique_ts)}) for a meaningful time split.")

    cut = int(np.floor(len(unique_ts) * (1.0 - test_fraction)))
    cut = max(1, min(cut, len(unique_ts) - 1))  # ensure both sides non-empty

    train_ts = set(unique_ts[:cut])
    test_ts = set(unique_ts[cut:])

    train_df = df[df[ts_col].isin(train_ts)].copy()
    test_df = df[df[ts_col].isin(test_ts)].copy()
    return train_df, test_df


def _prepare_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that can be used for supervised learning:
    - label exists (needs t+1 observation)
    - core lag exists (for baseline features)
    """
    needed = PRIMARY_KEYS + BASELINE_8_FEATURES + [LABEL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for training: {missing}")

    # Require label and lag to exist; other NaNs will be imputed
    out = df.dropna(subset=[LABEL, "count_lag1"]).copy()

    # Ensure numeric features are numeric
    for c in BASELINE_8_FEATURES + [LABEL]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop rows that became invalid after coercion
    out = out.dropna(subset=[LABEL, "count_lag1"])
    return out


def _transform_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.log1p(y)
    if mode == "none":
        return y
    raise ValueError(f"Unknown target_transform: {mode}")


def _inverse_transform_target(yhat: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(yhat)
    if mode == "none":
        return yhat
    raise ValueError(f"Unknown target_transform: {mode}")


def topk_hit_rate_by_ts(
    df_eval: pd.DataFrame,
    ts_col: str,
    trend_col: str,
    y_true_col: str,
    y_pred_col: str,
    k: int = 5,
) -> float:
    """
    For each timestamp ts, compare top-k trends by predicted vs top-k by true.
    Returns average hit rate across timestamps: |intersection|/k.
    """
    hits = []
    for ts, g in df_eval.groupby(ts_col):
        g = g.dropna(subset=[y_true_col, y_pred_col])
        if len(g) < k:
            continue
        top_true = set(g.nlargest(k, y_true_col)[trend_col].tolist())
        top_pred = set(g.nlargest(k, y_pred_col)[trend_col].tolist())
        hits.append(len(top_true & top_pred) / k)
    return float(np.mean(hits)) if hits else float("nan")


def train_ridge_next_count_model(
    features_df: pd.DataFrame,
    cfg: TrainConfig = TrainConfig(),
    model_path: str = "trend_nextcount_ridge.joblib",
    metrics_path: str = "trend_nextcount_metrics.json",
) -> Dict[str, Any]:
    """
    Trains a global model across all trends to predict next-window post_count.
    Saves the fitted pipeline and writes metrics json.
    Returns metrics + a small preview dataframe for inspection.
    """
    df = _prepare_training_rows(features_df)

    # Time-based split (strictly by ts)
    train_df, test_df = _time_based_split(df, ts_col="ts", test_fraction=cfg.test_fraction)

    X_train = train_df[BASELINE_8_FEATURES]
    y_train = train_df[LABEL].to_numpy(dtype=float)

    X_test = test_df[BASELINE_8_FEATURES]
    y_test = test_df[LABEL].to_numpy(dtype=float)

    y_train_t = _transform_target(y_train, cfg.target_transform)

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(alpha=cfg.ridge_alpha, random_state=42)),
    ])

    pipeline.fit(X_train, y_train_t)

    # Predict
    yhat_test_t = pipeline.predict(X_test)
    yhat_test = _inverse_transform_target(yhat_test_t, cfg.target_transform)

    if cfg.clip_negative_preds:
        yhat_test = np.maximum(yhat_test, 0.0)
    if cfg.round_preds:
        yhat_test = np.rint(yhat_test)

    # Metrics on count scale
    mae = mean_absolute_error(y_test, yhat_test)
    rmse = mean_squared_error(y_test, yhat_test)

    # Ranking utility: Top-5 hit rate by timestamp (using candidates present at ts)
    eval_df = test_df[[*PRIMARY_KEYS]].copy()
    eval_df["y_true_next"] = y_test
    eval_df["y_pred_next"] = yhat_test
    top5_hit = topk_hit_rate_by_ts(eval_df, ts_col="ts", trend_col="trend",
                                   y_true_col="y_true_next", y_pred_col="y_pred_next", k=5)
    # Persist artifacts
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": BASELINE_8_FEATURES,
            "label": LABEL,
            "target_transform": cfg.target_transform,
            "config": cfg.__dict__,
        },
        model_path,
    )

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "unique_ts_total": int(df["ts"].nunique()),
        "unique_ts_train": int(train_df["ts"].nunique()),
        "unique_ts_test": int(test_df["ts"].nunique()),
        "mae_count": float(mae),
        "rmse_count": float(rmse),
        "top5_hit_rate": float(top5_hit),
        # "feature_list": BASELINE_8_FEATURES,
        # "label": LABEL,
        # "target_transform": cfg.target_transform,
        # "ridge_alpha": float(cfg.ridge_alpha),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Small preview for sanity check
    preview = eval_df.sort_values(["ts", "y_pred_next"], ascending=[True, False]).head(20)

    return {"metrics": metrics, "preview": preview}
