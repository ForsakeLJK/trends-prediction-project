from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Trend",
    "time_stamp",
    "post_count",
    "avg_post_length",
    "token_volume",
    "growth_rate",
    "source_file",
]


@dataclass(frozen=True)
class FeatureConfig:
    bucket_minutes: int = 5
    rolling_windows: Tuple[int, ...] = (3, 12)  # 3=15min, 12=60min if bucket is 5min
    normalize_trend: bool = True               # lower-case + strip
    keep_original_trend: bool = True
    dedupe_strategy: str = "latest"            # "latest" within (trend, ts)
    std_ddof: int = 0                          # population std (stable for small windows)


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not np.issubdtype(df["time_stamp"].dtype, np.datetime64):
        raise TypeError("Column 'time_stamp' must be datetime64[ns]. "
                        "Use pd.to_datetime(df['time_stamp']).")


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # ts is the 5-min bucket timestamp (datetime64[ns])
    hour_float = df["ts"].dt.hour + (df["ts"].dt.minute / 60.0)
    angle = 2.0 * math.pi * (hour_float / 24.0)
    df["hour_sin"] = np.sin(angle)
    df["hour_cos"] = np.cos(angle)

    df["day_of_week"] = df["ts"].dt.dayofweek.astype("int16")  # Mon=0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    # Optional simple buckets (easy to demo); you can drop if not needed
    # midnight [0,6), morning [6,12), afternoon [12,18), evening [18,24)
    df["time_bucket"] = pd.cut(
        df["ts"].dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["midnight", "morning", "afternoon", "evening"],
    ).astype("string")
    return df


def build_feature_table(raw: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """
    Transform raw 5-minute snapshots into a leakage-safe feature table.

    Output keys:
      - trend (normalized)
      - ts (bucketed timestamp)

    Core outputs:
      - engineered features (lags, rolling stats, intensity, context, time)
      - labels for supervised learning:
          label_count_next = post_count at next bucket within same trend
          label_delta_next = label_count_next - post_count
    """
    _validate_input(raw)

    df = raw.copy()

    # Preserve your existing "growth_rate" but rename to reflect reality (difference, not %)
    df = df.rename(columns={"growth_rate": "delta_count_raw"})
    if cfg.keep_original_trend:
        df["trend_raw"] = df["Trend"].astype("string")

    # Normalize trend identifier for stable keys
    if cfg.normalize_trend:
        df["trend"] = (
            df["Trend"].astype("string")
            .str.strip()
            .str.lower()
        )
    else:
        df["trend"] = df["Trend"].astype("string").str.strip()

    # Bucket timestamps to consistent 5-min windows
    df["ts"] = df["time_stamp"].dt.floor(f"{cfg.bucket_minutes}min")

    # Dedupe: keep latest observation within each (trend, ts)
    if cfg.dedupe_strategy == "latest":
        df = (
            df.sort_values(["trend", "ts", "time_stamp"])
              .groupby(["trend", "ts"], as_index=False)
              .tail(1)
        )
    else:
        raise ValueError(f"Unsupported dedupe_strategy: {cfg.dedupe_strategy}")

    # print(df["Trend"].head(10))
    # Ensure deterministic ordering
    df = df.sort_values(["ts", "trend"]).reset_index(drop=True)
    # print(df["Trend"].head(10))

    # ---------- Context features per timestamp (snapshot-level) ----------
    df["rank_in_snapshot"] = (
        df.groupby("ts")["post_count"]
          .rank(method="first", ascending=False)
          .astype("int32")
    )
    df["total_count_all_trends_at_ts"] = df.groupby("ts")["post_count"].transform("sum").astype("int64")
    df["share_of_attention"] = df["post_count"] / df["total_count_all_trends_at_ts"].clip(lower=1)

    # print(df["Trend"].head(10))

    # ---------- Intensity features ----------
    df["tokens_per_post"] = df["token_volume"] / df["post_count"].clip(lower=1)

    # Stable transforms
    df["log_count"] = np.log1p(df["post_count"].astype(float))
    df["log_token_volume"] = np.log1p(df["token_volume"].astype(float))

    # ---------- Time features ----------
    df = _add_time_features(df)

    # ---------- Per-trend time-series features ----------
    df = df.sort_values(["trend", "ts"]).reset_index(drop=True)
    
    g = df.groupby("trend", sort=False)

    # Lags
    df["count_lag1"] = g["post_count"].shift(1)
    df["count_lag2"] = g["post_count"].shift(2)
    df["token_lag1"] = g["token_volume"].shift(1)
    df["avglen_lag1"] = g["avg_post_length"].shift(1)

    # Deltas / growth (computed from counts; this replaces ambiguous raw growth_rate)
    df["delta_count"] = df["post_count"] - df["count_lag1"]
    df["log_growth"] = df["log_count"] - g["log_count"].shift(1)  # log(1+c_t) - log(1+c_{t-1})

    # Time since last seen (helps when trends drop out and reappear)
    df["time_since_last_seen_min"] = (
        (df["ts"] - g["ts"].shift(1)).dt.total_seconds() / 60.0
    )

    # Rolling stats (include current time t; no future leakage for predicting t+1)
    for w in cfg.rolling_windows:
        df[f"count_ma{w}"] = g["post_count"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"count_std{w}"] = g["post_count"].transform(
            lambda s: s.rolling(w, min_periods=1).std(ddof=cfg.std_ddof)
        )

        df[f"token_ma{w}"] = g["token_volume"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"token_std{w}"] = g["token_volume"].transform(
            lambda s: s.rolling(w, min_periods=1).std(ddof=cfg.std_ddof)
        )

        df[f"avglen_ma{w}"] = g["avg_post_length"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"avglen_std{w}"] = g["avg_post_length"].transform(
            lambda s: s.rolling(w, min_periods=1).std(ddof=cfg.std_ddof)
        )

        df[f"share_ma{w}"] = g["share_of_attention"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"share_std{w}"] = g["share_of_attention"].transform(
            lambda s: s.rolling(w, min_periods=1).std(ddof=cfg.std_ddof)
        )

    # ---------- Labels (targets) ----------
    # Predict next-window count; from that you can derive delta/rank for your UI.
    df["label_count_next"] = g["post_count"].shift(-1)
    df["label_delta_next"] = df["label_count_next"] - df["post_count"]

    # Optional: percentage growth feature (as a feature; can be spiky for small denominators)
    df["pct_growth"] = df["delta_count"] / df["count_lag1"].clip(lower=1)

    df.sort_values(["ts","post_count","trend"], ascending=[True, False, True], inplace=True)
    # Clean up types (optional)
    # Keep NaNs in lag/label columns; you will drop them for training.
    return df.reset_index(drop=True)


def get_feature_columns(df_features: pd.DataFrame) -> List[str]:
    """
    Convenience helper: returns a reasonable default feature list.
    You can edit this list based on ablations for your report.
    """
    base = [
        # raw-ish
        "post_count",
        "avg_post_length",
        "token_volume",
        "tokens_per_post",
        "share_of_attention",
        "rank_in_snapshot",
        # lags/dynamics
        "count_lag1",
        "count_lag2",
        "delta_count",
        "log_growth",
        "time_since_last_seen_min",
        # time
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        # snapshot context
        "total_count_all_trends_at_ts",
    ]

    # Add rolling columns dynamically if present
    rolling = [c for c in df_features.columns if any(
        c.startswith(prefix) for prefix in ("count_ma", "count_std", "token_ma", "token_std", "avglen_ma", "avglen_std", "share_ma", "share_std")
    )]

    # Optional categorical time bucket (if you want to one-hot encode later)
    if "time_bucket" in df_features.columns:
        base.append("time_bucket")

    cols = base + sorted(rolling)
    # Keep only existing columns
    return [c for c in cols if c in df_features.columns]


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

LABELS = [
    "label_count_next",
    "label_delta_next",
]

PRIMARY_KEYS = ["trend", "ts"]
SORT_KEYS = ["ts", "post_count", "trend"]
SORT_ASCENDING = [True, False, True]

def select_feature_store_columns(df_features: pd.DataFrame) -> pd.DataFrame:
    # Optional metadata that can be useful for debugging / lineage
    optional_meta = [c for c in ["trend_raw", "source_file"] if c in df_features.columns]

    needed = PRIMARY_KEYS + BASELINE_8_FEATURES + LABELS + optional_meta
    missing = [c for c in needed if c not in df_features.columns]
    if missing:
        raise ValueError(f"Missing columns required for feature store selection: {missing}")

    out = df_features[needed].copy()

    # Feature store usually expects uniqueness per entity-time key
    dup = out.duplicated(subset=PRIMARY_KEYS, keep=False)
    if dup.any():
        examples = out.loc[dup, PRIMARY_KEYS].head(10).to_dict("records")
        raise ValueError(
            f"Duplicate (trend, ts) keys found. Example duplicates (first 10): {examples}"
        )

    # Sort for deterministic backfill writing
    out = out.sort_values(SORT_KEYS, ascending=SORT_ASCENDING).reset_index(drop=True)

    return out

# ---------------- Example usage ----------------
if __name__ == "__main__":
    # raw_df = pd.read_parquet("raw_snapshots.parquet")  # or however you load it
    # raw_df["time_stamp"] = pd.to_datetime(raw_df["time_stamp"])

    raw_df = pd.read_csv('./historical_data.csv',
                        dtype={
            "Trend": "string",
            "source_file": "string"
        },
                    parse_dates=["time_stamp"]
                    )
    cfg = FeatureConfig(bucket_minutes=5, rolling_windows=(3, 12))
    features_df = build_feature_table(raw_df, cfg)
    # features_df.sort_values(["ts", "post_count", "trend"], ascending=[True, False, True], inplace=True)
    print(features_df.head(10))
    # Training set: drop rows with missing label (no t+1 observation)
    # train_df = features_df.dropna(subset=["label_count_next"]).copy()

    # X / y
    # feature_cols = get_feature_columns(train_df)
    # X = train_df[feature_cols]
    # y = train_df["label_count_next"]  # or np.log1p(...) if you prefer log-target

    # Backfill feature store (offline): write partitioned parquet
    # features_df.to_parquet("feature_store_backfill.parquet", index=False)
    # pass
    print(features_df.info())