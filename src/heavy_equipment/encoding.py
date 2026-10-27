"""Imputation, out-of-fold target encoding and label encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from .config import ID_COL, NUMERIC_INTERACTION_CATS, NUMERIC_INTERACTION_NUMS, TEST_FLAG

EXCLUDE = {TEST_FLAG, ID_COL}


def categorical_columns(df: pd.DataFrame) -> list[str]:
    # "string" keeps this working with the pandas>=3 default string dtype
    return [c for c in df.select_dtypes(include=["object", "string"]).columns if c not in EXCLUDE]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]


def fill_missing(df: pd.DataFrame, train_len: int) -> tuple[pd.DataFrame, list[str]]:
    """Categoricals -> 'Unknown'; numerics -> train median plus a `<col>_miss` flag."""
    cat_cols = categorical_columns(df)
    num_cols = numeric_columns(df)

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    for col in num_cols:
        if df[col].isnull().any():
            flag = f"{col}_miss"
            if flag not in df.columns:
                df[flag] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df[col].iloc[:train_len].median())
    return df, cat_cols


def target_encode(
    df: pd.DataFrame,
    col: str,
    y: pd.Series,
    train_len: int,
    folds: KFold,
    global_mean: float,
    smoothing: float,
) -> pd.DataFrame:
    """Add `<col>_freq` (frequency) and `<col>_te` (smoothed mean of log-price).

    Train rows receive *out-of-fold* encodings so a row's own target never leaks
    into its feature. Test rows are encoded with statistics from all train rows.
    Smoothing shrinks rare categories toward the global mean:
        te = (mean * n + global_mean * m) / (n + m)
    """
    df[f"{col}_freq"] = df[col].map(df[col].value_counts() / len(df))
    te_col = f"{col}_te"
    te = np.full(len(df), np.nan)

    values = df[col].to_numpy()
    y_arr = y.to_numpy()

    def smoothed_means(cats, targets):
        s = pd.DataFrame({"c": cats, "y": targets}).groupby("c")["y"].agg(["mean", "count"])
        return (s["mean"] * s["count"] + global_mean * smoothing) / (s["count"] + smoothing)

    for tr_idx, val_idx in folds.split(np.arange(train_len)):
        mapping = smoothed_means(values[tr_idx], y_arr[tr_idx])
        te[val_idx] = pd.Series(values[val_idx]).map(mapping).to_numpy(dtype=float)

    mapping = smoothed_means(values[:train_len], y_arr)
    te[train_len:] = pd.Series(values[train_len:]).map(mapping).to_numpy(dtype=float)

    df[te_col] = te
    df[te_col] = df[te_col].fillna(global_mean)  # categories unseen in the fitting fold
    return df


def label_encode(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def add_numeric_interactions(df: pd.DataFrame) -> pd.DataFrame:
    for cat in NUMERIC_INTERACTION_CATS:
        if cat not in df.columns:
            continue
        for num in NUMERIC_INTERACTION_NUMS:
            df[f"{cat[:6]}_x_{num[:6]}"] = df[cat] * df[num]
    return df


def encode(
    df: pd.DataFrame,
    y: pd.Series,
    train_len: int,
    te_folds: int = 5,
    te_seed: int = 2024,
    smoothing: float = 10.0,
) -> pd.DataFrame:
    df, cat_cols = fill_missing(df, train_len)
    folds = KFold(n_splits=te_folds, shuffle=True, random_state=te_seed)
    global_mean = float(y.mean())
    for col in cat_cols:
        df = target_encode(df, col, y, train_len, folds, global_mean, smoothing)
    df = label_encode(df.copy(), cat_cols)  # .copy() de-fragments the wide frame
    df = add_numeric_interactions(df)
    return df.copy()


def split_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = df[df[TEST_FLAG] == 0].drop(columns=[TEST_FLAG, ID_COL])
    X_test = df[df[TEST_FLAG] == 1].drop(columns=[TEST_FLAG, ID_COL])
    return X_train, X_test
