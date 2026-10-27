"""Data loading and train/test stacking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_COL, TARGET_COL, TEST_FLAG


@dataclass
class Dataset:
    """Everything downstream steps need about the raw data."""

    combined: pd.DataFrame      # train rows first, then test rows
    y: pd.Series                # log1p(TargetValue) for the train rows
    train_len: int
    test_ids: np.ndarray
    price_min: float
    price_max: float


def load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", low_memory=False)
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    return train, test


def build_dataset(train: pd.DataFrame, test: pd.DataFrame) -> Dataset:
    """Concatenate train and test so every feature is built identically for both.

    The target is modelled in log space: minimising RMSE on log1p(price)
    is exactly minimising the competition metric (RMSLE).
    """
    target = train[TARGET_COL].to_numpy(dtype=float)
    y = pd.Series(np.log1p(target), name="log_target")

    df_train = train.drop(columns=[TARGET_COL]).copy()
    df_test = test.copy()
    df_train[TEST_FLAG] = 0
    df_test[TEST_FLAG] = 1

    combined = pd.concat([df_train, df_test], ignore_index=True)
    return Dataset(
        combined=combined,
        y=y,
        train_len=len(df_train),
        test_ids=test[ID_COL].to_numpy(),
        price_min=float(target.min()),
        price_max=float(target.max()),
    )


def load_dataset(data_dir: str | Path) -> Dataset:
    return build_dataset(*load_raw(data_dir))
