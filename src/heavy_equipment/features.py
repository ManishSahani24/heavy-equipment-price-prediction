"""Feature engineering that does not use the target.

Every function takes the combined (train + test) frame and returns it with new
columns. Statistics that are *imputed* (medians) are computed on the train rows
only; purely descriptive group statistics (counts, medians of hours/age per
configuration) use all rows since they never touch the target.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import (
    AGG_GROUP_COLS,
    CROSS_FEATURES,
    ID_COL,
    ID_LIKE_CATEGORICALS,
    MAX_MISSING_RATIO,
    TEST_FLAG,
)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def cast_id_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ProductConfigID / AssetID are identifiers, not quantities."""
    for col in ID_LIKE_CATEGORICALS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def drop_mostly_missing(df: pd.DataFrame, threshold: float = MAX_MISSING_RATIO) -> tuple[pd.DataFrame, list[str]]:
    ratio = df.isnull().mean()
    drop = [c for c in ratio.index if ratio[c] > threshold and c not in (ID_COL, TEST_FLAG)]
    return df.drop(columns=drop), drop


# --------------------------------------------------------------------------- #
# Date features
# --------------------------------------------------------------------------- #
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["SaleYear"] = date.dt.year
    df["SaleMonth"] = date.dt.month
    df["SaleDay"] = date.dt.day
    df["SaleDayOfWeek"] = date.dt.dayofweek
    df["SaleQuarter"] = date.dt.quarter
    df["SaleDayOfYear"] = date.dt.dayofyear
    df["DaysSinceStart"] = (date - date.min()).dt.days
    df["SaleIsYearEnd"] = (df["SaleMonth"] == 12).astype(int)
    df["SaleIsYearStart"] = (df["SaleMonth"] == 1).astype(int)
    df["SaleDecade"] = (df["SaleYear"] // 10).astype(int)
    df["SaleYear_sq"] = df["SaleYear"] ** 2
    return df.drop(columns=["TransactionDate"])


# --------------------------------------------------------------------------- #
# Manufacture year / equipment age
# --------------------------------------------------------------------------- #
def add_age_features(df: pd.DataFrame, train_len: int) -> pd.DataFrame:
    year = df["ManufactureYear"].replace({1000: np.nan, 1001: np.nan})  # placeholder years
    year = year.where((year >= 1900) & (year <= 2030))
    df["ManufactureYear_missing"] = year.isnull().astype(int)
    df["ManufactureYear"] = year.fillna(year.iloc[:train_len].median())

    age = (df["SaleYear"] - df["ManufactureYear"]).clip(0, 65)
    df["EquipmentAge"] = age
    df["Log_Age"] = np.log1p(age)
    df["Sqrt_Age"] = np.sqrt(age)
    df["Age_Squared"] = age ** 2
    df["Age_Decade"] = (age // 10).astype(int)
    df["Age_Bucket"] = pd.cut(age, bins=[-1, 2, 5, 10, 20, 65], labels=[0, 1, 2, 3, 4]).astype(int)
    df["IsNew"] = (age <= 2).astype(int)
    df["IsOld"] = (age >= 20).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Operational hours
# --------------------------------------------------------------------------- #
def add_hours_features(df: pd.DataFrame, train_len: int) -> pd.DataFrame:
    hours = df["OperationalHoursMeter"].replace(0, np.nan)  # 0 means "not recorded"
    df["Hours_missing"] = hours.isnull().astype(int)
    hours = hours.fillna(hours.iloc[:train_len].median())
    df["OperationalHoursMeter"] = hours

    df["Log_Hours"] = np.log1p(hours)
    df["Sqrt_Hours"] = np.sqrt(hours)
    df["Hours_per_year"] = hours / (df["EquipmentAge"] + 1)
    df["Log_Hours_per_year"] = np.log1p(df["Hours_per_year"])
    df["Hours_x_Age"] = df["Log_Hours"] * df["Log_Age"]
    df["Hours_x_SaleYear"] = df["Log_Hours"] * df["SaleYear"]
    return df


# --------------------------------------------------------------------------- #
# Model descriptor string
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"(\d+\.?\d*)")


def extract_spec_number(text) -> float:
    """First number in a model string, e.g. '950FII' -> 950.0, 'PC200LC6' -> 200.0.

    Within a manufacturer family the model number is a rough proxy for machine size.
    """
    if pd.isna(text):
        return np.nan
    match = _NUM_RE.search(str(text))
    return float(match.group(1)) if match else np.nan


def add_spec_features(df: pd.DataFrame, train_len: int) -> pd.DataFrame:
    if "Spec_FullDescriptor" in df.columns:
        num = df["Spec_FullDescriptor"].apply(extract_spec_number)
        df["Spec_Extracted_Num"] = num.fillna(num.iloc[:train_len].median())
        df["Spec_Num_x_Age"] = df["Spec_Extracted_Num"] * df["EquipmentAge"]

    if "ProductConfigID" in df.columns:
        med = df.groupby("ProductConfigID")["OperationalHoursMeter"].transform("median")
        df["Hours_Ratio_vs_Config"] = df["OperationalHoursMeter"] / (med + 1)

    if "Spec_BaseClass" in df.columns:
        med = df.groupby("Spec_BaseClass")["OperationalHoursMeter"].transform("median")
        df["Hours_Ratio_vs_BaseClass"] = df["OperationalHoursMeter"] / (med + 1)
    return df


def add_utilization_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    if "UtilizationTier" in df.columns:
        df["UtilizationTier"] = df["UtilizationTier"].fillna("Unknown")
        df["UtilizationTier_ord"] = df["UtilizationTier"].map({"Low": 0, "Medium": 1, "High": 2}).fillna(-1)
    return df


# --------------------------------------------------------------------------- #
# Cross / aggregate / relative features
# --------------------------------------------------------------------------- #
def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two categoricals so tree models can split on their combination."""
    for name, left, right in CROSS_FEATURES:
        df[name] = df[left].astype(str) + "_" + df[right].astype(str)
    return df


def add_group_aggregates(df: pd.DataFrame, group_cols: list[str] = AGG_GROUP_COLS) -> pd.DataFrame:
    """Hours / age statistics and group size for each grouping column."""
    for col in group_cols:
        if col not in df.columns:
            continue
        hours = df.groupby(col)["OperationalHoursMeter"].agg(["mean", "median", "std"])
        hours.columns = [f"{col}_Hours_{s}" for s in hours.columns]
        df = df.merge(hours, on=col, how="left")

        age = df.groupby(col)["EquipmentAge"].agg(["mean", "std", "min", "max"])
        age.columns = [f"{col}_Age_{s}" for s in age.columns]
        df = df.merge(age, on=col, how="left")

        count = df.groupby(col).size().rename(f"{col}_Count")
        df = df.merge(count, on=col, how="left")
    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Where does a unit sit relative to other units of the same configuration?"""
    g = df.groupby("ProductConfigID")
    df["Hours_pct_in_cfg"] = g["OperationalHoursMeter"].rank(pct=True)
    df["Age_pct_in_cfg"] = g["EquipmentAge"].rank(pct=True)
    df["Day_pct_in_cfg"] = g["DaysSinceStart"].rank(pct=True)
    df["Hours_dev_cfg"] = df["Log_Hours"] - g["Log_Hours"].transform("median")
    df["Age_dev_cfg"] = df["EquipmentAge"] - g["EquipmentAge"].transform("median")
    df["Day_dev_cfg"] = df["DaysSinceStart"] - g["DaysSinceStart"].transform("median")
    return df


def build_base_features(df: pd.DataFrame, train_len: int, verbose: bool = True) -> pd.DataFrame:
    """Run every target-free feature step in the same order as the notebook."""
    df = cast_id_like_columns(df)
    df, dropped = drop_mostly_missing(df)
    if verbose:
        print(f"Dropped mostly-missing columns: {dropped}")
    df = add_date_features(df)
    df = add_age_features(df, train_len)
    df = add_hours_features(df, train_len)
    df = add_spec_features(df, train_len)
    df = add_utilization_ordinal(df)
    df = add_cross_features(df)
    df = add_group_aggregates(df)
    df = add_relative_features(df)
    return df
