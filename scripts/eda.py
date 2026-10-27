"""Exploratory data analysis: summary tables and figures written to reports/.

    python scripts/eda.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = p.parse_args()

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv", low_memory=False)
    test = pd.read_csv(args.data_dir / "test.csv", low_memory=False)
    y = train["TargetValue"]

    # --- tables -------------------------------------------------------------
    missing = pd.concat([train.drop(columns="TargetValue"), test]).isnull().mean().sort_values(ascending=False)
    missing.rename("missing_ratio").to_csv(args.out_dir / "missing_ratio.csv")
    card = train.nunique().sort_values(ascending=False).rename("n_unique")
    card.to_csv(args.out_dir / "cardinality.csv")
    y.describe().to_csv(args.out_dir / "target_summary.csv")

    # --- figures ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(y, bins=80, color="#4C72B0")
    ax[0].set(title="Sale price (USD)", xlabel="TargetValue", ylabel="count")
    ax[1].hist(np.log1p(y), bins=80, color="#55A868")
    ax[1].set(title="log1p(price) - the modelled target", xlabel="log1p(TargetValue)")
    fig.tight_layout()
    fig.savefig(fig_dir / "target_distribution.png", dpi=120)
    plt.close(fig)

    top = missing[missing > 0].head(30)[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top.index, top.values, color="#C44E52")
    ax.set(title="Share of missing values (top 30 columns)", xlabel="missing ratio", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "missing_values.png", dpi=120)
    plt.close(fig)

    sale_year = pd.to_datetime(train["TransactionDate"], errors="coerce").dt.year
    yr = train["ManufactureYear"].where(train["ManufactureYear"].between(1900, 2030))
    age = (sale_year - yr).clip(0, 65)
    by_age = pd.DataFrame({"age": age, "logp": np.log1p(y)}).dropna().groupby("age")["logp"].median()
    by_year = pd.DataFrame({"year": sale_year, "logp": np.log1p(y)}).groupby("year")["logp"].median()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(by_age.index, by_age.values, marker="o", ms=3)
    ax[0].set(title="Median log-price vs equipment age", xlabel="age at sale (years)", ylabel="median log1p(price)")
    ax[1].plot(by_year.index, by_year.values, marker="o", ms=3, color="#8172B2")
    ax[1].set(title="Median log-price vs sale year", xlabel="sale year")
    fig.tight_layout()
    fig.savefig(fig_dir / "price_vs_age_and_year.png", dpi=120)
    plt.close(fig)

    cat = train.groupby("InventoryGroupCategory")["TargetValue"].median().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(cat.index.astype(str), cat.values, color="#DD8452")
    ax.set(title="Median price by inventory group", xlabel="median TargetValue (USD)")
    fig.tight_layout()
    fig.savefig(fig_dir / "price_by_inventory_group.png", dpi=120)
    plt.close(fig)

    print(f"Train {train.shape}, test {test.shape}")
    print(f"Wrote tables to {args.out_dir} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
