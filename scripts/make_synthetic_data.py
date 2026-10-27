"""Generate a small synthetic dataset with the competition's schema.

The real data is only available to competition participants, so this lets anyone
(and the test-suite) run the full pipeline end-to-end in seconds:

    python scripts/make_synthetic_data.py --out data/synthetic --n-train 3000 --n-test 500
    python scripts/train.py --data-dir data/synthetic --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CAT_COLS = ["col1", "CabinType", "Forks", "col3", "col4", "DrivetrainType", "col5", "col6", "col7",
            "col8", "col9", "col10", "col11", "col12", "col13", "col14", "col15", "col16", "col18",
            "col19", "col20", "col21", "col22", "col23", "col24", "col25", "col27", "col28", "col29",
            "col30"]


def make_frame(n: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    n_cfg = max(20, n // 40)
    cfg = rng.integers(1, n_cfg + 1, n)
    base_classes = np.array(["950", "310", "790", "416", "430", "988", "PC200", "420", "214"])
    base = base_classes[cfg % len(base_classes)]
    year = rng.integers(1985, 2011, n)
    year[rng.random(n) < 0.08] = 1000  # placeholder years, as in the real data
    sale = pd.to_datetime("1990-01-31") + pd.to_timedelta(rng.integers(0, 8000, n), unit="D")
    hours = rng.gamma(2.0, 3000, n).round()
    hours[rng.random(n) < 0.3] = 0
    hours = np.where(rng.random(n) < 0.3, np.nan, hours)

    df = pd.DataFrame({
        "TransactionID": np.arange(start_id, start_id + n),
        "AssetID": rng.integers(100000, 2000000, n),
        "ProductConfigID": cfg,
        "DataOriginCode": rng.choice(["ACH138", "BCX221", "ZZQ001"], n),
        "VendorPartnerID": rng.choice(["GTX", "KLM", "PQR", "UVW"], n),
        "ManufactureYear": year,
        "OperationalHoursMeter": hours,
        "UtilizationTier": rng.choice(["Low", "Medium", "High", None], n, p=[0.15, 0.1, 0.1, 0.65]),
        "TransactionDate": sale.strftime("%Y-%m-%d"),
        "Spec_FullDescriptor": [f"{b}{s}" for b, s in zip(base, rng.choice(["D", "G", "FII", "LC6", "E"], n))],
        "Spec_BaseClass": base,
        "Spec_SubClass": rng.choice(["D", "G", "E", None], n),
        "Spec_ReleaseSeries": rng.choice(["II", "6", None], n, p=[0.1, 0.1, 0.8]),
        "Spec_VariantModifier": rng.choice(["LC", "XL", None], n, p=[0.2, 0.2, 0.6]),
        "AssetScaleFactor": rng.choice(["Mini", "Small", "Medium", "Large", None], n),
        "FunctionalClassification": rng.choice(["Wheel Loader", "Track Excavator", "Motorgrader", "Backhoe"], n),
        "RegionCode": rng.choice(["Texas", "Florida", "Ohio", "California"], n),
        "InventoryGroupCategory": rng.choice(["WL", "TEX", "MG", "BL", "SSL", "TTT"], n),
        "InventoryGroupDescription": rng.choice(["Wheel Loader", "Track Excavators", "Motor Graders"], n),
    })
    for col in CAT_COLS:
        miss = 0.999 if col in ("col18", "col19") else rng.uniform(0.05, 0.9)
        vals = rng.choice(["None or Unspecified", "Standard", "Double", "EROPS"], n).astype(object)
        vals[rng.random(n) < miss] = None
        df[col] = vals

    age = np.clip(np.asarray(sale.year) - np.where(year < 1900, 1999, year), 0, 40)
    log_price = (10.8 - 0.045 * age + 0.03 * (cfg % 7) + 0.15 * (df["InventoryGroupCategory"] == "WL")
                 - 0.02 * np.log1p(np.nan_to_num(hours)) + rng.normal(0, 0.25, n))
    df.insert(1, "TargetValue", (np.round(np.expm1(log_price) / 500) * 500).clip(4750, 142000))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train = make_frame(args.n_train, rng, 1139300)
    test = make_frame(args.n_test, rng, 1139300 + args.n_train).drop(columns=["TargetValue"])
    train.to_csv(args.out / "train.csv", index=False)
    test.to_csv(args.out / "test.csv", index=False)
    pd.DataFrame({"TransactionID": test["TransactionID"], "TargetValue": 30000.0}) \
        .to_csv(args.out / "sample_submission.csv", index=False)
    print(f"Wrote synthetic train {train.shape} / test {test.shape} to {args.out}")


if __name__ == "__main__":
    main()
