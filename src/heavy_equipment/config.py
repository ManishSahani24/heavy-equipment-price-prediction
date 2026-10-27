"""Central configuration: paths, column groups and model hyper-parameters.

All values mirror the final competition notebook (Version 23, private LB RMSLE 0.18741)
so that running the pipeline end-to-end reproduces that submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

ID_COL = "TransactionID"
TARGET_COL = "TargetValue"
TEST_FLAG = "_is_test"

# High-cardinality IDs that look numeric but are really categorical.
ID_LIKE_CATEGORICALS = ["ProductConfigID", "AssetID"]

# Columns with more than this share of missing values are dropped (col18, col19).
MAX_MISSING_RATIO = 0.99

# Groups used for the non-target aggregate features.
AGG_GROUP_COLS = [
    "ProductConfigID",
    "Spec_BaseClass",
    "FunctionalClassification",
    "InventoryGroupCategory",
    "RegionCode",
    "Cfg_AgeBucket",
]

# (new column name, left column, right column) for string cross features.
CROSS_FEATURES = [
    ("Cfg_AgeBucket", "ProductConfigID", "Age_Bucket"),
    ("Cfg_SaleDecade", "ProductConfigID", "SaleDecade"),
    ("Cfg_SaleYear", "ProductConfigID", "SaleYear"),
    ("Base_AgeBucket", "Spec_BaseClass", "Age_Bucket"),
    ("Base_SaleDecade", "Spec_BaseClass", "SaleDecade"),
    ("Func_AgeBucket", "FunctionalClassification", "Age_Bucket"),
    ("InvCat_SaleYear", "InventoryGroupCategory", "SaleYear"),
    ("Region_InvCat", "RegionCode", "InventoryGroupCategory"),
    ("Vendor_InvCat", "VendorPartnerID", "InventoryGroupCategory"),
    ("Scale_InvCat", "AssetScaleFactor", "InventoryGroupCategory"),
    ("Cfg_Scale", "ProductConfigID", "AssetScaleFactor"),
]

# label-encoded categorical x numeric products
NUMERIC_INTERACTION_CATS = ["InventoryGroupCategory", "FunctionalClassification", "Spec_BaseClass", "RegionCode"]
NUMERIC_INTERACTION_NUMS = ["EquipmentAge", "Log_Hours", "SaleYear"]

# Predictions are snapped to the $500 price grid if that improves OOF RMSLE.
PRICE_GRID = 500


@dataclass
class TrainConfig:
    """Run-time options for the training pipeline."""

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    n_folds: int = 5
    cv_seed: int = 42
    te_folds: int = 5
    te_seed: int = 2024
    te_smoothing: float = 10.0
    models: tuple[str, ...] = ("lgb1", "lgb2", "xgb", "cat")
    # Multiply n_estimators / iterations by this factor (use < 1 for quick smoke runs).
    estimator_scale: float = 1.0
    early_stopping_rounds: int = 300
    n_jobs: int = -1

    lgb1_params: dict = field(default_factory=lambda: dict(
        n_estimators=10000, learning_rate=0.008, num_leaves=255, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        min_child_samples=20, reg_alpha=0.05, reg_lambda=0.2,
    ))
    lgb2_params: dict = field(default_factory=lambda: dict(
        n_estimators=10000, learning_rate=0.006, num_leaves=511, max_depth=-1,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        min_child_samples=30, reg_alpha=0.1, reg_lambda=0.5,
    ))
    xgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=10000, learning_rate=0.008, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
    ))
    cat_params: dict = field(default_factory=lambda: dict(
        iterations=8000, learning_rate=0.02, depth=8, l2_leaf_reg=3,
        bootstrap_type="Bernoulli", subsample=0.8, loss_function="RMSE",
    ))
    # per-model seed offsets (the notebook used seed + fold)
    seeds: dict = field(default_factory=lambda: dict(lgb1=42, lgb2=100, xgb=42, cat=42))
