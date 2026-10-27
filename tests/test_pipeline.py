"""Unit + smoke tests. Run with `pytest -q`. Uses synthetic data, no Kaggle access needed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heavy_equipment.config import TrainConfig  # noqa: E402
from heavy_equipment.data import build_dataset  # noqa: E402
from heavy_equipment.encoding import target_encode  # noqa: E402
from heavy_equipment.ensemble import optimize_weights, snap_to_grid  # noqa: E402
from heavy_equipment.features import add_age_features, build_base_features, extract_spec_number  # noqa: E402
from heavy_equipment.metrics import rmse, rmsle  # noqa: E402
from heavy_equipment.pipeline import make_features, run  # noqa: E402


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "make_synthetic_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dataset():
    gen = _load_generator()
    rng = np.random.default_rng(1)
    train = gen.make_frame(1200, rng, 1)
    test = gen.make_frame(300, rng, 5000).drop(columns=["TargetValue"])
    return build_dataset(train, test)


def test_rmsle_matches_definition():
    a, p = np.array([100.0, 2000.0]), np.array([110.0, 1800.0])
    expected = np.sqrt(np.mean((np.log1p(p) - np.log1p(a)) ** 2))
    assert rmsle(a, p) == pytest.approx(expected)
    assert rmse(np.log1p(a), np.log1p(p)) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [("950FII", 950.0), ("PC200LC6", 200.0), ("D6R", 6.0)])
def test_extract_spec_number(text, expected):
    assert extract_spec_number(text) == expected


def test_extract_spec_number_handles_missing():
    assert np.isnan(extract_spec_number(np.nan))
    assert np.isnan(extract_spec_number("ABC"))


def test_placeholder_manufacture_years_are_imputed():
    df = pd.DataFrame({"ManufactureYear": [1000, 1001, 2000, 2002], "SaleYear": [2005, 2006, 2005, 2006]})
    out = add_age_features(df, train_len=4)
    assert out["ManufactureYear_missing"].tolist() == [1, 1, 0, 0]
    assert out.loc[0, "ManufactureYear"] == 2001.0  # train median of the valid years
    assert (out["EquipmentAge"] >= 0).all()


def test_target_encoding_is_out_of_fold():
    from sklearn.model_selection import KFold
    # every category is unique -> an OOF encoding can never see its own target
    df = pd.DataFrame({"c": [f"u{i}" for i in range(50)] + ["u0"] * 5})
    y = pd.Series(np.arange(50, dtype=float))
    out = target_encode(df, "c", y, 50, KFold(5, shuffle=True, random_state=0), global_mean=y.mean(), smoothing=10)
    assert np.allclose(out["c_te"].iloc[1:50], y.mean())         # unseen in fitting fold -> global mean
    assert out["c_te"].iloc[50] == pytest.approx((0 * 1 + y.mean() * 10) / 11)  # test row uses all train rows


def test_snap_to_grid():
    out = snap_to_grid(np.array([1240.0, 1260.0, 10.0, 1e9]), 500, 1000, 5000)
    assert out.tolist() == [1000.0, 1500.0, 1000.0, 5000.0]


def test_optimize_weights_prefers_better_model():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=500))
    good, bad = y + rng.normal(0, 0.1, 500), y + rng.normal(0, 1.0, 500)
    w = optimize_weights(np.column_stack([good, bad]), y)
    assert w.sum() == pytest.approx(1.0) and w[0] > 0.9


def test_feature_matrix_shape_and_no_nans(dataset):
    X_train, X_test = make_features(dataset, TrainConfig())
    assert len(X_train) == dataset.train_len and len(X_test) == len(dataset.test_ids)
    assert list(X_train.columns) == list(X_test.columns)
    assert not X_train.isnull().any().any() and not X_test.isnull().any().any()
    assert all(np.issubdtype(t, np.number) for t in X_train.dtypes)
    assert "col18" not in X_train.columns  # >99% missing -> dropped


def test_base_features_preserve_row_order(dataset):
    df = build_base_features(dataset.combined.copy(), dataset.train_len, verbose=False)
    assert df["TransactionID"].tolist() == dataset.combined["TransactionID"].tolist()


def test_end_to_end_quick_run(dataset, tmp_path):
    cfg = TrainConfig(output_dir=tmp_path, n_folds=3, estimator_scale=0.01, early_stopping_rounds=20,
                      models=("lgb1", "lgb2", "xgb", "cat"))
    metrics = run(cfg, ds=dataset)
    sub = pd.read_csv(tmp_path / "submission.csv")
    assert len(sub) == len(dataset.test_ids) and (sub["TargetValue"] > 0).all()
    assert set(metrics["blend_weights"]) == {"lgb1", "lgb2", "xgb", "cat", "stack"}
    assert metrics["blend_oof_rmsle"] < 0.6
