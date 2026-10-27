"""End-to-end training pipeline: raw CSVs -> features -> CV models -> ensemble -> submission."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import ID_COL, PRICE_GRID, TARGET_COL, TrainConfig
from .data import Dataset, load_dataset
from .encoding import encode, split_matrices
from .ensemble import optimize_weights, snap_to_grid, stack_bayesian_ridge
from .features import build_base_features
from .metrics import rmse
from .models import train_cv

# initial Nelder-Mead guess used in the notebook: [lgb1, lgb2, xgb, cat, stack]
_INIT_WEIGHTS = {"lgb1": 1.0, "lgb2": 1.0, "xgb": 0.5, "cat": 0.5, "stack": 2.0}


def make_features(ds: Dataset, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_base_features(ds.combined.copy(), ds.train_len)
    df = encode(df, ds.y, ds.train_len, cfg.te_folds, cfg.te_seed, cfg.te_smoothing)
    X_train, X_test = split_matrices(df)
    print(f"Feature matrix: train {X_train.shape}, test {X_test.shape}")
    return X_train, X_test


def run(cfg: TrainConfig, ds: Dataset | None = None) -> dict:
    t_start = time.time()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ds if ds is not None else load_dataset(cfg.data_dir)
    print(f"Train rows: {ds.train_len:,} | Test rows: {len(ds.test_ids):,}")

    X_train, X_test = make_features(ds, cfg)
    y = ds.y
    folds = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.cv_seed)

    # ---- level 1 -------------------------------------------------------------
    results = [train_cv(name, X_train, y, X_test, folds, cfg) for name in cfg.models]
    names = [r.name for r in results]
    oof_l1 = np.column_stack([r.oof for r in results])
    test_l1 = np.column_stack([r.test for r in results])

    # ---- level 2 -------------------------------------------------------------
    oof_stack, test_stack, stack_scores = stack_bayesian_ridge(oof_l1, test_l1, y, folds)

    blend_names = names + ["stack"]
    oof_all = np.column_stack([oof_l1, oof_stack])
    test_all = np.column_stack([test_l1, test_stack])
    weights = optimize_weights(oof_all, y, [_INIT_WEIGHTS.get(n, 1.0) for n in blend_names])
    print("\nBlend weights: " + "  ".join(f"{n}={w:.3f}" for n, w in zip(blend_names, weights)))

    oof_blend = oof_all @ weights
    test_blend = test_all @ weights

    # ---- post-processing -----------------------------------------------------
    oof_raw = np.expm1(oof_blend)
    oof_snap = snap_to_grid(oof_raw, PRICE_GRID, ds.price_min, ds.price_max)
    score_raw = rmse(y, oof_blend)
    score_snap = rmse(y, np.log1p(oof_snap))
    use_snap = score_snap < score_raw
    print(f"\nOOF RMSLE raw: {score_raw:.4f} | snapped: {score_snap:.4f} -> using {'snapped' if use_snap else 'raw'}")

    test_raw = np.expm1(test_blend)
    final = snap_to_grid(test_raw, PRICE_GRID, ds.price_min, ds.price_max) if use_snap else test_raw.clip(0)

    # ---- save artefacts ------------------------------------------------------
    sub = pd.DataFrame({ID_COL: ds.test_ids, TARGET_COL: final})
    sub.to_csv(out_dir / "submission.csv", index=False)
    pd.DataFrame(oof_all, columns=blend_names).assign(blend=oof_blend, y=y.values) \
        .to_csv(out_dir / "oof_predictions.csv", index=False)

    metrics = {
        "n_features": int(X_train.shape[1]),
        "n_folds": cfg.n_folds,
        "models": {r.name: {"oof_rmsle": round(r.oof_score, 5),
                            "fold_rmsle": [round(s, 5) for s in r.fold_scores],
                            "best_iterations": r.best_iterations} for r in results},
        "stack": {"oof_rmsle": round(rmse(y, oof_stack), 5), "fold_rmsle": [round(s, 5) for s in stack_scores]},
        "blend_weights": {n: round(float(w), 4) for n, w in zip(blend_names, weights)},
        "blend_oof_rmsle": round(score_raw, 5),
        "blend_oof_rmsle_snapped": round(score_snap, 5),
        "used_snapping": bool(use_snap),
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved submission.csv, oof_predictions.csv, metrics.json to {out_dir}")
    print(sub[TARGET_COL].describe())
    return metrics
