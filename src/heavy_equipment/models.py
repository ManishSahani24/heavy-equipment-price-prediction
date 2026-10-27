"""K-fold training of the level-1 gradient-boosting models."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

from .config import TrainConfig
from .metrics import rmse

# lightgbm>=4.6 prefers eval_X/eval_y; eval_set keeps the code compatible with older versions
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


@dataclass
class CVResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    fold_scores: list[float]
    best_iterations: list[int]
    oof_score: float


def _scaled(params: dict, key: str, scale: float) -> dict:
    params = dict(params)
    params[key] = max(10, int(params[key] * scale))
    return params


def _fit_lgb(params, seed, cfg, X_tr, y_tr, X_val, y_val):
    m = lgb.LGBMRegressor(**_scaled(params, "n_estimators", cfg.estimator_scale),
                          random_state=seed, n_jobs=cfg.n_jobs, verbose=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), lgb.log_evaluation(-1)])
    return m, m.best_iteration_


def _fit_xgb(params, seed, cfg, X_tr, y_tr, X_val, y_val):
    m = xgb.XGBRegressor(**_scaled(params, "n_estimators", cfg.estimator_scale),
                         random_state=seed, n_jobs=cfg.n_jobs,
                         early_stopping_rounds=cfg.early_stopping_rounds,
                         eval_metric="rmse", verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m, m.best_iteration


def _fit_cat(params, seed, cfg, X_tr, y_tr, X_val, y_val):
    m = CatBoostRegressor(**_scaled(params, "iterations", cfg.estimator_scale),
                          random_seed=seed, early_stopping_rounds=cfg.early_stopping_rounds,
                          thread_count=cfg.n_jobs, allow_writing_files=False, verbose=False)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    return m, m.get_best_iteration()


MODEL_REGISTRY = {
    "lgb1": (_fit_lgb, "lgb1_params", "LightGBM v1 (255 leaves)"),
    "lgb2": (_fit_lgb, "lgb2_params", "LightGBM v2 (511 leaves)"),
    "xgb": (_fit_xgb, "xgb_params", "XGBoost (depth 7)"),
    "cat": (_fit_cat, "cat_params", "CatBoost (depth 8)"),
}


def train_cv(
    name: str,
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: KFold,
    cfg: TrainConfig,
) -> CVResult:
    """Train one model type on every fold; return OOF and fold-averaged test predictions."""
    fit_fn, params_attr, label = MODEL_REGISTRY[name]
    params = getattr(cfg, params_attr)
    base_seed = cfg.seeds[name]

    oof = np.zeros(len(X_train))
    test = np.zeros(len(X_test))
    scores, iters = [], []
    print(f"\n=== {label} ===")
    for fold, (tr_idx, val_idx) in enumerate(folds.split(X_train)):
        t0 = time.time()
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model, best_iter = fit_fn(params, base_seed + fold, cfg, X_tr, y_tr, X_val, y_val)
        oof[val_idx] = model.predict(X_val)
        test += model.predict(X_test) / folds.get_n_splits()
        score = rmse(y_val, oof[val_idx])
        scores.append(score)
        iters.append(int(best_iter) if best_iter is not None else -1)
        print(f"  Fold {fold + 1}: {score:.4f}  iter={best_iter}  ({time.time() - t0:.0f}s)")

    result = CVResult(name=name, oof=oof, test=test, fold_scores=scores,
                      best_iterations=iters, oof_score=rmse(y, oof))
    print(f"{name.upper()} OOF RMSLE: {result.oof_score:.4f}")
    return result
