"""Level-2 stacking, blend-weight optimisation and price-grid snapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold

from .metrics import rmse


def stack_bayesian_ridge(
    oof_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y: pd.Series,
    folds: KFold,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Fit a Bayesian Ridge meta-model on the level-1 OOF predictions (K-fold)."""
    oof = np.zeros(len(oof_matrix))
    test = np.zeros(len(test_matrix))
    scores = []
    print("\n=== Bayesian Ridge stacker ===")
    for fold, (tr_idx, val_idx) in enumerate(folds.split(oof_matrix)):
        model = BayesianRidge()
        model.fit(oof_matrix[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = model.predict(oof_matrix[val_idx])
        test += model.predict(test_matrix) / folds.get_n_splits()
        scores.append(rmse(y.iloc[val_idx], oof[val_idx]))
        print(f"  Fold {fold + 1}: {scores[-1]:.4f}")
    print(f"STACK OOF RMSLE: {rmse(y, oof):.4f}")
    return oof, test, scores


def optimize_weights(oof_matrix: np.ndarray, y: pd.Series, init: list[float] | None = None) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise OOF RMSLE (Nelder-Mead)."""
    n = oof_matrix.shape[1]
    init = np.ones(n) if init is None else np.asarray(init, dtype=float)
    y_arr = y.to_numpy()

    def loss(w):
        w = np.abs(w)
        s = w.sum()
        if s < 1e-10:
            return 1e10
        return rmse(y_arr, oof_matrix @ (w / s))

    res = minimize(loss, init, method="Nelder-Mead",
                   options={"maxiter": 30000, "xatol": 1e-7, "fatol": 1e-7})
    w = np.abs(res.x)
    return w / w.sum()


def snap_to_grid(prices: np.ndarray, grid: float, lo: float, hi: float) -> np.ndarray:
    """Round to the nearest `grid` dollars and clip to the observed price range.

    Sale prices in the training data are quoted in round amounts (e.g. 26,500 or
    21,000), so snapping *can* help; the pipeline keeps it only if it lowers OOF RMSLE.
    """
    return np.clip(np.round(prices / grid) * grid, lo, hi)
