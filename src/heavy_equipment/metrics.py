"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """RMSE. Applied to log1p-space values this *is* RMSLE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmsle(y_true, y_pred) -> float:
    """Competition metric on raw prices: sqrt(mean((log1p(p) - log1p(a))^2))."""
    return rmse(np.log1p(np.asarray(y_true, dtype=float)), np.log1p(np.asarray(y_pred, dtype=float)))
