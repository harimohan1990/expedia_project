"""Standardized regression and classification metrics for advertising forecasting models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, WAPE, and R2 safely."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(root_mean_squared_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.std(yt) > 0 else 0.0

    denom_sum = np.sum(np.abs(yt))
    wape = float(np.sum(np.abs(yt - yp)) / denom_sum) if denom_sum > 0 else 0.0

    nonzero = np.abs(yt) > 1e-8
    mape = (
        float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])))
        if np.any(nonzero)
        else 0.0
    )

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 6),
        "wape": round(wape, 6),
        "r2": round(r2, 6),
    }


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute Precision, Recall, F1, and ROC-AUC safely."""
    yt = np.asarray(y_true, dtype=int)
    yp_prob = np.asarray(y_prob, dtype=float)
    yp_cls = (yp_prob >= threshold).astype(int)

    roc = float(roc_auc_score(yt, yp_prob)) if len(np.unique(yt)) > 1 else 0.5
    return {
        "precision": round(float(precision_score(yt, yp_cls, zero_division=0)), 6),
        "recall": round(float(recall_score(yt, yp_cls, zero_division=0)), 6),
        "f1": round(float(f1_score(yt, yp_cls, zero_division=0)), 6),
        "roc_auc": round(roc, 6),
    }
