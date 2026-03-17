"""Evaluation metrics for IR drop prediction.

Implements the exact contest scoring formulas:
  - MAE: Mean Absolute Error (60% of score)
  - F1:  Hotspot detection F1 (30% of score)
         Hotspot threshold = 0.9 * max(actual_ir_drop)
         Same threshold applied to both prediction and ground truth
  - Runtime: inference time (10% of score)
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
)


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error across all pixels."""
    return mean_absolute_error(target.flatten(), pred.flatten())


def compute_hotspot_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict:
    """Compute hotspot classification metrics using contest formula.

    Contest definition:
      threshold = 0.9 * max(actual_ir_drop)
      A pixel is a hotspot if its value >= threshold
      The SAME threshold is applied to both prediction and ground truth.

    Args:
        pred: Predicted voltage drop map.
        target: Ground truth voltage drop map.

    Returns:
        Dictionary with f1, precision, recall, and threshold.
    """
    pred_flat = pred.flatten().astype(np.float64)
    target_flat = target.flatten().astype(np.float64)

    # Contest formula: 90% of the actual maximum IR drop
    actual_max = target_flat.max()
    threshold = 0.9 * actual_max

    target_binary = (target_flat >= threshold).astype(int)
    # Apply the SAME threshold to predictions (not a separate percentile)
    pred_binary = (pred_flat >= threshold).astype(int)

    return {
        "f1": f1_score(target_binary, pred_binary, zero_division=0),
        "precision": precision_score(target_binary, pred_binary, zero_division=0),
        "recall": recall_score(target_binary, pred_binary, zero_division=0),
        "threshold": float(threshold),
    }


def compute_auc_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute AUC-ROC and AUC-PR for hotspot detection."""
    pred_flat = pred.flatten().astype(np.float64)
    target_flat = target.flatten().astype(np.float64)

    # Use contest threshold for binary labels
    threshold = 0.9 * target_flat.max()
    target_binary = (target_flat >= threshold).astype(int)

    if target_binary.sum() == 0 or target_binary.sum() == len(target_binary):
        return {"auroc": 0.0, "auprc": 0.0}

    return {
        "auroc": roc_auc_score(target_binary, pred_flat),
        "auprc": average_precision_score(target_binary, pred_flat),
    }


def compute_per_testcase_metrics(
    preds: list,
    targets: list,
) -> dict:
    """Compute metrics per testcase and average (contest scoring).

    Args:
        preds: List of predicted arrays, one per testcase.
        targets: List of ground truth arrays, one per testcase.

    Returns:
        Dictionary with per-testcase and averaged metrics.
    """
    maes = []
    f1s = []
    per_testcase = []

    for i, (p, t) in enumerate(zip(preds, targets)):
        mae = compute_mae(p, t)
        hotspot = compute_hotspot_metrics(p, t)
        maes.append(mae)
        f1s.append(hotspot["f1"])
        per_testcase.append({
            "testcase": i,
            "mae": mae,
            "f1": hotspot["f1"],
            "precision": hotspot["precision"],
            "recall": hotspot["recall"],
            "threshold": hotspot["threshold"],
        })

    return {
        "mae_avg": float(np.mean(maes)),
        "f1_avg": float(np.mean(f1s)),
        "mae_per_testcase": maes,
        "f1_per_testcase": f1s,
        "details": per_testcase,
    }


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict:
    """Compute all evaluation metrics for a single sample or batch."""
    mae = compute_mae(pred, target)
    hotspot = compute_hotspot_metrics(pred, target)
    auc = compute_auc_metrics(pred, target)

    return {
        "mae": mae,
        **hotspot,
        **auc,
    }
