"""Evaluation metrics for few-shot defect classification.

Provides balanced accuracy, per-class F1, confusion matrix,
and learning curve computation (Area Under Learning Curve).
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Overall classification accuracy."""
    return (preds == targets).mean()


def compute_balanced_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Balanced accuracy (mean of per-class recall). Handles imbalanced classes."""
    return balanced_accuracy_score(targets, preds)


def compute_per_class_f1(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Per-class and macro-averaged F1 scores.

    Returns:
        Dictionary with 'per_class' (list of F1 per class) and 'macro' (averaged).
    """
    per_class = f1_score(targets, preds, average=None, zero_division=0)
    macro = f1_score(targets, preds, average="macro", zero_division=0)
    return {"per_class": per_class.tolist(), "macro": float(macro)}


def compute_confusion_matrix(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Confusion matrix."""
    return confusion_matrix(targets, preds)


def compute_classification_report(preds: np.ndarray, targets: np.ndarray, class_names=None) -> str:
    """Full classification report string."""
    return classification_report(targets, preds, target_names=class_names, zero_division=0)


def compute_area_under_learning_curve(accuracies: list) -> float:
    """Area Under the Learning Curve (AULC).

    A single scalar metric capturing overall learning efficiency.
    Higher = faster learning.

    Args:
        accuracies: List of accuracy values at each step.

    Returns:
        AULC normalized to [0, 1] range.
    """
    if len(accuracies) == 0:
        return 0.0
    return np.trapz(accuracies) / len(accuracies)


def compute_time_to_threshold(
    accuracies: list,
    threshold: float = 0.85,
) -> int:
    """Number of examples needed to reach an accuracy threshold.

    Args:
        accuracies: List of accuracy values at each step.
        threshold: Target accuracy (default 85%).

    Returns:
        Number of examples needed, or -1 if never reached.
    """
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            return i + 1
    return -1
