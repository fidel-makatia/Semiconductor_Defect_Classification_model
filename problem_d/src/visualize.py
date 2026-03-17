"""Visualization utilities for IR drop prediction results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use("Agg")


def plot_prediction(
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: str,
    title: str = "",
):
    """Plot ground truth, prediction, and error side-by-side.

    Args:
        gt: Ground truth IR drop map [H, W].
        pred: Predicted IR drop map [H, W].
        save_path: Path to save the figure.
        title: Optional title prefix.
    """
    error = np.abs(gt - pred)
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(gt, cmap="hot", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title(f"{title} Ground Truth IR Drop")
    plt.colorbar(im0, ax=axes[0], label="Voltage Drop")

    im1 = axes[1].imshow(pred, cmap="hot", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title(f"{title} Predicted IR Drop")
    plt.colorbar(im1, ax=axes[1], label="Voltage Drop")

    im2 = axes[2].imshow(error, cmap="Blues", interpolation="nearest")
    axes[2].set_title(f"Absolute Error (MAE={error.mean():.4f})")
    plt.colorbar(im2, ax=axes[2], label="Error")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(
    target_binary: np.ndarray,
    pred_scores: np.ndarray,
    save_path: str,
):
    """Plot ROC and Precision-Recall curves.

    Args:
        target_binary: Binary hotspot labels [N].
        pred_scores: Continuous prediction scores [N].
        save_path: Path to save the figure.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(target_binary, pred_scores)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC-ROC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve (Hotspot Detection)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # PR curve
    prec, rec, _ = precision_recall_curve(target_binary, pred_scores)
    pr_auc = auc(rec, prec)
    ax2.plot(rec, prec, "r-", linewidth=2, label=f"AUC-PR = {pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve (Hotspot Detection)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mae_distribution(
    mae_per_sample: list,
    save_path: str,
):
    """Plot histogram of MAE across test samples.

    Args:
        mae_per_sample: List of MAE values, one per test design.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mae_per_sample, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.mean(mae_per_sample),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean MAE = {np.mean(mae_per_sample):.4f}",
    )
    ax.set_xlabel("Mean Absolute Error")
    ax.set_ylabel("Count")
    ax.set_title("MAE Distribution Across Test Designs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_f1s: list,
    save_path: str,
):
    """Plot training and validation loss/metric curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        val_f1s: List of validation F1 scores per epoch.
        save_path: Path to save the figure.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, "g-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation Hotspot F1 Score")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
