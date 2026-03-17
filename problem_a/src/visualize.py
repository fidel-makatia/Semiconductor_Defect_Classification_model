"""Visualization utilities for few-shot defect classification.

Generates all contest-required plots:
  - Learning curves (accuracy vs examples seen)
  - Per-class learning curves
  - Accuracy vs defect class occurrence
  - Detection accuracy
  - Confusion matrices
  - K-shot comparison charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

matplotlib.use("Agg")


def plot_learning_curve(results, save_path, title="Learning Curve: Accuracy vs. Examples Seen"):
    """Plot learning curves with error bands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for i, (name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        steps = data["steps"]
        mean = data["mean"]
        ax.plot(steps, mean, "-o", color=color, label=name, linewidth=2, markersize=4)
        if "std" in data:
            std = data["std"]
            ax.fill_between(steps, np.array(mean) - np.array(std),
                            np.array(mean) + np.array(std), alpha=0.2, color=color)

    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="85% target")
    ax.set_xlabel("Number of Labeled Examples Seen", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_class_learning_curves(steps, per_class_accs, save_path):
    """Plot per-class learning curves (accuracy vs total examples seen)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab10")

    for i, (cls_name, accs) in enumerate(per_class_accs.items()):
        ax.plot(steps, accs, "-o", color=cmap(i), label=cls_name, linewidth=1.5, markersize=3)

    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="85% target")
    ax.set_xlabel("Total Examples Seen", fontsize=12)
    ax.set_ylabel("Per-Class Accuracy", fontsize=12)
    ax.set_title("Per-Class Learning Curves", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_class_occurrence(per_class_data, save_path):
    """Plot accuracy vs defect class occurrence (contest deliverable).

    X-axis: number of examples of each specific class the model has seen.
    Y-axis: classification accuracy for that class.

    Args:
        per_class_data: Dict mapping class_name -> {
            'occurrences': [int], 'accuracies': [float]
        }
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab10")

    for i, (cls_name, data) in enumerate(per_class_data.items()):
        occurrences = data["occurrences"]
        accuracies = data["accuracies"]
        ax.plot(occurrences, accuracies, "-o", color=cmap(i), label=cls_name,
                linewidth=1.5, markersize=4)

    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="85% target")
    ax.set_xlabel("Defect Class Occurrence (examples seen per class)", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title("Classification Accuracy vs. Defect Class Occurrence", fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_detection_accuracy(steps, detection_accs, classification_accs, save_path):
    """Plot detection AND classification accuracy vs examples seen."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, detection_accs, "-s", color="#FF5722", label="Detection Accuracy",
            linewidth=2, markersize=4)
    ax.plot(steps, classification_accs, "-o", color="#2196F3", label="Classification Accuracy",
            linewidth=2, markersize=4)

    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.3, label="85% target")
    ax.set_xlabel("Number of Labeled Examples Seen", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Detection & Classification Accuracy vs. Examples Seen", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_kshot_comparison(k_values, method_accuracies, save_path):
    """Bar chart comparing methods at fixed K-shot values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(k_values))
    width = 0.8 / len(method_accuracies)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, (name, accs) in enumerate(method_accuracies.items()):
        offset = (i - len(method_accuracies) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=name, color=colors[i % len(colors)])
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.1%}", ha="center", va="bottom", fontsize=8)

    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="85% target")
    ax.set_xlabel("K (examples per class)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("K-Shot Accuracy Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
