"""Evaluation script for Problem A: Few-Shot Defect Classification.

Generates all contest deliverables:
  1. Learning curves (accuracy vs examples seen) with error bands
  2. Per-class accuracy vs defect class occurrence plots
  3. Detection + classification accuracy plots
  4. Confusion matrices
  5. K-shot comparison charts

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.utils import set_seed, get_device, load_config, setup_logging, ensure_dir
from problem_a.src.dataset import DAGMDataset
from problem_a.src.backbone import get_backbone
from problem_a.src.protonet import PrototypicalNetwork, IncrementalPrototypeTracker
from problem_a.src.augmentations import get_eval_transform
from problem_a.src.metrics import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_per_class_f1,
    compute_confusion_matrix,
    compute_area_under_learning_curve,
    compute_time_to_threshold,
)
from problem_a.src.visualize import (
    plot_learning_curve,
    plot_per_class_learning_curves,
    plot_confusion_matrix,
    plot_kshot_comparison,
    plot_detection_accuracy,
    plot_accuracy_vs_class_occurrence,
)


def run_incremental_evaluation(
    model: PrototypicalNetwork,
    train_dataset: DAGMDataset,
    test_dataset: DAGMDataset,
    device: torch.device,
    max_examples_per_class: int = 50,
    seed: int = 42,
    good_dataset: DAGMDataset = None,
) -> dict:
    """Run one incremental evaluation trial with per-class tracking.

    Args:
        good_dataset: Optional dataset of "good"/non-defective images for
            computing meaningful detection accuracy (defective vs non-defective).

    Returns:
        Dictionary with 'steps', 'accuracies', 'per_class_accuracies',
        'per_class_counts', and 'detection_accuracies'.
    """
    rng = np.random.RandomState(seed)
    tracker = IncrementalPrototypeTracker(model, device)

    # Build randomized stream of training examples
    stream = []
    for cls_label in train_dataset.class_labels:
        indices = train_dataset.get_class_indices(cls_label)
        selected = rng.choice(indices, size=min(max_examples_per_class, len(indices)), replace=False)
        for idx in selected:
            stream.append(idx)
    rng.shuffle(stream)

    # Pre-compute test embeddings once (avoids re-running ViT backbone every step)
    test_labels_list = []
    test_embeddings_list = []
    model.eval()
    batch_size = 16
    all_imgs = []
    for idx in range(len(test_dataset)):
        img, lbl = test_dataset[idx]
        test_labels_list.append(lbl)
        all_imgs.append(img)
    test_labels = torch.tensor(test_labels_list)

    with torch.no_grad():
        for i in range(0, len(all_imgs), batch_size):
            batch = torch.stack(all_imgs[i:i+batch_size]).to(device)
            embs = model.embed(batch)
            test_embeddings_list.append(embs.cpu())
    test_embeddings = torch.cat(test_embeddings_list, dim=0)  # [N, proj_dim]
    del all_imgs  # free memory

    # Pre-compute "good" embeddings for detection accuracy (if available)
    good_embeddings = None
    if good_dataset is not None and len(good_dataset) > 0:
        good_imgs = []
        n_good = min(200, len(good_dataset))  # cap for speed
        good_indices = rng.choice(len(good_dataset), size=n_good, replace=False)
        for idx in good_indices:
            img, _ = good_dataset[idx]
            good_imgs.append(img)
        good_emb_list = []
        with torch.no_grad():
            for i in range(0, len(good_imgs), batch_size):
                batch = torch.stack(good_imgs[i:i+batch_size]).to(device)
                embs = model.embed(batch)
                good_emb_list.append(embs.cpu())
        good_embeddings = torch.cat(good_emb_list, dim=0)
        del good_imgs

    # Stream examples and evaluate using pre-computed embeddings
    steps = []
    accuracies = []
    detection_accuracies = []
    per_class_accuracies = []  # List of dicts: {class_label -> accuracy}
    per_class_counts_history = []  # List of dicts: {class_label -> count_seen}

    for i, train_idx in enumerate(stream):
        img, label = train_dataset[train_idx]
        tracker.add_example(img, label)

        # Evaluate at key steps
        if i < 10 or (i < 50 and i % 5 == 0) or i % 10 == 0 or i == len(stream) - 1:
            protos = tracker.prototypes
            lmap = tracker.label_map

            if protos is None or len(tracker.prototype_sums) < 2:
                steps.append(i + 1)
                accuracies.append(0.0)
                detection_accuracies.append(0.0)
                per_class_accuracies.append({})
                per_class_counts_history.append(dict(tracker.prototype_counts))
                continue

            # Classify using pre-computed embeddings with cosine similarity
            test_emb_dev = test_embeddings.to(device)
            log_probs = model.classify_embeddings(test_emb_dev, protos)
            preds = log_probs.argmax(dim=1).cpu()

            # Filter to known classes
            mask = torch.tensor([int(l) in lmap for l in test_labels])
            if mask.sum() == 0:
                acc = 0.0
                det_acc = 0.0
                per_cls = {}
            else:
                mapped_true = torch.tensor([lmap[int(l)] for l in test_labels[mask]])
                acc = (preds[mask] == mapped_true).float().mean().item()

                # Detection accuracy: binary defective-vs-good using cosine distance
                if good_embeddings is not None:
                    p_norm = torch.nn.functional.normalize(protos, dim=1)
                    # Defective test images: max cosine sim to any prototype
                    t_norm = torch.nn.functional.normalize(test_emb_dev[mask], dim=1)
                    defect_max_sim = torch.mm(t_norm, p_norm.t()).max(dim=1).values.cpu()
                    # Good images: max cosine sim to any prototype
                    g_norm = torch.nn.functional.normalize(good_embeddings.to(device), dim=1)
                    good_max_sim = torch.mm(g_norm, p_norm.t()).max(dim=1).values.cpu()
                    # Threshold: midpoint between defect and good mean similarity
                    threshold = (defect_max_sim.mean() + good_max_sim.mean()) / 2
                    # Detection: defect correctly above threshold + good correctly below
                    tp = (defect_max_sim >= threshold).float().sum()
                    tn = (good_max_sim < threshold).float().sum()
                    det_acc = (tp + tn).item() / (len(defect_max_sim) + len(good_max_sim))
                else:
                    det_acc = acc  # fallback when no good data available

                # Per-class accuracy
                per_cls = {}
                inv_map = {v: k for k, v in lmap.items()}
                for mapped_label in mapped_true.unique():
                    cls_mask = mapped_true == mapped_label
                    cls_correct = (preds[mask][cls_mask] == mapped_label).float().mean().item()
                    per_cls[inv_map[mapped_label.item()]] = cls_correct

            steps.append(i + 1)
            accuracies.append(acc)
            detection_accuracies.append(det_acc)
            per_class_accuracies.append(per_cls)
            per_class_counts_history.append(dict(tracker.prototype_counts))

    return {
        "steps": steps,
        "accuracies": accuracies,
        "detection_accuracies": detection_accuracies,
        "per_class_accuracies": per_class_accuracies,
        "per_class_counts": per_class_counts_history,
    }


def run_kshot_evaluation(
    model: PrototypicalNetwork,
    dataset: DAGMDataset,
    device: torch.device,
    n_way: int,
    k_shot: int,
    n_query: int = 15,
    n_episodes: int = 100,
    seed: int = 42,
) -> float:
    """Run standard N-way K-shot evaluation. Returns mean accuracy."""
    set_seed(seed)
    model.eval()
    accs = []
    rng = np.random.RandomState(seed)
    class_labels = dataset.class_labels

    for _ in range(n_episodes):
        selected_classes = rng.choice(class_labels, size=min(n_way, len(class_labels)), replace=False)
        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []

        for new_lbl, cls in enumerate(selected_classes):
            indices = dataset.get_class_indices(cls)
            n_available = len(indices)
            shuffled = rng.permutation(indices)

            if n_available >= k_shot + n_query:
                # Plenty of data: disjoint support and query
                support_idx = shuffled[:k_shot]
                query_idx = shuffled[k_shot:k_shot + n_query]
            elif n_available > k_shot:
                # Enough for unique support, fewer queries
                support_idx = shuffled[:k_shot]
                query_idx = shuffled[k_shot:]
            else:
                # Very rare class: use all unique for support, resample for query
                support_idx = shuffled[:n_available]
                if n_available < k_shot:
                    # Pad support with replacement to reach k_shot
                    extra = rng.choice(indices, size=k_shot - n_available, replace=True)
                    support_idx = np.concatenate([support_idx, extra])
                query_idx = rng.choice(indices, size=min(n_query, max(n_available, 3)), replace=True)

            for idx in support_idx:
                img, _ = dataset[idx]
                support_imgs.append(img)
                support_lbls.append(new_lbl)
            for idx in query_idx:
                img, _ = dataset[idx]
                query_imgs.append(img)
                query_lbls.append(new_lbl)

        support_imgs = torch.stack(support_imgs).to(device)
        support_lbls = torch.tensor(support_lbls).to(device)
        query_imgs = torch.stack(query_imgs).to(device)
        query_lbls = torch.tensor(query_lbls).to(device)

        with torch.no_grad():
            log_probs = model(support_imgs, support_lbls, query_imgs)
        preds = log_probs.argmax(dim=1).cpu()
        acc = (preds == query_lbls.cpu()).float().mean().item()
        accs.append(acc)

    return np.mean(accs)


def benchmark_inference_time(model, dataset, device, n_samples=20):
    """Benchmark inference time per image."""
    model.eval()
    times = []
    for i in range(min(n_samples, len(dataset))):
        img, _ = dataset[i]
        img = img.unsqueeze(0).to(device)

        # Warm up
        if i == 0:
            with torch.no_grad():
                _ = model.embed(img)
            if device.type == "cuda":
                torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.embed(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Few-Shot Defect Classifier")
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path(__file__).parent / "checkpoints" / "best_model.pt"))
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    results_dir = Path(__file__).parent / cfg["output"]["results_dir"]
    ensure_dir(str(results_dir))
    logger = setup_logging(str(results_dir))

    eval_tf = get_eval_transform(cfg["data"]["img_size"])
    data_root = str(Path(__file__).parent / cfg["data"]["root"])
    # Fall back to proxy data if primary root doesn't exist
    if not Path(data_root).exists() and "proxy_root" in cfg["data"]:
        data_root = str(Path(__file__).parent / cfg["data"]["proxy_root"])
        logger.info(f"Primary data root not found, using proxy: {data_root}")

    # Load datasets for evaluation — use ALL available classes for contest deliverables
    eval_classes = cfg["data"].get("all_classes", cfg["data"]["test_classes"])
    logger.info(f"Evaluating on classes: {eval_classes}")

    train_dataset = DAGMDataset(
        data_root=data_root, split="train", transform=eval_tf,
        defect_only=cfg["data"]["defect_only"],
        classes=eval_classes, val_fraction=0,
    )
    test_dataset = DAGMDataset(
        data_root=data_root, split="test", transform=eval_tf,
        defect_only=cfg["data"]["defect_only"],
        classes=eval_classes, val_fraction=0,
    )

    # If test split is empty (no Train/Test subdirs), use train split for both
    # and do 80/20 random split
    if len(test_dataset) == 0:
        logger.info("No separate test split found; splitting train data 80/20")
        full_dataset = DAGMDataset(
            data_root=data_root, split="train", transform=eval_tf,
            defect_only=cfg["data"]["defect_only"],
            classes=eval_classes, val_fraction=0,
        )
        n_total = len(full_dataset)
        n_test = max(1, n_total // 5)
        rng_split = np.random.RandomState(cfg["seed"])
        indices = rng_split.permutation(n_total)
        test_indices = set(indices[:n_test].tolist())

        # Rebuild train/test by filtering — preserve old->new index mapping
        # for correct image cache remapping
        from copy import deepcopy
        test_indexed = [(i, s) for i, s in enumerate(full_dataset.samples) if i in test_indices]
        train_indexed = [(i, s) for i, s in enumerate(full_dataset.samples) if i not in test_indices]

        train_dataset = deepcopy(full_dataset)
        test_dataset = deepcopy(full_dataset)

        train_dataset.samples = [s for _, s in train_indexed]
        test_dataset.samples = [s for _, s in test_indexed]

        # Rebuild class_to_indices AND image cache with correct new indices
        old_cache = full_dataset._image_cache
        for ds, indexed_samples in [(train_dataset, train_indexed), (test_dataset, test_indexed)]:
            ds.class_to_indices = {}
            ds._image_cache = {}
            for new_idx, (old_idx, s) in enumerate(indexed_samples):
                ds.class_to_indices.setdefault(s["label"], []).append(new_idx)
                if old_idx in old_cache:
                    ds._image_cache[new_idx] = old_cache[old_idx]

        logger.info(f"Split: {len(train_dataset)} train, {len(test_dataset)} test")

    # Build model (fully frozen backbone for evaluation)
    backbone = get_backbone(
        cfg["model"]["backbone"],
        size=cfg["model"].get("backbone_size", "small"),
        freeze=True,
        grad_checkpointing=False,
    )
    model = PrototypicalNetwork(
        backbone=backbone,
        proj_hidden=cfg["model"]["proj_hidden"],
        proj_dim=cfg["model"]["proj_dim"],
    ).to(device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, acc {ckpt.get('best_val_acc', '?')})")
        else:
            model.load_state_dict(ckpt)
    else:
        logger.warning(f"No checkpoint at {ckpt_path}. Using random projection head.")

    model.eval()
    ecfg = cfg["evaluation"]

    # =========================================================================
    # 0. Inference Time Benchmark
    # =========================================================================
    logger.info("Benchmarking inference time...")
    timing = benchmark_inference_time(model, test_dataset, device)
    logger.info(f"  Inference: {timing['mean_ms']:.1f} +/- {timing['std_ms']:.1f} ms (max: {timing['max_ms']:.1f} ms)")

    # Load "good" dataset for meaningful detection accuracy (defective vs non-defective)
    good_dataset = None
    try:
        good_dataset = DAGMDataset(
            data_root=data_root, split="train", transform=eval_tf,
            defect_only=False, classes=[0], val_fraction=0,
            cache_images=False,  # 7K+ good images, only sample 200
        )
        if len(good_dataset) > 0:
            logger.info(f"Loaded {len(good_dataset)} 'good' images for detection evaluation")
        else:
            good_dataset = None
    except Exception:
        pass

    # =========================================================================
    # 1. Incremental Learning Curves (KEY DELIVERABLE)
    # =========================================================================
    logger.info("Running incremental evaluation...")
    all_runs = []
    for seed_idx in range(ecfg["n_seeds"]):
        logger.info(f"  Seed {seed_idx + 1}/{ecfg['n_seeds']}")
        result = run_incremental_evaluation(
            model, train_dataset, test_dataset, device,
            max_examples_per_class=ecfg["max_examples"],
            seed=cfg["seed"] + seed_idx,
            good_dataset=good_dataset,
        )
        all_runs.append(result)

    # Aggregate across seeds
    common_steps = all_runs[0]["steps"]
    all_accs = np.array([r["accuracies"] for r in all_runs])
    mean_accs = all_accs.mean(axis=0).tolist()
    std_accs = all_accs.std(axis=0).tolist()

    aulc = compute_area_under_learning_curve(mean_accs)
    ttt = compute_time_to_threshold(mean_accs, ecfg["target_accuracy"])

    logger.info("=" * 60)
    logger.info("INCREMENTAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Final accuracy:  {mean_accs[-1]:.3f} +/- {std_accs[-1]:.3f}")
    logger.info(f"  AULC:            {aulc:.3f}")
    logger.info(f"  Time-to-{ecfg['target_accuracy']:.0%}:  {ttt} examples ({'never reached' if ttt < 0 else ''})")
    logger.info("=" * 60)

    # Plot overall learning curve
    plot_learning_curve(
        {"ProtoNet + DINOv2": {"steps": common_steps, "mean": mean_accs, "std": std_accs}},
        str(results_dir / "learning_curve.png"),
    )
    logger.info("Saved learning curve plot")

    # Detection vs Classification accuracy
    all_det_accs = np.array([r["detection_accuracies"] for r in all_runs])
    mean_det_accs = all_det_accs.mean(axis=0).tolist()
    plot_detection_accuracy(
        common_steps, mean_det_accs, mean_accs,
        str(results_dir / "detection_accuracy.png"),
    )
    logger.info("Saved detection accuracy plot")

    # =========================================================================
    # 2. Per-Class Accuracy vs Class Occurrence (CONTEST DELIVERABLE)
    # =========================================================================
    logger.info("Generating per-class learning curves...")

    # Use first run for per-class data
    first_run = all_runs[0]
    class_labels = train_dataset.class_labels
    class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else [f"Class {c}" for c in class_labels]

    # Build per-class accuracy curves
    per_class_accs_over_steps = {}
    for cls_label, cls_name in zip(class_labels, class_names):
        cls_accs = []
        for step_data in first_run["per_class_accuracies"]:
            cls_accs.append(step_data.get(cls_label, 0.0))
        per_class_accs_over_steps[cls_name] = cls_accs

    if per_class_accs_over_steps:
        plot_per_class_learning_curves(
            common_steps, per_class_accs_over_steps,
            str(results_dir / "per_class_learning_curves.png"),
        )
        logger.info("Saved per-class learning curves")

    # Accuracy vs class occurrence (how many examples of EACH class have been seen)
    # This is the exact deliverable: "accuracy vs defect class occurrence"
    per_class_occurrence = {}
    for cls_label, cls_name in zip(class_labels, class_names):
        occurrences = []
        accs = []
        for step_idx, counts in enumerate(first_run["per_class_counts"]):
            cls_count = counts.get(cls_label, 0)
            cls_acc = first_run["per_class_accuracies"][step_idx].get(cls_label, 0.0)
            occurrences.append(cls_count)
            accs.append(cls_acc)
        per_class_occurrence[cls_name] = {"occurrences": occurrences, "accuracies": accs}

    if per_class_occurrence:
        plot_accuracy_vs_class_occurrence(
            per_class_occurrence,
            str(results_dir / "accuracy_vs_class_occurrence.png"),
        )
        logger.info("Saved accuracy vs class occurrence plot")

    # =========================================================================
    # 3. K-Shot Bar Chart Comparison
    # =========================================================================
    logger.info("Running K-shot evaluation...")
    kshot_accs = []
    n_way = min(len(eval_classes), test_dataset.num_classes)
    for k in ecfg["kshot_values"]:
        acc = run_kshot_evaluation(
            model, test_dataset, device,
            n_way=n_way, k_shot=k, n_episodes=100,
        )
        kshot_accs.append(acc)
        logger.info(f"  K={k:2d}: accuracy = {acc:.3f}")

    plot_kshot_comparison(
        ecfg["kshot_values"],
        {"ProtoNet + DINOv2": kshot_accs},
        str(results_dir / "kshot_comparison.png"),
    )
    logger.info("Saved K-shot comparison plot")

    # =========================================================================
    # 4. Confusion Matrix
    # =========================================================================
    logger.info("Generating confusion matrix...")
    best_k = ecfg["kshot_values"][-1]
    tracker = IncrementalPrototypeTracker(model, device)

    rng = np.random.RandomState(cfg["seed"])
    for cls_label in train_dataset.class_labels:
        indices = train_dataset.get_class_indices(cls_label)
        selected = rng.choice(indices, size=min(best_k, len(indices)), replace=False)
        for idx in selected:
            img, lbl = train_dataset[idx]
            tracker.add_example(img, lbl)

    all_preds = []
    all_true = []
    lmap = tracker.label_map

    for idx in range(len(test_dataset)):
        img, true_label = test_dataset[idx]
        if true_label not in lmap:
            continue
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model.classify(img, tracker.prototypes)
        pred = log_probs.argmax(dim=1).item()
        all_preds.append(pred)
        all_true.append(lmap[true_label])

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    cm = compute_confusion_matrix(all_preds, all_true)
    # Use actual class names (e.g., "defect1") instead of generic "Class N"
    inv_lmap = {v: k for k, v in lmap.items()}
    cm_class_names = []
    for i in range(len(inv_lmap)):
        orig_label = inv_lmap[i]
        name = next((s["class_name"] for s in test_dataset.samples if s["label"] == orig_label), f"Class {orig_label}")
        cm_class_names.append(name)
    plot_confusion_matrix(
        cm, cm_class_names,
        str(results_dir / f"confusion_matrix_k{best_k}.png"),
        title=f"Confusion Matrix (K={best_k})",
    )

    f1_result = compute_per_class_f1(all_preds, all_true)
    bal_acc = compute_balanced_accuracy(all_preds, all_true)
    logger.info(f"Balanced accuracy (K={best_k}): {bal_acc:.3f}")
    logger.info(f"Macro F1 (K={best_k}): {f1_result['macro']:.3f}")
    for i, f1 in enumerate(f1_result['per_class']):
        logger.info(f"  {cm_class_names[i]}: F1={f1:.3f}")

    logger.info(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
