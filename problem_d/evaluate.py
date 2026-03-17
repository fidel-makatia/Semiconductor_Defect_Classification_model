"""Evaluation script for Problem D: Static IR Drop Prediction.

Runs per-testcase evaluation matching the exact contest scoring formula.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.utils import set_seed, get_device, load_config, setup_logging, ensure_dir
from problem_d.src.dataset import get_dataloaders
from problem_d.src.model import AttentionUNet
from problem_d.src.metrics import compute_all_metrics, compute_per_testcase_metrics, compute_mae
from problem_d.src.visualize import plot_prediction, plot_roc_pr_curves, plot_mae_distribution


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp=True):
    """Run evaluation, return per-sample predictions and targets."""
    model.eval()
    all_preds = []
    all_targets = []
    inference_times = []

    for features, targets in tqdm(loader, desc="Evaluating"):
        features = features.to(device)

        t0 = time.perf_counter()
        with autocast('cuda', enabled=use_amp):
            preds = model(features)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - t0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return all_preds, all_targets, inference_times


def main():
    parser = argparse.ArgumentParser(description="Evaluate IR Drop Prediction Model")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(Path(__file__).parent / "checkpoints" / "best_model.pt"),
    )
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent / "configs" / "default.yaml"),
    )
    parser.add_argument("--num-vis", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    results_dir = Path(__file__).parent / cfg["output"]["results_dir"]
    ensure_dir(str(results_dir))
    logger = setup_logging(str(results_dir))

    # Detect checkpoint's input channel count before building model/data
    in_channels = cfg["model"]["in_channels"]
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        for k, v in sd.items():
            if "enc1" in k and "weight" in k and v.dim() == 4:
                in_channels = v.shape[1]
                break
        if in_channels != cfg["model"]["in_channels"]:
            logger.warning(
                f"Checkpoint has {in_channels} input channels, "
                f"config has {cfg['model']['in_channels']}. "
                f"Using checkpoint's {in_channels} channels for evaluation."
            )

    # Data (channel count matches checkpoint)
    data_root = str(Path(__file__).parent / cfg["data"]["root"])
    loaders = get_dataloaders(
        data_root=data_root,
        batch_size=1,  # Per-testcase evaluation requires batch_size=1
        patch_size=cfg["data"]["patch_size"],
        num_workers=cfg["data"]["num_workers"],
        data_format=cfg["data"].get("format", "auto"),
        in_channels=in_channels,
    )

    # Model
    model = AttentionUNet(
        in_channels=in_channels,
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
        dropout=0.0,
    ).to(device)

    # Load checkpoint
    if ckpt_path.exists():
        if "model_state_dict" in ckpt:
            model.load_state_dict(sd)
            logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} ({in_channels}ch)")
        else:
            model.load_state_dict(sd)
    else:
        logger.warning(f"No checkpoint at {ckpt_path}. Using random weights.")

    # Run evaluation
    all_preds, all_targets, times = evaluate_model(
        model, loaders["test"], device
    )

    # Per-testcase metrics (CONTEST SCORING)
    preds_list = [all_preds[i] for i in range(len(all_preds))]
    targets_list = [all_targets[i] for i in range(len(all_targets))]
    tc_metrics = compute_per_testcase_metrics(preds_list, targets_list)

    logger.info("=" * 70)
    logger.info("TEST SET RESULTS (Per-Testcase Contest Scoring)")
    logger.info("=" * 70)
    for detail in tc_metrics["details"]:
        logger.info(
            f"  Testcase {detail['testcase']:2d} | "
            f"MAE: {detail['mae']:.6e} | "
            f"F1: {detail['f1']:.4f} | "
            f"P: {detail['precision']:.4f} | "
            f"R: {detail['recall']:.4f} | "
            f"Thresh: {detail['threshold']:.6e}"
        )
    logger.info("-" * 70)
    logger.info(f"  AVG MAE:  {tc_metrics['mae_avg']:.6e}")
    logger.info(f"  AVG F1:   {tc_metrics['f1_avg']:.4f}")
    logger.info(f"  Avg inference time: {np.mean(times)*1000:.1f}ms per sample")
    logger.info("=" * 70)

    # Visualizations
    for i in range(min(args.num_vis, len(all_preds))):
        gt = all_targets[i, 0]
        pred = all_preds[i, 0]
        plot_prediction(gt, pred, str(results_dir / f"prediction_sample_{i}.png"), title=f"Testcase {i}")
    logger.info(f"Saved {min(args.num_vis, len(all_preds))} prediction heatmaps")

    # ROC/PR curves using contest threshold
    target_flat = all_targets.flatten().astype(np.float64)
    threshold = 0.9 * target_flat.max()
    target_binary = (target_flat >= threshold).astype(int)
    if target_binary.sum() > 0 and target_binary.sum() < len(target_binary):
        plot_roc_pr_curves(target_binary, all_preds.flatten(), str(results_dir / "roc_pr_curves.png"))
        logger.info("Saved ROC/PR curves")

    # MAE distribution
    mae_per_sample = tc_metrics["mae_per_testcase"]
    plot_mae_distribution(mae_per_sample, str(results_dir / "mae_distribution.png"))
    logger.info("Saved MAE distribution plot")

    logger.info(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
