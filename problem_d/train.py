"""Training script for Problem D: Static IR Drop Prediction.

Usage:
    python train.py                          # Use default config
    python train.py --config configs/custom.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.utils import set_seed, get_device, load_config, setup_logging, ensure_dir, AverageMeter
from problem_d.src.dataset import get_dataloaders
from problem_d.src.model import AttentionUNet, count_parameters
from problem_d.src.losses import ContestAlignedLoss
from problem_d.src.metrics import compute_all_metrics, compute_per_testcase_metrics
from problem_d.src.visualize import plot_training_curves


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, grad_accum_steps=1):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    loss_meter = AverageMeter()

    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Train", leave=False)
    for step, (features, targets) in enumerate(pbar):
        features = features.to(device)
        targets = targets.to(device)

        with autocast('cuda', enabled=use_amp):
            preds = model(features)
            loss = criterion(preds, targets)
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, features.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    """Validate and compute per-testcase metrics (contest scoring)."""
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []

    for features, targets in tqdm(loader, desc="Val", leave=False):
        features = features.to(device)
        targets = targets.to(device)

        with autocast('cuda', enabled=use_amp):
            preds = model(features)
            loss = criterion(preds, targets)

        loss_meter.update(loss.item(), features.size(0))
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Per-testcase metrics (contest scoring)
    preds_list = [all_preds[i] for i in range(len(all_preds))]
    targets_list = [all_targets[i] for i in range(len(all_targets))]
    tc_metrics = compute_per_testcase_metrics(preds_list, targets_list)

    # Also compute aggregate for logging
    agg_metrics = compute_all_metrics(all_preds, all_targets)

    return {
        "loss": loss_meter.avg,
        "mae": tc_metrics["mae_avg"],
        "f1": tc_metrics["f1_avg"],
        "auroc": agg_metrics.get("auroc", 0.0),
        "auprc": agg_metrics.get("auprc", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train IR Drop Prediction Model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "default.yaml"),
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    # Setup output directories
    ckpt_dir = Path(__file__).parent / cfg["output"]["checkpoint_dir"]
    results_dir = Path(__file__).parent / cfg["output"]["results_dir"]
    ensure_dir(str(ckpt_dir))
    ensure_dir(str(results_dir))

    logger = setup_logging(str(results_dir))
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    logger.info(f"Config: {cfg}")

    # Data — use official ASU/ICCAD contest data only
    data_root = str(Path(__file__).parent / cfg["data"]["root"])
    logger.info(f"Data root: {data_root}")

    loaders = get_dataloaders(
        data_root=data_root,
        batch_size=cfg["data"]["batch_size"],
        patch_size=cfg["data"]["patch_size"],
        num_workers=cfg["data"]["num_workers"],
        data_format=cfg["data"].get("format", "auto"),
        cache_npy=cfg["data"].get("cache_npy", True),
        oversample=cfg["data"].get("oversample", 1),
    )
    logger.info(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)} | Test: {len(loaders['test'].dataset)}")

    # Model
    model = AttentionUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Loss (contest-aligned: SmoothL1 + hotspot-weighted L1 + BCE)
    criterion = ContestAlignedLoss(
        lambda_under=cfg["training"].get("lambda_under", 1.5),
        alpha=cfg["training"].get("loss_alpha", 0.3),
        hotspot_weight=cfg["training"].get("hotspot_weight", 5.0),
        use_smooth_l1=cfg["training"].get("use_smooth_l1", True),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"]["lr_min"],
    )
    scaler = GradScaler('cuda', enabled=cfg["training"]["use_amp"])

    # Training loop — track best by composite score (MAE 60% + F1 30%)
    best_score = -float("inf")
    patience_counter = 0
    train_losses, val_losses, val_f1s = [], [], []
    grad_accum_steps = cfg["training"].get("grad_accum_steps", 1)
    effective_bs = cfg["data"]["batch_size"] * grad_accum_steps
    logger.info(f"Batch size: {cfg['data']['batch_size']} x {grad_accum_steps} accum = {effective_bs} effective")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device,
            cfg["training"]["use_amp"], grad_accum_steps,
        )

        # Free fragmented GPU memory before validation
        torch.cuda.empty_cache()

        val_metrics = validate(
            model, loaders["val"], criterion, device, cfg["training"]["use_amp"],
        )
        scheduler.step()

        # Free fragmented GPU memory between epochs
        torch.cuda.empty_cache()

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        val_f1s.append(val_metrics["f1"])

        elapsed = time.time() - t0
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3

        # Composite score matching contest weights: lower MAE is better (negate), higher F1 is better
        # Normalize MAE to comparable scale: use -MAE so higher = better
        composite_score = -0.6 * val_metrics["mae"] + 0.3 * val_metrics["f1"]

        logger.info(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"MAE: {val_metrics['mae']:.6f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"AUROC: {val_metrics['auroc']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"VRAM: {vram_used:.1f}/{vram_reserved:.1f} GB | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model by composite score
        if composite_score > best_score:
            best_score = composite_score
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_mae": val_metrics["mae"],
                    "best_f1": val_metrics["f1"],
                    "config": cfg,
                },
                ckpt_dir / "best_model.pt",
            )
            logger.info(f"  -> New best: MAE={val_metrics['mae']:.6f}, F1={val_metrics['f1']:.4f}")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % cfg["output"]["save_every"] == 0:
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg},
                ckpt_dir / f"epoch_{epoch}.pt",
            )

        # Early stopping
        if patience_counter >= cfg["training"]["patience"]:
            logger.info(f"Early stopping at epoch {epoch} (patience={cfg['training']['patience']})")
            break

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, val_f1s,
        str(results_dir / "training_curves.png"),
    )
    logger.info(f"Training complete. Best composite score: {best_score:.6f}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    logger.info(f"Plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
