"""Meta-training script for Problem A: Few-Shot Defect Classification.

Trains a Prototypical Network on episodic tasks sampled from the
training classes, then validates on held-out classes.

Usage:
    python train.py
    python train.py --config configs/default.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.utils import set_seed, get_device, load_config, setup_logging, ensure_dir, AverageMeter
from problem_a.src.dataset import DAGMDataset, EpisodicSampler, collate_episode, episode_collate_fn
from problem_a.src.backbone import get_backbone
from problem_a.src.protonet import PrototypicalNetwork
from problem_a.src.augmentations import get_train_transform, get_eval_transform


def label_smoothing_loss(log_probs, targets, smoothing=0.1):
    """NLL loss with label smoothing for better calibration."""
    n_classes = log_probs.size(1)
    smooth = smoothing / n_classes
    targets_smooth = torch.full_like(log_probs, smooth)
    targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing + smooth)
    return -(targets_smooth * log_probs).sum(dim=1).mean()


def train_one_epoch(model, loader, optimizer, scaler, device, n_way, k_shot,
                    use_amp=False, grad_clip=0.0, label_smoothing=0.0):
    """Train for one epoch of episodes."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        support_imgs, support_lbls, query_imgs, query_lbls = collate_episode(
            batch, n_way, k_shot
        )
        support_imgs = support_imgs.to(device)
        support_lbls = support_lbls.to(device)
        query_imgs = query_imgs.to(device)
        query_lbls = query_lbls.to(device)

        optimizer.zero_grad()
        with autocast('cuda', enabled=use_amp):
            log_probs = model(support_imgs, support_lbls, query_imgs)
            if label_smoothing > 0:
                loss = label_smoothing_loss(log_probs, query_lbls, label_smoothing)
            else:
                loss = F.nll_loss(log_probs, query_lbls)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        preds = log_probs.argmax(dim=1)
        acc = (preds == query_lbls).float().mean().item()

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.3f}")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, device, n_way, k_shot, use_amp=False):
    """Validate on episodes from held-out classes."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in tqdm(loader, desc="Val", leave=False):
        support_imgs, support_lbls, query_imgs, query_lbls = collate_episode(
            batch, n_way, k_shot
        )
        support_imgs = support_imgs.to(device)
        support_lbls = support_lbls.to(device)
        query_imgs = query_imgs.to(device)
        query_lbls = query_lbls.to(device)

        with autocast('cuda', enabled=use_amp):
            log_probs = model(support_imgs, support_lbls, query_imgs)
            loss = F.nll_loss(log_probs, query_lbls)

        preds = log_probs.argmax(dim=1)
        acc = (preds == query_lbls).float().mean().item()

        loss_meter.update(loss.item())
        acc_meter.update(acc)

    return loss_meter.avg, acc_meter.avg


def make_episode_loader(dataset, n_way, k_shot, n_query, n_episodes):
    """Create a DataLoader that yields episodes."""
    sampler = EpisodicSampler(dataset, n_way, k_shot, n_query, n_episodes)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=episode_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Few-Shot Defect Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    ckpt_dir = Path(__file__).parent / cfg["output"]["checkpoint_dir"]
    results_dir = Path(__file__).parent / cfg["output"]["results_dir"]
    ensure_dir(str(ckpt_dir))
    ensure_dir(str(results_dir))

    logger = setup_logging(str(results_dir))
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")

    # Transforms
    train_tf = get_train_transform(cfg["data"]["img_size"])
    eval_tf = get_eval_transform(cfg["data"]["img_size"])

    # Datasets
    data_root = str(Path(__file__).parent / cfg["data"]["root"])
    logger.info(f"Data root: {data_root}")

    train_dataset = DAGMDataset(
        data_root=data_root,
        split="train",
        transform=train_tf,
        defect_only=cfg["data"]["defect_only"],
        classes=cfg["data"]["train_classes"],
    )
    # Validate on held-out 20% of images (stratified per class) with eval transforms.
    # This gives a meaningful accuracy signal for early stopping.
    val_dataset = DAGMDataset(
        data_root=data_root,
        split="val",
        transform=eval_tf,
        defect_only=cfg["data"]["defect_only"],
        classes=cfg["data"]["train_classes"],
    )

    logger.info(f"Train: {len(train_dataset)} samples, {train_dataset.num_classes} classes")
    logger.info(f"Val:   {len(val_dataset)} samples, {val_dataset.num_classes} classes")
    for cls_label in train_dataset.class_labels:
        n = len(train_dataset.get_class_indices(cls_label))
        logger.info(f"  Class {cls_label}: {n} samples")

    # Episode loaders
    tcfg = cfg["training"]
    train_loader = make_episode_loader(
        train_dataset, tcfg["n_way"], tcfg["k_shot"], tcfg["n_query"], tcfg["n_episodes_train"]
    )
    val_loader = make_episode_loader(
        val_dataset, min(tcfg["n_way"], val_dataset.num_classes),
        tcfg["k_shot"], tcfg["n_query"], tcfg["n_episodes_val"]
    )

    # Model
    mcfg = cfg["model"]
    backbone = get_backbone(
        mcfg["backbone"],
        size=mcfg.get("backbone_size", "small"),
        freeze=mcfg["freeze_backbone"],
        unfreeze_last_n=mcfg.get("unfreeze_last_n", 0),
        grad_checkpointing=mcfg.get("grad_checkpointing", False),
    )
    model = PrototypicalNetwork(
        backbone=backbone,
        proj_hidden=mcfg["proj_hidden"],
        proj_dim=mcfg["proj_dim"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {mcfg['backbone']} {mcfg.get('backbone_size', 'small')}")
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Optimizer: differential LR
    use_amp = tcfg.get("use_amp", False)
    lr_backbone = tcfg.get("lr_backbone", tcfg["lr"] * 0.01)
    grad_clip = tcfg.get("gradient_clip", 1.0)
    label_smoothing = tcfg.get("label_smoothing", 0.0)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "backbone" not in n]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    param_groups.append({"params": head_params, "lr": tcfg["lr"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=tcfg["weight_decay"])

    # Scheduler: warmup + cosine annealing
    warmup_epochs = tcfg.get("warmup_epochs", 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"] - warmup_epochs, eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    scaler = GradScaler('cuda', enabled=use_amp)

    # Training loop
    best_val_acc = 0.0
    patience = tcfg.get("patience", 30)
    no_improve = 0

    logger.info(f"Training: {tcfg['epochs']} epochs, {tcfg['n_episodes_train']} episodes/epoch")
    logger.info(f"Episodes: {tcfg['n_way']}-way {tcfg['k_shot']}-shot {tcfg['n_query']}-query")
    logger.info(f"LR: head={tcfg['lr']}, backbone={lr_backbone}, warmup={warmup_epochs}")
    logger.info(f"AMP: {use_amp}, grad_clip: {grad_clip}, label_smoothing: {label_smoothing}")

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            tcfg["n_way"], tcfg["k_shot"], use_amp=use_amp,
            grad_clip=grad_clip, label_smoothing=label_smoothing,
        )

        torch.cuda.empty_cache()

        val_loss, val_acc = validate(
            model, val_loader, device,
            min(tcfg["n_way"], val_dataset.num_classes), tcfg["k_shot"],
            use_amp=use_amp,
        )
        scheduler.step()

        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        vram_used = torch.cuda.memory_allocated() / 1024**3
        current_lr = optimizer.param_groups[-1]['lr']
        logger.info(
            f"Epoch {epoch:3d}/{tcfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"LR: {current_lr:.2e} | "
            f"VRAM: {vram_used:.1f} GB | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": cfg,
                },
                ckpt_dir / "best_model.pt",
            )
            logger.info(f"  -> New best val accuracy: {best_val_acc:.3f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
