"""Dataset loader for IR drop prediction.

Supports the ASU/ICCAD contest CSV format:
  - 3 input features: current_map.csv, eff_dist_map.csv, pdn_density.csv
  - 1 target: ir_drop_map.csv
  - Optional: netlist.sp (SPICE netlist)

Also retains CircuitNet-N14 .npz support as a fallback for proxy training.

Contest data layout:
    data_root/
        fake-circuit-data/
            current_map00_current.csv
            current_map00_eff_dist.csv
            current_map00_pdn_density.csv
            current_map00_ir_drop.csv
            ...
        real-circuit-data/
            testcase1/
                current_map.csv
                eff_dist_map.csv
                pdn_density.csv
                ir_drop_map.csv
                netlist.sp
            ...
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------------

def load_csv_matrix(path: str, cache: bool = True) -> np.ndarray:
    """Load a CSV matrix (scientific notation) into a 2D float32 array.

    When cache=True, saves a .npy file next to the CSV on first load.
    Subsequent loads use the .npy file which is ~100x faster.
    """
    npy_path = path + ".npy"
    if cache:
        try:
            return np.load(npy_path).astype(np.float32)
        except FileNotFoundError:
            pass
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64).astype(np.float32)
    if cache:
        try:
            np.save(npy_path, arr)
        except OSError:
            pass  # read-only filesystem, skip caching
    return arr


def save_csv_matrix(arr: np.ndarray, path: str):
    """Save a 2D array as CSV (same format as contest)."""
    np.savetxt(path, arr, delimiter=",", fmt="%.6e")


# ---------------------------------------------------------------------------
# Contest CSV Dataset
# ---------------------------------------------------------------------------

class ContestIRDropDataset(Dataset):
    """Contest-format dataset: CSV feature maps -> IR drop map.

    Each sample has 3 input channels (current, eff_dist, pdn_density)
    and 1 target (ir_drop). Spatial dimensions vary per sample.

    For training we resize to a fixed patch_size. For inference, the
    model handles arbitrary dimensions via padding.
    """

    FEATURE_FILES = ["current_map", "eff_dist_map", "pdn_density"]
    TARGET_FILE = "ir_drop_map"

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = False,
        normalize_targets: bool = False,
        cache_npy: bool = True,
        oversample: int = 1,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == "train")
        self.normalize_targets = normalize_targets
        self.cache_npy = cache_npy
        self.oversample = oversample if (split == "train") else 1

        self.samples = self._discover_samples()

    def _discover_samples(self):
        """Find all valid samples from contest directory structure."""
        samples = []

        # --- Fake circuit data (flat naming convention) ---
        fake_dir = self.data_root / "fake-circuit-data"
        if fake_dir.is_dir():
            # Files: current_mapNN_current.csv, current_mapNN_ir_drop.csv, etc.
            seen = set()
            for f in sorted(fake_dir.glob("*_current.csv")):
                prefix = f.name.replace("_current.csv", "")
                # Check all required files exist
                ir_file = fake_dir / f"{prefix}_ir_drop.csv"
                eff_file = fake_dir / f"{prefix}_eff_dist.csv"
                pdn_file = fake_dir / f"{prefix}_pdn_density.csv"
                if ir_file.exists() and eff_file.exists() and pdn_file.exists():
                    if prefix not in seen:
                        seen.add(prefix)
                        samples.append({
                            "type": "fake",
                            "id": prefix,
                            "current": str(fake_dir / f"{prefix}_current.csv"),
                            "eff_dist": str(eff_file),
                            "pdn_density": str(pdn_file),
                            "ir_drop": str(ir_file),
                        })

        # --- Real circuit data (per-testcase directories) ---
        real_dir = self.data_root / "real-circuit-data"
        if real_dir.is_dir():
            for tc_dir in sorted(real_dir.iterdir()):
                if not tc_dir.is_dir():
                    continue
                cur = tc_dir / "current_map.csv"
                eff = tc_dir / "eff_dist_map.csv"
                pdn = tc_dir / "pdn_density.csv"
                ir = tc_dir / "ir_drop_map.csv"
                if cur.exists() and eff.exists() and pdn.exists() and ir.exists():
                    samples.append({
                        "type": "real",
                        "id": tc_dir.name,
                        "current": str(cur),
                        "eff_dist": str(eff),
                        "pdn_density": str(pdn),
                        "ir_drop": str(ir),
                        "netlist": str(tc_dir / "netlist.sp") if (tc_dir / "netlist.sp").exists() else None,
                    })

        # --- Hidden test data (no ir_drop labels) ---
        hidden_dir = self.data_root / "hidden-real-circuit-data"
        if hidden_dir.is_dir():
            for tc_dir in sorted(hidden_dir.iterdir()):
                if not tc_dir.is_dir():
                    continue
                cur = tc_dir / "current_map.csv"
                eff = tc_dir / "eff_dist_map.csv"
                pdn = tc_dir / "pdn_density.csv"
                ir = tc_dir / "ir_drop_map.csv"
                if cur.exists() and eff.exists() and pdn.exists():
                    samples.append({
                        "type": "hidden",
                        "id": tc_dir.name,
                        "current": str(cur),
                        "eff_dist": str(eff),
                        "pdn_density": str(pdn),
                        "ir_drop": str(ir) if ir.exists() else None,
                    })

        # Split logic
        fake_samples = [s for s in samples if s["type"] == "fake"]
        real_samples = [s for s in samples if s["type"] == "real"]
        hidden_samples = [s for s in samples if s["type"] == "hidden"]

        if self.split == "train":
            # Exclude validation samples from training to prevent data leakage
            if real_samples:
                n_val = max(1, len(real_samples) // 5)
                train_real = real_samples[:-n_val]
            else:
                train_real = []
            if real_samples:
                return fake_samples + train_real
            else:
                return fake_samples[:-10] if len(fake_samples) > 10 else fake_samples
        elif self.split == "val":
            # Use real circuits for validation (held out from train)
            if real_samples:
                n_val = max(1, len(real_samples) // 5)
                return real_samples[-n_val:]
            return fake_samples[-10:]  # Fallback: last 10 fake
        elif self.split == "test":
            return hidden_samples if hidden_samples else real_samples
        elif self.split == "all":
            return samples
        else:
            return samples

    def __len__(self):
        return len(self.samples) * self.oversample

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]

        # Load 3 feature CSVs (with .npy caching for speed)
        current = load_csv_matrix(sample["current"], cache=self.cache_npy)
        eff_dist = load_csv_matrix(sample["eff_dist"], cache=self.cache_npy)
        pdn_density = load_csv_matrix(sample["pdn_density"], cache=self.cache_npy)

        # Stack as [3, H, W]
        features = np.stack([current, eff_dist, pdn_density], axis=0)

        # Load target if available
        if sample.get("ir_drop"):
            target = load_csv_matrix(sample["ir_drop"], cache=self.cache_npy)[np.newaxis]  # [1, H, W]
        else:
            h, w = features.shape[1], features.shape[2]
            target = np.zeros((1, h, w), dtype=np.float32)

        # Convert to tensors
        features = torch.from_numpy(features)
        target = torch.from_numpy(target)

        h, w = features.shape[1], features.shape[2]

        # Random crop if image is larger than patch_size, else resize
        if self.augment and self.patch_size > 0 and h > self.patch_size and w > self.patch_size:
            top = torch.randint(0, h - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, w - self.patch_size + 1, (1,)).item()
            features = features[:, top:top+self.patch_size, left:left+self.patch_size]
            target = target[:, top:top+self.patch_size, left:left+self.patch_size]
        elif self.patch_size > 0:
            if h != self.patch_size or w != self.patch_size:
                features = TF.resize(features, [self.patch_size, self.patch_size],
                                     antialias=True, interpolation=TF.InterpolationMode.BILINEAR)
                target = TF.resize(target, [self.patch_size, self.patch_size],
                                   antialias=True, interpolation=TF.InterpolationMode.BILINEAR)

        # Normalize features per-channel to [0, 1]
        for c in range(features.shape[0]):
            fmin, fmax = features[c].min(), features[c].max()
            if fmax - fmin > 1e-8:
                features[c] = (features[c] - fmin) / (fmax - fmin)

        # DO NOT normalize target — contest evaluates on raw voltage values
        if self.normalize_targets:
            tmin, tmax = target.min(), target.max()
            if tmax - tmin > 1e-8:
                target = (target - tmin) / (tmax - tmin)

        # Data augmentation (geometric only — preserves voltage values)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                features = TF.hflip(features)
                target = TF.hflip(target)
            if torch.rand(1).item() > 0.5:
                features = TF.vflip(features)
                target = TF.vflip(target)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                features = torch.rot90(features, k, [1, 2])
                target = torch.rot90(target, k, [1, 2])

        return features, target


# ---------------------------------------------------------------------------
# CircuitNet-N14 fallback (proxy data for development)
# ---------------------------------------------------------------------------

class CircuitNetIRDropDataset(Dataset):
    """Load CircuitNet-N14 .npz data as proxy training data.

    By default maps to 3 contest-compatible channels:
      power_all -> current_map (proxy)
      power_i   -> eff_dist_map (proxy)
      power_s   -> pdn_density (proxy)

    Set in_channels=4 to use all 4 original CircuitNet channels
    (for compatibility with older checkpoints).
    """

    FEATURE_DIRS_3 = ["power_all", "power_i", "power_s"]  # 3-channel contest mapping
    FEATURE_DIRS_4 = ["power_i", "power_s", "power_sca", "power_all"]  # Original 4-channel
    TARGET_DIR = "IR_drop"

    def __init__(self, data_root, split="train", img_size=256, augment=False, in_channels=3):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.feature_dirs = self.FEATURE_DIRS_4 if in_channels == 4 else self.FEATURE_DIRS_3
        self.samples = self._discover_samples()

    def _discover_samples(self):
        designs = []
        if self.data_root.exists():
            for d in sorted(self.data_root.iterdir()):
                if d.is_dir() and (d / self.TARGET_DIR).is_dir():
                    designs.append(d.name)

        if not designs:
            return []

        all_samples = []
        for design in designs:
            target_dir = self.data_root / design / self.TARGET_DIR
            for npz_file in sorted(target_dir.glob("*.npz")):
                sample_name = npz_file.stem
                if all(
                    (self.data_root / design / feat / f"{sample_name}.npz").exists()
                    for feat in self.feature_dirs
                ):
                    all_samples.append((design, sample_name))

        if not all_samples:
            return []

        n = len(all_samples)
        n_train = max(1, int(0.6 * n))
        n_val = max(1, int(0.2 * n))
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        split_map = {
            "train": set(indices[:n_train].tolist()),
            "val": set(indices[n_train:n_train + n_val].tolist()),
            "test": set(indices[n_train + n_val:].tolist()),
        }
        selected = split_map.get(self.split, split_map["test"])
        return [all_samples[i] for i in sorted(selected)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        design, sample_name = self.samples[idx]
        feat_arrays = []
        for feat_name in self.feature_dirs:
            path = self.data_root / design / feat_name / f"{sample_name}.npz"
            data = np.load(str(path))
            key = list(data.keys())[0]
            feat_arrays.append(data[key].astype(np.float32))
        features = np.stack(feat_arrays, axis=0)  # [3, H, W]

        target_path = self.data_root / design / self.TARGET_DIR / f"{sample_name}.npz"
        data = np.load(str(target_path))
        key = list(data.keys())[0]
        target = data[key].astype(np.float32)[np.newaxis]  # [1, H, W]

        features = torch.from_numpy(features)
        target = torch.from_numpy(target)

        if features.shape[-1] != self.img_size or features.shape[-2] != self.img_size:
            features = TF.resize(features, [self.img_size, self.img_size], antialias=True)
            target = TF.resize(target, [self.img_size, self.img_size], antialias=True)

        for c in range(features.shape[0]):
            fmin, fmax = features[c].min(), features[c].max()
            if fmax - fmin > 1e-8:
                features[c] = (features[c] - fmin) / (fmax - fmin)

        # DO NOT normalize target
        if self.augment:
            if torch.rand(1).item() > 0.5:
                features = TF.hflip(features)
                target = TF.hflip(target)
            if torch.rand(1).item() > 0.5:
                features = TF.vflip(features)
                target = TF.vflip(target)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                features = torch.rot90(features, k, [1, 2])
                target = torch.rot90(target, k, [1, 2])

        return features, target


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    patch_size: int = 256,
    num_workers: int = 4,
    data_format: str = "auto",
    in_channels: int = 3,
    cache_npy: bool = True,
    oversample: int = 8,
) -> dict:
    """Create train/val/test data loaders.

    Args:
        data_format: "contest" for CSV, "circuitnet" for .npz, "auto" to detect.
        in_channels: Number of input channels (3 for contest, 4 for legacy CircuitNet).
        oversample: Multiply training set size (each copy gets different random crops/augmentations).
    """
    root = Path(data_root)

    # Auto-detect format
    if data_format == "auto":
        if (root / "fake-circuit-data").is_dir() or (root / "real-circuit-data").is_dir():
            data_format = "contest"
        else:
            data_format = "circuitnet"

    loaders = {}
    for split in ["train", "val", "test"]:
        if data_format == "contest":
            ds = ContestIRDropDataset(
                data_root=data_root,
                split=split,
                patch_size=patch_size,
                augment=(split == "train"),
                cache_npy=cache_npy,
                oversample=oversample if (split == "train") else 1,
            )
        else:
            ds = CircuitNetIRDropDataset(
                data_root=data_root,
                split=split,
                img_size=patch_size,
                augment=(split == "train"),
                in_channels=in_channels,
            )

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train" and len(ds) > batch_size),
        )
    return loaders
