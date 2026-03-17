"""Dataset loaders for few-shot defect classification.

Supports:
  - Intel contest dataset (8 defect classes, grayscale, up to 1500x2500)
  - DAGM 2007 as proxy dataset (10 texture classes with defects)

Provides standard, episodic (N-way K-shot), and incremental evaluation modes.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image


class IntelDefectDataset(Dataset):
    """Intel contest defect dataset.

    Expected structure (adapt once actual data format is known):
        data_root/
            Class1/
                img_001.png (or .bmp, .tif, etc.)
                ...
            Class2/
                ...
            Class8/
                ...

    OR flat structure:
        data_root/
            images/
                img_001.png
                ...
            labels.csv  (filename, class_id)

    Grayscale images up to 1500x2500 pixels, 8 defect classes.
    Supports both defective-only mode (classification) and
    defective+normal mode (detection + classification).

    split="val" holds out 20% of images per class for validation
    (stratified, deterministic seed=42).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform=None,
        defect_only: bool = False,
        classes: Optional[list] = None,
        cache_images: bool = True,
        val_fraction: float = 0.2,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.defect_only = defect_only
        self.val_fraction = val_fraction

        self.samples = []
        self.class_to_indices = {}
        self._image_cache = {}

        self._load_samples(classes)

        if cache_images:
            self._preload_images()

    def _load_samples(self, classes):
        """Discover images from Intel dataset structure."""
        # Try class-folder structure first
        class_dirs = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]) if self.data_root.exists() else []

        if class_dirs:
            self._load_from_class_dirs(class_dirs, classes)
        else:
            # Try flat structure with labels file
            self._load_from_labels_file(classes)

        if not self.samples and self.split != "test":
            self._generate_synthetic(classes)

        # Stratified train/val split within each class
        if self.split in ("train", "val") and self.val_fraction > 0:
            self._apply_stratified_split()

    def _apply_stratified_split(self):
        """Hold out val_fraction of images per class for validation."""
        rng = np.random.RandomState(42)
        keep_indices = set()
        for cls_label, indices in self.class_to_indices.items():
            n = len(indices)
            n_val = max(1, int(n * self.val_fraction))
            perm = rng.permutation(n)
            if self.split == "val":
                selected = perm[:n_val]
            else:  # train
                selected = perm[n_val:]
            for i in selected:
                keep_indices.add(indices[i])

        # Rebuild samples and class_to_indices with only kept indices
        old_samples = self.samples
        self.samples = []
        self.class_to_indices = {}
        for old_idx in sorted(keep_indices):
            new_idx = len(self.samples)
            sample = old_samples[old_idx]
            self.samples.append(sample)
            self.class_to_indices.setdefault(sample["label"], []).append(new_idx)

    def _load_from_class_dirs(self, class_dirs, classes):
        """Load from Class1/, defect1/, etc. directory structure.

        Handles non-contiguous class numbering (e.g. 1,2,3,4,5,8,9,10)
        and special folders like 'good' (mapped to class 0).
        Sorts numerically by extracted class number to avoid alphabetic
        ordering issues (defect10 before defect2).
        """
        import re

        # Build list of (class_num, class_dir) to sort numerically
        numbered_dirs = []
        for class_dir in class_dirs:
            class_name = class_dir.name
            nums = re.findall(r'\d+', class_name)
            if nums:
                class_num = int(nums[0])
            elif class_name.lower() == 'good':
                class_num = 0  # 'good' / non-defect class
            else:
                continue  # skip unrecognized directories
            numbered_dirs.append((class_num, class_dir))

        # Sort by class number (numeric order)
        numbered_dirs.sort(key=lambda x: x[0])

        contiguous_label = 0
        for class_num, class_dir in numbered_dirs:
            class_name = class_dir.name

            if classes is not None and class_num not in classes:
                continue

            label_idx = contiguous_label
            contiguous_label += 1

            # Check for train/test split subdirectories
            if self.split == "train" and (class_dir / "Train").is_dir():
                img_dir = class_dir / "Train"
            elif self.split == "test" and (class_dir / "Test").is_dir():
                img_dir = class_dir / "Test"
            elif self.split == "test":
                # No dedicated Test subdir — return empty so caller
                # can do its own split (e.g. eval.py 80/20 fallback)
                continue
            else:
                img_dir = class_dir

            # Detect defective images via label masks
            label_dir = img_dir / "Label"
            defective_stems = set()
            if label_dir and label_dir.exists():
                for lf in label_dir.glob("*_label.*"):
                    stem = lf.stem.replace("_label", "")
                    defective_stems.add(stem)

            img_exts = ("*.png", "*.bmp", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
            seen_paths = set()
            for ext in img_exts:
                for img_path in sorted(img_dir.glob(ext)):
                    # Deduplicate (Windows glob is case-insensitive)
                    resolved = str(img_path.resolve())
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)

                    if img_path.parent.name == "Label":
                        continue

                    is_defective = img_path.stem in defective_stems if defective_stems else True
                    if self.defect_only and not is_defective:
                        continue

                    idx = len(self.samples)
                    self.samples.append({
                        "path": str(img_path),
                        "label": label_idx,
                        "class_name": class_name,
                        "is_defective": is_defective,
                    })
                    self.class_to_indices.setdefault(label_idx, []).append(idx)

    def _load_from_labels_file(self, classes):
        """Load from flat structure with CSV labels."""
        labels_file = self.data_root / "labels.csv"
        if not labels_file.exists():
            return

        import csv
        with open(labels_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    filename, class_id = row[0].strip(), int(row[1].strip())
                    if classes is not None and class_id not in classes:
                        continue
                    img_path = self.data_root / "images" / filename
                    if not img_path.exists():
                        img_path = self.data_root / filename
                    if img_path.exists():
                        idx = len(self.samples)
                        self.samples.append({
                            "path": str(img_path),
                            "label": class_id,
                            "class_name": f"Class{class_id}",
                            "is_defective": True,
                        })
                        self.class_to_indices.setdefault(class_id, []).append(idx)

    def _generate_synthetic(self, classes):
        """Generate synthetic data for development."""
        n_classes = 8 if classes is None else len(classes)
        class_labels = list(range(n_classes))
        n_per_class = 30 if self.split == "train" else 15

        for label in class_labels:
            for i in range(n_per_class):
                idx = len(self.samples)
                self.samples.append({
                    "path": f"synthetic_{label}_{i}",
                    "label": label,
                    "class_name": f"Class{label+1}",
                    "is_defective": True,
                })
                self.class_to_indices.setdefault(label, []).append(idx)

    def _preload_images(self):
        """Preload all images into RAM for fast episodic sampling."""
        for idx, sample in enumerate(self.samples):
            path = sample["path"]
            if path.startswith("synthetic"):
                continue
            self._image_cache[idx] = Image.open(path).convert("L")
        # Detach from file handles
        for idx in self._image_cache:
            self._image_cache[idx] = self._image_cache[idx].copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        label = sample["label"]

        if path.startswith("synthetic"):
            rng = np.random.RandomState(hash(path) % (2**31))
            img = rng.randint(50, 200, (224, 224), dtype=np.uint8)
            center = 112
            radius = 20 + label * 5
            y, x = np.ogrid[-center:224-center, -center:224-center]
            mask = x**2 + y**2 <= radius**2
            img[mask] = np.clip(img[mask].astype(int) + 80, 0, 255).astype(np.uint8)
            img = Image.fromarray(img, mode="L")
        elif idx in self._image_cache:
            img = self._image_cache[idx].copy()
        else:
            img = Image.open(path).convert("L")

        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0)

        return img, label

    def get_class_indices(self, label: int) -> list:
        return self.class_to_indices.get(label, [])

    @property
    def num_classes(self) -> int:
        return len(self.class_to_indices)

    @property
    def class_labels(self) -> list:
        return sorted(self.class_to_indices.keys())

    @property
    def class_names(self) -> list:
        """Get human-readable class names."""
        names = {}
        for s in self.samples:
            names[s["label"]] = s["class_name"]
        return [names.get(l, f"Class{l}") for l in self.class_labels]


# Keep DAGM as alias for backward compatibility
class DAGMDataset(IntelDefectDataset):
    """DAGM 2007 defect dataset (backward-compatible alias).

    Same as IntelDefectDataset — the loader auto-detects DAGM's
    ClassN/Train/... ClassN/Test/... directory structure.
    """
    pass


class EpisodicSampler(Sampler):
    """Sample N-way K-shot episodes for meta-learning."""

    def __init__(self, dataset, n_way=5, k_shot=5, n_query=15, n_episodes=100):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            all_labels = self.dataset.class_labels
            n_select = min(self.n_way, len(all_labels))
            classes = np.random.choice(all_labels, size=n_select, replace=False)
            episode_indices = []
            for cls in classes:
                cls_indices = self.dataset.get_class_indices(cls)
                needed = self.k_shot + self.n_query
                # Always use replacement for rare classes to reach needed count
                selected = np.random.choice(
                    cls_indices,
                    size=needed,
                    replace=len(cls_indices) < needed,
                )
                episode_indices.extend(selected.tolist())
            yield episode_indices

    def __len__(self):
        return self.n_episodes


def episode_collate_fn(batch):
    """Custom collate that keeps (image, label) tuples as a list."""
    return batch


def collate_episode(batch, n_way, k_shot):
    """Collate an episode into support and query sets."""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])

    unique_labels = labels.unique()
    label_map = {int(l): i for i, l in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_map[int(l)] for l in labels])

    support_imgs, support_lbls = [], []
    query_imgs, query_lbls = [], []

    for new_label in range(min(n_way, len(unique_labels))):
        mask = mapped_labels == new_label
        cls_imgs = images[mask]
        cls_lbls = mapped_labels[mask]

        support_imgs.append(cls_imgs[:k_shot])
        support_lbls.append(cls_lbls[:k_shot])
        query_imgs.append(cls_imgs[k_shot:])
        query_lbls.append(cls_lbls[k_shot:])

    return (
        torch.cat(support_imgs),
        torch.cat(support_lbls),
        torch.cat(query_imgs),
        torch.cat(query_lbls),
    )
