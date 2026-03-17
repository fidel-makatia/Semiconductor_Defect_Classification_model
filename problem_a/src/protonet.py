"""Prototypical Network for few-shot classification.

Uses L2-normalized embeddings with learned temperature scaling
(cosine similarity) for more robust distance comparisons.
Supports incremental prototype updates and anomaly detection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Learnable projection from backbone embeddings to prototype space.

    3-layer MLP with LayerNorm and GELU for richer representations.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrototypicalNetwork(nn.Module):
    """Prototypical Network with cosine similarity and learned temperature.

    Embeddings are L2-normalized so distance comparisons use cosine similarity,
    scaled by a learned temperature parameter for sharper/softer softmax.
    """

    def __init__(self, backbone, proj_hidden=512, proj_dim=256):
        super().__init__()
        self.backbone = backbone
        self.projection = ProjectionHead(backbone.embed_dim, proj_hidden, proj_dim)
        self.proj_dim = proj_dim
        # Learned log-temperature for cosine similarity scaling
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    @property
    def scale(self):
        return self.log_scale.exp()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract, project, and L2-normalize features [B, C, H, W] -> [B, proj_dim]."""
        features = self.backbone(x)
        projected = self.projection(features)
        return F.normalize(projected, dim=1)

    def compute_prototypes(self, support_images, support_labels):
        """Compute L2-normalized class prototypes from support set."""
        embeddings = self.embed(support_images)
        n_classes = support_labels.unique().shape[0]
        prototypes = torch.zeros(n_classes, self.proj_dim, device=embeddings.device)
        for c in range(n_classes):
            mask = support_labels == c
            prototypes[c] = embeddings[mask].mean(dim=0)
        return F.normalize(prototypes, dim=1)

    def classify(self, query_images, prototypes):
        """Classify queries via cosine similarity -> log-probabilities [Q, N]."""
        query_embeddings = self.embed(query_images)
        return self.classify_embeddings(query_embeddings, prototypes)

    def classify_embeddings(self, query_embeddings, prototypes):
        """Classify pre-computed embeddings against prototypes.

        Both inputs are L2-normalized before computing cosine similarity.
        """
        q_norm = F.normalize(query_embeddings, dim=1)
        p_norm = F.normalize(prototypes, dim=1)
        logits = torch.mm(q_norm, p_norm.t()) * self.scale
        return F.log_softmax(logits, dim=1)

    def detect_and_classify(self, query_images, prototypes, distance_threshold=None):
        """Classify with defect detection (anomaly = far from all prototypes).

        Returns:
            log_probs: [Q, N] log-probabilities
            min_dists: [Q] cosine distance to nearest prototype (1 - max_cos_sim)
            is_known: [Q] boolean mask (True = within threshold, classified as known defect)
        """
        query_embeddings = self.embed(query_images)
        q_norm = F.normalize(query_embeddings, dim=1)
        p_norm = F.normalize(prototypes, dim=1)
        sims = torch.mm(q_norm, p_norm.t())  # [Q, N]
        max_sims, _ = sims.max(dim=1)
        min_dists = 1 - max_sims  # Cosine distance
        logits = sims * self.scale
        log_probs = F.log_softmax(logits, dim=1)

        if distance_threshold is not None:
            is_known = min_dists <= distance_threshold
        else:
            is_known = torch.ones(len(query_images), dtype=torch.bool,
                                  device=query_images.device)

        return log_probs, min_dists, is_known

    def forward(self, support_images, support_labels, query_images):
        prototypes = self.compute_prototypes(support_images, support_labels)
        return self.classify(query_images, prototypes)


class IncrementalPrototypeTracker:
    """Track prototypes incrementally as new examples arrive.

    Stores L2-normalized embeddings and computes mean prototypes.
    Uses the model's cosine similarity for classification.
    """

    def __init__(self, model: PrototypicalNetwork, device: torch.device):
        self.model = model
        self.device = device
        self.prototype_sums = {}
        self.prototype_counts = {}
        self._prototypes_cache = None

    @torch.no_grad()
    def add_example(self, image: torch.Tensor, label: int):
        """Add a single example and update the class prototype."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        embedding = self.model.embed(image).squeeze(0)  # L2-normalized

        if label not in self.prototype_sums:
            self.prototype_sums[label] = torch.zeros_like(embedding)
            self.prototype_counts[label] = 0

        self.prototype_sums[label] += embedding
        self.prototype_counts[label] += 1
        self._prototypes_cache = None

    @property
    def prototypes(self) -> torch.Tensor:
        if self._prototypes_cache is not None:
            return self._prototypes_cache
        if not self.prototype_sums:
            return None
        labels = sorted(self.prototype_sums.keys())
        protos = [self.prototype_sums[l] / self.prototype_counts[l] for l in labels]
        self._prototypes_cache = torch.stack(protos)
        return self._prototypes_cache

    @property
    def label_map(self) -> dict:
        return {l: i for i, l in enumerate(sorted(self.prototype_sums.keys()))}

    @property
    def num_examples_seen(self) -> int:
        return sum(self.prototype_counts.values())

    def get_per_class_counts(self) -> dict:
        """Get number of examples seen per class."""
        return dict(self.prototype_counts)

    @torch.no_grad()
    def evaluate(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """Evaluate accuracy on a batch using current prototypes."""
        if self.prototypes is None or len(self.prototype_sums) < 2:
            return 0.0

        images = images.to(self.device)
        lmap = self.label_map

        mask = torch.tensor([int(l) in lmap for l in labels])
        if mask.sum() == 0:
            return 0.0

        filtered_images = images[mask]
        filtered_labels = torch.tensor([lmap[int(l)] for l in labels[mask]])

        log_probs = self.model.classify(filtered_images, self.prototypes)
        preds = log_probs.argmax(dim=1).cpu()
        return (preds == filtered_labels).float().mean().item()

    @torch.no_grad()
    def evaluate_per_class(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """Evaluate per-class accuracy using current prototypes.

        Returns dict mapping original_label -> accuracy.
        """
        if self.prototypes is None or len(self.prototype_sums) < 2:
            return {}

        images = images.to(self.device)
        lmap = self.label_map

        mask = torch.tensor([int(l) in lmap for l in labels])
        if mask.sum() == 0:
            return {}

        filtered_images = images[mask]
        filtered_orig_labels = labels[mask]
        filtered_mapped = torch.tensor([lmap[int(l)] for l in filtered_orig_labels])

        log_probs = self.model.classify(filtered_images, self.prototypes)
        preds = log_probs.argmax(dim=1).cpu()

        per_class = {}
        inv_map = {v: k for k, v in lmap.items()}
        for mapped_label in filtered_mapped.unique():
            cls_mask = filtered_mapped == mapped_label
            cls_correct = (preds[cls_mask] == mapped_label).float().mean().item()
            orig_label = inv_map[mapped_label.item()]
            per_class[orig_label] = cls_correct

        return per_class

    @torch.no_grad()
    def detect(self, images: torch.Tensor, distance_threshold: float = None) -> tuple:
        """Detect whether images contain known defects."""
        if self.prototypes is None:
            return None, None, None

        images = images.to(self.device)
        log_probs, min_dists, is_known = self.model.detect_and_classify(
            images, self.prototypes, distance_threshold
        )
        return log_probs.argmax(dim=1).cpu(), min_dists.cpu(), is_known.cpu()

    def reset(self):
        self.prototype_sums.clear()
        self.prototype_counts.clear()
        self._prototypes_cache = None
