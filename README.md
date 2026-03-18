# Semiconductor Defect Classification Model

**Few-Shot Semiconductor Defect Classification using DINOv2 and Prototypical Networks**

A desktop application and deep learning pipeline for classifying semiconductor wafer defects from grayscale microscopy images. Built for the **Intel Semiconductor Solutions Challenge 2026**, this system uses a DINOv2 Vision Transformer backbone with a Prototypical Network head to achieve high accuracy with as few as 1-5 example images per defect class.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Qt](https://img.shields.io/badge/Qt-PySide6-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

### Pre-trained Model

**Download the model checkpoint (1.2 GB) from Hugging Face:**

https://huggingface.co/Makatia/semiconductor-defect-classifier

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Desktop Application](#desktop-application)
- [Packaging and Distribution](#packaging-and-distribution)
- [Results](#results)
- [Technical Details](#technical-details)

---

## Overview

Semiconductor manufacturing requires rapid, accurate detection and classification of wafer defects. Traditional approaches need thousands of labeled examples per class, which is impractical for rare defect types. This project solves this with **few-shot learning** -- the model can classify defects accurately using only 1-20 reference images per class.

The system classifies images into **9 categories**: 8 defect types + 1 "good" (non-defective) class.

### Key Capabilities

- **Few-shot classification**: Accurate with as few as 1-5 support images per class
- **8 defect classes + good detection**: Handles both defective and non-defective wafer images
- **Attention visualization**: Shows which image regions drive the classification decision
- **Real-time inference**: ~700ms per image on GPU, ~3s on CPU
- **Desktop GUI**: Drag-and-drop image analysis with PySide6/QML interface
- **Incremental learning**: Add new support examples on the fly without retraining

---

## Features

### Deep Learning Pipeline
- DINOv2 ViT-L/14 backbone (304M parameters) with fine-tuned last 6 transformer blocks
- 3-layer projection head: 1024 to 768 to 512 dimensions (L2-normalized)
- Cosine similarity with learned temperature parameter for classification
- Episodic meta-learning training: 9-way 5-shot 10-query episodes
- Mixed precision (AMP) training with gradient checkpointing for memory efficiency
- Differential learning rates: backbone (5e-6) vs projection head (3e-4)
- Label smoothing (0.1) for robust generalization
- Aspect-ratio-preserving resize to DINOv2's native 518x518 resolution

### Desktop Application
- Modern PySide6/QML interface with dark theme
- Dashboard showing GPU status, model info, and system diagnostics
- Defect Detection page with drag-and-drop image analysis
- Attention heatmap and overlay visualization
- Defect knowledge base with root cause analysis and prevention steps
- Configurable settings for paths and parameters

### Good Image Detection
- Hybrid approach: prototypical classification + cosine similarity gap threshold
- Automatically detects non-defective wafer images
- 20 diverse "good" support images for robust prototype estimation

---

## Architecture

```
Input Image (grayscale, up to 7000x5600)
    |
    v
+----------------------------------+
|  Preprocessing                    |
|  - LongestMaxSize(518)           |
|  - PadIfNeeded(518x518)          |
|  - Normalize(mean=0.5, std=0.5)  |
+-----------------+----------------+
                  |
                  v
+----------------------------------+
|  DINOv2 ViT-L/14 Backbone        |
|  - 304M parameters (frozen)      |
|  - Last 6 blocks fine-tuned      |
|  - Gradient checkpointing        |
|  - Output: 1024-dim CLS token    |
+-----------------+----------------+
                  |
                  v
+----------------------------------+
|  Projection Head                  |
|  - Linear(1024, 768)             |
|  - LayerNorm + GELU              |
|  - Linear(768, 768)              |
|  - LayerNorm + GELU              |
|  - Linear(768, 512)              |
|  - L2 Normalization              |
+-----------------+----------------+
                  |
                  v
+----------------------------------+
|  Prototypical Classification      |
|  - Compute class prototypes from  |
|    support set (mean embeddings)  |
|  - Cosine similarity x learned   |
|    temperature (log_scale)        |
|  - Softmax -> class probabilities|
|  - Good-detection gap threshold   |
+----------------------------------+
```

### Prototypical Network

The model learns an embedding space where images of the same class cluster together. At inference time:

1. **Support set**: K reference images per class are embedded and averaged to form **class prototypes**
2. **Query image**: The test image is embedded into the same space
3. **Classification**: Cosine similarity between the query embedding and each prototype determines the predicted class
4. **Good detection**: If the gap between the top defect similarity and the "good" prototype similarity is below a threshold (0.20), the image is classified as non-defective

### Incremental Prototype Tracking

The `IncrementalPrototypeTracker` maintains running averages of class prototypes, allowing new support examples to be added without recomputing from scratch. This enables real-time learning in the desktop application.

---

## Dataset

The model is trained on the Intel Semiconductor Solutions Challenge dataset:

| Class | Name | Samples | Description |
|-------|------|---------|-------------|
| 0 | Good | 7,135 | Non-defective wafer surface |
| 1 | Defect 1 | 253 | Scratch-type defect |
| 2 | Defect 2 | 178 | Particle contamination |
| 3 | Defect 3 | 9 | Micro-crack (rare) |
| 4 | Defect 4 | 14 | Edge defect (rare) |
| 5 | Defect 5 | 411 | Pattern anomaly |
| 8 | Defect 8 | 803 | Surface roughness |
| 9 | Defect 9 | 319 | Deposition defect |
| 10 | Defect 10 | 674 | Etch residue |

**Note**: Defect classes 6 and 7 do not exist in the dataset. Classes 3 and 4 are extremely rare (9 and 14 samples respectively), making this a challenging few-shot scenario.

### Data Characteristics
- **Image format**: Grayscale PNG
- **Resolution**: Up to ~7000x5600 pixels
- **Total samples**: 9,796
- **Class imbalance**: 793:1 ratio (good vs defect3)
- **Critical pairs**: defect3 and defect9 (0.963 cosine similarity), defect4 and defect8 (0.889) -- nearly identical without explicit training

---

## Project Structure

```
.
├── app/                              # Desktop application
│   ├── main.py                       # Application entry point (PySide6)
│   ├── bridge.py                     # QML <-> Python bridge (signals/slots)
│   ├── imageprovider.py              # QML image provider for heatmaps
│   ├── processmanager.py             # QProcess wrapper for inference subprocess
│   ├── python/
│   │   └── inference_server.py       # Inference server (stdin/stdout JSON protocol)
│   ├── qml/
│   │   ├── main.qml                  # Main window layout
│   │   ├── components/               # Reusable QML components
│   │   │   ├── Sidebar.qml           # Navigation sidebar
│   │   │   ├── StatusBar.qml         # Bottom status bar
│   │   │   └── ImageDropZone.qml     # Drag-and-drop image input
│   │   └── pages/
│   │       ├── DashboardPage.qml     # System info and model status
│   │       ├── DefectPage.qml        # Defect classification interface
│   │       └── SettingsPage.qml      # Configuration page
│   └── data/
│       └── support/                  # Bundled support images (5-20 per class)
│           ├── good/                 # 20 non-defective reference images
│           ├── defect1/              # 5 reference images per defect class
│           ├── defect2/
│           └── ...
│
├── problem_a/                        # Training and evaluation pipeline
│   ├── train.py                      # Training script (episodic meta-learning)
│   ├── evaluate.py                   # K-shot evaluation and confusion matrices
│   ├── configs/
│   │   └── default.yaml              # Training hyperparameters
│   ├── src/
│   │   ├── backbone.py               # DINOv2 backbone wrapper
│   │   ├── protonet.py               # Prototypical Network + tracker
│   │   ├── dataset.py                # Few-shot episode sampler
│   │   ├── augmentations.py          # Train/eval transforms
│   │   └── visualize.py              # Plotting utilities
│   └── checkpoints/
│       └── best_model.pt             # Trained model weights (~1.2 GB, download from HF)
│
├── package.py                        # Distribution packager
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- NVIDIA GPU with CUDA support (recommended; CPU fallback available)
- ~4 GB disk space for dependencies
- ~1.2 GB for the model checkpoint

### Setup

```bash
# Clone the repository
git clone https://github.com/fidel-makatia/Semiconductor_Defect_Classification_model.git
cd Semiconductor_Defect_Classification_model

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm>=1.0 albumentations>=1.3 opencv-python-headless>=4.8
pip install Pillow>=10.0 numpy>=1.24 matplotlib>=3.7 PyYAML>=6.0
pip install PySide6>=6.5 scikit-learn>=1.3

# Download pre-trained model checkpoint (~1.2 GB)
# From Hugging Face:
mkdir -p problem_a/checkpoints
# Download best_model.pt from https://huggingface.co/Makatia/semiconductor-defect-classifier
# and place it in problem_a/checkpoints/best_model.pt
```

> **Pre-trained Model**: The model checkpoint (~1.2 GB) is hosted on [Hugging Face](https://huggingface.co/Makatia/semiconductor-defect-classifier). Download `best_model.pt` and place it in `problem_a/checkpoints/`.
>
> **Packaged App**: A ready-to-run distribution (includes model + dependencies launcher) is available on [Google Drive](https://drive.google.com/drive/folders/12migcKrLSo5II0LhCG046wkpJsL-dZkY?usp=sharing). Just unzip and run `run.bat` (Windows) or `run.sh` (Linux).

---

## Training

### Configuration

All training hyperparameters are in `problem_a/configs/default.yaml`:

```yaml
data:
  root: "../challenge/dataset/Dataset/Data/"
  img_size: 518                    # DINOv2 native resolution
  train_classes: [0, 1, 2, 3, 4, 5, 8, 9, 10]

model:
  backbone: "dinov2"
  backbone_size: "large"           # ViT-L/14 (1024-dim, 304M params)
  freeze_backbone: true
  unfreeze_last_n: 6               # Fine-tune last 6 transformer blocks
  grad_checkpointing: true
  proj_hidden: 768
  proj_dim: 512

training:
  n_way: 9                         # All 9 classes per episode
  k_shot: 5
  n_query: 10
  n_episodes_train: 500
  epochs: 100
  lr: 3.0e-4                       # Projection head learning rate
  lr_backbone: 5.0e-6              # Backbone learning rate (differential)
  warmup_epochs: 5
  label_smoothing: 0.1
  patience: 20                     # Early stopping
```

### Run Training

```bash
cd problem_a
python train.py --config configs/default.yaml
```

Training uses **episodic meta-learning**: each epoch consists of 500 episodes, where each episode samples 9 classes with 5 support + 10 query images per class. The model learns to classify by comparing query embeddings to support prototypes.

**Hardware requirements**:
- GPU: ~2 GB VRAM with gradient checkpointing enabled
- Training time: ~17 minutes/epoch on NVIDIA RTX PRO 6000
- Convergence: ~7-30 epochs (early stopping with patience=20)

### Training Features

- **Differential learning rates**: Backbone layers use 60x lower LR than the projection head
- **Gradient checkpointing**: Reduces VRAM usage from ~24 GB to ~2 GB
- **Mixed precision (AMP)**: Faster training with reduced memory footprint
- **Cosine annealing**: Learning rate schedule with warmup
- **Label smoothing**: Prevents overconfidence on easy classes
- **Gradient clipping**: Max norm 1.0 for training stability
- **Sampling with replacement**: Ensures rare classes (defect3=9 samples) appear in every episode

---

## Evaluation

```bash
cd problem_a
python evaluate.py --config configs/default.yaml
```

The evaluation script runs K-shot testing across multiple values of K (1, 3, 5, 10, 20) and generates:
- Per-class accuracy breakdown
- Confusion matrix
- K-shot comparison plots
- Accuracy vs class occurrence analysis

---

## Desktop Application

### Launch

```bash
python app/main.py
```

Or use the packaged distribution:
- **Windows**: Double-click `run.bat`
- **Linux**: `chmod +x run.sh && ./run.sh`

### Application Pages

1. **Dashboard**: System information, GPU status, model status, and architecture details
2. **Defect Detection**:
   - Drag and drop or browse for a wafer image
   - View classification result with confidence scores
   - Attention heatmap showing which regions drive the prediction
   - Overlay visualization (attention map blended with original image)
   - Defect knowledge base: description, severity, root causes, prevention steps
3. **Settings**: Configure data paths and inference parameters

### Inference Architecture

The application uses a **subprocess architecture**:

```
+--------------------+    JSON/stdin     +-----------------------+
|  PySide6/QML UI    | ----------------> |  inference_server.py   |
|  (main process)    |                   |  (subprocess)          |
|                    | <---------------- |                        |
|  bridge.py         |    JSON/stdout    |  PyTorch model         |
|  processmanager    |                   |  DINOv2 + ProtoNet     |
+--------------------+                   +-----------------------+
```

This design isolates the heavy PyTorch inference from the UI thread, preventing freezes and ensuring responsive interaction.

### Defect Knowledge Base

The application includes a built-in knowledge base for each defect class with:
- **Description**: What the defect looks like and its characteristics
- **Severity**: Critical / High / Medium / Low / None
- **Root Causes**: Common manufacturing issues that produce this defect
- **Prevention**: Process control recommendations

---

## Packaging and Distribution

Create a self-contained distribution folder:

```bash
python package.py
```

This generates `dist/SemiAI/` containing:
- Application source code and QML interface
- Pre-trained model checkpoint
- Bundled support images (5-20 per class)
- `run.bat` (Windows) and `run.sh` (Linux) launcher scripts
- `requirements.txt` for automatic dependency installation
- `README.txt` with quick-start instructions

The launcher scripts automatically create a virtual environment and install dependencies on first run.

---

## Results

### Classification Accuracy

| K-Shot | Accuracy |
|--------|----------|
| K=1 | 99.5% |
| K=3 | 99.7% |
| K=5 | 99.7% |
| K=10 | 99.7% |
| K=20 | 99.8% |

### Per-Class F1 Scores (K=20)

| Class | F1 Score |
|-------|----------|
| Defect 1 | 1.000 |
| Defect 2 | 1.000 |
| Defect 3 | 1.000 |
| Defect 4 | 1.000 |
| Defect 5 | 0.994 |
| Defect 8 | 1.000 |
| Defect 9 | 1.000 |
| Defect 10 | 0.996 |

### Good Image Detection

| Metric | Value |
|--------|-------|
| Good accuracy | ~90% |
| Defect accuracy | ~97% |
| Method | Cosine similarity gap threshold (0.20) |

### Model Specifications

| Property | Value |
|----------|-------|
| Backbone | DINOv2 ViT-L/14 |
| Parameters | 306M total, 77M trainable (25.3%) |
| Embedding dim | 512 (L2-normalized) |
| Input size | 518x518 (aspect-ratio preserved) |
| Inference time | ~700ms (GPU) / ~3s (CPU) |
| VRAM usage | ~2 GB (inference) |
| Checkpoint size | 1.2 GB |

---

## Technical Details

### Why DINOv2?

DINOv2 (ViT-L/14) was chosen because:
1. **Self-supervised pre-training** on 142M images gives rich, general-purpose visual features
2. **Excellent few-shot transfer**: DINOv2 features cluster semantically without fine-tuning
3. **518x518 native resolution** preserves detail in high-resolution wafer images
4. **Patch-level features** enable attention visualization for interpretability

### Why Prototypical Networks?

Prototypical Networks are ideal for this task because:
1. **Sample-efficient**: Work with as few as 1 example per class
2. **Non-parametric classifier**: No class-specific parameters to learn
3. **Incremental learning**: New classes/examples can be added without retraining
4. **Interpretable**: Cosine similarity provides natural confidence scores

### Handling Class Imbalance

The dataset has extreme imbalance (7135 good vs 9 defect3). The training strategy addresses this through:
1. **Episodic sampling**: Each episode samples equally from all classes (with replacement for rare classes)
2. **Good-detection threshold**: A cosine similarity gap threshold compensates for the high diversity of "good" images
3. **Label smoothing**: Prevents the model from becoming overconfident on the dominant class

### Handling Visually Similar Classes

Defect3 and Defect9, as well as Defect4 and Defect8, are nearly identical without training. The model separates them by:
1. **Training on all 9 classes simultaneously**: The backbone learns discriminative features for similar pairs
2. **Fine-tuning last 6 transformer blocks**: Allows the backbone to specialize for semiconductor domain
3. **High-dimensional projection** (512-dim): Provides sufficient capacity for fine-grained discrimination

---

## License

MIT License

---

## Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI Research
- [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman
- Intel Corporation for the Semiconductor Solutions Challenge 2026 dataset
