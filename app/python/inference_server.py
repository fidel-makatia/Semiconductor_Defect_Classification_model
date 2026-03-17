"""Inference server: long-running Python process for ML model inference.

Reads JSON commands from stdin, writes JSON responses to stdout.
Image data is saved to temp files and paths returned in responses.
"""

import argparse
import json
import sys
import time
import os
import tempfile
from pathlib import Path

# Parse project root from command line before any project imports
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--support-dir", default=None,
                        help="Directory with bundled support images (Class1/, Class2/, ...)")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
else:
    # When imported as a module, use defaults (caller sets PROJECT_ROOT)
    args = argparse.Namespace(project_root=None, support_dir=None)

if args.project_root:
    PROJECT_ROOT = str(Path(args.project_root).resolve())
else:
    PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

from problem_a.src.backbone import get_backbone
from problem_a.src.protonet import PrototypicalNetwork, IncrementalPrototypeTracker
from problem_a.src.augmentations import get_eval_transform
import torchvision.transforms.functional as TF

TEMP_DIR = os.path.join(tempfile.gettempdir(), "semiai_images")
os.makedirs(TEMP_DIR, exist_ok=True)

# DAGM 2007 defect knowledge base - maps class numbers to defect information
DEFECT_KNOWLEDGE = {
    0: {
        "name": "No Defect (Good)",
        "description": "This image shows a normal, defect-free semiconductor surface. No anomalies or process deviations were detected. The surface texture and uniformity are within expected manufacturing tolerances.",
        "severity": "none",
        "causes": [],
        "prevention": [
            "Continue current process controls and monitoring",
            "Maintain regular equipment calibration schedules",
            "Document this baseline for future comparison",
        ],
    },
    1: {
        "name": "Surface Scratch",
        "description": "Linear scratch marks on a smooth metallic texture. These appear as thin, elongated anomalies that disrupt the uniform surface pattern, typically caused by physical contact with hard particles or tooling.",
        "severity": "medium",
        "causes": [
            "Mechanical contact with handling equipment or tooling edges",
            "Contaminated polishing pads introducing abrasive particles",
            "Improper wafer transport between processing stages",
        ],
        "prevention": [
            "Implement automated non-contact wafer handling systems",
            "Schedule regular polishing pad replacement and inspection",
            "Install particle counters at transfer stations to detect contamination early",
        ],
    },
    2: {
        "name": "Spot Contamination",
        "description": "Isolated dark spot defects on a woven-texture surface. These localized anomalies indicate particle deposition or chemical residue that was not removed during cleaning.",
        "severity": "low",
        "causes": [
            "Airborne particle contamination in the cleanroom environment",
            "Residual photoresist from incomplete strip or rinse cycles",
            "Chemical splash or micro-droplet deposition during wet processing",
        ],
        "prevention": [
            "Upgrade HEPA/ULPA filtration and monitor cleanroom particle counts",
            "Optimize rinse cycle duration and verify strip completeness",
            "Use enclosed chemical delivery systems with splash guards",
        ],
    },
    3: {
        "name": "Smudge Defect",
        "description": "Diffuse, irregular smudge-like blemishes on a cross-hatched texture. These spread-out anomalies suggest contact contamination or incomplete drying leaving residue films.",
        "severity": "medium",
        "causes": [
            "Finger or glove contact transferring oils to the surface",
            "Incomplete spin-dry leaving solvent residue films",
            "Cross-contamination from adjacent processing equipment",
        ],
        "prevention": [
            "Enforce strict gloving protocols and no-touch handling procedures",
            "Calibrate spin-dry RPM and duration for complete solvent removal",
            "Implement dedicated equipment cleaning between process lots",
        ],
    },
    4: {
        "name": "Edge Delamination",
        "description": "Film peeling or delamination defects near edges on a striped texture. These indicate adhesion failure between deposited layers, often triggered by thermal or mechanical stress.",
        "severity": "high",
        "causes": [
            "Poor adhesion due to surface contamination before deposition",
            "Thermal stress from rapid temperature cycling during annealing",
            "Film stress exceeding adhesion strength at wafer edges",
        ],
        "prevention": [
            "Ensure thorough pre-deposition surface cleaning and activation",
            "Optimize anneal ramp rates to minimize thermal shock",
            "Add adhesion promotion layers (e.g., Ti/TiN) at critical interfaces",
        ],
    },
    5: {
        "name": "Pit Cluster",
        "description": "Clusters of small pits or voids on a granular textured surface. These grouped micro-defects often result from localized etching anomalies or gas bubble entrapment during deposition.",
        "severity": "high",
        "causes": [
            "Non-uniform etch chemistry causing localized over-etching",
            "Gas bubble entrapment during chemical vapor deposition",
            "Micro-masking by particle contamination during etch",
        ],
        "prevention": [
            "Optimize etchant flow uniformity and temperature distribution",
            "Improve gas flow dynamics in CVD chamber to prevent bubble trapping",
            "Add pre-etch megasonic clean to remove micro-particles",
        ],
    },
    6: {
        "name": "Surface Stain",
        "description": "Broad discoloration stains on a fine-grain texture. These large-area defects indicate chemical residue, watermark formation, or non-uniform oxidation across the surface.",
        "severity": "medium",
        "causes": [
            "Watermark formation from incomplete drying after wet clean",
            "Chemical bath exhaustion causing non-uniform etch or clean",
            "Atmospheric oxidation during extended queue time between steps",
        ],
        "prevention": [
            "Implement IPA vapor dry or Marangoni drying to prevent watermarks",
            "Monitor chemical bath concentration and schedule timely replenishment",
            "Minimize queue time between wet clean and subsequent processing",
        ],
    },
    7: {
        "name": "Fiber Inclusion",
        "description": "Foreign fiber or thread-like inclusions embedded in a fabric-weave texture. These elongated contaminants were likely introduced during material handling or from degraded cleanroom garments.",
        "severity": "critical",
        "causes": [
            "Degraded cleanroom garment shedding synthetic fibers",
            "Contaminated packaging or carrier materials releasing fibers",
            "HVAC filter bypass allowing external fiber ingress",
        ],
        "prevention": [
            "Enforce cleanroom garment replacement schedule and quality checks",
            "Use certified low-particulate wafer carriers and packaging",
            "Perform regular HVAC filter integrity testing and replacement",
        ],
    },
    8: {
        "name": "Groove Anomaly",
        "description": "Irregular groove or channel defects on a directionally textured surface. These linear depressions deviate from the expected pattern and indicate process tool wear or misalignment.",
        "severity": "medium",
        "causes": [
            "Worn or damaged CMP (chemical mechanical polishing) pad grooves",
            "Misalignment of scribing or dicing equipment",
            "Non-uniform pressure distribution during planarization",
        ],
        "prevention": [
            "Monitor CMP pad condition with in-situ sensors and replace proactively",
            "Calibrate scribing and dicing tool alignment before each lot",
            "Use closed-loop pressure control for uniform planarization",
        ],
    },
    9: {
        "name": "Ring Mark",
        "description": "Circular or arc-shaped marks on a smooth gradient texture. These concentric defects typically originate from spin-coating non-uniformities or chuck contamination leaving ring patterns.",
        "severity": "high",
        "causes": [
            "Spin-coater chuck contamination leaving ring impressions",
            "Non-uniform resist dispensing causing thickness variation rings",
            "Edge bead removal process leaving residual ring marks",
        ],
        "prevention": [
            "Clean spin-coater chucks before each lot with automated scrub",
            "Optimize resist dispense volume, acceleration, and spread timing",
            "Tune edge bead removal solvent flow and nozzle positioning",
        ],
    },
    10: {
        "name": "Mottled Region",
        "description": "Patchy, mottled discoloration on a complex multi-scale texture. These irregular blotchy areas indicate non-uniform processing conditions affecting large surface regions.",
        "severity": "critical",
        "causes": [
            "Non-uniform temperature distribution in furnace or chamber",
            "Exhaust flow turbulence creating dead zones during deposition",
            "Chemical bath agitation failure causing concentration gradients",
        ],
        "prevention": [
            "Map and correct temperature uniformity in processing chambers",
            "Redesign exhaust baffles to eliminate turbulent dead zones",
            "Install real-time agitation monitoring with automatic fault detection",
        ],
    },
}


def analyze_irdrop_prediction(pred_np, input_np):
    """Analyze IR drop prediction and return severity, causes, and recommendations."""
    max_val = float(pred_np.max())
    mean_val = float(pred_np.mean())

    # Severity based on max IR drop
    if max_val > 0.15:
        severity = "critical"
    elif max_val > 0.10:
        severity = "high"
    elif max_val > 0.05:
        severity = "medium"
    else:
        severity = "low"

    # Hotspot analysis: pixels above 80th percentile
    threshold = float(np.percentile(pred_np, 80))
    hotspot_mask = pred_np > threshold
    hotspot_pct = float(hotspot_mask.sum()) / pred_np.size * 100

    # Channel statistics (from normalized input)
    channel_names = ["current_map", "eff_dist_map", "pdn_density"]
    channel_summary = {}
    dominant_channel = "current_map"
    max_channel_val = 0
    for i, name in enumerate(channel_names):
        ch = input_np[i]
        ch_mean = float(ch.mean())
        ch_max = float(ch.max())
        channel_summary[name] = {"mean": round(ch_mean, 4), "max": round(ch_max, 4)}
        if ch_max > max_channel_val:
            max_channel_val = ch_max
            dominant_channel = name

    # Build description
    severity_desc = {
        "critical": ("The predicted IR drop map reveals severe voltage violations across the design. "
                     "The maximum drop of {max:.4f} significantly exceeds safe operating margins, "
                     "with hotspots covering {hot:.1f}% of the chip area. Immediate power grid redesign is required "
                     "to prevent functional failures and reliability degradation."),
        "high": ("The prediction shows concerning IR drop levels that approach design limits. "
                 "With a peak drop of {max:.4f} and hotspots spanning {hot:.1f}% of the area, "
                 "targeted power delivery improvements are needed to ensure adequate voltage margins."),
        "medium": ("The IR drop prediction indicates moderate voltage variations within acceptable bounds. "
                   "The peak value of {max:.4f} leaves some margin, but {hot:.1f}% of the chip "
                   "shows elevated drop that should be monitored during design optimization."),
        "low": ("The predicted IR drop is well within safe operating limits. "
                "With a maximum of {max:.4f} and only {hot:.1f}% of the area showing "
                "elevated values, the power delivery network appears adequate for this design."),
    }
    description = severity_desc[severity].format(max=max_val, hot=hotspot_pct)

    # Dominant power source context
    dominant_labels = {
        "current_map": "current draw",
        "eff_dist_map": "effective distance to power pins",
        "pdn_density": "power delivery network density",
    }
    dominant_label = dominant_labels.get(dominant_channel, "current draw")

    # Causes based on severity and dominant channel
    causes_pool = {
        "critical": [
            "Severely insufficient power grid metal density for the power demand",
            f"Concentrated {dominant_label} creating extreme local current draw",
            "Power pin placement unable to supply adequate current to hotspot regions",
            "Missing or broken power straps creating resistive bottlenecks",
        ],
        "high": [
            "Power grid routing too narrow or sparse in high-drop regions",
            f"Elevated {dominant_label} density without proportional power delivery",
            "Long resistive paths between power pins and active circuitry",
            "Insufficient via connections between power grid layers",
        ],
        "medium": [
            "Moderate power grid density imbalance between chip regions",
            f"Localized {dominant_label} peaks creating minor voltage sag",
            "Suboptimal power pin distribution across the floorplan",
        ],
        "low": [
            "Minor power grid resistance contributing to small voltage variation",
            "Nominal current density within power delivery capacity",
        ],
    }

    # Recommendations based on severity
    recs_pool = {
        "critical": [
            "Increase power grid metal width and density in hotspot regions by 2-3x",
            "Add dedicated power straps connecting directly to high-drop areas",
            "Insert decoupling capacitors (decap cells) adjacent to hotspot locations",
            "Redistribute power pin placement to reduce maximum current path length",
            "Consider adding a dedicated power mesh layer for critical regions",
        ],
        "high": [
            "Widen power straps in high-drop areas to reduce resistive losses",
            "Add decoupling capacitors near regions with elevated IR drop",
            "Optimize power pin assignment to balance current distribution",
            "Increase via density between power grid layers in affected areas",
        ],
        "medium": [
            "Fine-tune power grid density to balance drop across the chip",
            "Add targeted decoupling capacitance in moderate-drop regions",
            "Review power pin placement for potential redistribution improvements",
        ],
        "low": [
            "Current power delivery network is adequate for this design",
            "Monitor IR drop during future design iterations as power increases",
            "Consider adding design margin with preventive decap cell placement",
        ],
    }

    return {
        "severity": severity,
        "description": description,
        "hotspot_percentage": round(hotspot_pct, 1),
        "causes": causes_pool[severity],
        "recommendations": recs_pool[severity],
        "channel_summary": channel_summary,
    }


def numpy_to_png(arr: np.ndarray, path: str, colormap: str = "hot"):
    """Save a 2D float array as a colormapped PNG."""
    arr = arr.astype(np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-8:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)
    cmap = matplotlib.colormaps.get_cmap(colormap)
    rgba = (cmap(arr) * 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)


def grayscale_to_png(arr: np.ndarray, path: str):
    """Save a 2D array as a grayscale PNG."""
    arr = arr.astype(np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-8:
        normalized = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(arr, dtype=np.uint8)
    Image.fromarray(normalized, "L").save(path)


def create_colorbar_png(vmin: float, vmax: float, path: str,
                        colormap: str = "hot", width: int = 30, height: int = 256):
    """Generate a vertical colorbar PNG."""
    gradient = np.linspace(1, 0, height).reshape(-1, 1).repeat(width, axis=1)
    numpy_to_png(gradient, path, colormap)


class InferenceServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.defect_model = None
        self.defect_tracker = None
        self.defect_transform = None
        self.irdrop_model = None
        self.defect_loaded = False
        self.irdrop_loaded = False
        self.defect_epoch = "?"
        self.defect_acc = "?"
        self.irdrop_epoch = "?"
        self.irdrop_f1 = "?"
        self.support_images = {}

    def respond(self, data: dict):
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

    def handle_init(self, cmd):
        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_mem = f"{mem:.1f} GB"

        self.respond({
            "status": "ok",
            "cmd": "init",
            "device": device_name,
            "cuda": torch.cuda.is_available(),
            "gpu_memory": gpu_mem,
            "torch_version": torch.__version__,
        })

    def handle_load_models(self, cmd):
        errors = []

        # Load defect model
        try:
            ckpt_path = Path(PROJECT_ROOT) / "problem_a" / "checkpoints" / "best_model.pt"
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
                # Read architecture from checkpoint config (matches training)
                mcfg = ckpt.get("config", {}).get("model", {})
                backbone_size = mcfg.get("backbone_size", "large")
                proj_hidden = mcfg.get("proj_hidden", 768)
                proj_dim = mcfg.get("proj_dim", 512)
                img_size = ckpt.get("config", {}).get("data", {}).get("img_size", 518)
            else:
                backbone_size, proj_hidden, proj_dim, img_size = "large", 768, 512, 518
                ckpt = None

            backbone = get_backbone("dinov2", size=backbone_size, freeze=True)
            self.defect_model = PrototypicalNetwork(backbone, proj_hidden=proj_hidden, proj_dim=proj_dim)
            if ckpt is not None:
                if "model_state_dict" in ckpt:
                    self.defect_model.load_state_dict(ckpt["model_state_dict"])
                    self.defect_epoch = str(ckpt.get("epoch", "?"))
                    acc = ckpt.get("best_val_acc", "?")
                    self.defect_acc = f"{acc:.1%}" if isinstance(acc, float) else str(acc)
                else:
                    self.defect_model.load_state_dict(ckpt)
            self.defect_model.to(self.device).eval()
            self.defect_tracker = IncrementalPrototypeTracker(self.defect_model, self.device)
            self.defect_transform = get_eval_transform(img_size)
            self.defect_loaded = True
        except Exception as e:
            errors.append(f"Defect model: {e}")

        # IR Drop model disabled (Problem A only)
        self.irdrop_loaded = False

        self.respond({
            "status": "ok" if not errors else "partial",
            "cmd": "load_models",
            "defect_loaded": self.defect_loaded,
            "irdrop_loaded": self.irdrop_loaded,
            "defect_epoch": self.defect_epoch,
            "defect_accuracy": self.defect_acc,
            "irdrop_epoch": self.irdrop_epoch,
            "irdrop_f1": self.irdrop_f1,
            "errors": errors,
        })

    def handle_add_support(self, cmd):
        path = cmd["path"]
        label = cmd["label"]

        img = Image.open(path).convert("L")
        img_arr = np.array(img)
        self.support_images.setdefault(label, []).append(img_arr)

        tensor = self.defect_transform(img_arr)
        self.defect_tracker.add_example(tensor, label)

        # Save thumbnail
        thumb = np.array(img.resize((64, 64)))
        thumb_path = os.path.join(TEMP_DIR, f"support_{label}_{len(self.support_images[label])-1}.png")
        grayscale_to_png(thumb, thumb_path)

        summary = {str(k): len(v) for k, v in self.support_images.items()}
        self.respond({
            "status": "ok",
            "cmd": "add_support",
            "label": label,
            "thumbnail": thumb_path,
            "summary": summary,
        })

    def handle_classify(self, cmd):
        path = cmd["path"]
        t0 = time.perf_counter()

        img = Image.open(path).convert("L")
        img_arr = np.array(img)

        tensor = self.defect_transform(img_arr)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.defect_model.classify(tensor, self.defect_tracker.prototypes)
            # Raw cosine similarities for good-detection threshold
            import torch.nn.functional as F
            emb = self.defect_model.embed(tensor)
            q_norm = F.normalize(emb, dim=1)
            p_norm = F.normalize(self.defect_tracker.prototypes, dim=1)
            raw_cosine_sims = torch.mm(q_norm, p_norm.t()).squeeze(0)

        elapsed = time.perf_counter() - t0
        probs = torch.exp(log_probs).squeeze(0).cpu().tolist()
        label_map = self.defect_tracker.label_map
        reverse_map = {v: k for k, v in label_map.items()}
        prob_dict = {str(reverse_map.get(i, i)): round(p, 4) for i, p in enumerate(probs)}

        predicted = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[predicted]

        # Good-detection threshold
        GOOD_GAP_THRESHOLD = 0.20
        good_class_idx = label_map.get(0)
        if good_class_idx is not None and int(predicted) != 0:
            top_sim = raw_cosine_sims.max().item()
            good_sim = raw_cosine_sims[good_class_idx].item()
            if (top_sim - good_sim) < GOOD_GAP_THRESHOLD:
                predicted = "0"
                confidence = prob_dict.get("0", good_sim)

        # Save query image
        query_path = os.path.join(TEMP_DIR, "query.png")
        grayscale_to_png(img_arr, query_path)

        self.respond({
            "status": "ok",
            "cmd": "classify",
            "predicted_class": predicted,
            "confidence": confidence,
            "probabilities": prob_dict,
            "query_image": query_path,
            "inference_time": elapsed,
        })

    def handle_predict_irdrop(self, cmd):
        file_path = cmd["path"]
        t0 = time.perf_counter()

        path = Path(file_path)

        # Support two formats: contest CSV and CircuitNet .npz
        if path.suffix == ".csv":
            # Contest CSV format: detect naming convention
            parent = path.parent
            name = path.stem

            # Try per-testcase directory format (real-circuit-data/testcaseN/)
            cur_file = parent / "current_map.csv"
            eff_file = parent / "eff_dist_map.csv"
            pdn_file = parent / "pdn_density.csv"

            if cur_file.exists() and eff_file.exists() and pdn_file.exists():
                current = np.loadtxt(str(cur_file), delimiter=",", dtype=np.float64).astype(np.float32)
                eff_dist = np.loadtxt(str(eff_file), delimiter=",", dtype=np.float64).astype(np.float32)
                pdn_density = np.loadtxt(str(pdn_file), delimiter=",", dtype=np.float64).astype(np.float32)
            else:
                # Flat naming: current_mapNN_current.csv, etc.
                prefix = name.replace("_current", "").replace("_eff_dist", "").replace("_pdn_density", "").replace("_ir_drop", "")
                current = np.loadtxt(str(parent / f"{prefix}_current.csv"), delimiter=",", dtype=np.float64).astype(np.float32)
                eff_dist = np.loadtxt(str(parent / f"{prefix}_eff_dist.csv"), delimiter=",", dtype=np.float64).astype(np.float32)
                pdn_density = np.loadtxt(str(parent / f"{prefix}_pdn_density.csv"), delimiter=",", dtype=np.float64).astype(np.float32)

            feat_arrays = [current, eff_dist, pdn_density]
        else:
            # CircuitNet .npz fallback (3-channel proxy)
            sample_name = path.stem
            design_dir = path.parent.parent
            FEATURE_DIRS = ["power_all", "power_i", "power_s"]

            feat_arrays = []
            for feat_name in FEATURE_DIRS:
                feat_path = design_dir / feat_name / f"{sample_name}.npz"
                if feat_path.exists():
                    data = np.load(str(feat_path))
                    feat_arrays.append(data[list(data.keys())[0]].astype(np.float32))
                else:
                    feat_arrays.append(np.zeros((256, 256), dtype=np.float32))

        features = np.stack(feat_arrays, axis=0)  # [3, H, W]
        features_t = torch.from_numpy(features).unsqueeze(0)

        # Normalize per-channel to [0, 1]
        for c in range(3):
            fmin, fmax = features_t[0, c].min(), features_t[0, c].max()
            if fmax - fmin > 1e-8:
                features_t[0, c] = (features_t[0, c] - fmin) / (fmax - fmin)

        features_t = features_t.to(self.device)
        with torch.no_grad():
            pred = self.irdrop_model(features_t)

        elapsed = time.perf_counter() - t0
        input_np = features_t.squeeze(0).cpu().numpy()
        pred_np = pred.squeeze().cpu().numpy()

        # Save input channel images
        channel_names = ["current_map", "eff_dist_map", "pdn_density"]
        input_paths = []
        for i in range(3):
            p = os.path.join(TEMP_DIR, f"irdrop_input_{i}.png")
            numpy_to_png(input_np[i], p, "viridis")
            input_paths.append(p)

        # Save prediction
        pred_path = os.path.join(TEMP_DIR, "irdrop_prediction.png")
        numpy_to_png(pred_np, pred_path, "hot")

        # Save colorbar
        colorbar_path = os.path.join(TEMP_DIR, "irdrop_colorbar.png")
        create_colorbar_png(float(pred_np.min()), float(pred_np.max()), colorbar_path)

        # Analyze prediction for severity, causes, recommendations
        analysis = analyze_irdrop_prediction(pred_np, input_np)

        self.respond({
            "status": "ok",
            "cmd": "predict_irdrop",
            "input_images": input_paths,
            "prediction_image": pred_path,
            "colorbar_image": colorbar_path,
            "min_val": float(pred_np.min()),
            "max_val": float(pred_np.max()),
            "mean_val": float(pred_np.mean()),
            "inference_time": elapsed,
            "severity": analysis["severity"],
            "description": analysis["description"],
            "hotspot_percentage": analysis["hotspot_percentage"],
            "causes": analysis["causes"],
            "recommendations": analysis["recommendations"],
            "channel_summary": analysis["channel_summary"],
        })

    def handle_auto_load_support(self, cmd):
        """Auto-load defective images from DAGM training data as support set.

        Supports two modes:
        1. Bundled support dir (--support-dir flag): loads from support_dir/Class{N}/*.PNG
        2. DAGM dataset: scans Train/Label/ for defective images
        """
        k_shot = cmd.get("k_shot", 5)

        # Reset existing support
        if self.defect_tracker and self.defect_model:
            self.defect_tracker = IncrementalPrototypeTracker(self.defect_model, self.device)
        self.support_images.clear()

        class_counts = {}
        total = 0

        # Check for bundled support directory first
        support_dir = args.support_dir
        if support_dir and Path(support_dir).is_dir():
            # Bundled mode: load from support_dir/Class{N}/ or defect{N}/ or good/
            import re
            support_path = Path(support_dir)
            class_dirs = sorted([
                d for d in support_path.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ], key=lambda d: d.name)

            for class_dir in class_dirs:
                # Extract class index: "Class1"->1, "defect3"->3, "good"->0
                nums = re.findall(r'\d+', class_dir.name)
                if nums:
                    class_idx = int(nums[0])
                elif class_dir.name.lower() == 'good':
                    class_idx = 0
                else:
                    continue

                count = 0
                img_files = sorted(class_dir.glob("*.PNG")) + sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
                for img_path in img_files[:k_shot]:
                    img = Image.open(str(img_path)).convert("L")
                    img_arr = np.array(img)
                    self.support_images.setdefault(class_idx, []).append(img_arr)
                    tensor = self.defect_transform(img_arr)
                    self.defect_tracker.add_example(tensor, class_idx)
                    count += 1
                    total += 1

                if count > 0:
                    class_counts[str(class_idx)] = count
        else:
            # Dataset mode: scan for class directories with Train/Label structure (DAGM)
            # or flat class-folder structure (Intel contest)
            # Try Intel contest data first, fall back to DAGM
            default_root = str(Path(PROJECT_ROOT) / "challenge" / "dataset" / "Dataset" / "Data")
            if not Path(default_root).is_dir():
                default_root = str(Path(PROJECT_ROOT) / "problem_a" / "data" / "DAGM_KaggleUpload")
            dagm_root = cmd.get("data_root", default_root)
            root_path = Path(dagm_root)

            # Discover class directories dynamically (Class*, defect*, good)
            import re as re2
            class_dirs = sorted([
                d for d in root_path.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ], key=lambda d: d.name) if root_path.is_dir() else []

            for class_dir in class_dirs:
                nums = re2.findall(r'\d+', class_dir.name)
                if nums:
                    class_idx = int(nums[0])
                elif class_dir.name.lower() == 'good':
                    class_idx = 0
                else:
                    continue

                train_dir = class_dir / "Train"
                label_dir = train_dir / "Label"

                if label_dir.exists():
                    # DAGM format: find defective images via label masks
                    defective_stems = sorted({
                        lf.stem.replace("_label", "")
                        for lf in label_dir.glob("*_label.*")
                    })

                    count = 0
                    for stem in defective_stems:
                        if count >= k_shot:
                            break
                        img_path = train_dir / f"{stem}.PNG"
                        if not img_path.exists():
                            continue
                        img = Image.open(str(img_path)).convert("L")
                        img_arr = np.array(img)
                        self.support_images.setdefault(class_idx, []).append(img_arr)
                        tensor = self.defect_transform(img_arr)
                        self.defect_tracker.add_example(tensor, class_idx)
                        count += 1
                        total += 1

                    if count > 0:
                        class_counts[str(class_idx)] = count
                else:
                    # Intel/flat format: all images in the class folder are defective
                    count = 0
                    img_files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.PNG")) + sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.bmp"))
                    for img_path in img_files[:k_shot]:
                        img = Image.open(str(img_path)).convert("L")
                        img_arr = np.array(img)
                        self.support_images.setdefault(class_idx, []).append(img_arr)
                        tensor = self.defect_transform(img_arr)
                        self.defect_tracker.add_example(tensor, class_idx)
                        count += 1
                        total += 1

                    if count > 0:
                        class_counts[str(class_idx)] = count

        self.respond({
            "status": "ok",
            "cmd": "auto_load_support",
            "classes_loaded": len(class_counts),
            "total_images": total,
            "class_counts": class_counts,
        })

    def handle_analyze(self, cmd):
        """Classify image with attention map extraction and knowledge base lookup."""
        path = cmd["path"]
        t0 = time.perf_counter()

        img = Image.open(path).convert("L")
        img_arr = np.array(img)

        tensor = self.defect_transform(img_arr)
        tensor = tensor.unsqueeze(0).to(self.device)

        # Hook into last attention block's QKV to extract attention weights
        attn_module = self.defect_model.backbone.backbone.blocks[-1].attn
        qkv_output = []

        def qkv_hook(module, input, output):
            qkv_output.append(output.detach())

        hook_handle = attn_module.qkv.register_forward_hook(qkv_hook)

        with torch.no_grad():
            log_probs = self.defect_model.classify(tensor, self.defect_tracker.prototypes)
            # Also compute raw cosine similarities for good-detection threshold
            emb = self.defect_model.embed(tensor)
            import torch.nn.functional as F
            q_norm = F.normalize(emb, dim=1)
            p_norm = F.normalize(self.defect_tracker.prototypes, dim=1)
            raw_cosine_sims = torch.mm(q_norm, p_norm.t()).squeeze(0)  # [N]

        hook_handle.remove()

        # Extract attention map from QKV
        if qkv_output:
            qkv = qkv_output[0]  # [B, N, 3*embed_dim]
            B, N, _ = qkv.shape
            num_heads = attn_module.num_heads
            head_dim = attn_module.head_dim

            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)  # each [B, num_heads, N, head_dim]

            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]

            # CLS token attention to all patches (average across heads)
            cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # [num_patches]
            num_patches_side = int(cls_attn.shape[0] ** 0.5)
            attn_map = cls_attn.reshape(num_patches_side, num_patches_side)

            # Upsample to original image size
            attn_map = torch.nn.functional.interpolate(
                attn_map.unsqueeze(0).unsqueeze(0),
                size=(img_arr.shape[0], img_arr.shape[1]),
                mode="bilinear", align_corners=False,
            )
            attn_np = attn_map.squeeze().cpu().numpy()
        else:
            attn_np = np.zeros_like(img_arr, dtype=np.float32)

        # Save attention heatmap
        attn_path = os.path.join(TEMP_DIR, "defect_attention.png")
        numpy_to_png(attn_np, attn_path, "jet")

        # Save blended overlay (attention over original image)
        overlay_path = os.path.join(TEMP_DIR, "defect_overlay.png")
        img_norm = img_arr.astype(np.float64)
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
        attn_norm = attn_np.astype(np.float64)
        attn_norm = (attn_norm - attn_norm.min()) / (attn_norm.max() - attn_norm.min() + 1e-8)
        cmap = matplotlib.colormaps.get_cmap("jet")
        attn_rgba = cmap(attn_norm)  # [H, W, 4]
        base_rgb = np.stack([img_norm] * 3, axis=-1)
        alpha = 0.5 * attn_norm[..., np.newaxis]
        blended = base_rgb * (1 - alpha) + attn_rgba[:, :, :3] * alpha
        blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(blended, "RGB").save(overlay_path)

        # Classification results
        elapsed = time.perf_counter() - t0
        probs = torch.exp(log_probs).squeeze(0).cpu().tolist()
        label_map = self.defect_tracker.label_map
        reverse_map = {v: k for k, v in label_map.items()}
        prob_dict = {str(reverse_map.get(i, i)): round(p, 4) for i, p in enumerate(probs)}

        predicted = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[predicted]

        # Good-detection threshold: if predicted is a defect class but the gap
        # between top cosine similarity and good-class similarity is small,
        # reclassify as "good" (class 0). This compensates for the high visual
        # diversity of the "good" class.
        GOOD_GAP_THRESHOLD = 0.20
        good_class_idx = label_map.get(0)  # prototype index for class 0 (good)
        if good_class_idx is not None and int(predicted) != 0:
            top_sim = raw_cosine_sims.max().item()
            good_sim = raw_cosine_sims[good_class_idx].item()
            gap = top_sim - good_sim
            if gap < GOOD_GAP_THRESHOLD:
                predicted = "0"
                confidence = prob_dict.get("0", good_sim)

        # Save query image
        query_path = os.path.join(TEMP_DIR, "query.png")
        grayscale_to_png(img_arr, query_path)

        # Knowledge base lookup
        pred_int = int(predicted)
        knowledge = DEFECT_KNOWLEDGE.get(pred_int, {
            "name": f"Unknown Defect (Class {predicted})",
            "description": "No detailed information available for this defect class.",
            "severity": "unknown",
            "causes": [],
            "prevention": [],
        })

        self.respond({
            "status": "ok",
            "cmd": "analyze",
            "predicted_class": predicted,
            "confidence": confidence,
            "probabilities": prob_dict,
            "query_image": query_path,
            "attention_image": attn_path,
            "overlay_image": overlay_path,
            "inference_time": elapsed,
            "defect_name": knowledge["name"],
            "defect_description": knowledge["description"],
            "severity": knowledge["severity"],
            "causes": knowledge["causes"],
            "prevention": knowledge["prevention"],
        })

    def handle_reset_support(self, cmd):
        if self.defect_tracker and self.defect_model:
            self.defect_tracker = IncrementalPrototypeTracker(self.defect_model, self.device)
        self.support_images.clear()
        self.respond({"status": "ok", "cmd": "reset_support"})

    def handle_status(self, cmd):
        has_prototypes = (self.defect_tracker is not None and
                          self.defect_tracker.prototypes is not None)
        summary = {str(k): len(v) for k, v in self.support_images.items()}
        self.respond({
            "status": "ok",
            "cmd": "status",
            "defect_loaded": self.defect_loaded,
            "irdrop_loaded": self.irdrop_loaded,
            "has_prototypes": has_prototypes,
            "support_summary": summary,
        })

    def run(self):
        # Signal ready
        self.respond({"status": "ready"})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                self.respond({"status": "error", "message": f"Invalid JSON: {e}"})
                continue

            try:
                handler = {
                    "init": self.handle_init,
                    "load_models": self.handle_load_models,
                    "add_support": self.handle_add_support,
                    "classify": self.handle_classify,
                    "auto_load_support": self.handle_auto_load_support,
                    "analyze": self.handle_analyze,
                    "predict_irdrop": self.handle_predict_irdrop,
                    "reset_support": self.handle_reset_support,
                    "status": self.handle_status,
                }.get(cmd.get("cmd"))

                if handler:
                    handler(cmd)
                else:
                    self.respond({"status": "error", "message": f"Unknown command: {cmd.get('cmd')}"})
            except Exception as e:
                self.respond({"status": "error", "cmd": cmd.get("cmd", "?"), "message": str(e)})


if __name__ == "__main__":
    server = InferenceServer()
    server.run()
