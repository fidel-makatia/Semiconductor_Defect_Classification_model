"""Package SemiAI for distribution.

Creates a self-contained folder that can be zipped and run on any
Windows or Linux machine with Python 3.11+ installed.

Usage:
    python package.py
"""

import shutil
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "SemiAI"


def main():
    # Clean
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    # Files to copy
    items = [
        # App core
        "app/main.py",
        "app/bridge.py",
        "app/imageprovider.py",
        "app/processmanager.py",
        "app/python/inference_server.py",

        # Problem A source (needed by inference server)
        "problem_a/__init__.py",
        "problem_a/src/__init__.py",
        "problem_a/src/backbone.py",
        "problem_a/src/protonet.py",
        "problem_a/src/augmentations.py",
        "problem_a/src/dataset.py",
        "problem_a/configs/default.yaml",

        # Checkpoint
        "problem_a/checkpoints/best_model.pt",
    ]

    for item in items:
        src = ROOT / item
        dst = DIST / item
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Copied {item}")
        else:
            print(f"  MISSING: {item}")

    # Copy QML directory (exclude IR Drop page)
    qml_src = ROOT / "app" / "qml"
    if qml_src.exists():
        shutil.copytree(qml_src, DIST / "app" / "qml",
                        ignore=shutil.ignore_patterns("IRDropPage.qml", "__pycache__"))
        print("  Copied app/qml/")

    # Copy support images (Intel defect data)
    support_src = ROOT / "app" / "data" / "support"
    if support_src.exists():
        shutil.copytree(support_src, DIST / "app" / "data" / "support",
                        ignore=shutil.ignore_patterns("__pycache__"))
        print("  Copied app/data/support/")

    # ── requirements.txt ─────────────────────────────────────────────
    (DIST / "requirements.txt").write_text(
        "torch>=2.0\n"
        "torchvision>=0.15\n"
        "timm>=1.0\n"
        "albumentations>=1.3\n"
        "opencv-python-headless>=4.8\n"
        "Pillow>=10.0\n"
        "numpy>=1.24\n"
        "matplotlib>=3.7\n"
        "PyYAML>=6.0\n"
        "PySide6>=6.5\n"
        "scikit-learn>=1.3\n"
    )
    print("  Created requirements.txt")

    # ── run.bat (Windows) ────────────────────────────────────────────
    (DIST / "run.bat").write_text(
        '@echo off\r\n'
        'setlocal\r\n'
        'title SemiAI - Semiconductor Defect Classification\r\n'
        'echo.\r\n'
        'echo  ============================================\r\n'
        'echo   SemiAI - Few-Shot Defect Classification\r\n'
        'echo   Semiconductor Solutions Challenge 2026\r\n'
        'echo  ============================================\r\n'
        'echo.\r\n'
        '\r\n'
        'cd /d "%~dp0"\r\n'
        '\r\n'
        'where python >nul 2>&1\r\n'
        'if %errorlevel% neq 0 (\r\n'
        '    echo ERROR: Python not found. Install Python 3.11+ from python.org\r\n'
        '    pause & exit /b 1\r\n'
        ')\r\n'
        '\r\n'
        'python -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>nul\r\n'
        'if %errorlevel% neq 0 (\r\n'
        '    echo ERROR: Python 3.11+ required. & python --version & pause & exit /b 1\r\n'
        ')\r\n'
        '\r\n'
        'if not exist "venv\\Scripts\\python.exe" (\r\n'
        '    echo Creating virtual environment...\r\n'
        '    python -m venv venv\r\n'
        '    if %errorlevel% neq 0 ( echo Failed to create venv & pause & exit /b 1 )\r\n'
        ')\r\n'
        '\r\n'
        'call venv\\Scripts\\activate.bat\r\n'
        '\r\n'
        'if not exist "venv\\.installed" (\r\n'
        '    echo Installing dependencies (this may take a few minutes)...\r\n'
        '    pip install --upgrade pip >nul 2>&1\r\n'
        '    pip install -r requirements.txt\r\n'
        '    if %errorlevel% neq 0 ( echo Failed to install deps & pause & exit /b 1 )\r\n'
        '    echo. > venv\\.installed\r\n'
        ')\r\n'
        '\r\n'
        'echo Starting SemiAI...\r\n'
        'python app\\main.py\r\n'
        'if %errorlevel% neq 0 ( echo. & echo App exited with error. & pause )\r\n'
    )
    print("  Created run.bat")

    # ── run.sh (Linux / macOS) ───────────────────────────────────────
    run_sh = DIST / "run.sh"
    run_sh.write_text(
        '#!/usr/bin/env bash\n'
        'set -e\n'
        'echo ""\n'
        'echo "  ============================================"\n'
        'echo "   SemiAI - Few-Shot Defect Classification"\n'
        'echo "   Semiconductor Solutions Challenge 2026"\n'
        'echo "  ============================================"\n'
        'echo ""\n'
        '\n'
        'cd "$(dirname "$0")"\n'
        '\n'
        '# Find python3\n'
        'PYTHON=""\n'
        'for cmd in python3 python; do\n'
        '    if command -v "$cmd" &>/dev/null; then\n'
        '        ver=$("$cmd" -c "import sys; print(sys.version_info >= (3,11))" 2>/dev/null)\n'
        '        if [ "$ver" = "True" ]; then\n'
        '            PYTHON="$cmd"\n'
        '            break\n'
        '        fi\n'
        '    fi\n'
        'done\n'
        '\n'
        'if [ -z "$PYTHON" ]; then\n'
        '    echo "ERROR: Python 3.11+ not found. Install from python.org or your package manager."\n'
        '    exit 1\n'
        'fi\n'
        'echo "Using: $PYTHON ($($PYTHON --version))"\n'
        '\n'
        '# Create venv if needed\n'
        'if [ ! -f "venv/bin/python" ]; then\n'
        '    echo "Creating virtual environment..."\n'
        '    $PYTHON -m venv venv\n'
        'fi\n'
        '\n'
        'source venv/bin/activate\n'
        '\n'
        '# Install deps if needed\n'
        'if [ ! -f "venv/.installed" ]; then\n'
        '    echo "Installing dependencies (this may take a few minutes)..."\n'
        '    pip install --upgrade pip > /dev/null 2>&1\n'
        '    pip install -r requirements.txt\n'
        '    touch venv/.installed\n'
        'fi\n'
        '\n'
        'echo "Starting SemiAI..."\n'
        'python app/main.py\n'
    )
    # Make run.sh executable
    run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("  Created run.sh")

    # ── README.txt ───────────────────────────────────────────────────
    (DIST / "README.txt").write_text(
        "SemiAI - Few-Shot Semiconductor Defect Classification\n"
        "=====================================================\n"
        "Semiconductor Solutions Challenge 2026 (Intel)\n"
        "\n"
        "QUICK START\n"
        "-----------\n"
        "  Windows:  Double-click run.bat\n"
        "  Linux:    chmod +x run.sh && ./run.sh\n"
        "\n"
        "First run creates a virtual environment and installs dependencies\n"
        "automatically (~4 GB for PyTorch + PySide6). Subsequent runs start\n"
        "instantly.\n"
        "\n"
        "REQUIREMENTS\n"
        "------------\n"
        "- Python 3.11 or higher (python.org)\n"
        "- NVIDIA GPU with CUDA (recommended, falls back to CPU)\n"
        "- ~4 GB disk space for Python dependencies\n"
        "\n"
        "WHAT'S INCLUDED\n"
        "---------------\n"
        "- PySide6/QML desktop application\n"
        "- DINOv2 ViT-L/14 backbone + Prototypical Network (pre-trained)\n"
        "- 8-class defect classification with attention visualization\n"
        "- Bundled support images (5 per class) for instant inference\n"
        "- Defect knowledge base with root cause analysis\n"
        "\n"
        "USING THE APP\n"
        "-------------\n"
        "1. Dashboard  - System info, GPU status, model status\n"
        "2. Defect Detection - Drop/browse an image to classify\n"
        "   - Shows: predicted class, confidence, attention map, overlay\n"
        "   - Shows: defect description, root causes, prevention steps\n"
        "3. Settings   - Configure paths and parameters\n"
        "\n"
        "ARCHITECTURE\n"
        "------------\n"
        "- Backbone: DINOv2 ViT-L/14 (304M params)\n"
        "- Projection: 1024 -> 768 -> 512 (L2-normalized)\n"
        "- Classifier: Cosine similarity + learned temperature\n"
        "- Training: Episodic meta-learning (8-way 5-shot)\n"
        "- Accuracy: 99.8% with 5 examples per class\n"
        "- Inference: ~700ms per image (GPU), ~3s (CPU)\n"
    )
    print("  Created README.txt")

    # ── Summary ──────────────────────────────────────────────────────
    total_size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"\nDone! Distribution: {DIST}")
    print(f"Size: {total_size / 1024 / 1024:.1f} MB")
    print(f"\nTo distribute: zip dist/SemiAI/ and share.")
    print(f"  Windows: double-click run.bat")
    print(f"  Linux:   ./run.sh")


if __name__ == "__main__":
    main()
