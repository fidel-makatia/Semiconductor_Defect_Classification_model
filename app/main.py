"""SemiAI - PySide6 entry point.

Cross-platform alternative to the C++ Qt6 frontend (app/src/main.cpp).
Loads the same QML files using PySide6 instead of compiled C++.

Usage:
    python app/main.py
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from imageprovider import HeatmapProvider
from processmanager import ProcessManager
from bridge import AppBridge


def find_python() -> str:
    """Find a working Python executable."""
    candidates = ["python3", "python"]

    if platform.system() == "Windows":
        candidates = ["python", "python3"]
        home = Path.home()
        for ver in ["314", "313", "312", "311"]:
            candidates.append(
                str(home / "AppData" / "Local" / "Programs" / "Python"
                    / f"Python{ver}" / "python.exe")
            )

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, timeout=3,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return sys.executable


def find_paths(app_dir: Path):
    """Find QML directory and inference script, searching multiple locations."""
    qml_dir = None
    python_script = None

    search_paths = [
        app_dir,                    # Same directory as main.py
        app_dir.parent,             # Project root
        Path.cwd() / "app",        # CWD/app
        Path.cwd(),                 # CWD
    ]

    for base in search_paths:
        if qml_dir is None and (base / "qml").is_dir():
            qml_dir = str((base / "qml").resolve())
        if python_script is None and (base / "python" / "inference_server.py").is_file():
            python_script = str((base / "python" / "inference_server.py").resolve())
        if qml_dir and python_script:
            break

    return qml_dir, python_script


def find_project_root(app_dir: Path) -> str:
    """Find the project root (directory containing problem_a/)."""
    candidates = [
        Path.cwd(),
        app_dir.parent,
        app_dir.parent.parent,
    ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "problem_a").is_dir():
            return str(resolved)

    return str(Path.cwd())


def main():
    app = QGuiApplication(sys.argv)
    app.setApplicationName("SemiAI")
    app.setOrganizationName("SemiAI")

    QQuickStyle.setStyle("Basic")

    # Resolve paths
    app_dir = Path(__file__).resolve().parent

    qml_dir, python_script = find_paths(app_dir)

    if not qml_dir:
        print(f"ERROR: Could not find QML directory", file=sys.stderr)
        return 1
    if not python_script:
        print(f"ERROR: Could not find inference_server.py", file=sys.stderr)
        return 1

    project_root = find_project_root(app_dir)

    print(f"QML dir: {qml_dir}")
    print(f"Python script: {python_script}")
    print(f"Project root: {project_root}")

    # Create engine
    engine = QQmlApplicationEngine()

    # Image provider (engine takes ownership)
    image_provider = HeatmapProvider()
    engine.addImageProvider("heatmap", image_provider)

    # Inference manager (subprocess)
    python_path = find_python()
    print(f"Using Python: {python_path}")

    # Check for bundled support images
    support_dir = ""
    for candidate in [app_dir / "data" / "support", Path(project_root) / "app" / "data" / "support"]:
        if candidate.is_dir():
            support_dir = str(candidate.resolve())
            break
    if support_dir:
        print(f"Support dir: {support_dir}")

    inference = ProcessManager(python_path, python_script, project_root, support_dir, app)

    # Bridge
    bridge = AppBridge(app)
    bridge.setImageProvider(image_provider)
    bridge.setInferenceManager(inference)

    engine.rootContext().setContextProperty("bridge", bridge)
    engine.addImportPath(qml_dir)

    # Load QML
    main_qml = os.path.join(qml_dir, "main.qml")
    engine.load(QUrl.fromLocalFile(main_qml))

    if not engine.rootObjects():
        print(f"ERROR: Failed to load QML from: {main_qml}", file=sys.stderr)
        return 1

    # Start inference server
    inference.start()

    ret = app.exec()

    # Clean shutdown
    inference.shutdown()

    return ret


if __name__ == "__main__":
    sys.exit(main())
