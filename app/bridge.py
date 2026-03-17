"""AppBridge: QObject bridge between QML UI and inference subprocess.

Mirrors app/src/appbridge.h/cpp.
Exposes properties, signals, and slots that QML binds to.
"""

import json

from PySide6.QtCore import QObject, Property, Signal, Slot, QUrl
from PySide6.QtGui import QImage


class AppBridge(QObject):
    # --- Signals ---
    deviceInfoChanged = Signal()
    modelStatusChanged = Signal()
    supportStatusChanged = Signal()
    defectResultReady = Signal(str, float, str)       # cls, confidence, probsJson
    defectAnalysisReady = Signal()
    irdropResultReady = Signal(float, float, float, float)  # min, max, mean, inferenceTime
    irdropAnalysisReady = Signal()
    supportSetChanged = Signal(str)                    # summaryJson
    errorOccurred = Signal(str, str)                   # title, message
    inferenceStarted = Signal(str)                     # model
    inferenceFinished = Signal(str)                    # model
    imageUpdated = Signal(str, int)                    # imageId, version

    def __init__(self, parent=None):
        super().__init__(parent)
        self._provider = None
        self._inference = None

        # Device info
        self._deviceName = "Initializing..."
        self._cudaAvailable = False
        self._gpuMemory = "N/A"
        self._torchVersion = "N/A"

        # Model status
        self._defectLoaded = False
        self._irdropLoaded = False
        self._modelsLoading = False
        self._defectEpoch = "?"
        self._defectAccuracy = "?"
        self._irdropEpoch = "?"
        self._irdropF1 = "?"

        # Support status
        self._supportLoaded = False
        self._supportLoading = False
        self._supportClassCount = 0
        self._supportImageCount = 0

        # IR drop analysis results
        self._irdropSeverity = ""
        self._irdropDescription = ""
        self._irdropCauses = []
        self._irdropRecommendations = []
        self._irdropHotspotPct = 0.0

        # Defect analysis results
        self._defectName = ""
        self._defectDescription = ""
        self._defectSeverity = ""
        self._defectCauses = []
        self._defectPrevention = []
        self._predictedDefectClass = ""
        self._defectConfidence = 0.0
        self._defectProbsJson = ""
        self._defectInferenceTime = 0.0

    def setImageProvider(self, provider):
        self._provider = provider

    def setInferenceManager(self, manager):
        self._inference = manager
        self._inference.ready.connect(self._on_server_ready)
        self._inference.responseReceived.connect(self._on_response)
        self._inference.errorOccurred.connect(self._on_server_error)

    # --- Device Properties ---

    def _get_deviceName(self):
        return self._deviceName

    def _get_cudaAvailable(self):
        return self._cudaAvailable

    def _get_gpuMemory(self):
        return self._gpuMemory

    def _get_torchVersion(self):
        return self._torchVersion

    deviceName = Property(str, _get_deviceName, notify=deviceInfoChanged)
    cudaAvailable = Property(bool, _get_cudaAvailable, notify=deviceInfoChanged)
    gpuMemory = Property(str, _get_gpuMemory, notify=deviceInfoChanged)
    torchVersion = Property(str, _get_torchVersion, notify=deviceInfoChanged)

    # --- Model Status Properties ---

    def _get_defectLoaded(self):
        return self._defectLoaded

    def _get_irdropLoaded(self):
        return self._irdropLoaded

    def _get_modelsLoading(self):
        return self._modelsLoading

    def _get_defectEpoch(self):
        return self._defectEpoch

    def _get_defectAccuracy(self):
        return self._defectAccuracy

    def _get_irdropEpoch(self):
        return self._irdropEpoch

    def _get_irdropF1(self):
        return self._irdropF1

    defectLoaded = Property(bool, _get_defectLoaded, notify=modelStatusChanged)
    irdropLoaded = Property(bool, _get_irdropLoaded, notify=modelStatusChanged)
    modelsLoading = Property(bool, _get_modelsLoading, notify=modelStatusChanged)
    defectEpoch = Property(str, _get_defectEpoch, notify=modelStatusChanged)
    defectAccuracy = Property(str, _get_defectAccuracy, notify=modelStatusChanged)
    irdropEpoch = Property(str, _get_irdropEpoch, notify=modelStatusChanged)
    irdropF1 = Property(str, _get_irdropF1, notify=modelStatusChanged)

    # --- Support Status Properties ---

    def _get_supportLoaded(self):
        return self._supportLoaded

    def _get_supportLoading(self):
        return self._supportLoading

    def _get_supportClassCount(self):
        return self._supportClassCount

    def _get_supportImageCount(self):
        return self._supportImageCount

    supportLoaded = Property(bool, _get_supportLoaded, notify=supportStatusChanged)
    supportLoading = Property(bool, _get_supportLoading, notify=supportStatusChanged)
    supportClassCount = Property(int, _get_supportClassCount, notify=supportStatusChanged)
    supportImageCount = Property(int, _get_supportImageCount, notify=supportStatusChanged)

    # --- Defect Analysis Properties ---

    def _get_defectName(self):
        return self._defectName

    def _get_defectDescription(self):
        return self._defectDescription

    def _get_defectSeverity(self):
        return self._defectSeverity

    def _get_defectCauses(self):
        return self._defectCauses

    def _get_defectPrevention(self):
        return self._defectPrevention

    def _get_predictedDefectClass(self):
        return self._predictedDefectClass

    def _get_defectConfidence(self):
        return self._defectConfidence

    def _get_defectProbsJson(self):
        return self._defectProbsJson

    def _get_defectInferenceTime(self):
        return self._defectInferenceTime

    defectName = Property(str, _get_defectName, notify=defectAnalysisReady)
    defectDescription = Property(str, _get_defectDescription, notify=defectAnalysisReady)
    defectSeverity = Property(str, _get_defectSeverity, notify=defectAnalysisReady)
    defectCauses = Property(list, _get_defectCauses, notify=defectAnalysisReady)
    defectPrevention = Property(list, _get_defectPrevention, notify=defectAnalysisReady)
    predictedDefectClass = Property(str, _get_predictedDefectClass, notify=defectAnalysisReady)
    defectConfidence = Property(float, _get_defectConfidence, notify=defectAnalysisReady)
    defectProbsJson = Property(str, _get_defectProbsJson, notify=defectAnalysisReady)
    defectInferenceTime = Property(float, _get_defectInferenceTime, notify=defectAnalysisReady)

    # --- IR Drop Analysis Properties ---

    def _get_irdropSeverity(self):
        return self._irdropSeverity

    def _get_irdropDescription(self):
        return self._irdropDescription

    def _get_irdropCauses(self):
        return self._irdropCauses

    def _get_irdropRecommendations(self):
        return self._irdropRecommendations

    def _get_irdropHotspotPct(self):
        return self._irdropHotspotPct

    irdropSeverity = Property(str, _get_irdropSeverity, notify=irdropAnalysisReady)
    irdropDescription = Property(str, _get_irdropDescription, notify=irdropAnalysisReady)
    irdropCauses = Property(list, _get_irdropCauses, notify=irdropAnalysisReady)
    irdropRecommendations = Property(list, _get_irdropRecommendations, notify=irdropAnalysisReady)
    irdropHotspotPct = Property(float, _get_irdropHotspotPct, notify=irdropAnalysisReady)

    # --- QML Invokable Slots ---

    @Slot()
    def loadModels(self):
        self._modelsLoading = True
        self.modelStatusChanged.emit()
        self._inference.sendCommand({"cmd": "load_models"})

    @Slot(str, int)
    def addSupportImage(self, path, label):
        if not self._defectLoaded:
            self.errorOccurred.emit("Not Ready", "Defect model not loaded yet")
            return
        # Convert QUrl to local path if needed
        path = self._to_local_path(path)
        self._inference.sendCommand({"cmd": "add_support", "path": path, "label": label})

    @Slot(str)
    def classifyImage(self, path):
        if not self._defectLoaded:
            self.errorOccurred.emit("Not Ready", "Defect model not loaded yet")
            return
        path = self._to_local_path(path)
        self.inferenceStarted.emit("defect")
        self._inference.sendCommand({"cmd": "classify", "path": path})

    @Slot(str)
    def analyzeDefect(self, path):
        print(f"[bridge] analyzeDefect called: path={path} defectLoaded={self._defectLoaded} supportLoaded={self._supportLoaded}", flush=True)
        if not self._defectLoaded:
            self.errorOccurred.emit("Not Ready", "Defect model not loaded yet")
            return
        if not self._supportLoaded:
            self.errorOccurred.emit("Not Ready", "Support set is still loading. Please wait.")
            return
        path = self._to_local_path(path)
        print(f"[bridge] Sending analyze: {path}", flush=True)
        self.inferenceStarted.emit("defect")
        self._inference.sendCommand({"cmd": "analyze", "path": path})

    @Slot()
    def autoLoadSupport(self):
        if not self._defectLoaded:
            self.errorOccurred.emit("Not Ready", "Defect model not loaded yet")
            return
        self._supportLoading = True
        self.supportStatusChanged.emit()
        self._inference.sendCommand({"cmd": "auto_load_support", "k_shot": 20})

    @Slot(str)
    def predictIRDrop(self, path):
        if not self._irdropLoaded:
            self.errorOccurred.emit("Not Ready", "IR Drop model not loaded yet")
            return
        path = self._to_local_path(path)
        self.inferenceStarted.emit("irdrop")
        self._inference.sendCommand({"cmd": "predict_irdrop", "path": path})

    @Slot()
    def resetSupportSet(self):
        self._inference.sendCommand({"cmd": "reset_support"})
        self.supportSetChanged.emit("{}")

    @Slot(str, result=int)
    def getImageVersion(self, image_id):
        return self._provider.getVersion(image_id) if self._provider else 0

    # --- Internal Handlers ---

    def _to_local_path(self, path):
        """Convert QML file:/// URL to local path if needed."""
        if path.startswith("file:///"):
            url = QUrl(path)
            return url.toLocalFile()
        return path

    def _on_server_ready(self):
        print("Inference server ready")
        self._inference.sendCommand({"cmd": "init"})

    def _on_response(self, response):
        status = response.get("status", "")
        cmd = response.get("cmd", "")
        print(f"[bridge] Response: cmd={cmd} status={status}", flush=True)

        if status == "error":
            message = response.get("message", "Unknown error")
            print(f"[bridge] ERROR: {message}", flush=True)
            self.errorOccurred.emit("Server Error", message)
            if cmd in ("classify", "analyze"):
                self.inferenceFinished.emit("defect")
            if cmd == "predict_irdrop":
                self.inferenceFinished.emit("irdrop")
            return

        handlers = {
            "init": self._handle_init,
            "load_models": self._handle_load_models,
            "add_support": self._handle_add_support,
            "classify": self._handle_classify,
            "analyze": self._handle_analyze,
            "auto_load_support": self._handle_auto_load_support,
            "predict_irdrop": self._handle_predict_irdrop,
        }

        handler = handlers.get(cmd)
        if handler:
            handler(response)

    def _on_server_error(self, message):
        self.errorOccurred.emit("Server Error", message)

    def _handle_init(self, data):
        self._deviceName = data.get("device", "CPU")
        self._cudaAvailable = data.get("cuda", False)
        self._gpuMemory = data.get("gpu_memory", "N/A")
        self._torchVersion = data.get("torch_version", "N/A")
        self.deviceInfoChanged.emit()
        self.loadModels()

    def _handle_load_models(self, data):
        self._defectLoaded = data.get("defect_loaded", False)
        self._irdropLoaded = data.get("irdrop_loaded", False)
        self._defectEpoch = data.get("defect_epoch", "?")
        self._defectAccuracy = data.get("defect_accuracy", "?")
        self._irdropEpoch = data.get("irdrop_epoch", "?")
        self._irdropF1 = data.get("irdrop_f1", "?")
        self._modelsLoading = False
        self.modelStatusChanged.emit()

        errors = data.get("errors", [])
        for err in errors:
            self.errorOccurred.emit("Model Load Error", err)

        if self._defectLoaded:
            self.autoLoadSupport()

    def _handle_add_support(self, data):
        thumb_path = data.get("thumbnail", "")
        label = data.get("label", 0)

        if thumb_path:
            summary = data.get("summary", {})
            count = summary.get(str(label), 0)
            image_id = f"defect_support_{label}_{count - 1}"
            self._load_image_to_provider(thumb_path, image_id)

        summary = data.get("summary", {})
        self.supportSetChanged.emit(json.dumps(summary))

    def _handle_classify(self, data):
        query_path = data.get("query_image", "")
        if query_path:
            self._load_image_to_provider(query_path, "defect_query")

        cls = data.get("predicted_class", "")
        confidence = data.get("confidence", 0.0)
        probs = data.get("probabilities", {})
        probs_json = json.dumps(probs)

        self.defectResultReady.emit(cls, confidence, probs_json)
        self.inferenceFinished.emit("defect")

    def _handle_analyze(self, data):
        # Load images to provider
        query_path = data.get("query_image", "")
        if query_path:
            self._load_image_to_provider(query_path, "defect_query")

        attention_path = data.get("attention_image", "")
        if attention_path:
            self._load_image_to_provider(attention_path, "defect_attention")

        overlay_path = data.get("overlay_image", "")
        if overlay_path:
            self._load_image_to_provider(overlay_path, "defect_overlay")

        # Classification results
        self._predictedDefectClass = data.get("predicted_class", "")
        self._defectConfidence = data.get("confidence", 0.0)
        probs = data.get("probabilities", {})
        self._defectProbsJson = json.dumps(probs)
        self._defectInferenceTime = data.get("inference_time", 0.0)

        # Knowledge base
        self._defectName = data.get("defect_name", "")
        self._defectDescription = data.get("defect_description", "")
        self._defectSeverity = data.get("severity", "")
        self._defectCauses = data.get("causes", [])
        self._defectPrevention = data.get("prevention", [])

        self.defectAnalysisReady.emit()
        self.inferenceFinished.emit("defect")

    def _handle_auto_load_support(self, data):
        self._supportLoading = False
        self._supportLoaded = True
        self._supportClassCount = data.get("classes_loaded", 0)
        self._supportImageCount = data.get("total_images", 0)
        self.supportStatusChanged.emit()

    def _handle_predict_irdrop(self, data):
        # Load input channel images
        input_paths = data.get("input_images", [])
        for i, path in enumerate(input_paths[:4]):
            self._load_image_to_provider(path, f"irdrop_input_{i}")

        # Load prediction image
        pred_path = data.get("prediction_image", "")
        if pred_path:
            self._load_image_to_provider(pred_path, "irdrop_prediction")

        # Load colorbar
        colorbar_path = data.get("colorbar_image", "")
        if colorbar_path:
            self._load_image_to_provider(colorbar_path, "irdrop_colorbar")

        min_val = data.get("min_val", 0.0)
        max_val = data.get("max_val", 0.0)
        mean_val = data.get("mean_val", 0.0)
        inference_time = data.get("inference_time", 0.0)

        self.irdropResultReady.emit(min_val, max_val, mean_val, inference_time)

        # Analysis results
        self._irdropSeverity = data.get("severity", "")
        self._irdropDescription = data.get("description", "")
        self._irdropCauses = data.get("causes", [])
        self._irdropRecommendations = data.get("recommendations", [])
        self._irdropHotspotPct = data.get("hotspot_percentage", 0.0)

        self.irdropAnalysisReady.emit()
        self.inferenceFinished.emit("irdrop")

    def _load_image_to_provider(self, file_path, image_id):
        img = QImage(file_path)
        if not img.isNull() and self._provider:
            self._provider.setImage(image_id, img)
            self.imageUpdated.emit(image_id, self._provider.getVersion(image_id))
