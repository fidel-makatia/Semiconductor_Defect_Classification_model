"""HeatmapProvider: QQuickImageProvider for serving dynamic images to QML.

Mirrors app/src/heatmapprovider.h/cpp.
Images are accessed via: image://heatmap/<id>?v=<version>
"""

import threading

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QImage
from PySide6.QtQuick import QQuickImageProvider


class HeatmapProvider(QQuickImageProvider):
    def __init__(self):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._lock = threading.Lock()
        self._images: dict[str, QImage] = {}
        self._versions: dict[str, int] = {}

    def requestImage(self, id: str, size: QSize, requestedSize: QSize):
        with self._lock:
            # Strip version query param: "id?v=3" -> "id"
            clean_id = id.split("?")[0] if "?" in id else id

            img = self._images.get(clean_id)
            if img is None or img.isNull():
                img = QImage(256, 256, QImage.Format.Format_RGBA8888)
                img.fill(QColor(26, 30, 46, 255))

            if requestedSize.width() > 0 and requestedSize.height() > 0:
                img = img.scaled(
                    requestedSize,
                    mode=img.TransformationMode.SmoothTransformation,
                )

            return img

    def setImage(self, image_id: str, image: QImage):
        with self._lock:
            self._images[image_id] = image
            self._versions[image_id] = self._versions.get(image_id, 0) + 1

    def getVersion(self, image_id: str) -> int:
        with self._lock:
            return self._versions.get(image_id, 0)
