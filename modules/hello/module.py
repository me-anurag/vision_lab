from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from core.base_module import BaseModule
from core.utils import frame_to_qimage


class HelloModule(BaseModule):
    """Minimal reference module: shows the live camera feed.

    Copy this folder to start a new module.
    """

    name = "Hello"
    description = "Minimal example module with a live camera feed."

    def __init__(self, camera):
        super().__init__(camera)
        self._active = False

    def create_widget(self, parent):
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet("background: black; color: white; font-size: 16px;")
        layout.addWidget(self.video)
        return widget

    def on_open(self):
        self._active = True
        self.video.setText("Starting camera…")
        self.camera.frame_ready.connect(self._show_frame)
        self.camera.error.connect(self._show_error)
        self.camera.start()

    def on_close(self):
        self._active = False
        self.camera.frame_ready.disconnect(self._show_frame)
        self.camera.error.disconnect(self._show_error)
        self.camera.stop()
        self.video.clear()

    def _show_frame(self, frame):
        if not self._active:
            return
        pixmap = QPixmap.fromImage(frame_to_qimage(frame))
        self.video.setPixmap(
            pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _show_error(self, message):
        self.video.clear()
        self.video.setText(message)
