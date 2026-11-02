import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.base_module import BaseModule
from core.utils import UiCallback, frame_to_qimage
from modules.face_recognition.detector import FaceDetector
from modules.face_recognition.recognizer import FaceRecognizer
from modules.face_recognition.storage import FaceStore
from modules.face_recognition.worker import FaceWorker

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DETECTOR_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = ROOT / "models" / "face_recognition_sface_2021dec.onnx"

TARGET_SAMPLES = 15
AUTO_CAPTURE_MS = 500
GREEN, RED = (0, 200, 0), (0, 0, 255)


class FaceRecognitionModule(BaseModule):
    name = "Face Recognition"
    description = "Enroll people with the webcam and recognize them live."

    def __init__(self, camera):
        super().__init__(camera)
        self.store = None
        self.worker = None
        self.worker_thread = None
        self._active = False
        self._busy = False             # a frame is being processed by the worker
        self._capture_pending = False  # the next frame sent to the worker is a capture
        self._capture_name = None      # name at the time the in-flight capture was sent
        self._faces = np.empty((0, 15), np.float32)  # latest detections, drawn on every frame
        self._session_count = 0        # samples captured for the current name

    # ----- UI -----

    def create_widget(self, parent):
        self.store = FaceStore(DATA_DIR)

        widget = QWidget(parent)
        layout = QHBoxLayout(widget)

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet("background: black; color: white; font-size: 16px;")
        layout.addWidget(self.video, stretch=1)

        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(320)
        self.tabs.addTab(self._build_enroll_tab(), "Enroll")
        layout.addWidget(self.tabs)

        self.auto_timer = QTimer(widget)
        self.auto_timer.setInterval(AUTO_CAPTURE_MS)
        self.auto_timer.timeout.connect(self._request_capture)

        self._refresh_people()
        self._update_counter()
        return widget

    def _build_enroll_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Person's name")
        self.name_edit.textChanged.connect(self._reset_session)
        layout.addWidget(self.name_edit)

        self.capture_button = QPushButton("Capture")
        self.capture_button.clicked.connect(self._on_capture_clicked)
        layout.addWidget(self.capture_button)

        self.auto_check = QCheckBox("Auto-capture (every 0.5 s)")
        self.auto_check.toggled.connect(self._on_auto_toggled)
        layout.addWidget(self.auto_check)

        self.counter_label = QLabel()
        self.counter_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(self.counter_label)

        self.hint_label = QLabel("Slowly turn your head a little")
        self.hint_label.setStyleSheet("color: #1a73e8; font-weight: bold;")
        self.hint_label.hide()
        layout.addWidget(self.hint_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addSpacing(12)
        layout.addWidget(QLabel("Enrolled people"))
        self.people_list = QListWidget()
        layout.addWidget(self.people_list, stretch=1)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._on_delete)
        layout.addWidget(self.delete_button)
        return tab

    def _refresh_people(self):
        self.people_list.clear()
        for safe, display, count in self.store.people():
            item = QListWidgetItem(f"{display}  ({count} samples)")
            item.setData(Qt.UserRole, safe)
            self.people_list.addItem(item)

    def _update_counter(self):
        self.counter_label.setText(f"Samples: {self._session_count} / {TARGET_SAMPLES}")

    def _update_hint(self):
        capturing = self.auto_check.isChecked() or 0 < self._session_count < TARGET_SAMPLES
        self.hint_label.setVisible(capturing)

    def _set_status(self, text, warning=False):
        self.status_label.setStyleSheet("color: #d93025;" if warning else "color: #188038;")
        self.status_label.setText(text)

    # ----- lifecycle -----

    def on_open(self):
        try:
            self._ensure_worker()
        except Exception as exc:
            log.exception("Could not load face models")
            self.video.setText(f"Could not load face models:\n{exc}")
            return

        self._active = True
        self._busy = False
        self._capture_pending = False
        self._faces = np.empty((0, 15), np.float32)
        self.worker_thread.start()
        self.video.setText("Starting camera…")
        self.camera.frame_ready.connect(self._on_frame)
        self.camera.error.connect(self._on_camera_error)
        self.camera.start()

    def on_close(self):
        self.auto_check.setChecked(False)
        if not self._active:
            return
        self._active = False
        self.camera.frame_ready.disconnect(self._on_frame)
        self.camera.error.disconnect(self._on_camera_error)
        self.camera.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.video.clear()

    def _ensure_worker(self):
        if self.worker is not None:
            return
        detector = FaceDetector(DETECTOR_MODEL)
        recognizer = FaceRecognizer(RECOGNIZER_MODEL)
        self.worker = FaceWorker(detector, recognizer)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        # Deliver results on the UI thread (see UiCallback).
        self.result_relay = UiCallback(self._on_result)
        self.worker.result.connect(self.result_relay.call)

    # ----- frames and results -----

    def _on_frame(self, frame):
        if not self._active:
            return
        if not self._busy:
            self._busy = True
            capture, self._capture_pending = self._capture_pending, False
            if capture:
                self._capture_name = self._name()
            self.worker.request.emit(frame, capture)
        self._show(frame)

    def _show(self, frame):
        view = frame.copy()
        color = GREEN if len(self._faces) == 1 else RED
        for face in self._faces:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        pixmap = QPixmap.fromImage(frame_to_qimage(view))
        self.video.setPixmap(
            pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_camera_error(self, message):
        self.auto_check.setChecked(False)
        self.video.clear()
        self.video.setText(message)

    def _on_result(self, res):
        if not self._active:
            return
        self._busy = False
        self._faces = res.faces
        if res.error:
            self._set_status(f"Processing error: {res.error}", warning=True)
        elif res.capture:
            self._handle_capture(res)

    # ----- enrollment -----

    def _name(self):
        return self.name_edit.text().strip()

    def _check_name(self):
        if not self._name():
            self._set_status("Enter a name first.", warning=True)
            return False
        return True

    def _on_capture_clicked(self):
        if self._check_name():
            self._capture_pending = True

    def _on_auto_toggled(self, on):
        if on:
            if not self._check_name():
                self.auto_check.setChecked(False)
                return
            if self._session_count >= TARGET_SAMPLES:
                self._session_count = 0  # start a new round
                self._update_counter()
            self.name_edit.setEnabled(False)
            self.auto_timer.start()
        else:
            self.auto_timer.stop()
            self._capture_pending = False  # drop a tick that fired just before stopping
            self.name_edit.setEnabled(True)
        self._update_hint()

    def _request_capture(self):
        self._capture_pending = True

    def _reset_session(self):
        self._capture_pending = False
        self._session_count = 0
        self._update_counter()
        self._update_hint()
        self.status_label.clear()

    def _handle_capture(self, res):
        name, self._capture_name = self._capture_name, None
        if not name or name != self._name():
            return  # the name changed while this capture was being processed
        count = len(res.faces)
        if count != 1:
            if count == 0:
                self._set_status("No face visible.", warning=True)
            else:
                self._set_status(f"{count} faces visible. Only one person should be in view.", warning=True)
            return
        try:
            total = self.store.add_sample(name, res.crop, res.embedding)
        except OSError as exc:
            log.exception("Could not save face sample")
            self._set_status(f"Could not save sample: {exc}", warning=True)
            self.auto_check.setChecked(False)
            return

        self._session_count += 1
        self._update_counter()
        self._refresh_people()
        if self._session_count >= TARGET_SAMPLES:
            self.auto_check.setChecked(False)
            self._set_status(f"Done: {TARGET_SAMPLES} samples captured for {name} ({total} total).")
        else:
            self._set_status(f"Saved sample {self._session_count} for {name}.")
        self._update_hint()

    def _on_delete(self):
        item = self.people_list.currentItem()
        if item is None:
            self._set_status("Select a person to delete.", warning=True)
            return
        safe = item.data(Qt.UserRole)
        display = self.store.names.get(safe, safe)
        answer = QMessageBox.question(
            self.people_list, "Delete person",
            f"Delete {display} and all their samples?",
        )
        if answer != QMessageBox.Yes:
            return
        try:
            self.store.delete(safe)
        except OSError as exc:
            log.exception("Could not delete %r", safe)
            self._set_status(f"Could not delete {display}: {exc}", warning=True)
            return
        self._refresh_people()
        self._set_status(f"Deleted {display}.")
