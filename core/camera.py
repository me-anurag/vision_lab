import logging
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)

# Consecutive failed reads before the camera is considered disconnected.
MAX_FAILED_READS = 10


class Camera(QThread):
    """Shared webcam. Reads frames on its own thread and emits them as BGR arrays.

    Call start() to open the camera and stop() to release it (the webcam light
    turns off). It can be started again after it has been stopped.
    """

    frame_ready = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, index=0, parent=None):
        super().__init__(parent)
        self.index = index
        self._running = False

    def start(self):
        if self.isRunning():
            return
        self._running = True
        super().start()

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        try:
            if not cap.isOpened():
                log.error("Could not open camera %d", self.index)
                self.error.emit(
                    f"No camera found (index {self.index}).\n"
                    "Check that a webcam is connected and not used by another app."
                )
                return

            failed = 0
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    failed += 1
                    if failed >= MAX_FAILED_READS:
                        log.error("Camera %d stopped delivering frames", self.index)
                        self.error.emit("Camera disconnected or not delivering frames.")
                        return
                    time.sleep(0.05)
                    continue
                failed = 0
                self.frame_ready.emit(frame.copy())
        finally:
            cap.release()
