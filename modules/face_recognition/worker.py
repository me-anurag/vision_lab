import logging
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

log = logging.getLogger(__name__)


@dataclass
class FaceResult:
    faces: np.ndarray                    # (N, 15) YuNet detections
    capture: bool                        # True if this frame was a capture request
    crop: np.ndarray | None = None       # aligned crop, only for a capture with exactly one face
    embedding: np.ndarray | None = None  # its embedding
    error: str | None = None


class FaceWorker(QObject):
    """Runs detection and embedding on its own thread.

    Emit `request(frame, capture)` from the UI thread; the answer arrives on `result`.
    The UI sends a new frame only after the previous result came back, so frames
    are dropped while the worker is busy and no lag builds up.
    """

    request = Signal(object, bool)
    result = Signal(object)

    def __init__(self, detector, recognizer):
        super().__init__()
        self.detector = detector
        self.recognizer = recognizer
        self.request.connect(self.process)

    @Slot(object, bool)
    def process(self, frame, capture):
        try:
            faces = self.detector.detect(frame)
            res = FaceResult(faces, capture)
            if capture and len(faces) == 1:
                res.crop = self.recognizer.align_crop(frame, faces[0])
                res.embedding = self.recognizer.embed(res.crop)
        except Exception as exc:
            log.exception("Face processing failed")
            res = FaceResult(np.empty((0, 15), np.float32), capture, error=str(exc))
        self.result.emit(res)
