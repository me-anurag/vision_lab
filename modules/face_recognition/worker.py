import logging
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

log = logging.getLogger(__name__)

# Request tasks
DETECT = "detect"        # boxes only
CAPTURE = "capture"      # boxes, plus crop and embedding if exactly one face
RECOGNIZE = "recognize"  # boxes, plus the best match for every face


@dataclass
class FaceResult:
    faces: np.ndarray                    # (N, 15) YuNet detections
    task: str                            # the task that produced this result
    crop: np.ndarray | None = None       # CAPTURE with exactly one face: aligned crop
    embedding: np.ndarray | None = None  # ... and its embedding
    matches: list[tuple[str | None, float]] | None = None  # RECOGNIZE: (name, score) per face
    error: str | None = None


class FaceWorker(QObject):
    """Runs detection, embedding and matching on its own thread.

    Emit `request(frame, task)` from the UI thread; the answer arrives on `result`.
    The UI sends a new frame only after the previous result came back, so frames
    are dropped while the worker is busy and no lag builds up.
    Emit `set_gallery(embeddings, names)` to update the people used for RECOGNIZE.
    """

    request = Signal(object, str)
    set_gallery = Signal(object, object)
    result = Signal(object)

    def __init__(self, detector, recognizer):
        super().__init__()
        self.detector = detector
        self.recognizer = recognizer
        self.gallery = np.empty((0, 128), np.float32)
        self.names: list[str] = []
        self.request.connect(self.process)
        self.set_gallery.connect(self._set_gallery)

    @Slot(object, object)
    def _set_gallery(self, gallery, names):
        self.gallery, self.names = gallery, names

    @Slot(object, str)
    def process(self, frame, task):
        try:
            faces = self.detector.detect(frame)
            res = FaceResult(faces, task)
            if task == CAPTURE and len(faces) == 1:
                res.crop = self.recognizer.align_crop(frame, faces[0])
                res.embedding = self.recognizer.embed(res.crop)
            elif task == RECOGNIZE:
                res.matches = [
                    self.recognizer.match(
                        self.recognizer.embed(self.recognizer.align_crop(frame, face)),
                        self.gallery, self.names,
                    )
                    for face in faces
                ]
        except Exception as exc:
            log.exception("Face processing failed")
            res = FaceResult(np.empty((0, 15), np.float32), task, error=str(exc))
        self.result.emit(res)
