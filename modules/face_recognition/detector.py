from pathlib import Path

import cv2
import numpy as np


class FaceDetector:
    """YuNet face detector.

    detect() returns an (N, 15) float32 array, one row per face:
    x, y, w, h, 5 landmark (x, y) pairs, score.
    """

    def __init__(self, model_path: Path, score_threshold=0.9, nms_threshold=0.3, top_k=5000):
        # Load as bytes so paths with any characters work.
        model = np.frombuffer(Path(model_path).read_bytes(), dtype=np.uint8)
        self._detector = cv2.FaceDetectorYN.create(
            "onnx", model, np.empty(0, np.uint8), (320, 320),
            score_threshold, nms_threshold, top_k,
        )
        self._input_size = None

    def detect(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._input_size != (w, h):
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self._detector.detect(frame)
        if faces is None:
            return np.empty((0, 15), np.float32)
        return faces
