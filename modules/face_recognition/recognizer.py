from pathlib import Path

import cv2
import numpy as np


class FaceRecognizer:
    """SFace face recognizer: aligns a detected face and turns it into a 128-d embedding."""

    def __init__(self, model_path: Path):
        # Load as bytes so paths with any characters work.
        model = np.frombuffer(Path(model_path).read_bytes(), dtype=np.uint8)
        self._recognizer = cv2.FaceRecognizerSF.create("onnx", model, np.empty(0, np.uint8))

    def align_crop(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        """Return the aligned 112x112 face crop for one YuNet detection row."""
        return self._recognizer.alignCrop(frame, face)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return the L2-normalized 128-d embedding of an aligned crop."""
        feature = self._recognizer.feature(crop).flatten().astype(np.float32)
        return feature / np.linalg.norm(feature)

    @staticmethod
    def match(embedding: np.ndarray, gallery: np.ndarray, names: list[str]) -> tuple[str | None, float]:
        """Best cosine score of `embedding` against every stored embedding.

        `gallery` is (M, 128) with one name per row. Embeddings are L2-normalized,
        so the cosine similarity is a dot product. Returns (name, score).
        """
        if len(gallery) == 0:
            return None, 0.0
        scores = gallery @ embedding
        best = int(np.argmax(scores))
        return names[best], float(scores[best])
