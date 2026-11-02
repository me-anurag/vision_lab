from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """Convert a BGR frame to a QImage (the image owns a copy of the data)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def read_image(path: Path) -> np.ndarray | None:
    """Read an image file; works with any characters in the path. Returns None on failure."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image file (format from the extension); works with any characters in the path."""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    path.write_bytes(buf.tobytes())
