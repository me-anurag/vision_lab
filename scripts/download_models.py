"""Download the face models from the official OpenCV Zoo into models/.

Uses only the standard library. Files that already exist are skipped.
Run: python scripts/download_models.py
"""

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{BASE_URL}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE_URL}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
# A real model is far larger; anything smaller is an error page or a Git LFS pointer.
MIN_SIZE = 100_000


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        while chunk := response.read(1 << 16):
            f.write(chunk)
    if tmp.stat().st_size < MIN_SIZE:
        tmp.unlink()
        raise RuntimeError(f"Downloaded file is too small, not a model: {url}")
    tmp.replace(dest)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"[skip] {filename} already exists")
            continue
        print(f"[get ] {filename} ...", flush=True)
        try:
            download(url, dest)
            print(f"[ ok ] {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"[fail] {filename}: {exc}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
