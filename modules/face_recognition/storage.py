import json
import re
import shutil
import time
import unicodedata
from pathlib import Path

import numpy as np

from core.utils import write_image

# Folder names Windows does not allow, plus names that clash with np.savez() arguments.
RESERVED_NAMES = (
    {"con", "prn", "aux", "nul", "file", "allow_pickle"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def sanitize_name(name: str) -> str:
    """Turn a display name into a safe, lowercase ASCII folder name."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name).strip("_").lower()
    if not safe:
        safe = "person"
    if safe in RESERVED_NAMES:
        safe += "_"
    return safe


def replace_file(src: Path, dest: Path, attempts=10, delay=0.05):
    """os.replace with retries: on Windows it fails while another program (antivirus,
    indexer, an editor) briefly has the target open."""
    for attempt in range(attempts):
        try:
            src.replace(dest)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


class FaceStore:
    """Enrolled people on disk.

    data/faces/<safe_name>/0001.jpg ...   aligned face crops
    data/embeddings.npz                    safe_name -> (N, 128) embeddings
    data/people.json                       safe_name -> original display name
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.faces_dir = self.data_dir / "faces"
        self.embeddings_path = self.data_dir / "embeddings.npz"
        self.people_path = self.data_dir / "people.json"
        self.names: dict[str, str] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.load()

    def load(self):
        if self.people_path.exists():
            self.names = json.loads(self.people_path.read_text(encoding="utf-8"))
        if self.embeddings_path.exists():
            with np.load(self.embeddings_path) as npz:
                self.embeddings = {key: npz[key] for key in npz.files}

    def save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first so a crash never leaves a half-written file.
        tmp = self.embeddings_path.with_name("embeddings.tmp.npz")
        with open(tmp, "wb") as f:
            np.savez(f, **self.embeddings)
        replace_file(tmp, self.embeddings_path)

        tmp = self.people_path.with_name("people.tmp.json")
        tmp.write_text(json.dumps(self.names, ensure_ascii=False, indent=2), encoding="utf-8")
        replace_file(tmp, self.people_path)

    def safe_name_for(self, name: str) -> str:
        """Folder name for a display name: the existing one if this exact name is
        already enrolled, otherwise a new unique one (with a _2, _3... suffix on collision)."""
        for safe, display in self.names.items():
            if display == name:
                return safe
        base = sanitize_name(name)
        safe, i = base, 2
        while safe in self.names:
            safe, i = f"{base}_{i}", i + 1
        return safe

    def add_sample(self, name: str, crop: np.ndarray, embedding: np.ndarray) -> int:
        """Save one face crop and its embedding. Returns the person's total sample count."""
        safe = self.safe_name_for(name)
        folder = self.faces_dir / safe
        folder.mkdir(parents=True, exist_ok=True)
        numbers = [int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit()]
        write_image(folder / f"{max(numbers, default=0) + 1:04d}.jpg", crop)

        previous = self.embeddings.get(safe, np.empty((0, embedding.size), np.float32))
        self.embeddings[safe] = np.vstack([previous, embedding[None, :]])
        self.names[safe] = name
        self.save()
        return len(self.embeddings[safe])

    def delete(self, safe: str):
        """Remove a person's images and embeddings."""
        folder = self.faces_dir / safe
        if folder.exists():
            shutil.rmtree(folder)
        self.names.pop(safe, None)
        self.embeddings.pop(safe, None)
        self.save()

    def people(self) -> list[tuple[str, str, int]]:
        """(safe_name, display_name, sample_count) for everyone, sorted by display name."""
        rows = [
            (safe, display, len(self.embeddings.get(safe, ())))
            for safe, display in self.names.items()
        ]
        return sorted(rows, key=lambda row: row[1].lower())
