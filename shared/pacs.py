from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _layout_root(candidate: Path) -> Path | None:
    candidate = candidate.expanduser().resolve()
    variants = [
        candidate, candidate / "images", candidate / "PACS", candidate / "PACS_Original",
        candidate / "kfold", candidate / "PACS" / "PACS_Original",
        candidate / "PACS" / "kfold", candidate / "images" / "kfold",
    ]
    for root in variants:
        if all((root / domain).is_dir() for domain in DOMAINS):
            return root
    return None


def find_pacs_root(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("PACS_ROOT"):
        candidates.append(Path(os.environ["PACS_ROOT"]))
    cwd = Path.cwd()
    candidates += [cwd / "PACS", cwd / "data" / "PACS", cwd.parent / "PACS"]
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        candidates += list(kaggle.glob("*/PACS")) + list(kaggle.glob("*/pacs"))
        candidates += list(kaggle.glob("*/PACS/PACS_Original")) + list(kaggle.glob("*/PACS/images"))
        candidates += [path for path in kaggle.glob("*/*") if path.is_dir()]
    for candidate in candidates:
        root = _layout_root(candidate)
        if root:
            return root
    raise FileNotFoundError(
        "PACS was not found. Attach a Kaggle PACS dataset or set PACS_ROOT to the directory "
        "containing photo/, art_painting/, cartoon/, and sketch/."
    )


def labeled_records(root: Path, domain: str):
    records = []
    for label, class_name in enumerate(CLASSES):
        class_dir = root / domain / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing PACS class directory: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                records.append({"path": str(path.resolve()), "relative_path": str(path.relative_to(root)),
                                "label": label, "class_name": class_name, "domain": domain})
    return records


def unlabeled_paths(root: Path, domain: str):
    return sorted(path.resolve() for path in (root / domain).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


class PACSLabeledDataset(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = list(records), transform

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        return self.transform(image), int(record["label"]), record["path"]


class PACSUnlabeledDataset(Dataset):
    """Returns no class label; training cannot access Sketch labels through this object."""
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), index
