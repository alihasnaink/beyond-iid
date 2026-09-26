from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 6304
    image_size: int = 224
    test_per_class: int = 50
    batch_size: int = 64
    # Notebook runtimes (including Kaggle) often have a small /dev/shm.  A
    # 64-image float batch at 224x224 is ~38 MiB, so multiprocessing prefetch
    # can exhaust shared memory and leave DataLoader workers in a broken state.
    num_workers: int = 0
    max_head_epochs: int = 50
    head_patience: int = 5
    head_lr: float = 1e-3
    head_weight_decay: float = 1e-4
    hue_factor: float = 0.25
    style_alpha: float = 0.80
    conflicts_per_direction: int = 30
    min_conflicts_per_direction: int = 20
    tsne_pairs: int = 150
    tsne_perplexity: int = 30
    tsne_iterations: int = 1000


CLASS_NAMES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
CUE_PAIRS = [("airplane", "bird"), ("car", "truck"), ("cat", "dog"), ("deer", "horse"), ("monkey", "ship")]
TRANSLATION_DELTAS = [0, 8, 16, 32]
TRANSLATION_DIRECTIONS = ["left", "right", "up", "down"]


def resolve_task_dir(start: Path | None = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    candidates = [cwd / "inductive-bias-representations", cwd, cwd.parent]
    candidates.extend(cwd.glob("*/inductive-bias-representations"))
    return next(
        (path for path in candidates if path.exists() and path.name == "inductive-bias-representations"),
        cwd / "inductive-bias-representations",
    )
