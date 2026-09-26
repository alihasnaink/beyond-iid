from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def make_splits(train_dataset, test_dataset, class_names, seed: int, test_per_class: int, results_dir: Path):
    """Create and persist the required stratified train/validation split and test subset."""
    train_targets = np.asarray(train_dataset.labels)
    test_targets = np.asarray(test_dataset.labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(train_targets)), train_targets))

    rng = np.random.default_rng(seed)
    selected = []
    for class_idx in range(len(class_names)):
        available = np.flatnonzero(test_targets == class_idx)
        selected.extend(rng.choice(available, size=min(test_per_class, len(available)), replace=False).tolist())
    selected = np.asarray(sorted(selected), dtype=int)

    payload = {"seed": seed, "train_official_indices": train_idx.tolist(), "validation_official_indices": val_idx.tolist()}
    (results_dir / "train_validation_split.json").write_text(json.dumps(payload, indent=2))

    manifest = pd.DataFrame({
        "eval_position": np.arange(len(selected)),
        "official_test_index": selected,
        "class_index": test_targets[selected],
    })
    manifest["class_name"] = manifest.class_index.map(dict(enumerate(class_names)))
    manifest.to_csv(results_dir / "selected_test_identifiers.csv", index=False)
    return train_idx, val_idx, selected, manifest
