from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def make_or_load_split(targets, split_path: Path, seed=6304, validation_fraction=0.10):
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        if payload["seed"] != seed or payload["validation_fraction"] != validation_fraction:
            raise ValueError("Existing split metadata does not match configuration.")
        train_idx = np.asarray(payload["train_indices"])
        val_idx = np.asarray(payload["validation_indices"])
        if len(train_idx) + len(val_idx) != len(targets) or np.intersect1d(train_idx, val_idx).size:
            raise ValueError("Existing split is incomplete or overlapping.")
        return train_idx, val_idx
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_fraction, random_state=seed, stratify=np.asarray(targets),
    )
    payload = {"seed": seed, "validation_fraction": validation_fraction,
               "train_indices": sorted(map(int, train_idx)), "validation_indices": sorted(map(int, val_idx))}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2))
    return np.asarray(payload["train_indices"]), np.asarray(payload["validation_indices"])
