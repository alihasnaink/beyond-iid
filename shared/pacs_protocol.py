from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from shared.pacs import CLASSES, labeled_records


SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"


def build_or_load_protocol(pacs_root: Path, split_path: Path, seed: int = 6304):
    """Persist portable relative-path source splits; never loads the target domain."""
    current = {domain: labeled_records(pacs_root, domain) for domain in SOURCE_DOMAINS}
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        if payload.get("seed") != seed:
            raise ValueError(f"Existing split uses seed {payload.get('seed')}, expected {seed}.")
        lookup = {record["relative_path"]: record for records in current.values() for record in records}
        protocol = {"train": {}, "val": {}}
        for split in ["train", "val"]:
            for domain in SOURCE_DOMAINS:
                missing = [path for path in payload[split][domain] if path not in lookup]
                if missing:
                    raise FileNotFoundError(f"Split paths missing under PACS_ROOT, e.g. {missing[0]}")
                protocol[split][domain] = [lookup[path] for path in payload[split][domain]]
        return protocol

    protocol, serializable = {"train": {}, "val": {}}, {"seed": seed, "train": {}, "val": {}}
    for domain, records in current.items():
        indices = np.arange(len(records))
        labels = [record["label"] for record in records]
        train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=seed, stratify=labels)
        for split, chosen in [("train", train_idx), ("val", val_idx)]:
            protocol[split][domain] = [records[int(index)] for index in sorted(chosen)]
            serializable[split][domain] = [record["relative_path"] for record in protocol[split][domain]]
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(serializable, indent=2))
    return protocol


def protocol_summary(protocol):
    return {
        split: {domain: {class_name: sum(r["class_name"] == class_name for r in records) for class_name in CLASSES}
                for domain, records in protocol[split].items()}
        for split in ["train", "val"]
    }
