from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data.cifar10 import CIFARArrayDataset, make_transform


NEAR_UNKNOWN_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_UNKNOWN_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]


def build_unknown_datasets(cifar100_test, results_dir: Path):
    """Called only after all checkpoints and score definitions have been fixed."""
    class_to_index = {name: index for index, name in enumerate(cifar100_test.classes)}
    transform = make_transform(train=False)
    datasets, manifest = {}, []
    for group, names in [("near", NEAR_UNKNOWN_CLASSES), ("far", FAR_UNKNOWN_CLASSES)]:
        labels = {class_to_index[name] for name in names}
        indices = [i for i, label in enumerate(cifar100_test.targets) if label in labels]
        datasets[group] = CIFARArrayDataset(
            cifar100_test.data, cifar100_test.targets, indices, transform,
            classes=cifar100_test.classes, identifier_prefix="cifar100_test",
        )
        for index in indices:
            manifest.append({"group": group, "official_test_index": index,
                             "fine_label": int(cifar100_test.targets[index]),
                             "class_name": cifar100_test.classes[cifar100_test.targets[index]]})
        counts = Counter(cifar100_test.classes[cifar100_test.targets[index]] for index in indices)
        if counts != Counter({name: 100 for name in names}):
            raise RuntimeError(f"Unexpected {group} class counts: {dict(counts)}")
    if len(datasets["near"]) != 800 or len(datasets["far"]) != 800:
        raise RuntimeError(f"Expected 800 near and 800 far unknowns, got {len(datasets['near'])}, {len(datasets['far'])}.")
    import pandas as pd
    pd.DataFrame(manifest).to_csv(results_dir / "unknown_evaluation_manifest.csv", index=False)
    (results_dir / "unknown_class_groups.json").write_text(json.dumps({"near": NEAR_UNKNOWN_CLASSES, "far": FAR_UNKNOWN_CLASSES}, indent=2))
    return datasets
