from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.cifar10 import CIFAR10_CLASSES


PLAUSIBILITY = {
    "bus": "plausible: road vehicle", "pickup_truck": "plausible: road vehicle",
    "motorcycle": "plausible: road vehicle", "tractor": "plausible: road vehicle",
    "wolf": "plausible: canine", "fox": "plausible: canine",
    "leopard": "plausible: animal", "camel": "plausible: animal",
    "bottle": "surprising: household object", "bowl": "surprising: household object",
    "chair": "surprising: furniture", "clock": "surprising: household object",
    "keyboard": "surprising: electronic object", "mushroom": "surprising: fungus",
    "sunflower": "surprising: plant", "wardrobe": "surprising: furniture",
}


def select_failures(unknown_outputs, unknown_scores, threshold, count=3):
    rows = []
    for group in ["near", "far"]:
        output = unknown_outputs[group]
        scores = np.asarray(unknown_scores[group])
        accepted = np.flatnonzero(scores <= threshold)
        if len(accepted) < count:
            raise RuntimeError(f"Vanilla MLS accepted only {len(accepted)} {group} unknowns; "
                               f"cannot produce the required {count} failure examples.")
        chosen = accepted[np.argsort(scores[accepted])[:count]]
        predictions = output["logits"].argmax(1).numpy()
        for rank, position in enumerate(chosen, 1):
            identifier = output["identifiers"][int(position)]
            rows.append({
                "group": group, "rank": rank, "identifier": identifier,
                "official_test_index": int(identifier.rsplit(":", 1)[1]),
                "unknown_class": output["class_names"][int(position)],
                "predicted_known_class": CIFAR10_CLASSES[int(predictions[position])],
                "unknownness_score": float(scores[position]), "threshold": float(threshold),
                "interpretation": PLAUSIBILITY.get(output["class_names"][int(position)], "inspect manually"),
            })
    return pd.DataFrame(rows)


def plot_failures(table, cifar100_test, output_path: Path):
    if table.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, row in zip(axes.flat, table.itertuples(index=False)):
        axis.imshow(cifar100_test.data[row.official_test_index])
        axis.set_title(f"{row.group}: {row.unknown_class} → {row.predicted_known_class}\n"
                       f"score={row.unknownness_score:.3f}, τ={row.threshold:.3f}", fontsize=9)
        axis.set_xlabel(row.interpretation, fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle("Vanilla MLS: confidently accepted unknown failure cases")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
