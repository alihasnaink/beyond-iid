from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix


def per_class_table(predictions_by_method, class_names, source_only="source_only"):
    baseline = predictions_by_method[source_only]
    rows = []
    for method, output in predictions_by_method.items():
        for class_index, class_name in enumerate(class_names):
            mask = output["labels"] == class_index
            accuracy = (output["predictions"][mask] == class_index).mean()
            baseline_mask = baseline["labels"] == class_index
            baseline_accuracy = (baseline["predictions"][baseline_mask] == class_index).mean()
            incorrect = output["predictions"][mask & (output["predictions"] != class_index)]
            dominant = "none" if len(incorrect) == 0 else class_names[int(np.bincount(incorrect, minlength=len(class_names)).argmax())]
            rows.append({"method": method, "class_name": class_name, "target_class_accuracy": accuracy,
                         "change_vs_source_only": accuracy - baseline_accuracy,
                         "dominant_incorrect_prediction": dominant, "class_count": int(mask.sum())})
    return pd.DataFrame(rows)


def save_confusion_matrices(predictions_by_method, class_names, methods, results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for axis, method in zip(axes.flat, methods):
        output = predictions_by_method[method]
        matrix = confusion_matrix(output["labels"], output["predictions"], normalize="true")
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=class_names, yticklabels=class_names, ax=axis, cbar=False)
        axis.set_title(method)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.suptitle("Normalized Sketch confusion matrices")
    fig.tight_layout()
    fig.savefig(results_dir / "target_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_failure_examples(predictions_by_method, class_table, class_names, results_dir, seed=6304):
    gains = class_table.groupby("method").change_vs_source_only.mean().drop("source_only", errors="ignore")
    comparison = gains.idxmax()
    baseline, adapted = predictions_by_method["source_only"], predictions_by_method[comparison]
    improvement_class = class_table.query("method == @comparison").sort_values("change_vs_source_only", ascending=False).iloc[0].class_name
    degradation_class = class_table.query("method == @comparison").sort_values("change_vs_source_only").iloc[0].class_name
    candidates = []
    for category, class_name in [("improvement", improvement_class), ("degradation", degradation_class)]:
        class_idx = class_names.index(class_name)
        class_mask = baseline["labels"] == class_idx
        if category == "improvement":
            mask = class_mask & (baseline["predictions"] != class_idx) & (adapted["predictions"] == class_idx)
            if not mask.any():
                mask = class_mask & (baseline["predictions"] != class_idx)
        else:
            mask = class_mask & (baseline["predictions"] == class_idx) & (adapted["predictions"] != class_idx)
            if not mask.any():
                mask = class_mask & (adapted["predictions"] != class_idx)
        for index in np.flatnonzero(mask)[:6]:
            candidates.append((category, class_name, int(index)))
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for axis in axes.flat:
        axis.axis("off")
    rows = []
    for axis, (category, class_name, index) in zip(axes.flat, candidates):
        path = baseline["identifiers"][index]
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(
            f"{category}: {class_name}\nERM={class_names[baseline['predictions'][index]]}\n"
            f"{comparison}={class_names[adapted['predictions'][index]]}", fontsize=8,
        )
        rows.append({"category": category, "class_name": class_name, "path": path,
                     "source_only_prediction": class_names[baseline["predictions"][index]],
                     "comparison_method": comparison,
                     "adapted_prediction": class_names[adapted["predictions"][index]]})
    fig.suptitle(f"Selected positive/negative transfer examples: source_only vs {comparison}")
    fig.tight_layout()
    fig.savefig(results_dir / "selected_target_failure_cases.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(results_dir / "selected_target_failure_cases.csv", index=False)
    return comparison, pd.DataFrame(rows)
