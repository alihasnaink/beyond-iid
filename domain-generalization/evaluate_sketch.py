from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parent
for path in [TASK_DIR, REPO_ROOT]:
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from configs.load_config import load_config  # noqa: E402
from evaluation.domain_metrics import classification_metrics, predict  # noqa: E402
from models.backbone import build_model  # noqa: E402
from shared.pacs import CLASSES, PACSLabeledDataset, find_pacs_root, labeled_records  # noqa: E402
from train import MAIN_METHODS, checkpoint_paths  # noqa: E402
from training.loaders import make_transforms  # noqa: E402


def evaluate_sketch_suite(pacs_root=None, task_dir=TASK_DIR, configurations_locked=False):
    """The only Task 3 entry point that enumerates or loads Sketch images."""
    if not configurations_locked:
        raise RuntimeError(
            "Lock all Task 3 methods, hyperparameters, checkpoints, and source-side conclusions before "
            "calling final Sketch evaluation. Pass configurations_locked=True only after doing so."
        )
    task_dir = Path(task_dir).resolve()
    results_dir = task_dir / "results"
    manifest_path = results_dir / "run_manifest.csv"
    source_path = results_dir / "source_validation_comparison.csv"
    if not manifest_path.exists() or not source_path.exists():
        raise FileNotFoundError("Run train_suite and evaluate_source_suite before unlocking Sketch.")
    manifest = pd.read_csv(manifest_path)
    source_comparison = pd.read_csv(source_path)
    paths = checkpoint_paths(task_dir)
    missing = [run for run in manifest.run_name if not paths[run].exists()]
    incomplete = [run for run in manifest.run_name if not (results_dir / "completed" / f"{run}.json").exists()]
    if missing or incomplete:
        raise FileNotFoundError(f"All runs must be fixed first. Missing checkpoints={missing}; incomplete={incomplete}")

    # No target records are enumerated before all lock/completeness checks above pass.
    pacs_root = find_pacs_root(pacs_root)
    config = load_config(task_dir, "erm")
    _train_transform, eval_transform = make_transforms(config)
    target_dataset = PACSLabeledDataset(labeled_records(pacs_root, config["target_domain"]), eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Configurations locked. Beginning final Sketch evaluation on {device}.", flush=True)

    rows, outputs = [], {}
    for index, run in enumerate(manifest.itertuples(), start=1):
        print(f"[{index}/{len(manifest)}] Sketch evaluation: {run.run_name}", flush=True)
        model = build_model(config["num_classes"], pretrained=False).to(device)
        state = torch.load(paths[run.run_name], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        output = predict(model, target_dataset, device, num_workers=config["num_workers"])
        outputs[run.run_name] = output
        metrics = classification_metrics(output["labels"], output["predictions"])
        rows.append({"method": run.run_name, "sketch_accuracy": metrics["accuracy"],
                     "sketch_macro_f1": metrics["macro_f1"]})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_results = pd.DataFrame(rows)
    comparison = source_comparison.merge(target_results, on="method", validate="one_to_one")
    baseline_accuracy = comparison.loc[comparison.method == "erm", "sketch_accuracy"].iloc[0]
    comparison["sketch_accuracy_change_vs_erm"] = comparison.sketch_accuracy - baseline_accuracy
    comparison["mean_source_to_sketch_gap"] = comparison.mean_source_accuracy - comparison.sketch_accuracy
    comparison.to_csv(results_dir / "final_comparison_all_runs.csv", index=False)
    main = comparison[comparison.method.isin(MAIN_METHODS)].copy()
    main.to_csv(results_dir / "main_method_comparison.csv", index=False)

    per_class = _per_class_table(outputs, manifest.run_name.tolist())
    per_class.to_csv(results_dir / "sketch_per_class_changes.csv", index=False)
    _save_predictions(outputs, results_dir)
    _save_confusions(outputs, results_dir)
    failures = _save_failure_examples(outputs, per_class, main, results_dir)
    _save_per_class_plot(per_class, results_dir)
    study = _save_sam_study(comparison, manifest, results_dir)
    cross_task = _save_task2_comparison(main, task_dir, results_dir)
    _completion_audit(results_dir, main, per_class, cross_task)
    return {"comparison": comparison, "main_comparison": main, "per_class": per_class,
            "study": study, "cross_task": cross_task, "failure_examples": failures}


def _per_class_table(outputs, methods):
    baseline = outputs["erm"]
    rows = []
    for method in methods:
        output = outputs[method]
        for class_index, class_name in enumerate(CLASSES):
            mask = output["labels"] == class_index
            baseline_mask = baseline["labels"] == class_index
            accuracy = (output["predictions"][mask] == class_index).mean()
            baseline_accuracy = (baseline["predictions"][baseline_mask] == class_index).mean()
            incorrect = output["predictions"][mask & (output["predictions"] != class_index)]
            dominant = "none" if len(incorrect) == 0 else CLASSES[int(np.bincount(incorrect, minlength=len(CLASSES)).argmax())]
            rows.append({"method": method, "class_name": class_name,
                         "sketch_class_accuracy": accuracy,
                         "change_vs_erm": accuracy - baseline_accuracy,
                         "dominant_incorrect_prediction": dominant,
                         "class_count": int(mask.sum())})
    return pd.DataFrame(rows)


def _save_predictions(outputs, results_dir):
    rows = []
    for method, output in outputs.items():
        for path, label, prediction in zip(output["identifiers"], output["labels"], output["predictions"]):
            rows.append({"method": method, "path": path, "true_label": int(label),
                         "true_class": CLASSES[int(label)], "prediction": int(prediction),
                         "predicted_class": CLASSES[int(prediction)]})
    pd.DataFrame(rows).to_csv(results_dir / "sketch_predictions.csv", index=False)


def _save_confusions(outputs, results_dir):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for axis, method in zip(axes, MAIN_METHODS):
        output = outputs[method]
        matrix = confusion_matrix(output["labels"], output["predictions"], normalize="true")
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=CLASSES, yticklabels=CLASSES, cbar=False, ax=axis)
        axis.set_title(method)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    figure.suptitle("Normalized Sketch confusion matrices")
    figure.tight_layout()
    figure.savefig(results_dir / "sketch_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_failure_examples(outputs, per_class, main, results_dir):
    candidates = main[main.method != "erm"].sort_values("sketch_accuracy", ascending=False)
    comparison_method = candidates.iloc[0].method
    baseline, comparison = outputs["erm"], outputs[comparison_method]
    method_rows = per_class[per_class.method == comparison_method]
    classes = [
        ("improvement", method_rows.nlargest(1, "change_vs_erm").iloc[0].class_name),
        ("degradation", method_rows.nsmallest(1, "change_vs_erm").iloc[0].class_name),
    ]
    chosen = []
    for category, class_name in classes:
        class_index = CLASSES.index(class_name)
        class_mask = baseline["labels"] == class_index
        if category == "improvement":
            mask = class_mask & (baseline["predictions"] != class_index) & (comparison["predictions"] == class_index)
            if not mask.any():
                mask = class_mask & (baseline["predictions"] != class_index)
        else:
            mask = class_mask & (baseline["predictions"] == class_index) & (comparison["predictions"] != class_index)
            if not mask.any():
                mask = class_mask & (comparison["predictions"] != class_index)
        for item_index in np.flatnonzero(mask)[:6]:
            chosen.append((category, class_name, int(item_index)))

    figure, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for axis in axes.flat:
        axis.axis("off")
    rows = []
    for axis, (category, class_name, item_index) in zip(axes.flat, chosen):
        path = baseline["identifiers"][item_index]
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"{category}: {class_name}\nERM={CLASSES[baseline['predictions'][item_index]]}\n"
            f"{comparison_method}={CLASSES[comparison['predictions'][item_index]]}", fontsize=8,
        )
        rows.append({"category": category, "class_name": class_name, "path": path,
                     "erm_prediction": CLASSES[baseline["predictions"][item_index]],
                     "comparison_method": comparison_method,
                     "comparison_prediction": CLASSES[comparison["predictions"][item_index]]})
    figure.suptitle(f"Selected Sketch changes: ERM vs {comparison_method}")
    figure.tight_layout()
    figure.savefig(results_dir / "selected_sketch_failure_cases.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "selected_sketch_failure_cases.csv", index=False)
    return table


def _save_per_class_plot(per_class, results_dir):
    data = per_class[per_class.method.isin(["dan_dg", "sam"])]
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.barplot(data=data, x="class_name", y="change_vs_erm", hue="method", ax=axis)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_title("Sketch per-class accuracy change relative to ERM")
    figure.tight_layout()
    figure.savefig(results_dir / "sketch_per_class_accuracy_changes.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_sam_study(comparison, manifest, results_dir):
    study_names = ["sam_rho_0p01", "sam", "sam_rho_0p1"]
    study = comparison[comparison.method.isin(study_names)].copy()
    rho_map = manifest.set_index("run_name").rho
    study["rho"] = study.method.map(rho_map)
    study = study.sort_values("rho")
    study.to_csv(results_dir / "sam_radius_study.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for metric, axis in zip(["mean_source_macro_f1", "sharpness_increase", "sketch_accuracy"], axes):
        sns.lineplot(data=study, x="rho", y=metric, marker="o", ax=axis)
        axis.set_title(metric.replace("_", " "))
    figure.suptitle("Controlled SAM radius study (main setting rho=0.05)")
    figure.tight_layout()
    figure.savefig(results_dir / "sam_radius_study.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return study


def _save_task2_comparison(main, task_dir, results_dir):
    task2_path = task_dir.parent / "domain-adaptation" / "results" / "main_method_comparison.csv"
    rows = []
    for method in ["erm", "dan_dg"]:
        row = main[main.method == method].iloc[0]
        rows.append({"task": "Task 3 DG", "method": method,
                     "target_access_during_training": False,
                     "sketch_accuracy": row.sketch_accuracy,
                     "sketch_macro_f1": row.sketch_macro_f1})
    if task2_path.exists():
        task2 = pd.read_csv(task2_path)
        for method in ["source_only", "dan"]:
            selected = task2[task2.method == method]
            if not selected.empty:
                row = selected.iloc[0]
                rows.append({"task": "Task 2 UDA", "method": method,
                             "target_access_during_training": method == "dan",
                             "sketch_accuracy": row.target_accuracy,
                             "sketch_macro_f1": row.target_macro_f1})
    else:
        print(f"WARNING: Task 2 final comparison not found at {task2_path}; cross-task table is incomplete.", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "task2_dan_vs_task3_dan_dg.csv", index=False)
    return table


def _completion_audit(results_dir, main, per_class, cross_task):
    required = [
        "source_split_summary.json", "run_manifest.csv", "training_history_all_runs.csv",
        "main_training_curves.png", "sam_radius_training_curves.png",
        "source_validation_comparison.csv", "source_domain_separability.csv", "sharpness_proxy.csv",
        "source_diagnostics.png", "main_method_comparison.csv", "final_comparison_all_runs.csv",
        "sketch_per_class_changes.csv", "sketch_predictions.csv", "sketch_confusion_matrices.png",
        "selected_sketch_failure_cases.csv", "selected_sketch_failure_cases.png",
        "sketch_per_class_accuracy_changes.png", "sam_radius_study.csv", "sam_radius_study.png",
        "task2_dan_vs_task3_dan_dg.csv",
    ]
    missing = [name for name in required if not (results_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing required Task 3 outputs: {missing}")
    payload = {
        "required_outputs": required,
        "main_methods": MAIN_METHODS,
        "main_rows": len(main),
        "per_class_rows": len(per_class),
        "task2_comparison_available": set(cross_task.method) >= {"source_only", "dan"},
        "sketch_loaded_only_after_configuration_lock": True,
        "sketch_used_during_training_selection_or_source_diagnostics": False,
    }
    (results_dir / "completion_audit.json").write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Locked final Task 3 evaluation; this script loads Sketch labels")
    parser.add_argument("--pacs-root")
    parser.add_argument("--configurations-locked", action="store_true")
    args = parser.parse_args()
    output = evaluate_sketch_suite(args.pacs_root, configurations_locked=args.configurations_locked)
    print(output["main_comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
