from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parent
for path in [TASK_DIR, REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.load_config import load_config  # noqa: E402
from evaluation.class_analysis import per_class_table, save_confusion_matrices, save_failure_examples  # noqa: E402
from evaluation.domain_separability import domain_separability  # noqa: E402
from evaluation.metrics import classification_metrics, predict  # noqa: E402
from models.backbone import build_model  # noqa: E402
from shared.pacs import CLASSES, PACSLabeledDataset, find_pacs_root, labeled_records  # noqa: E402
from shared.pacs_protocol import SOURCE_DOMAINS, build_or_load_protocol  # noqa: E402
from train import MAIN_METHODS, checkpoint_paths  # noqa: E402
from training.loaders import make_transforms  # noqa: E402


def evaluate_suite(pacs_root=None, task_dir=TASK_DIR):
    task_dir = Path(task_dir).resolve()
    results_dir = task_dir / "results"
    pacs_root = find_pacs_root(pacs_root)
    config = load_config(task_dir, "source_only")
    split_path = task_dir.parent / "shared" / "splits" / "pacs_sketch_seed6304.json"
    protocol = build_or_load_protocol(pacs_root, split_path, config["seed"])
    _, eval_transform = make_transforms(config)
    source_val = {domain: PACSLabeledDataset(protocol["val"][domain], eval_transform) for domain in SOURCE_DOMAINS}
    # This is the first place in the workflow where Sketch labels are loaded.
    target_dataset = PACSLabeledDataset(labeled_records(pacs_root, "sketch"), eval_transform)
    run_manifest = pd.read_csv(results_dir / "run_manifest.csv")
    paths = checkpoint_paths(task_dir)
    missing = [run for run in run_manifest.run_name if not paths[run].exists()]
    if missing:
        raise FileNotFoundError(f"All settings must be trained/fixed before target evaluation. Missing: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, target_outputs, separability_rows = [], {}, []
    for run in run_manifest.itertuples():
        model = build_model(config["num_classes"]).to(device)
        state = torch.load(paths[run.run_name], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        source_features, row = [], {"method": run.run_name, "main_comparison": bool(run.main_comparison)}
        source_accuracies, source_f1s = [], []
        for domain, dataset in source_val.items():
            output = predict(model, dataset, device, num_workers=config["num_workers"], return_features=True)
            metrics = classification_metrics(output["labels"], output["predictions"])
            row[f"{domain}_accuracy"], row[f"{domain}_macro_f1"] = metrics["accuracy"], metrics["macro_f1"]
            source_accuracies.append(metrics["accuracy"]); source_f1s.append(metrics["macro_f1"])
            source_features.append(output["features"].numpy())
        row["mean_source_accuracy"] = sum(source_accuracies) / len(source_accuracies)
        row["mean_source_macro_f1"] = sum(source_f1s) / len(source_f1s)

        target = predict(model, target_dataset, device, num_workers=config["num_workers"], return_features=True)
        target_outputs[run.run_name] = target
        target_metrics = classification_metrics(target["labels"], target["predictions"])
        row["target_accuracy"], row["target_macro_f1"] = target_metrics["accuracy"], target_metrics["macro_f1"]
        import numpy as np
        source_feature_array = np.concatenate(source_features)
        separability, count = domain_separability(source_feature_array, target["features"].numpy(), config["seed"])
        row["domain_separability"] = separability
        separability_rows.append({"method": run.run_name, "balanced_examples_per_domain": count, "heldout_accuracy": separability})
        rows.append(row)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    comparison = pd.DataFrame(rows)
    baseline_accuracy = comparison.query("method == 'source_only'").target_accuracy.iloc[0]
    comparison["target_accuracy_change_vs_source_only"] = comparison.target_accuracy - baseline_accuracy
    comparison["source_to_target_gap"] = comparison.mean_source_accuracy - comparison.target_accuracy
    comparison.to_csv(results_dir / "final_comparison_all_runs.csv", index=False)
    comparison.query("main_comparison").to_csv(results_dir / "main_method_comparison.csv", index=False)
    pd.DataFrame(separability_rows).to_csv(results_dir / "domain_separability.csv", index=False)
    main_plot = comparison.query("main_comparison")
    fig, axis = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=main_plot, x="domain_separability", y="target_accuracy", hue="method", s=130, ax=axis)
    for row in main_plot.itertuples():
        axis.annotate(row.method, (row.domain_separability, row.target_accuracy), xytext=(5, 4), textcoords="offset points")
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="chance separability")
    axis.set_title("Residual domain information vs Sketch recognition")
    fig.tight_layout()
    fig.savefig(results_dir / "domain_separability_vs_target_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    main_outputs = {method: target_outputs[method] for method in MAIN_METHODS}
    class_results = per_class_table(main_outputs, CLASSES)
    class_results.to_csv(results_dir / "target_per_class_changes.csv", index=False)
    extremes = []
    for method in [name for name in MAIN_METHODS if name != "source_only"]:
        method_rows = class_results.query("method == @method")
        extremes.append(method_rows.nlargest(2, "change_vs_source_only").assign(transfer_type="largest_improvement"))
        extremes.append(method_rows.nsmallest(2, "change_vs_source_only").assign(transfer_type="largest_degradation"))
    pd.concat(extremes, ignore_index=True).to_csv(results_dir / "class_transfer_extremes.csv", index=False)
    save_confusion_matrices(main_outputs, CLASSES, MAIN_METHODS, results_dir)
    selected_method, failure_examples = save_failure_examples(main_outputs, class_results, CLASSES, results_dir, config["seed"])
    _save_per_class_plot(class_results, results_dir)
    study = _save_study(comparison, run_manifest, results_dir)
    _save_predictions(target_outputs, results_dir)
    _completion_audit(results_dir, comparison, class_results, selected_method)
    return {"comparison": comparison, "main_comparison": comparison.query("main_comparison").copy(),
            "per_class": class_results, "study": study, "failure_examples": failure_examples}


def _save_predictions(outputs, results_dir):
    rows = []
    for method, output in outputs.items():
        for path, label, prediction in zip(output["identifiers"], output["labels"], output["predictions"]):
            rows.append({"method": method, "path": path, "true_label": int(label), "true_class": CLASSES[int(label)],
                         "prediction": int(prediction), "predicted_class": CLASSES[int(prediction)]})
    pd.DataFrame(rows).to_csv(results_dir / "target_predictions.csv", index=False)


def _save_per_class_plot(table, results_dir):
    data = table.query("method != 'source_only'")
    fig, axis = plt.subplots(figsize=(12, 5))
    sns.barplot(data=data, x="class_name", y="change_vs_source_only", hue="method", ax=axis)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_title("Sketch per-class accuracy change relative to Source-only")
    fig.tight_layout()
    fig.savefig(results_dir / "target_per_class_accuracy_changes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_study(comparison, run_manifest, results_dir):
    study_names = ["dann_alpha_0p25", "dann_alpha_0p5", "dann"]
    study = comparison[comparison.method.isin(study_names)].copy()
    strength_map = run_manifest.set_index("run_name").grl_max_strength
    study["grl_max_strength"] = study.method.map(strength_map)
    study = study.sort_values("grl_max_strength")
    study.to_csv(results_dir / "dann_strength_study.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for metric, axis in zip(["mean_source_macro_f1", "target_accuracy", "domain_separability"], axes):
        sns.lineplot(data=study, x="grl_max_strength", y=metric, marker="o", ax=axis)
        axis.set_title(metric.replace("_", " "))
    fig.suptitle("Controlled DANN alignment-strength study")
    fig.tight_layout()
    fig.savefig(results_dir / "dann_strength_study.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return study


def _completion_audit(results_dir, comparison, class_results, selected_method):
    required = [
        "source_split_summary.json", "run_manifest.csv", "training_history_all_runs.csv",
        "main_training_curves.png", "dann_strength_training_curves.png", "main_method_comparison.csv",
        "final_comparison_all_runs.csv", "domain_separability.csv", "target_per_class_changes.csv",
        "domain_separability_vs_target_accuracy.png",
        "class_transfer_extremes.csv",
        "target_per_class_accuracy_changes.png", "target_confusion_matrices.png",
        "selected_target_failure_cases.png", "selected_target_failure_cases.csv", "target_predictions.csv",
        "dann_strength_study.csv", "dann_strength_study.png",
    ]
    missing = [name for name in required if not (results_dir / name).exists()]
    if missing: raise RuntimeError(f"Missing required outputs: {missing}")
    payload = {"required_outputs": required, "main_methods": MAIN_METHODS,
               "main_rows": len(comparison.query("main_comparison")), "per_class_rows": len(class_results),
               "failure_case_comparison_method": selected_method,
               "target_labels_used_only_in_final_evaluation": True}
    (results_dir / "completion_audit.json").write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Final Task 2 evaluation; this script loads Sketch labels")
    parser.add_argument("--pacs-root")
    args = parser.parse_args()
    outputs = evaluate_suite(args.pacs_root)
    print(outputs["main_comparison"].to_string(index=False))
    print(f"Final results saved to {TASK_DIR / 'results'}")


if __name__ == "__main__":
    main()
