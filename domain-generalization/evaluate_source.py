from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parent
for path in [TASK_DIR, REPO_ROOT]:
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from configs.load_config import load_config  # noqa: E402
from evaluation.sharpness import fixed_validation_batches, local_sharpness_proxy  # noqa: E402
from evaluation.source_domain_separability import source_domain_separability  # noqa: E402
from models.backbone import build_model  # noqa: E402
from selection.source_validation import aggregate_source_metrics, evaluate_source_domains  # noqa: E402
from shared.pacs import find_pacs_root  # noqa: E402
from shared.pacs_protocol import build_or_load_protocol  # noqa: E402
from train import MAIN_METHODS, checkpoint_paths  # noqa: E402
from training.loaders import make_source_loaders  # noqa: E402


def evaluate_source_suite(pacs_root=None, task_dir=TASK_DIR):
    """Run all model-selection diagnostics using Photo/Art/Cartoon only."""
    task_dir = Path(task_dir).resolve()
    results_dir = task_dir / "results"
    pacs_root = find_pacs_root(pacs_root)
    config = load_config(task_dir, "erm")
    split_path = task_dir.parent / "shared" / "splits" / "pacs_sketch_seed6304.json"
    protocol = build_or_load_protocol(pacs_root, split_path, config["seed"])
    _source_loaders, val_datasets, _eval_transform = make_source_loaders(protocol, config)
    diagnostic_batches = fixed_validation_batches(val_datasets, config["seed"], examples_per_domain=32)
    manifest_path = results_dir / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Run train_suite before source-only evaluation.")
    manifest = pd.read_csv(manifest_path)
    paths = checkpoint_paths(task_dir)
    missing = [run for run in manifest.run_name if not paths[run].exists()]
    if missing:
        raise FileNotFoundError(f"Missing fixed checkpoints: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, separability_rows, sharpness_rows = [], [], []
    for index, run in enumerate(manifest.itertuples(), start=1):
        print(f"[{index}/{len(manifest)}] source diagnostics: {run.run_name}", flush=True)
        model = build_model(config["num_classes"], pretrained=False).to(device)
        state = torch.load(paths[run.run_name], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        metrics_by_domain, outputs = evaluate_source_domains(
            model, val_datasets, device, config["num_workers"], return_features=True,
        )
        row = {"method": run.run_name, "main_comparison": bool(run.main_comparison)}
        for domain, metrics in metrics_by_domain.items():
            row[f"{domain}_accuracy"] = metrics["accuracy"]
            row[f"{domain}_macro_f1"] = metrics["macro_f1"]
        row.update(aggregate_source_metrics(metrics_by_domain))
        features = {domain: output["features"].numpy() for domain, output in outputs.items()}
        separability, count = source_domain_separability(features, config["seed"])
        sharpness = local_sharpness_proxy(model, diagnostic_batches, device, radius=0.05)
        row["source_domain_separability"] = separability
        row["sharpness_increase"] = sharpness["sharpness_increase"]
        rows.append(row)
        separability_rows.append({"method": run.run_name, "heldout_accuracy": separability,
                                  "balanced_examples_per_domain": count, "chance_accuracy": 1 / 3})
        sharpness_rows.append({"method": run.run_name, **sharpness, "examples_per_source_domain": 32})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = pd.DataFrame(rows)
    separability_table = pd.DataFrame(separability_rows)
    sharpness_table = pd.DataFrame(sharpness_rows)
    comparison.to_csv(results_dir / "source_validation_comparison.csv", index=False)
    separability_table.to_csv(results_dir / "source_domain_separability.csv", index=False)
    sharpness_table.to_csv(results_dir / "sharpness_proxy.csv", index=False)
    _plot_source_diagnostics(comparison, results_dir)
    print("Source-only diagnostics complete; Sketch has not been loaded.", flush=True)
    return {"comparison": comparison, "separability": separability_table, "sharpness": sharpness_table}


def _plot_source_diagnostics(comparison, results_dir):
    main = comparison[comparison.method.isin(MAIN_METHODS)]
    long = main.melt(
        id_vars="method",
        value_vars=["mean_source_macro_f1", "worst_source_macro_f1",
                    "source_domain_separability", "sharpness_increase"],
        var_name="metric",
        value_name="value",
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, metric in zip(axes.flat, long.metric.unique()):
        sns.barplot(data=long[long.metric == metric], x="method", y="value", ax=axis)
        axis.set_title(metric.replace("_", " "))
    figure.suptitle("Task 3 source-only model-selection diagnostics")
    figure.tight_layout()
    figure.savefig(results_dir / "source_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Task 3 source-only diagnostics; does not load Sketch")
    parser.add_argument("--pacs-root")
    args = parser.parse_args()
    output = evaluate_source_suite(args.pacs_root)
    print(output["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()

