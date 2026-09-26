from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
from models.backbone import build_model  # noqa: E402
from shared.pacs import find_pacs_root  # noqa: E402
from shared.pacs_protocol import build_or_load_protocol, protocol_summary  # noqa: E402
from training.loaders import make_source_loaders  # noqa: E402
from training.trainer import train_method  # noqa: E402


MAIN_METHODS = ["erm", "dan_dg", "sam"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def experiment_specs(task_dir, include_study=True):
    specs = [
        ("erm", load_config(task_dir, "erm"), True),
        ("dan_dg", load_config(task_dir, "dan_dg"), True),
        ("sam", load_config(task_dir, "sam"), True),
    ]
    if include_study:
        for rho in [0.01, 0.1]:  # rho=0.05 is the required main SAM run.
            config = load_config(task_dir, "sam", {"rho": rho})
            specs.append((f"sam_rho_{str(rho).replace('.', 'p')}", config, False))
    return specs


def checkpoint_paths(task_dir):
    paths = {"erm": task_dir.parent / "shared" / "checkpoints" / "source_only_pacs_sketch_seed6304.pt"}
    for run_name, _config, _is_main in experiment_specs(task_dir, include_study=True):
        paths.setdefault(run_name, task_dir / "results" / "checkpoints" / f"{run_name}.pt")
    return paths


def _completion_path(results_dir, run_name):
    return results_dir / "completed" / f"{run_name}.json"


def _atomic_json_write(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _load_erm_history(task_dir):
    history_path = task_dir.parent / "domain-adaptation" / "results" / "histories" / "source_only.csv"
    if not history_path.exists():
        raise FileNotFoundError(
            "Task 3 must reuse the completed Task 2 ERM run, but its history is missing: "
            f"{history_path}. Finish Task 2 train_suite first."
        )
    history = pd.read_csv(history_path).copy()
    history["method"] = "erm"
    return history


def train_suite(pacs_root=None, task_dir=TASK_DIR, include_study=True, force=False):
    """Train using source domains only. This function never enumerates or loads Sketch."""
    task_dir = Path(task_dir).resolve()
    pacs_root = find_pacs_root(pacs_root)
    results_dir = task_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    config_base = load_config(task_dir, "erm")
    split_path = task_dir.parent / "shared" / "splits" / "pacs_sketch_seed6304.json"
    protocol = build_or_load_protocol(pacs_root, split_path, config_base["seed"])
    (results_dir / "source_split_summary.json").write_text(json.dumps(protocol_summary(protocol), indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = checkpoint_paths(task_dir)
    if not paths["erm"].exists():
        raise FileNotFoundError(
            "The shared Task 2 ERM checkpoint is required and must not be retrained by Task 3: "
            f"{paths['erm']}. Finish Task 2 train_suite first."
        )

    specs = experiment_specs(task_dir, include_study)
    train_counts = {domain: len(records) for domain, records in protocol["train"].items()}
    val_counts = {domain: len(records) for domain, records in protocol["val"].items()}
    print(
        f"Task 3 source-only training | device: {device} | runs: {len(specs)}\n"
        f"Source train images: {train_counts} | source validation images: {val_counts}\n"
        "Sketch images are not loaded by this stage.",
        flush=True,
    )

    histories, run_records = [], []
    for run_index, (run_name, config, is_main) in enumerate(specs, start=1):
        config["run_name"] = run_name
        checkpoint = paths[run_name]
        history_path = results_dir / "histories" / f"{run_name}.csv"
        completion_path = _completion_path(results_dir, run_name)
        if run_name == "erm":
            print(f"[{run_index}/{len(specs)}] erm: reusing unchanged Task 2 checkpoint", flush=True)
            history = _load_erm_history(task_dir)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(history_path, index=False)
            _atomic_json_write({"checkpoint": str(checkpoint), "reused_from_task2": True}, completion_path)
        elif checkpoint.exists() and history_path.exists() and completion_path.exists() and not force:
            print(f"[{run_index}/{len(specs)}] {run_name}: loading completed run", flush=True)
            history = pd.read_csv(history_path)
        else:
            if checkpoint.exists() and not completion_path.exists():
                print(f"[{run_index}/{len(specs)}] {run_name}: incomplete run found; restarting", flush=True)
            else:
                print(f"[{run_index}/{len(specs)}] {run_name}: training", flush=True)
            seed_everything(config["seed"])
            source_loaders, val_datasets, _ = make_source_loaders(protocol, config)
            model = build_model(config["num_classes"], pretrained=True).to(device)
            try:
                model, history, state = train_method(
                    model,
                    config["name"],
                    config,
                    source_loaders,
                    val_datasets,
                    device,
                    checkpoint,
                    history_path,
                )
                history.to_csv(history_path, index=False)
                _atomic_json_write(
                    {"checkpoint": str(checkpoint), "best_epoch": state["epoch"],
                     "mean_source_val_macro_f1": state["mean_source_val_macro_f1"]},
                    completion_path,
                )
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        histories.append(history)
        run_records.append({
            "run_name": run_name,
            "method": config["name"],
            "main_comparison": is_main,
            "checkpoint": str(checkpoint),
            "alignment_weight": config.get("alignment_weight"),
            "rho": config.get("rho"),
            "uses_sketch_during_training_or_selection": False,
        })

    combined = pd.concat(histories, ignore_index=True)
    manifest = pd.DataFrame(run_records)
    combined.to_csv(results_dir / "training_history_all_runs.csv", index=False)
    manifest.to_csv(results_dir / "run_manifest.csv", index=False)
    plot_training_curves(combined, results_dir)
    return {"pacs_root": pacs_root, "protocol": protocol, "device": device,
            "histories": combined, "run_manifest": manifest, "checkpoints": paths}


def plot_training_curves(history, results_dir):
    main = history[history.method.isin(MAIN_METHODS)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for column, axis, title in [
        ("classification_loss", axes[0], "Classification loss"),
        ("alignment_loss", axes[1], "Pairwise source MMD"),
        ("mean_source_val_macro_f1", axes[2], "Mean source-validation macro-F1"),
    ]:
        sns.lineplot(data=main, x="epoch", y=column, hue="method", marker="o", ax=axis)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(results_dir / "main_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    study = history[history.method.isin(["sam", "sam_rho_0p01", "sam_rho_0p1"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for column, axis in zip(["classification_loss", "mean_source_val_macro_f1"], axes):
        sns.lineplot(data=study, x="epoch", y=column, hue="method", marker="o", ax=axis)
        axis.set_title(column.replace("_", " "))
    fig.tight_layout()
    fig.savefig(results_dir / "sam_radius_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Task 3 without loading Sketch")
    parser.add_argument("--pacs-root")
    parser.add_argument("--no-study", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = train_suite(args.pacs_root, include_study=not args.no_study, force=args.force)
    print(output["run_manifest"].to_string(index=False))


if __name__ == "__main__":
    main()

