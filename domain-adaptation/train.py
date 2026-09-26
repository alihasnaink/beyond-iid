from __future__ import annotations

import argparse
import json
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
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.load_config import load_config  # noqa: E402
from models.backbone import build_model  # noqa: E402
from shared.pacs import find_pacs_root  # noqa: E402
from shared.pacs_protocol import build_or_load_protocol, protocol_summary  # noqa: E402
from training.loaders import make_loaders  # noqa: E402
from training.trainer import train_method  # noqa: E402


MAIN_METHODS = ["source_only", "dan", "dann", "cdan"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def experiment_specs(task_dir, include_study=True):
    specs = [(name, load_config(task_dir, name), True) for name in MAIN_METHODS]
    if include_study:
        for strength in [0.25, 0.5]:  # 1.0 is the required main DANN run.
            config = load_config(task_dir, "dann", {"grl_max_strength": strength})
            specs.append((f"dann_alpha_{str(strength).replace('.', 'p')}", config, False))
    return specs


def checkpoint_paths(task_dir):
    shared_checkpoint = task_dir.parent / "shared" / "checkpoints" / "source_only_pacs_sketch_seed6304.pt"
    paths = {"source_only": shared_checkpoint}
    for run_name, _, _ in experiment_specs(task_dir, include_study=True):
        paths.setdefault(run_name, task_dir / "results" / "checkpoints" / f"{run_name}.pt")
    return paths


def train_suite(pacs_root=None, task_dir=TASK_DIR, include_study=True, force=False):
    task_dir = Path(task_dir).resolve()
    pacs_root = find_pacs_root(pacs_root)
    results_dir = task_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    split_path = task_dir.parent / "shared" / "splits" / "pacs_sketch_seed6304.json"
    base = load_config(task_dir, "source_only")
    protocol = build_or_load_protocol(pacs_root, split_path, base["seed"])
    (results_dir / "source_split_summary.json").write_text(json.dumps(protocol_summary(protocol), indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = checkpoint_paths(task_dir)
    histories, run_records = [], []
    specs = experiment_specs(task_dir, include_study)
    train_counts = {domain: len(records) for domain, records in protocol["train"].items()}
    val_counts = {domain: len(records) for domain, records in protocol["val"].items()}
    print(
        f"PACS root: {pacs_root} | device: {device} | runs: {len(specs)}\n"
        f"Source train images: {train_counts} | source validation images: {val_counts}",
        flush=True,
    )

    for run_index, (run_name, config, is_main) in enumerate(specs, start=1):
        config["run_name"] = run_name
        method = config["name"]
        checkpoint = paths[run_name]
        history_path = results_dir / "histories" / f"{run_name}.csv"
        seed_everything(config["seed"])
        source_loaders, target_loader, val_datasets, _ = make_loaders(
            protocol, pacs_root, config, include_target=method != "source_only",
        )
        model = build_model(config["num_classes"]).to(device)
        if checkpoint.exists() and history_path.exists() and not force:
            print(f"[{run_index}/{len(specs)}] {run_name}: loading completed checkpoint", flush=True)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            history = pd.read_csv(history_path)
        else:
            if checkpoint.exists() and not history_path.exists():
                print(f"[{run_index}/{len(specs)}] {run_name}: incomplete prior run found; restarting", flush=True)
            else:
                print(f"[{run_index}/{len(specs)}] {run_name}: training", flush=True)
            model, history, state = train_method(
                model, method, config, source_loaders, target_loader, val_datasets, device, checkpoint,
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(history_path, index=False)
        histories.append(history)
        run_records.append({"run_name": run_name, "method": method, "main_comparison": is_main,
                            "checkpoint": str(checkpoint), "grl_max_strength": config.get("grl_max_strength"),
                            "alignment_weight": config.get("alignment_weight")})

    combined = pd.concat(histories, ignore_index=True)
    combined.to_csv(results_dir / "training_history_all_runs.csv", index=False)
    pd.DataFrame(run_records).to_csv(results_dir / "run_manifest.csv", index=False)
    plot_training_curves(combined, results_dir)
    return {"pacs_root": pacs_root, "protocol": protocol, "device": device,
            "histories": combined, "run_manifest": pd.DataFrame(run_records), "checkpoints": paths}


def plot_training_curves(history, results_dir):
    main = history[history.method.isin(MAIN_METHODS)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for column, axis, title in [
        ("classification_loss", axes[0, 0], "Source classification loss"),
        ("alignment_loss", axes[0, 1], "Alignment/domain loss"),
        ("mean_source_val_macro_f1", axes[1, 0], "Mean source-validation macro-F1"),
        ("domain_accuracy", axes[1, 1], "Training domain-discriminator accuracy"),
    ]:
        sns.lineplot(data=main, x="epoch", y=column, hue="method", marker="o", ax=axis)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(results_dir / "main_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    study = history[history.method.str.startswith("dann")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for column, axis in zip(["classification_loss", "alignment_loss", "mean_source_val_macro_f1"], axes):
        sns.lineplot(data=study, x="epoch", y=column, hue="method", marker="o", ax=axis)
        axis.set_title(column.replace("_", " "))
    fig.tight_layout()
    fig.savefig(results_dir / "dann_strength_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train leakage-safe PACS Task 2 experiments")
    parser.add_argument("--pacs-root")
    parser.add_argument("--no-study", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = train_suite(args.pacs_root, include_study=not args.no_study, force=args.force)
    print(output["run_manifest"].to_string(index=False))
    print("Training complete. Sketch labels were not loaded.")


if __name__ == "__main__":
    main()
