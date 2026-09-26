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
from torchvision.datasets import CIFAR10

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path: sys.path.insert(0, str(TASK_DIR))

from configs.load_config import load_config  # noqa: E402
from data.cifar10 import CIFARArrayDataset, make_transform  # noqa: E402
from data.make_splits import make_or_load_split  # noqa: E402
from models.resnet_cifar import CIFARResNet18, PROSERResNet18  # noqa: E402
from training.trainer import train_closed_set, train_proser  # noqa: E402


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def prepare_known_data(task_dir=TASK_DIR, download=True, method="vanilla"):
    config = load_config(task_dir, method)
    raw = Path(task_dir) / "data/raw"
    base = CIFAR10(raw, train=True, download=download)
    split_path = Path(task_dir) / "data/splits/cifar10_seed6304.json"
    train_idx, val_idx = make_or_load_split(base.targets, split_path, config["seed"], config["validation_fraction"])
    train_transform = make_transform(
        True, randaugment=config.get("randaugment", False),
        num_ops=config.get("randaugment_num_ops", 2), magnitude=config.get("randaugment_magnitude", 9),
    )
    eval_transform = make_transform(False)
    train_dataset = CIFARArrayDataset(base.data, base.targets, train_idx, train_transform)
    val_dataset = CIFARArrayDataset(base.data, base.targets, val_idx, eval_transform)
    return base, train_idx, val_idx, train_dataset, val_dataset


def train_all(task_dir=TASK_DIR, force=False):
    task_dir = Path(task_dir).resolve(); results = task_dir / "results"; results.mkdir(parents=True, exist_ok=True)
    checkpoints = {name: task_dir / f"checkpoints/{name}.pt" for name in ["vanilla", "gcsc", "proser"]}
    histories = []
    selections = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for method in ["vanilla", "gcsc"]:
        config = load_config(task_dir, method); seed_everything(config["seed"])
        _base, _train_idx, _val_idx, train_dataset, val_dataset = prepare_known_data(task_dir, method=method)
        model = CIFARResNet18(config["num_known_classes"]).to(device)
        history_path = results / f"{method}_training_history.csv"
        if checkpoints[method].exists() and history_path.exists() and not force:
            selected_state = torch.load(checkpoints[method], map_location=device, weights_only=False)
            model.load_state_dict(selected_state["model"])
            history = pd.read_csv(history_path)
        else:
            model, history, selected_state = train_closed_set(
                model, train_dataset, val_dataset, config, device, checkpoints[method])
            history.to_csv(history_path, index=False)
        selections[method] = {"checkpoint": str(checkpoints[method]),
                              "selected_epoch": int(selected_state["epoch"]),
                              "validation_accuracy": float(selected_state["validation_accuracy"]),
                              "selection_data": "CIFAR-10 validation only"}
        histories.append(history)

    config = load_config(task_dir, "proser"); seed_everything(config["seed"])
    _base, _train_idx, _val_idx, train_dataset, val_dataset = prepare_known_data(task_dir, method="vanilla")
    model = PROSERResNet18(config["num_known_classes"], config["dummy_classifiers"]).to(device)
    vanilla_state = torch.load(checkpoints["vanilla"], map_location=device, weights_only=False)["model"]
    incompatible = model.load_state_dict(vanilla_state, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {"dummy_classifier.weight", "dummy_classifier.bias"}:
        raise RuntimeError(f"Unexpected Vanilla-to-PROSER initialization mismatch: {incompatible}")
    history_path = results / "proser_training_history.csv"
    if checkpoints["proser"].exists() and history_path.exists() and not force:
        selected_state = torch.load(checkpoints["proser"], map_location=device, weights_only=False)
        model.load_state_dict(selected_state["model"])
        history = pd.read_csv(history_path)
    else:
        model, history, selected_state = train_proser(
            model, train_dataset, val_dataset, config, device, checkpoints["proser"])
        history.to_csv(history_path, index=False)
    selections["proser"] = {"checkpoint": str(checkpoints["proser"]),
                            "selected_epoch": int(selected_state["epoch"]),
                            "validation_accuracy": float(selected_state["validation_accuracy"]),
                            "selection_data": "CIFAR-10 validation only"}
    histories.append(history)

    combined = pd.concat(histories, ignore_index=True, sort=False)
    combined.to_csv(results / "training_history_all_models.csv", index=False)
    _plot_histories(combined, results)
    manifest = {"cifar100_used": False, "models": selections}
    (results / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame.from_dict(selections, orient="index").rename_axis("method").reset_index().to_csv(
        results / "checkpoint_selection.csv", index=False)
    return {"device": device, "checkpoints": checkpoints, "histories": combined, "selections": selections}


def _plot_histories(history, results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.lineplot(data=history, x="epoch", y="train_loss", hue="method", ax=axes[0])
    sns.lineplot(data=history, x="epoch", y="train_accuracy", hue="method", ax=axes[1])
    sns.lineplot(data=history, x="epoch", y="validation_accuracy", hue="method", ax=axes[2])
    axes[0].set_title("Training objective"); axes[1].set_title("Known training accuracy"); axes[2].set_title("CIFAR-10 validation accuracy")
    fig.tight_layout(); fig.savefig(results / "training_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    proser = history.query("method == 'proser'")
    fig, axis = plt.subplots(figsize=(9, 5))
    for column in ["known_ce", "classifier_boundary_loss", "data_placeholder_loss"]:
        axis.plot(proser.epoch, proser[column], label=column)
    axis.legend(); axis.set_xlabel("Epoch"); axis.set_title("PROSER loss components")
    fig.tight_layout(); fig.savefig(results / "proser_loss_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Task 4 models using CIFAR-10 only")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = train_all(force=args.force)
    print(output["histories"].groupby("method").tail(1).to_string(index=False))
    print("Training complete. CIFAR-100 was not loaded.")


if __name__ == "__main__": main()
