from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from configs.load_config import load_config  # noqa: E402
from data.cifar10 import CIFARArrayDataset, make_transform  # noqa: E402
from data.cifar100_unknowns import build_unknown_datasets  # noqa: E402
from data.make_splits import make_or_load_split  # noqa: E402
from models.resnet_cifar import CIFARResNet18, PROSERResNet18  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def extract_dataset(model, dataset, device, num_workers=2, proser=False):
    model.eval()
    output = {"logits": [], "features": [], "labels": [], "identifiers": [], "class_names": []}
    if proser:
        output["dummy_logits"] = []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    for images, labels, identifiers, class_names in loader:
        images = images.to(device, non_blocking=True)
        if proser:
            logits, dummy, features = model(images, return_dummy=True, return_features=True)
            output["dummy_logits"].append(dummy.cpu())
        else:
            logits, features = model(images, return_features=True)
        output["logits"].append(logits.cpu())
        output["features"].append(features.cpu())
        output["labels"].append(labels.cpu())
        output["identifiers"].extend(identifiers)
        output["class_names"].extend(class_names)
    for key in ["logits", "features", "labels"] + (["dummy_logits"] if proser else []):
        output[key] = torch.cat(output[key])
    return output


def _load_model(name, checkpoint, device, task_dir):
    config = load_config(task_dir, name)
    if name == "proser":
        model = PROSERResNet18(config["num_known_classes"], config["dummy_classifiers"])
    else:
        model = CIFARResNet18(config["num_known_classes"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), config, state


def extract_all_outputs(task_dir=TASK_DIR, download=True, force=False):
    """Cache outputs only after the three selected checkpoints already exist.

    The order here is deliberate: score code and checkpoints are fixed before the
    CIFAR-100 test set is instantiated. CIFAR-100 train is never instantiated.
    """
    task_dir = Path(task_dir).resolve()
    cache_dir, results_dir = task_dir / "cache", task_dir / "results"
    cache_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {name: task_dir / "checkpoints" / f"{name}.pt" for name in ["vanilla", "gcsc", "proser"]}
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Train and select all CIFAR-10 checkpoints before unknown evaluation: " + ", ".join(missing))
    checkpoint_hashes = {name: _sha256(path) for name, path in checkpoints.items()}
    lock_path = results_dir / "protocol_lock.json"
    old_lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    old_hashes = old_lock.get("checkpoint_sha256", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known_config = load_config(task_dir, "vanilla")
    raw = task_dir / "data/raw"
    cifar10_train = CIFAR10(raw, train=True, download=download)
    train_idx, val_idx = make_or_load_split(
        cifar10_train.targets, task_dir / "data/splits/cifar10_seed6304.json",
        known_config["seed"], known_config["validation_fraction"],
    )
    transform = make_transform(False)
    known_datasets = {
        "train": CIFARArrayDataset(cifar10_train.data, cifar10_train.targets, train_idx, transform),
        "validation": CIFARArrayDataset(cifar10_train.data, cifar10_train.targets, val_idx, transform),
    }
    cifar10_test = CIFAR10(raw, train=False, download=download)
    known_datasets["test"] = CIFARArrayDataset(
        cifar10_test.data, cifar10_test.targets, range(len(cifar10_test)), transform,
        identifier_prefix="cifar10_test",
    )

    models = {}
    selection = {}
    for name, checkpoint in checkpoints.items():
        models[name], config, state = _load_model(name, checkpoint, device, task_dir)
        selection[name] = {"epoch": int(state["epoch"]), "validation_accuracy": float(state["validation_accuracy"])}
        cache_path = cache_dir / f"{name}_known_outputs.pt"
        if force or not cache_path.exists() or old_hashes.get(name) != checkpoint_hashes[name]:
            splits = ["train", "validation", "test"] if name == "vanilla" else ["validation", "test"]
            torch.save({split: extract_dataset(models[name], known_datasets[split], device,
                                               config["num_workers"], name == "proser")
                        for split in splits}, cache_path)

    score_files = sorted((task_dir / "scores").glob("*.py"))
    protocol_lock = {
        "statement": "Checkpoints and score definitions fixed before CIFAR-100 evaluation.",
        "checkpoint_sha256": checkpoint_hashes,
        "score_definition_sha256": {path.name: _sha256(path) for path in score_files},
        "selection": selection,
    }
    lock_path.write_text(json.dumps(protocol_lock, indent=2))

    # Protocol boundary: this is the first and only dataset call involving CIFAR-100.
    cifar100_test = CIFAR100(raw, train=False, download=download)
    unknown_datasets = build_unknown_datasets(cifar100_test, results_dir)
    for name, model in models.items():
        cache_path = cache_dir / f"{name}_unknown_outputs.pt"
        if force or not cache_path.exists() or old_hashes.get(name) != checkpoint_hashes[name]:
            config = load_config(task_dir, name)
            torch.save({group: extract_dataset(model, dataset, device, config["num_workers"], name == "proser")
                        for group, dataset in unknown_datasets.items()}, cache_path)

    manifest = {
        "device": str(device), "selection": selection,
        "known_train_count": len(train_idx), "known_validation_count": len(val_idx),
        "known_test_count": len(cifar10_test), "near_unknown_count": 800, "far_unknown_count": 800,
        "cifar100_train_used": False,
        "protocol_lock": str(results_dir / "protocol_lock.json"),
    }
    (results_dir / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Extract Task 4 logits/features after checkpoint lock")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(extract_all_outputs(force=args.force), indent=2))


if __name__ == "__main__":
    main()
