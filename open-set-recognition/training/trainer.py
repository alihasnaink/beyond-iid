from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from methods.proser import proser_loss


@torch.inference_mode()
def validation_accuracy(model, dataset, device, config):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=config["num_workers"], pin_memory=True)
    correct, total = 0, 0
    for images, labels, _identifiers, _class_names in loader:
        logits = model(images.to(device))
        correct += logits.argmax(1).eq(labels.to(device)).sum().item()
        total += len(labels)
    return correct / total


def _optimizer_and_scheduler(model, config):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    return optimizer, scheduler


def train_closed_set(model, train_dataset, val_dataset, config, device, checkpoint_path: Path):
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"],
        pin_memory=True, worker_init_fn=_seed_worker, generator=generator,
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, config)
    best_accuracy, best_state, history = -1.0, None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels, _identifiers, _class_names in tqdm(loader, desc=f"{config['name']} epoch {epoch}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += len(labels)
        val_accuracy = validation_accuracy(model, val_dataset, device, config)
        history.append({"method": config["name"], "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0],
                        "train_loss": loss_sum / total, "train_accuracy": correct / total,
                        "validation_accuracy": val_accuracy})
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {"model": copy.deepcopy(model.state_dict()), "epoch": epoch,
                          "validation_accuracy": val_accuracy, "config": dict(config)}
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
        scheduler.step()
    model.load_state_dict(best_state["model"])
    return model.eval(), pd.DataFrame(history), best_state


def train_proser(model, train_dataset, val_dataset, config, device, checkpoint_path: Path):
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"],
        pin_memory=True, worker_init_fn=_seed_worker, generator=generator,
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, config)
    best_accuracy, best_state, history = -1.0, None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        sums = {"train_loss": 0.0, "known_ce": 0.0, "classifier_boundary_loss": 0.0,
                "data_placeholder_loss": 0.0, "mean_mix_lambda": 0.0, "different_class_pairs": 0.0}
        correct, total, batches = 0, 0, 0
        for images, labels, _identifiers, _class_names in tqdm(loader, desc=f"proser epoch {epoch}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            loss, components = proser_loss(
                model, images, labels, config["classifier_placeholder_beta"],
                config["data_placeholder_gamma"], config["mixup_beta_alpha"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums["train_loss"] += loss.item()
            for key, value in components.items(): sums[key] += value.item()
            with torch.inference_mode():
                known_logits = model(images)
                correct += known_logits.argmax(1).eq(labels).sum().item()
            total += len(labels); batches += 1
        val_accuracy = validation_accuracy(model, val_dataset, device, config)
        row = {"method": "proser", "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0],
               "train_accuracy": correct / total, "validation_accuracy": val_accuracy}
        row.update({key: value / batches for key, value in sums.items()})
        history.append(row)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {"model": copy.deepcopy(model.state_dict()), "epoch": epoch,
                          "validation_accuracy": val_accuracy, "config": dict(config)}
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
        scheduler.step()
    model.load_state_dict(best_state["model"])
    return model.eval(), pd.DataFrame(history), best_state


def _seed_worker(worker_id):
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
