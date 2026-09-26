from __future__ import annotations

import copy
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from methods.dan_dg import pairwise_source_mmd
from methods.erm import erm_loss
from methods.sam import ascent_perturbation, restore_parameters
from models.backbone import freeze_batchnorm_stats
from selection.source_validation import aggregate_source_metrics, evaluate_source_domains


def _next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _atomic_torch_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _source_batch(source_loaders, iterators, device):
    images_by_domain, labels_by_domain = [], []
    for domain, loader in source_loaders.items():
        batch, iterators[domain] = _next_or_restart(iterators[domain], loader)
        images, labels, _identifiers = batch
        images_by_domain.append(images.to(device, non_blocking=True))
        labels_by_domain.append(labels.to(device, non_blocking=True))
    return images_by_domain, labels_by_domain


def _forward_source(model, images_by_domain, labels_by_domain):
    sizes = [len(images) for images in images_by_domain]
    images = torch.cat(images_by_domain)
    labels = torch.cat(labels_by_domain)
    logits, features = model(images, return_features=True)
    return erm_loss(logits, labels), list(features.split(sizes))


def _standard_step(model, optimizer, method, config, images_by_domain, labels_by_domain):
    classification_loss, features_by_domain = _forward_source(model, images_by_domain, labels_by_domain)
    alignment_loss = classification_loss.new_zeros(())
    if method == "dan_dg":
        alignment_loss = pairwise_source_mmd(features_by_domain, config["mmd_bandwidth_multipliers"])
    total_loss = classification_loss + config.get("alignment_weight", 0.0) * alignment_loss
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    return classification_loss.item(), alignment_loss.item(), total_loss.item()


def _sam_step(model, optimizer, config, images_by_domain, labels_by_domain):
    optimizer.zero_grad(set_to_none=True)
    first_loss, _ = _forward_source(model, images_by_domain, labels_by_domain)
    first_loss.backward()
    perturbations = ascent_perturbation(model.parameters(), config["rho"])
    try:
        optimizer.zero_grad(set_to_none=True)
        second_loss, _ = _forward_source(model, images_by_domain, labels_by_domain)
        second_loss.backward()
    finally:
        restore_parameters(perturbations)
    optimizer.step()
    return first_loss.item(), 0.0, second_loss.item()


def train_method(model, method, config, source_loaders, val_datasets, device, checkpoint_path: Path, live_history_path: Path):
    if method not in {"dan_dg", "sam"}:
        raise ValueError(f"Task 3 only trains dan_dg and sam; got {method!r}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    steps_per_epoch = max(len(loader) for loader in source_loaders.values())
    best_f1, best_state, stale, history = -1.0, None, 0, []

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        model.apply(freeze_batchnorm_stats)
        iterators = {domain: iter(loader) for domain, loader in source_loaders.items()}
        sums = {"classification_loss": 0.0, "alignment_loss": 0.0, "total_loss": 0.0}
        batches = tqdm(
            range(steps_per_epoch),
            desc=f"{config['run_name']} epoch {epoch}/{config['max_epochs']}",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in batches:
            images_by_domain, labels_by_domain = _source_batch(source_loaders, iterators, device)
            if method == "sam":
                values = _sam_step(model, optimizer, config, images_by_domain, labels_by_domain)
            else:
                values = _standard_step(model, optimizer, method, config, images_by_domain, labels_by_domain)
            for key, value in zip(sums, values):
                sums[key] += value
            batches.set_postfix(loss=f"{values[2]:.3f}", refresh=False)

        validation, _ = evaluate_source_domains(model, val_datasets, device, config["num_workers"])
        aggregate = aggregate_source_metrics(validation)
        mean_f1 = aggregate["mean_source_macro_f1"]
        row = {"method": config["run_name"], "epoch": epoch, **aggregate}
        row.update({key: value / steps_per_epoch for key, value in sums.items()})
        for domain, metrics in validation.items():
            row[f"val_{domain}_accuracy"] = metrics["accuracy"]
            row[f"val_{domain}_macro_f1"] = metrics["macro_f1"]
        history.append(row)
        live_history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(live_history_path, index=False)
        print(
            f"  {config['run_name']} epoch {epoch}/{config['max_epochs']}: "
            f"loss={row['total_loss']:.4f}, mean-source macro-F1={mean_f1:.4f}, "
            f"best={max(best_f1, mean_f1):.4f}",
            flush=True,
        )

        if mean_f1 > best_f1 + 1e-12:
            best_f1, stale = mean_f1, 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "mean_source_val_macro_f1": mean_f1,
                "config": dict(config),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(best_state, checkpoint_path)
        else:
            stale += 1
            if stale >= config["patience"]:
                print(f"  {config['run_name']}: early stopping after epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"{config['run_name']} did not produce a valid checkpoint.")
    model.load_state_dict(best_state["model"])
    return model.eval(), pd.DataFrame(history), best_state

