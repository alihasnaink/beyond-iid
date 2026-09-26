from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from evaluation.metrics import evaluate_domains
from methods.cdan import conditional_outer_product
from methods.dan import mmd_rbf
from methods.dann import gradient_reverse, grl_schedule
from models.backbone import freeze_batchnorm_stats
from models.domain_discriminator import DomainDiscriminator


def _next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _domain_discriminator(method, config, device):
    if method == "dann":
        input_dim = 512
    elif method == "cdan":
        input_dim = 512 * config["num_classes"]
    else:
        return None
    return DomainDiscriminator(input_dim, config["discriminator_hidden"], config["discriminator_dropout"]).to(device)


def train_method(model, method, config, source_loaders, target_loader, val_datasets, device, checkpoint_path: Path):
    discriminator = _domain_discriminator(method, config, device)
    parameters = list(model.parameters()) + ([] if discriminator is None else list(discriminator.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    steps_per_epoch = max(len(loader) for loader in source_loaders.values())
    total_budget_steps = config["max_epochs"] * steps_per_epoch
    best_f1, best_state, stale, history, global_step = -1.0, None, 0, [], 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        model.apply(freeze_batchnorm_stats)
        if discriminator is not None:
            discriminator.train()
        source_iterators = {domain: iter(loader) for domain, loader in source_loaders.items()}
        target_iterator = iter(target_loader) if target_loader is not None else None
        sums = {"classification_loss": 0.0, "alignment_loss": 0.0, "total_loss": 0.0, "domain_accuracy": 0.0}

        batches = tqdm(
            range(steps_per_epoch),
            desc=f"{config['run_name']} epoch {epoch}/{config['max_epochs']}",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in batches:
            source_images, source_labels = [], []
            for domain, loader in source_loaders.items():
                batch, source_iterators[domain] = _next_or_restart(source_iterators[domain], loader)
                images, labels, _ = batch
                source_images.append(images)
                source_labels.append(labels)
            source_images = torch.cat(source_images).to(device, non_blocking=True)
            source_labels = torch.cat(source_labels).to(device, non_blocking=True)
            source_logits, source_features = model(source_images, return_features=True)
            classification_loss = F.cross_entropy(source_logits, source_labels)
            alignment_loss = source_features.new_zeros(())
            domain_accuracy = 0.0
            progress = global_step / max(total_budget_steps - 1, 1)

            if method != "source_only":
                target_batch, target_iterator = _next_or_restart(target_iterator, target_loader)
                target_images, _target_identifiers = target_batch
                target_images = target_images.to(device, non_blocking=True)
                target_logits, target_features = model(target_images, return_features=True)
                if method == "dan":
                    alignment_loss = mmd_rbf(source_features, target_features, config["mmd_bandwidth_multipliers"])
                else:
                    strength = grl_schedule(progress, config["grl_max_strength"])
                    combined_features = torch.cat([source_features, target_features])
                    if method == "dann":
                        discriminator_input = combined_features
                    elif method == "cdan":
                        combined_logits = torch.cat([source_logits, target_logits])
                        discriminator_input = conditional_outer_product(combined_features, combined_logits.softmax(1))
                    else:
                        raise ValueError(method)
                    domain_logits = discriminator(gradient_reverse(discriminator_input, strength))
                    domain_labels = torch.cat([
                        torch.zeros(len(source_features), dtype=torch.long, device=device),
                        torch.ones(len(target_features), dtype=torch.long, device=device),
                    ])
                    alignment_loss = F.cross_entropy(domain_logits, domain_labels)
                    domain_accuracy = domain_logits.argmax(1).eq(domain_labels).float().mean().item()

            total_loss = classification_loss + config["alignment_weight"] * alignment_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            sums["classification_loss"] += classification_loss.item()
            sums["alignment_loss"] += alignment_loss.item()
            sums["total_loss"] += total_loss.item()
            sums["domain_accuracy"] += domain_accuracy
            global_step += 1
            batches.set_postfix(loss=f"{total_loss.item():.3f}", refresh=False)

        validation = evaluate_domains(model, val_datasets, device, num_workers=config["num_workers"])
        mean_val_f1 = sum(metrics["macro_f1"] for metrics in validation.values()) / len(validation)
        row = {"method": config["run_name"], "epoch": epoch, "mean_source_val_macro_f1": mean_val_f1}
        row.update({key: value / steps_per_epoch for key, value in sums.items()})
        for domain, metrics in validation.items():
            row[f"val_{domain}_accuracy"] = metrics["accuracy"]
            row[f"val_{domain}_macro_f1"] = metrics["macro_f1"]
        history.append(row)
        print(
            f"  {config['run_name']} epoch {epoch}/{config['max_epochs']}: "
            f"loss={row['total_loss']:.4f}, source-val macro-F1={mean_val_f1:.4f}, "
            f"best={max(best_f1, mean_val_f1):.4f}",
            flush=True,
        )

        if mean_val_f1 > best_f1 + 1e-12:
            best_f1, stale = mean_val_f1, 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "discriminator": None if discriminator is None else copy.deepcopy(discriminator.state_dict()),
                "epoch": epoch, "mean_source_val_macro_f1": mean_val_f1, "config": dict(config),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
        else:
            stale += 1
            if stale >= config["patience"]:
                break

    model.load_state_dict(best_state["model"])
    return model.eval(), pd.DataFrame(history), best_state
