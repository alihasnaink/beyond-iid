from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from methods.sam import ascent_perturbation, restore_parameters


def fixed_validation_batches(val_datasets, seed=6304, examples_per_domain=32):
    """Materialize the exact source-only diagnostic batch without loading Sketch."""
    batches = []
    for offset, (domain, dataset) in enumerate(val_datasets.items()):
        if len(dataset) < examples_per_domain:
            raise ValueError(f"{domain} has only {len(dataset)} validation examples; need {examples_per_domain}.")
        rng = np.random.default_rng(seed + offset)
        indices = rng.choice(len(dataset), examples_per_domain, replace=False)
        examples = [dataset[int(index)] for index in indices]
        images = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples], dtype=torch.long)
        batches.append((domain, images, labels))
    return batches


def _mean_batch_loss(model, batches, device, backward=False):
    total = 0.0
    for _domain, images, labels in batches:
        logits = model(images.to(device), return_features=False)
        loss = F.cross_entropy(logits, labels.to(device)) / len(batches)
        if backward:
            loss.backward()
        total += loss.detach().item()
    return total


def local_sharpness_proxy(model, batches, device, radius=0.05):
    """Loss increase after the PDF-specified normalized ascent perturbation."""
    model.eval()
    model.zero_grad(set_to_none=True)
    base_loss = _mean_batch_loss(model, batches, device, backward=True)
    perturbations = ascent_perturbation(model.parameters(), radius)
    try:
        with torch.no_grad():
            perturbed_loss = _mean_batch_loss(model, batches, device, backward=False)
    finally:
        restore_parameters(perturbations)
        model.zero_grad(set_to_none=True)
    return {
        "base_validation_loss": base_loss,
        "perturbed_validation_loss": perturbed_loss,
        "sharpness_increase": perturbed_loss - base_loss,
        "radius": float(radius),
    }

