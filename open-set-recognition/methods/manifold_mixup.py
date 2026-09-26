from __future__ import annotations

import torch


def different_class_permutation(labels: torch.Tensor, generator=None):
    if labels.unique().numel() < 2:
        raise ValueError("Manifold mixup requires at least two classes in the data-placeholder half-batch.")
    permutation = []
    for label in labels:
        candidates = torch.nonzero(labels != label, as_tuple=False).flatten()
        choice = torch.randint(len(candidates), (1,), device=labels.device, generator=generator)
        permutation.append(candidates[choice].squeeze(0))
    return torch.stack(permutation)


def mix_different_classes(features, labels, alpha=2.0):
    permutation = different_class_permutation(labels)
    concentration = torch.full((len(features),), float(alpha), device=features.device)
    lambdas = torch.distributions.Beta(concentration, concentration).sample().view(-1, 1, 1, 1)
    mixed = lambdas * features + (1 - lambdas) * features[permutation]
    return mixed, permutation, lambdas.flatten()
