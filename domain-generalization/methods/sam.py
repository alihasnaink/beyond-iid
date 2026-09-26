from __future__ import annotations

import torch


@torch.no_grad()
def ascent_perturbation(parameters, rho, epsilon=1e-12):
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        raise RuntimeError("SAM cannot perturb a model with no gradients.")
    norm = torch.linalg.vector_norm(torch.stack([parameter.grad.norm(2) for parameter in parameters]))
    scale = float(rho) / (norm + epsilon)
    perturbations = []
    for parameter in parameters:
        perturbation = parameter.grad * scale
        parameter.add_(perturbation)
        perturbations.append((parameter, perturbation))
    return perturbations


@torch.no_grad()
def restore_parameters(perturbations):
    for parameter, perturbation in perturbations:
        parameter.sub_(perturbation)

