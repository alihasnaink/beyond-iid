from __future__ import annotations

import torch


def _pairwise_squared_distance(x, y):
    return (x.pow(2).sum(1, keepdim=True) + y.pow(2).sum(1).unsqueeze(0) - 2 * x @ y.T).clamp_min(0)


def mmd_rbf(source, target, multipliers=(0.5, 1.0, 2.0)):
    """Biased, nonnegative multi-kernel MMD using current-batch median squared distance."""
    combined = torch.cat([source, target], dim=0)
    distances = _pairwise_squared_distance(combined, combined)
    off_diagonal = distances[~torch.eye(len(combined), dtype=torch.bool, device=distances.device)]
    bandwidth = off_diagonal.detach().median().clamp_min(1e-6)

    def kernel(x, y):
        squared = _pairwise_squared_distance(x, y)
        return sum(torch.exp(-squared / (2 * bandwidth * multiplier)) for multiplier in multipliers)

    return kernel(source, source).mean() + kernel(target, target).mean() - 2 * kernel(source, target).mean()
