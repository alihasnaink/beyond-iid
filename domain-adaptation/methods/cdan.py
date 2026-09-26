from __future__ import annotations

import torch


def conditional_outer_product(features, probabilities):
    """vec(f outer p), retaining gradients through both f and p."""
    return torch.bmm(features.unsqueeze(2), probabilities.unsqueeze(1)).flatten(1)
