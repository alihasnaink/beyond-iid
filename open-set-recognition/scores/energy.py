import torch


def energy_unknownness(logits: torch.Tensor) -> torch.Tensor:
    """Unit-temperature negative log-sum-exp, as fixed by the assignment."""
    return -torch.logsumexp(logits, dim=1)
